//! Camera capture thread.
//!
//! Spawns a background thread pulling frames from the first available camera
//! via `nokhwa`. The most recent frame is made available through a single-slot
//! mailbox (`parking_lot::Mutex<Option<RgbImage>>`) so the UI never blocks
//! on camera I/O and always reads the freshest frame.

use anyhow::{Context, Result};
use image::RgbImage;
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};
use parking_lot::Mutex;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Shared handle to the most recent camera frame.
#[derive(Clone, Default)]
pub struct CameraFeed {
    inner: Arc<Mutex<Option<RgbImage>>>,
    stop: Arc<Mutex<bool>>,
}

impl CameraFeed {
    pub fn latest(&self) -> Option<RgbImage> {
        self.inner.lock().clone()
    }

    pub fn stop(&self) {
        *self.stop.lock() = true;
    }
}

/// Start a background capture thread. Returns immediately; frames become
/// available through the returned `CameraFeed` handle.
pub fn start() -> Result<CameraFeed> {
    // nokhwa requires an init call on some platforms (macOS). Harmless elsewhere.
    nokhwa::nokhwa_initialize(|_granted| {});

    let feed = CameraFeed::default();
    let slot = feed.inner.clone();
    let stop = feed.stop.clone();

    thread::Builder::new()
        .name("camera".into())
        .spawn(move || {
            if let Err(err) = run(slot, stop) {
                log::error!("camera thread died: {err:?}");
            }
        })
        .context("spawn camera thread")?;

    Ok(feed)
}

fn run(slot: Arc<Mutex<Option<RgbImage>>>, stop: Arc<Mutex<bool>>) -> Result<()> {
    let index = CameraIndex::Index(0);
    let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);

    let mut camera = Camera::new(index, format).context("open camera 0")?;
    camera.open_stream().context("open camera stream")?;
    log::info!(
        "camera opened: {}x{} @ {}fps",
        camera.resolution().width_x,
        camera.resolution().height_y,
        camera.frame_rate()
    );

    loop {
        if *stop.lock() {
            break;
        }
        match camera.frame() {
            Ok(frame) => match frame.decode_image::<RgbFormat>() {
                Ok(rgb) => {
                    *slot.lock() = Some(rgb);
                }
                Err(e) => log::warn!("decode: {e}"),
            },
            Err(e) => {
                log::warn!("frame: {e}");
                thread::sleep(Duration::from_millis(50));
            }
        }
    }

    let _ = camera.stop_stream();
    Ok(())
}
