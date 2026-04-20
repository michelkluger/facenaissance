//! facenaissance — fit your face into classical paintings.
//!
//! Architecture (single-binary, pure Rust):
//!   camera frames  →  face detect (SCRFD)  →  arcface embedding
//!      →  rank painting gallery by cosine similarity
//!      →  face-swap into top-N paintings (inswapper_128)
//!      →  egui gallery UI
//!
//! All inference runs through `ort` (ONNX Runtime). See `models/README.md`
//! for where to obtain the ONNX files.

mod align;
mod app;
mod camera;
mod downloader;
mod face;
mod paintings;
mod pipeline;
mod swap;

use anyhow::Result;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 820.0])
            .with_title("facenaissance — fit your face into classical paintings"),
        ..Default::default()
    };

    eframe::run_native(
        "facenaissance",
        native_options,
        Box::new(|cc| Ok(Box::new(app::FacenaissanceApp::new(cc)?))),
    )
    .map_err(|e| anyhow::anyhow!("eframe: {e}"))
}
