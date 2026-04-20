//! High-level orchestration: given a camera frame, produce a set of
//! face-swapped paintings ranked by similarity to the user.
//!
//! Runs on a dedicated worker thread so the UI thread never blocks.

use crate::face::{largest_face, Face, FaceAnalyzer};
use crate::paintings::{self, Painting};
use crate::swap::Swapper;
use anyhow::{anyhow, Result};
use crossbeam_channel::{bounded, Receiver, Sender};
use image::RgbImage;
use std::path::PathBuf;
use std::thread;

/// One finished result shown in the UI.
pub struct SwapResult {
    pub painting_title: String,
    pub painting_artist: String,
    pub similarity: f32,
    pub image: RgbImage,
}

/// Request sent from the UI to the worker.
pub struct SwapRequest {
    pub frame: RgbImage,
    pub top_n: usize,
}

/// Anything the UI can ask the worker to do.
pub enum WorkerReq {
    Swap(SwapRequest),
    /// Re-scan `assets/paintings/`, rebuild embeddings cache, replace
    /// the in-memory gallery. Sent after the starter downloader finishes.
    Reindex,
}

/// Message from the worker back to the UI.
pub enum WorkerMsg {
    /// Free-text status line + optional (done, total) progress tuple.
    Progress { text: String, done: Option<(usize, usize)> },
    /// The 112×112 aligned ArcFace crop of the captured user face, sent once
    /// per capture as soon as it's been extracted.
    UserFace(RgbImage),
    Results(Vec<SwapResult>),
    Error(String),
}

impl WorkerMsg {
    pub fn text(s: impl Into<String>) -> Self {
        WorkerMsg::Progress { text: s.into(), done: None }
    }
    pub fn step(s: impl Into<String>, done: usize, total: usize) -> Self {
        WorkerMsg::Progress { text: s.into(), done: Some((done, total)) }
    }
}

pub struct Worker {
    pub tx_req: Sender<WorkerReq>,
    pub rx_msg: Receiver<WorkerMsg>,
}

pub fn spawn(models_dir: PathBuf, paintings_dir: PathBuf, cache_path: PathBuf) -> Worker {
    let (tx_req, rx_req) = bounded::<WorkerReq>(4);
    let (tx_msg, rx_msg) = bounded::<WorkerMsg>(16);

    thread::Builder::new()
        .name("pipeline".into())
        .spawn(move || {
            if let Err(err) = run(rx_req, tx_msg.clone(), models_dir, paintings_dir, cache_path) {
                let _ = tx_msg.send(WorkerMsg::Error(format!("{err:?}")));
            }
        })
        .expect("spawn pipeline worker");

    Worker { tx_req, rx_msg }
}

fn run(
    rx: Receiver<WorkerReq>,
    tx: Sender<WorkerMsg>,
    models_dir: PathBuf,
    paintings_dir: PathBuf,
    cache_path: PathBuf,
) -> Result<()> {
    let _ = tx.send(WorkerMsg::text("Loading models..."));
    let mut analyzer = FaceAnalyzer::load(&models_dir)?;
    let mut swapper = Swapper::load(&models_dir)?;

    let _ = tx.send(WorkerMsg::text("Indexing paintings..."));
    // Don't fail if paintings dir is empty — the starter downloader may
    // still be running. We'll pick them up on a later Reindex message.
    let mut gallery = paintings::load_or_build(&paintings_dir, &cache_path, &mut analyzer)
        .unwrap_or_default();
    let _ = tx.send(if gallery.is_empty() {
        WorkerMsg::text("Waiting for paintings to download...")
    } else {
        WorkerMsg::text(format!("Ready ({} paintings).", gallery.len()))
    });

    // In-process cache of decoded painting RgbImages keyed by name. First
    // time a painting is used we decode its JPEG (~50 ms); subsequent
    // captures reuse the cached buffer.
    let mut image_cache: std::collections::HashMap<String, RgbImage> =
        std::collections::HashMap::new();

    while let Ok(work) = rx.recv() {
        let req = match work {
            WorkerReq::Reindex => {
                if gallery.is_empty() {
                    // First time: do the (potentially) full build.
                    gallery = paintings::load_or_build(
                        &paintings_dir,
                        &cache_path,
                        &mut analyzer,
                    )
                    .unwrap_or_default();
                } else {
                    // Incremental: only touch paintings not yet indexed.
                    match paintings::append_new(
                        &mut gallery,
                        &paintings_dir,
                        &cache_path,
                        &mut analyzer,
                    ) {
                        Ok(n) if n > 0 => log::info!("appended {n} paintings to gallery"),
                        Ok(_) => {}
                        Err(e) => log::warn!("append_new failed: {e:?}"),
                    }
                }
                let _ = tx.send(WorkerMsg::text(format!(
                    "Ready ({} paintings).",
                    gallery.len()
                )));
                continue;
            }
            WorkerReq::Swap(r) => r,
        };

        if gallery.is_empty() {
            let _ = tx.send(WorkerMsg::Error(
                "No paintings indexed yet — let the starter download finish.".into(),
            ));
            continue;
        }

        let _ = tx.send(WorkerMsg::text("Detecting your face..."));
        let faces = analyzer.detect(&req.frame)?;
        let Some(user_face) = largest_face(&faces).cloned() else {
            let _ = tx.send(WorkerMsg::Error("No face detected. Move closer.".into()));
            continue;
        };

        let _ = tx.send(WorkerMsg::text("Computing embedding..."));
        let (embedding, user_crop) = analyzer.embed_with_crop(&req.frame, &user_face)?;
        let _ = tx.send(WorkerMsg::UserFace(user_crop));

        let ranked = paintings::rank(&embedding, &gallery);
        let selected: Vec<_> = ranked.into_iter().take(req.top_n).collect();
        let total = selected.len();

        // Pre-decode in parallel any paintings we don't yet have cached.
        // This overlaps disk I/O + JPEG decode with the first inswapper run.
        use rayon::prelude::*;
        let missing: Vec<&crate::paintings::Painting> = selected
            .iter()
            .map(|(i, _)| &gallery[*i])
            .filter(|p| !image_cache.contains_key(&p.name))
            .collect();
        if !missing.is_empty() {
            let _ = tx.send(WorkerMsg::step(
                format!("Preloading {} paintings...", missing.len()),
                0,
                total,
            ));
            let decoded: Vec<(String, Result<RgbImage>)> = missing
                .par_iter()
                .map(|p| {
                    (
                        p.name.clone(),
                        paintings::load_image(p).map_err(|e| anyhow!("{e:?}")),
                    )
                })
                .collect();
            for (name, img) in decoded {
                match img {
                    Ok(i) => {
                        image_cache.insert(name, i);
                    }
                    Err(e) => log::warn!("preload {name}: {e:?}"),
                }
            }
        }

        // Pre-compute the source latent once per capture (it's the same for
        // every painting in this batch; previously we were recomputing a
        // 512×512 matmul per swap).
        let latent = swapper.latent_from_embedding(&embedding);

        let t_batch = std::time::Instant::now();
        let mut timings = crate::swap::Timings::default();
        let mut results = Vec::with_capacity(selected.len());
        for (done, (i, score)) in selected.into_iter().enumerate() {
            let p = &gallery[i];
            // Fixed-width status text so the UI doesn't jiggle.
            let _ = tx.send(WorkerMsg::step(
                format!("Swapping {}/{total}", done + 1),
                done,
                total,
            ));
            let base = match image_cache.get(&p.name) {
                Some(b) => b,
                None => match paintings::load_image(p) {
                    Ok(img) => {
                        image_cache.insert(p.name.clone(), img);
                        image_cache.get(&p.name).unwrap()
                    }
                    Err(e) => {
                        log::warn!("load {}: {e:?}", p.name);
                        continue;
                    }
                },
            };
            // Iterate every detected face in this painting, chaining swaps
            // so group portraits get all heads replaced.
            let mut current: Option<RgbImage> = None;
            for face_meta in &p.faces {
                let target_face: Face = face_meta.into();
                let input: &RgbImage = current.as_ref().unwrap_or(base);
                match swapper.swap_with_latent(
                    input,
                    &target_face,
                    &latent,
                    Some(&mut timings),
                ) {
                    Ok(img) => current = Some(img),
                    Err(e) => log::warn!(
                        "swap failed for {} face: {e:?}",
                        p.title
                    ),
                }
            }
            if let Some(img) = current {
                results.push(SwapResult {
                    painting_title: p.title.clone(),
                    painting_artist: p.artist.clone(),
                    similarity: score,
                    image: img,
                });
            }
        }
        let wall = t_batch.elapsed();
        if timings.count > 0 {
            let c = timings.count as f32;
            log::info!(
                "batch {} paintings in {:.1}s  —  per-painting avg: align {:.0}ms  inference {:.0}ms  paste {:.0}ms",
                timings.count,
                wall.as_secs_f32(),
                timings.align.as_secs_f32() * 1000.0 / c,
                timings.infer.as_secs_f32() * 1000.0 / c,
                timings.paste.as_secs_f32() * 1000.0 / c,
            );
            let _ = tx.send(WorkerMsg::text(format!(
                "Done in {:.1}s — inference avg {:.0}ms, paste {:.0}ms, align {:.0}ms",
                wall.as_secs_f32(),
                timings.infer.as_secs_f32() * 1000.0 / c,
                timings.paste.as_secs_f32() * 1000.0 / c,
                timings.align.as_secs_f32() * 1000.0 / c,
            )));
        }

        let _ = tx.send(WorkerMsg::Results(results));
    }

    Ok(())
}
