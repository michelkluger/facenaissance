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

/// Per-capture timing summary, attached to the Results message so the UI
/// can show how long the batch took and where the time went.
#[derive(Clone, Copy, Debug, Default)]
pub struct BatchTiming {
    pub wall_ms: u32,
    pub count: u32,
    pub align_avg_ms: u32,
    pub infer_avg_ms: u32,
    pub paste_avg_ms: u32,
}

/// Message from the worker back to the UI.
pub enum WorkerMsg {
    /// Free-text status line + optional (done, total) progress tuple.
    Progress { text: String, done: Option<(usize, usize)> },
    /// The 112×112 aligned ArcFace crop of the captured user face, sent once
    /// per capture as soon as it's been extracted.
    UserFace(RgbImage),
    Results {
        items: Vec<SwapResult>,
        timing: BatchTiming,
    },
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

    // Pool of inswapper sessions. On CPU we run two sessions in parallel,
    // splitting the cores between them. On GPU (Windows + DirectML) we drop
    // to a single session — two concurrent sessions would just serialise on
    // the GPU command queue while still fighting over CPU-side prep work.
    let total_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    #[cfg(target_os = "windows")]
    let using_gpu = std::env::var("FACENAISSANCE_NO_DML")
        .map(|v| !(v == "1" || v.eq_ignore_ascii_case("true")))
        .unwrap_or(true);
    #[cfg(not(target_os = "windows"))]
    let using_gpu = false;

    let pool_size = if using_gpu {
        1
    } else if total_cores >= 4 {
        2
    } else {
        1
    };
    let per_session_threads = (total_cores / pool_size).max(1);
    let mut swapper_pool: Vec<Swapper> = Vec::with_capacity(pool_size);
    for _ in 0..pool_size {
        swapper_pool.push(Swapper::load_with_threads(&models_dir, per_session_threads)?);
    }
    log::info!(
        "inswapper pool: {} session(s), {} intra-op threads each{}",
        pool_size,
        per_session_threads,
        if using_gpu { " (GPU/DML)" } else { " (CPU)" }
    );

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
        let latent = swapper_pool[0].latent_from_embedding(&embedding);

        // Any painting still missing from the cache (e.g. preload failed) is
        // loaded now — we need the cache to be read-only across the parallel
        // shard threads below.
        let selected_vec: Vec<(usize, f32)> = selected.into_iter().collect();
        for (i, _) in &selected_vec {
            let p = &gallery[*i];
            if !image_cache.contains_key(&p.name) {
                match paintings::load_image(p) {
                    Ok(img) => {
                        image_cache.insert(p.name.clone(), img);
                    }
                    Err(e) => log::warn!("load {}: {e:?}", p.name),
                }
            }
        }

        // Partition the selected paintings round-robin across the swapper
        // pool. Each shard runs on its own scoped thread using its own ONNX
        // session, so inswapper inference overlaps across paintings.
        let pool_n = swapper_pool.len();
        let mut shards: Vec<Vec<(usize, usize, f32)>> =
            (0..pool_n).map(|_| Vec::new()).collect();
        for (ordinal, (i, score)) in selected_vec.into_iter().enumerate() {
            shards[ordinal % pool_n].push((ordinal, i, score));
        }

        let t_batch = std::time::Instant::now();
        let done_counter = std::sync::atomic::AtomicUsize::new(0);
        let results_mu: std::sync::Mutex<Vec<(usize, SwapResult)>> =
            std::sync::Mutex::new(Vec::with_capacity(total));
        let timings_mu = std::sync::Mutex::new(crate::swap::Timings::default());

        std::thread::scope(|s| {
            for (shard, sw) in shards.into_iter().zip(swapper_pool.iter_mut()) {
                let tx = tx.clone();
                let gallery_ref: &Vec<paintings::Painting> = &gallery;
                let cache_ref = &image_cache;
                let latent_ref = &latent;
                let done_counter = &done_counter;
                let results_mu = &results_mu;
                let timings_mu = &timings_mu;
                s.spawn(move || {
                    let mut local_timings = crate::swap::Timings::default();
                    for (ordinal, i, score) in shard {
                        let p = &gallery_ref[i];
                        let done = done_counter
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let _ = tx.send(WorkerMsg::step(
                            format!("Swapping {}/{total}", done + 1),
                            done,
                            total,
                        ));
                        let Some(base) = cache_ref.get(&p.name) else {
                            log::warn!("skip {} (not loaded)", p.name);
                            continue;
                        };
                        let mut current: Option<RgbImage> = None;
                        for face_meta in &p.faces {
                            let target_face: Face = face_meta.into();
                            let input: &RgbImage = current.as_ref().unwrap_or(base);
                            match sw.swap_with_latent(
                                input,
                                &target_face,
                                latent_ref,
                                Some(&mut local_timings),
                            ) {
                                Ok(img) => current = Some(img),
                                Err(e) => log::warn!(
                                    "swap failed for {} face: {e:?}",
                                    p.title
                                ),
                            }
                        }
                        if let Some(img) = current {
                            results_mu.lock().unwrap().push((
                                ordinal,
                                SwapResult {
                                    painting_title: p.title.clone(),
                                    painting_artist: p.artist.clone(),
                                    similarity: score,
                                    image: img,
                                },
                            ));
                        }
                    }
                    let mut g = timings_mu.lock().unwrap();
                    g.align += local_timings.align;
                    g.infer += local_timings.infer;
                    g.paste += local_timings.paste;
                    g.count += local_timings.count;
                });
            }
        });

        let mut collected = results_mu.into_inner().unwrap();
        collected.sort_by_key(|(o, _)| *o);
        let results: Vec<SwapResult> =
            collected.into_iter().map(|(_, r)| r).collect();
        let timings = timings_mu.into_inner().unwrap();
        let wall = t_batch.elapsed();
        let batch_timing = if timings.count > 0 {
            let c = timings.count as f32;
            let t = BatchTiming {
                wall_ms: wall.as_millis() as u32,
                count: timings.count as u32,
                align_avg_ms: (timings.align.as_secs_f32() * 1000.0 / c).round() as u32,
                infer_avg_ms: (timings.infer.as_secs_f32() * 1000.0 / c).round() as u32,
                paste_avg_ms: (timings.paste.as_secs_f32() * 1000.0 / c).round() as u32,
            };
            log::info!(
                "batch {} paintings in {:.1}s  —  per-painting avg: align {}ms  inference {}ms  paste {}ms",
                t.count, wall.as_secs_f32(), t.align_avg_ms, t.infer_avg_ms, t.paste_avg_ms,
            );
            t
        } else {
            BatchTiming {
                wall_ms: wall.as_millis() as u32,
                ..Default::default()
            }
        };

        let _ = tx.send(WorkerMsg::Results {
            items: results,
            timing: batch_timing,
        });
    }

    Ok(())
}
