//! Painting gallery.
//!
//! At startup we scan `assets/paintings/` for `<name>.jpg` (+ optional
//! `<name>.json` metadata), detect the face in each painting, compute its
//! ArcFace embedding, and cache the result to `cache/paintings.json` so we
//! don't pay the cost on every launch.
//!
//! At runtime we rank paintings by cosine similarity to the live user's
//! embedding and expose the top-N for face-swapping.

use crate::face::{cosine_similarity, Face, FaceAnalyzer};
use anyhow::{Context, Result};
use image::RgbImage;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Painting {
    pub name: String,
    pub title: String,
    pub artist: String,
    pub path: PathBuf,
    pub face: PaintingFace,
    pub embedding: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaintingFace {
    pub bbox: [f32; 4],
    pub landmarks: [[f32; 2]; 5],
    pub score: f32,
}

impl From<&Face> for PaintingFace {
    fn from(f: &Face) -> Self {
        Self {
            bbox: f.bbox,
            landmarks: f.landmarks,
            score: f.score,
        }
    }
}

impl From<&PaintingFace> for Face {
    fn from(f: &PaintingFace) -> Self {
        Face {
            bbox: f.bbox,
            landmarks: f.landmarks,
            score: f.score,
        }
    }
}

#[derive(Serialize, Deserialize, Default)]
struct Manifest {
    title: Option<String>,
    artist: Option<String>,
}

pub fn load_or_build(
    paintings_dir: &Path,
    cache_path: &Path,
    analyzer: &mut FaceAnalyzer,
) -> Result<Vec<Painting>> {
    if let Ok(bytes) = std::fs::read(cache_path) {
        if let Ok(mut cached) = serde_json::from_slice::<Vec<Painting>>(&bytes) {
            // Old caches may have stored relative paths (resolved from a
            // specific cwd). Always rewrite to absolute `paintings_dir /
            // <filename>` so the cache is portable across launch dirs.
            let mut migrated = 0usize;
            for p in cached.iter_mut() {
                let needs_fix = !p.path.is_absolute() || !p.path.exists();
                if needs_fix {
                    if let Some(fname) = p.path.file_name() {
                        let candidate = paintings_dir.join(fname);
                        if candidate.exists() {
                            p.path = candidate;
                            migrated += 1;
                        }
                    }
                }
            }
            if migrated > 0 {
                log::info!(
                    "migrated {} painting paths to absolute, rewriting cache",
                    migrated
                );
                if let Some(parent) = cache_path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let _ = std::fs::write(cache_path, serde_json::to_vec_pretty(&cached)?);
            }
            if !cached.is_empty() && cached.iter().all(|p| p.path.exists()) {
                log::info!("loaded {} paintings from cache", cached.len());
                return Ok(cached);
            }
            log::warn!("cache paths still invalid, reindexing from scratch");
        }
    }

    let mut out = Vec::new();
    let entries = std::fs::read_dir(paintings_dir)
        .with_context(|| format!("read {}", paintings_dir.display()))?;

    let mut rejected = 0usize;
    for entry in entries {
        let Ok(entry) = entry else {
            continue;
        };
        let path = entry.path();
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        if !matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png") {
            continue;
        }
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("painting")
            .to_string();
        log::debug!("processing {}", name);

        let img = match image::open(&path) {
            Ok(i) => i.to_rgb8(),
            Err(e) => {
                log::warn!("cannot decode {}: {e}", path.display());
                rejected += 1;
                continue;
            }
        };
        let faces = match analyzer.detect(&img) {
            Ok(f) => f,
            Err(e) => {
                log::warn!("detect failed on {}: {e:?}", path.display());
                rejected += 1;
                continue;
            }
        };
        let Some(face) = crate::face::largest_face(&faces).cloned() else {
            rejected += 1;
            continue;
        };
        let embedding = match analyzer.embed(&img, &face) {
            Ok(e) => e,
            Err(e) => {
                log::warn!("embed failed on {}: {e:?}", path.display());
                rejected += 1;
                continue;
            }
        };

        let manifest_path = path.with_extension("json");
        let manifest: Manifest = std::fs::read(&manifest_path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();

        out.push(Painting {
            name: name.clone(),
            title: manifest.title.unwrap_or_else(|| prettify(&name)),
            artist: manifest.artist.unwrap_or_else(|| "Unknown".into()),
            path,
            face: (&face).into(),
            embedding,
        });
        if out.len() % 25 == 0 {
            log::info!("indexed {} so far ({} rejected)...", out.len(), rejected);
        }
    }
    log::info!("raw indexing: {} kept, {} rejected", out.len(), rejected);

    // Dedupe. Two mechanisms:
    //   1. Normalised title match — the same portrait often exists on
    //      Commons with and without a trailing catalogue number.
    //   2. Cosine similarity of face embeddings > 0.95 — catches
    //      visually-identical images uploaded under entirely different
    //      filenames.
    // We keep the first occurrence (highest detection score) and drop
    // later near-duplicates.
    out.sort_by(|a, b| {
        b.face.score
            .partial_cmp(&a.face.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept: Vec<Painting> = Vec::with_capacity(out.len());
    let mut seen_titles: std::collections::HashSet<String> = Default::default();
    let mut dup_title = 0usize;
    let mut dup_embed = 0usize;
    for p in out {
        let norm = normalize_title(&p.title);
        if !norm.is_empty() && !seen_titles.insert(norm) {
            dup_title += 1;
            continue;
        }
        let is_dup = kept.iter().any(|q| {
            crate::face::cosine_similarity(&p.embedding, &q.embedding) > 0.95
        });
        if is_dup {
            dup_embed += 1;
            continue;
        }
        kept.push(p);
    }
    log::info!(
        "after dedupe: {} paintings ({} title dupes, {} embedding dupes)",
        kept.len(),
        dup_title,
        dup_embed
    );
    let mut out = kept;

    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(cache_path, serde_json::to_vec_pretty(&out)?)?;
    Ok(out)
}

/// Normalise a title for dedupe: lowercase, strip trailing catalogue numbers
/// like "SK-A-1234", "NPG 15", "WGA10020", strip parenthetical years, collapse
/// whitespace/punctuation.
fn normalize_title(title: &str) -> String {
    let mut s = title.to_lowercase();
    // Strip catalogue suffixes.
    let cat_re = [
        " sk-a-", " sk a ", " npg ", " wga", " ng-", " ng ", " ngi ", " n0", " n.",
    ];
    for sep in &cat_re {
        if let Some(i) = s.find(sep) {
            s.truncate(i);
        }
    }
    // Strip parenthetical year ranges "(1641-1717)" etc.
    let re_strip_parens =
        regex_lite_strip_parens(&s);
    let s = re_strip_parens;
    // Collapse non-alphanumerics to single spaces, trim.
    let mut norm = String::with_capacity(s.len());
    let mut last_space = true;
    for c in s.chars() {
        if c.is_alphanumeric() {
            norm.push(c);
            last_space = false;
        } else if !last_space {
            norm.push(' ');
            last_space = true;
        }
    }
    norm.trim().to_string()
}

/// Cheap paren stripper — we avoid adding a regex dep for a 10-line job.
fn regex_lite_strip_parens(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth = 0i32;
    for c in s.chars() {
        match c {
            '(' => depth += 1,
            ')' => {
                if depth > 0 {
                    depth -= 1;
                }
            }
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

fn prettify(name: &str) -> String {
    name.replace(['_', '-'], " ")
        .split_whitespace()
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(first) => first.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Rank the gallery by cosine similarity to `user_embedding`.
/// Returns indices sorted by similarity descending (highest = most similar).
pub fn rank(user_embedding: &[f32], gallery: &[Painting]) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = gallery
        .iter()
        .enumerate()
        .map(|(i, p)| (i, cosine_similarity(user_embedding, &p.embedding)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored
}

pub fn load_image(p: &Painting) -> Result<RgbImage> {
    Ok(image::open(&p.path)?.to_rgb8())
}
