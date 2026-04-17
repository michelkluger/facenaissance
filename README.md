# facenaissance

[![CI](https://github.com/michelkluger/facenaissance/actions/workflows/ci.yml/badge.svg)](https://github.com/michelkluger/facenaissance/actions/workflows/ci.yml)

Upload a photo (or snap one with your webcam) and see yourself fitted into
the classical paintings whose subjects look most like you — Mona Lisa, Girl
with a Pearl Earring, Napoleon, Van Gogh's self-portrait, and ~440 more
portraits pulled from Wikimedia Commons.

All inference runs on-device through ONNX Runtime. Single Rust binary, no
Python at runtime. Camera is only activated if you choose that source —
otherwise it stays off.

## How it works

```
 webcam frame
    │
    ▼
 SCRFD ────► face bbox + 5 landmarks
    │
    ▼
 ArcFace ──► 512-d embedding
    │
    ├── cosine similarity vs. each painting's cached embedding
    │         │
    │         ▼
    │    top-N paintings
    │         │
    ▼         ▼
 inswapper_128 ──► swapped 128×128 face
    │
    ▼
 paste back with feathered oval mask
    │
    ▼
 egui gallery
```

- **Face detection / landmarks**: SCRFD 2.5g (InsightFace)
- **Embedding**: ArcFace w600k_r50 (InsightFace)
- **Face swap**: inswapper_128 (InsightFace)
- **Matching**: cosine similarity of ArcFace embeddings
- **UI**: egui + eframe
- **Camera**: nokhwa

## Setup

1. Install a Rust toolchain (stable, 1.78+).
2. Put paintings into `assets/paintings/` — see `assets/paintings/README.md`.
3. Fetch the three ONNX models into `models/` — see `models/README.md`, or
   run one of the helper scripts (they require Python + `insightface`):

   ```powershell
   # Windows
   powershell -ExecutionPolicy Bypass -File scripts\download_models.ps1
   ```

   ```bash
   # macOS / Linux
   bash scripts/download_models.sh
   ```

4. Build and run:

   ```bash
   cargo run --release
   ```

First launch will take a few seconds — it indexes every painting (detects
the face, computes the embedding) and caches to `cache/paintings.json`.
Subsequent launches are instant.

## Controls

| Action | Effect |
|--------|--------|
| `📸 Capture & fit into paintings` | Runs the pipeline on the current live frame |
| `results` slider | How many of the top-matching paintings to render |

## Project layout

```
art/
├── Cargo.toml
├── src/
│   ├── main.rs          # eframe bootstrap
│   ├── app.rs           # egui UI (live preview + gallery)
│   ├── camera.rs        # nokhwa capture thread
│   ├── face.rs          # SCRFD detection + ArcFace embedding
│   ├── align.rs         # 5-point similarity warp (Umeyama)
│   ├── swap.rs          # inswapper_128 + paste-back blending
│   ├── paintings.rs     # gallery scan, cache, ranking
│   └── pipeline.rs      # worker thread orchestrating the above
├── models/              # (you drop ONNX files here)
├── assets/paintings/    # (you drop painting JPGs here)
├── cache/               # created at runtime (painting embeddings)
└── scripts/             # helpers to fetch ONNX models via Python
```

## Known limitations

- **Blending is simple**: an oval feather mask, not Poisson blending, so
  harsh lighting in the painting (e.g. Caravaggio) may leave a visible
  seam. A `poisson` blend pass would be a nice upgrade.
- **Lookalike scoring is a proxy**: cosine similarity between ArcFace
  embeddings is a reasonable "does this face look like that face?" metric,
  but it was trained for recognition (same person vs. different person),
  not aesthetic resemblance. Results can still feel arbitrary — run with a
  larger gallery and higher `top_n` for more variety.
- **inswapper licence is non-commercial.** Don't ship this in a product.

## Troubleshooting

- *No face detected*: move closer, face the camera directly, ensure good
  light. SCRFD is picky about very small faces.
- *`cannot find scrfd_2.5g_bnkps.onnx`*: models aren't in `models/`. Run
  the download script.
- *`no paintings found`*: `assets/paintings/` is empty or your images
  don't have detectable faces. Check `assets/paintings/README.md`.
- *Webcam doesn't open on Windows*: the first call may prompt for camera
  permission. Also try setting a specific camera: edit `camera.rs` and
  change `CameraIndex::Index(0)` to a different index.
