//! egui application shell.
//!
//! Layout:
//!   ┌────────────────────────┬────────────────────────┐
//!   │  Live camera preview   │  Gallery of swapped    │
//!   │  + Capture button      │  painting results      │
//!   └────────────────────────┴────────────────────────┘

use crate::camera::{self, CameraFeed};
use crate::pipeline::{self, SwapRequest, SwapResult, Worker, WorkerMsg};
use anyhow::Result;
use eframe::CreationContext;
use egui::{ColorImage, TextureHandle, TextureOptions};
use image::RgbImage;
use std::path::PathBuf;

/// Where the source face comes from, as a state machine so the camera is
/// *only* active while we need it.
enum InputMode {
    /// Nothing chosen yet — camera off. User picks Upload or Camera.
    Welcome,
    /// Camera is streaming; user sees a live preview and can snap a photo
    /// (which transitions to `HasPhoto` and stops the camera).
    CameraLive { feed: CameraFeed },
    /// We have a photo. Camera is off. User can now Capture & fit.
    HasPhoto {
        image: RgbImage,
        tex: Option<TextureHandle>,
    },
}

pub struct ClassicMeApp {
    mode: InputMode,
    worker: Worker,
    status: String,
    /// Optional (done, total) for the currently running step.
    progress: Option<(usize, usize)>,
    results: Vec<GalleryItem>,
    live_tex: Option<TextureHandle>,
    user_face_tex: Option<TextureHandle>,
    top_n: usize,
    /// Target width for each gallery thumbnail, in pixels. The grid uses
    /// this to compute how many columns to show, like a Windows Explorer
    /// folder view.
    cell_size: f32,
    processing: bool,
}

struct GalleryItem {
    title: String,
    artist: String,
    similarity: f32,
    tex: TextureHandle,
    /// Original swapped image kept on the heap so we can save it to disk
    /// without round-tripping through the GPU texture.
    image: RgbImage,
    /// 1.0 = image fits the cell width. Higher = zoomed in.
    zoom: f32,
    /// Pan offset, in cell pixels, from the cell centre.
    pan: egui::Vec2,
    /// Transient toast after a save — fades after a moment.
    last_save: Option<std::time::Instant>,
}

impl ClassicMeApp {
    pub fn new(_cc: &CreationContext<'_>) -> Result<Self> {
        // Look for assets relative to cwd first, then walk up from the
        // executable so double-clicking the .exe from anywhere still works.
        let root = resolve_project_root();
        let models_dir = root.join("models");
        let paintings_dir = root.join("assets/paintings");
        let cache_path = root.join("cache/paintings.json");
        log::info!("using project root: {}", root.display());

        let worker = pipeline::spawn(models_dir, paintings_dir, cache_path);

        Ok(Self {
            mode: InputMode::Welcome,
            worker,
            status: "Choose a source to begin.".into(),
            progress: None,
            results: Vec::new(),
            live_tex: None,
            user_face_tex: None,
            top_n: 4,
            cell_size: 260.0,
            processing: false,
        })
    }

    fn refresh_live(&mut self, ctx: &egui::Context) {
        // Only pull from the camera when it's actually running.
        if let InputMode::CameraLive { feed } = &self.mode {
            if let Some(frame) = feed.latest() {
                self.live_tex = Some(rgb_to_texture(ctx, &frame, "live"));
            }
        } else {
            self.live_tex = None;
        }
    }

    fn start_camera(&mut self) {
        match camera::start() {
            Ok(feed) => {
                self.mode = InputMode::CameraLive { feed };
                self.status = "Camera on — frame it and hit Snap.".into();
            }
            Err(e) => {
                self.status = format!("Can't open camera: {e}");
            }
        }
    }

    fn stop_camera(&mut self) {
        if let InputMode::CameraLive { feed } = &self.mode {
            feed.stop();
        }
        self.live_tex = None;
    }

    /// Snap the current camera frame, stop the camera, and move to `HasPhoto`.
    fn snap_photo(&mut self, ctx: &egui::Context) {
        let img = if let InputMode::CameraLive { feed } = &self.mode {
            feed.latest()
        } else {
            None
        };
        match img {
            Some(rgb) => {
                self.stop_camera();
                let tex = Some(rgb_to_texture(ctx, &rgb, "photo"));
                self.mode = InputMode::HasPhoto { image: rgb, tex };
                self.status = "Snapped. Ready to fit.".into();
            }
            None => self.status = "No camera frame yet, try again.".into(),
        }
    }

    fn back_to_welcome(&mut self) {
        self.stop_camera();
        self.mode = InputMode::Welcome;
        self.status = "Choose a source.".into();
    }

    fn drain_worker(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.worker.rx_msg.try_recv() {
            match msg {
                WorkerMsg::Progress { text, done } => {
                    self.status = text;
                    self.progress = done;
                }
                WorkerMsg::UserFace(img) => {
                    self.user_face_tex = Some(rgb_to_texture(ctx, &img, "user_face"));
                }
                WorkerMsg::Results(r) => {
                    self.results.clear();
                    for res in r {
                        let tex = rgb_to_texture(ctx, &res.image, &res.painting_title);
                        self.results.push(GalleryItem {
                            title: res.painting_title,
                            artist: res.painting_artist,
                            similarity: res.similarity,
                            tex,
                            image: res.image,
                            zoom: 1.0,
                            pan: egui::Vec2::ZERO,
                            last_save: None,
                        });
                    }
                    self.processing = false;
                    self.progress = None;
                    self.status = format!("Done — {} results", self.results.len());
                }
                WorkerMsg::Error(e) => {
                    self.status = format!("Error: {e}");
                    self.processing = false;
                    self.progress = None;
                }
            }
        }
    }

    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped: Vec<egui::DroppedFile> =
            ctx.input(|i| i.raw.dropped_files.clone());
        for f in dropped {
            if let Some(path) = &f.path {
                self.load_photo(path, ctx);
                break;
            } else if let Some(bytes) = &f.bytes {
                if let Ok(img) = image::load_from_memory(bytes) {
                    self.set_photo(img.to_rgb8(), ctx);
                    break;
                }
            }
        }
    }

    fn load_photo(&mut self, path: &std::path::Path, ctx: &egui::Context) {
        match image::open(path) {
            Ok(img) => {
                self.set_photo(img.to_rgb8(), ctx);
                self.status = format!("Loaded: {}", path.display());
            }
            Err(e) => self.status = format!("Couldn't load {}: {e}", path.display()),
        }
    }

    fn set_photo(&mut self, img: RgbImage, ctx: &egui::Context) {
        self.stop_camera();
        let tex = Some(rgb_to_texture(ctx, &img, "photo"));
        self.mode = InputMode::HasPhoto { image: img, tex };
    }

    fn browse_for_photo(&mut self, ctx: &egui::Context) {
        let file = rfd::FileDialog::new()
            .add_filter("Image", &["jpg", "jpeg", "png", "bmp", "webp"])
            .pick_file();
        if let Some(p) = file {
            self.load_photo(&p, ctx);
        }
    }

    fn export_all(&mut self) {
        let dir = match output_run_dir() {
            Ok(d) => d,
            Err(e) => {
                self.status = format!("Export failed: {e}");
                return;
            }
        };
        let mut ok = 0usize;
        for item in &mut self.results {
            let path = dir.join(format!("{}.jpg", sanitize_filename(&item.title)));
            match item.image.save(&path) {
                Ok(_) => {
                    ok += 1;
                    item.last_save = Some(std::time::Instant::now());
                }
                Err(e) => log::warn!("save {} failed: {e}", path.display()),
            }
        }
        self.status = format!("Exported {ok} images to {}", dir.display());
        open_in_file_manager(&dir);
    }

    fn capture(&mut self) {
        let frame = match &self.mode {
            InputMode::HasPhoto { image, .. } => image.clone(),
            _ => {
                self.status = "Load or snap a photo first.".into();
                return;
            }
        };
        self.user_face_tex = None;
        self.processing = true;
        self.progress = None;
        self.status = "Processing...".into();
        let req = SwapRequest {
            frame,
            top_n: self.top_n,
        };
        if let Err(e) = self.worker.tx_req.try_send(req) {
            self.status = format!("Busy: {e}");
            self.processing = false;
        }
    }
}

impl ClassicMeApp {
    fn draw_source_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        // Snapshot the data we need so the &self.mode borrow drops before
        // we call any mutating helper on `self`.
        enum Stage {
            Welcome,
            Live(Option<TextureHandle>),
            Photo(Option<TextureHandle>),
        }
        let stage = match &self.mode {
            InputMode::Welcome => Stage::Welcome,
            InputMode::CameraLive { .. } => Stage::Live(self.live_tex.clone()),
            InputMode::HasPhoto { tex, .. } => Stage::Photo(tex.clone()),
        };

        match stage {
            Stage::Welcome => {
                ui.add_space(12.0);
                ui.heading("Pick a source");
                ui.label(
                    egui::RichText::new(
                        "The camera is off until you turn it on. \
                         You can also drag-and-drop an image on the window.",
                    )
                    .weak()
                    .small(),
                );
                ui.add_space(16.0);
                let w = ui.available_width();
                if ui
                    .add_sized([w, 48.0], egui::Button::new("📸  Use camera"))
                    .clicked()
                {
                    self.start_camera();
                }
                ui.add_space(8.0);
                if ui
                    .add_sized([w, 48.0], egui::Button::new("📁  Upload photo"))
                    .clicked()
                {
                    self.browse_for_photo(ctx);
                }
            }

            Stage::Live(tex_opt) => {
                let mut back = false;
                let mut snap = false;
                ui.horizontal(|ui| {
                    ui.heading("Live camera");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("✕ Cancel").clicked() {
                            back = true;
                        }
                    });
                });
                if let Some(tex) = tex_opt.as_ref() {
                    let image_size = tex.size_vec2();
                    let avail = ui.available_width();
                    let scale = (avail / image_size.x).min(360.0 / image_size.y);
                    ui.add(
                        egui::Image::from_texture(tex).fit_to_exact_size(image_size * scale),
                    );
                } else {
                    ui.label(egui::RichText::new("Opening camera...").weak());
                }
                ui.add_space(6.0);
                let w = ui.available_width();
                if ui
                    .add_sized([w, 44.0], egui::Button::new("📸  Snap photo"))
                    .clicked()
                {
                    snap = true;
                }
                if snap {
                    self.snap_photo(ctx);
                } else if back {
                    self.back_to_welcome();
                }
            }

            Stage::Photo(tex_opt) => {
                let mut back = false;
                let mut cap = false;
                ui.horizontal(|ui| {
                    ui.heading("Your photo");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("↩ Change").clicked() {
                            back = true;
                        }
                    });
                });
                if let Some(tex) = tex_opt.as_ref() {
                    let image_size = tex.size_vec2();
                    let avail = ui.available_width();
                    let scale = (avail / image_size.x).min(360.0 / image_size.y);
                    ui.add(
                        egui::Image::from_texture(tex).fit_to_exact_size(image_size * scale),
                    );
                }
                ui.add_space(6.0);
                let w = ui.available_width();
                let enabled = !self.processing;
                ui.add_enabled_ui(enabled, |ui| {
                    if ui
                        .add_sized(
                            [w, 44.0],
                            egui::Button::new("🎨  Capture & fit into paintings"),
                        )
                        .clicked()
                    {
                        cap = true;
                    }
                });
                if cap {
                    self.capture();
                } else if back {
                    self.back_to_welcome();
                }
            }
        }
    }
}

impl eframe::App for ClassicMeApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.refresh_live(ctx);
        self.drain_worker(ctx);
        self.handle_dropped_files(ctx);

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            // Row 1: title + right-aligned controls. Status goes on its own
            // row below so variable-length text can't shove things around.
            ui.horizontal(|ui| {
                ui.heading("classic-me");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    const STEPS: [usize; 8] = [2, 4, 8, 16, 32, 64, 128, 256];
                    egui::ComboBox::from_label("results")
                        .selected_text(format!("{}", self.top_n))
                        .show_ui(ui, |ui| {
                            for &n in &STEPS {
                                ui.selectable_value(&mut self.top_n, n, n.to_string());
                            }
                        });
                    ui.add_enabled_ui(!self.results.is_empty(), |ui| {
                        if ui.button("💾  Export all").clicked() {
                            self.export_all();
                        }
                    });
                    ui.add(
                        egui::Slider::new(&mut self.cell_size, 100.0..=520.0)
                            .text("size")
                            .fixed_decimals(0),
                    );
                });
            });

            // Row 2: fixed-width progress bar slot + truncated status text.
            ui.horizontal(|ui| {
                let frac = self
                    .progress
                    .map(|(d, t)| if t > 0 { d as f32 / t as f32 } else { 0.0 })
                    .unwrap_or(0.0);
                let text = self
                    .progress
                    .map(|(d, t)| format!("{d}/{t}"))
                    .unwrap_or_else(|| "—".into());
                // Fixed 220 px slot so the bar never shifts.
                ui.allocate_ui_with_layout(
                    egui::vec2(220.0, 20.0),
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| {
                        ui.add(
                            egui::ProgressBar::new(frac.clamp(0.0, 1.0))
                                .desired_width(220.0)
                                .text(text),
                        );
                    },
                );
                ui.add(egui::Label::new(&self.status).truncate());
            });
        });

        egui::SidePanel::left("left")
            .resizable(true)
            .default_width(340.0)
            .width_range(260.0..=520.0)
            .show(ctx, |ui| {
                self.draw_source_panel(ui, ctx);

                ui.separator();
                ui.heading("Extracted face");
                if let Some(tex) = &self.user_face_tex {
                    let size = tex.size_vec2();
                    ui.add(egui::Image::from_texture(tex).fit_to_exact_size(size * 2.0));
                    ui.label(
                        egui::RichText::new(
                            "What the swap model sees as you. \
                             Off-centre/rotated = detection is bad.",
                        )
                        .weak()
                        .small(),
                    );
                } else {
                    ui.label("No face captured yet.");
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("You in classical paintings");
            if self.results.is_empty() {
                ui.label("Hit Capture to see yourself fitted into the best-matching paintings.");
                return;
            }
            ui.label(
                egui::RichText::new(
                    "Pinch or ctrl-scroll over any image to zoom, drag to pan, double-click to reset.",
                )
                .weak()
                .small(),
            );

            let spacing = 12.0;
            let available = ui.available_width();
            // Columns derived from target cell size, like Windows Explorer.
            let cols = (((available + spacing) / (self.cell_size + spacing))
                .floor() as usize)
                .max(1);
            let cell_w = (available - spacing * (cols as f32 - 1.0)) / cols as f32;

            egui::ScrollArea::vertical().show(ui, |ui| {
                let n = self.results.len();
                let mut i = 0;
                while i < n {
                    ui.horizontal_top(|ui| {
                        for col in 0..cols {
                            if i >= n {
                                break;
                            }
                            let idx = i;
                            i += 1;
                            ui.allocate_ui_with_layout(
                                egui::vec2(cell_w, cell_w * 1.35),
                                egui::Layout::top_down(egui::Align::LEFT),
                                |ui| {
                                    zoomable_image_cell(ui, &mut self.results[idx], cell_w);
                                },
                            );
                            if col + 1 < cols {
                                ui.add_space(spacing);
                            }
                        }
                    });
                    ui.add_space(spacing * 0.5);
                }
            });
        });

        // Keep refreshing ~30 fps so the live preview stays smooth.
        ctx.request_repaint_after(std::time::Duration::from_millis(33));
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.stop_camera();
    }
}

/// Render one gallery cell: fixed-size viewport showing the painting, with
/// per-cell pinch/ctrl-scroll zoom and drag-to-pan. Contents are clipped
/// to the cell so zoom stays inside its frame.
fn zoomable_image_cell(ui: &mut egui::Ui, item: &mut GalleryItem, cell_w: f32) {
    // Viewport is a *fixed* square so every gallery cell aligns to the same
    // grid regardless of the painting's aspect ratio. The image is then
    // letterboxed inside at zoom=1.0.
    let viewport_h = cell_w;
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(cell_w, viewport_h),
        egui::Sense::click_and_drag(),
    );
    let img_size = item.tex.size_vec2();
    let fit_scale = (rect.width() / img_size.x).min(rect.height() / img_size.y);

    // ---- input: pinch, ctrl/cmd + scroll, drag, double-click ----
    let (pinch, scroll, mods, pointer_pos) = ui.ctx().input(|i| {
        (
            i.zoom_delta(),
            i.raw_scroll_delta.y,
            i.modifiers,
            i.pointer.hover_pos(),
        )
    });

    let mut zoom_mul = 1.0;
    if resp.hovered() {
        if pinch != 1.0 {
            zoom_mul *= pinch;
        }
        if (mods.command || mods.ctrl) && scroll != 0.0 {
            zoom_mul *= (scroll * 0.005).exp();
        }
    }

    if zoom_mul != 1.0 {
        let pivot = pointer_pos.unwrap_or_else(|| rect.center()) - rect.center();
        let old = item.zoom;
        item.zoom = (item.zoom * zoom_mul).clamp(1.0, 20.0);
        let real_mul = item.zoom / old;
        item.pan = pivot + (item.pan - pivot) * real_mul;
    }

    if resp.dragged() {
        item.pan += resp.drag_delta();
    }
    if resp.double_clicked() {
        item.zoom = 1.0;
        item.pan = egui::Vec2::ZERO;
    }

    // Clamp pan so the image edge can't drag past the cell centre.
    let scaled = img_size * fit_scale * item.zoom;
    let max_pan = egui::vec2(
        (scaled.x - rect.width()).max(0.0) * 0.5,
        (scaled.y - rect.height()).max(0.0) * 0.5,
    );
    item.pan.x = item.pan.x.clamp(-max_pan.x, max_pan.x);
    item.pan.y = item.pan.y.clamp(-max_pan.y, max_pan.y);

    // ---- render: draw via a clipped painter so zoom stays inside the cell ----
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, egui::Color32::from_gray(20));
    let center = rect.center() + item.pan;
    let image_rect = egui::Rect::from_center_size(center, scaled);
    painter.image(
        item.tex.id(),
        image_rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );

    // Small zoom indicator in the corner if not at 1.0.
    if (item.zoom - 1.0).abs() > 0.01 {
        let text = format!("{:.1}×", item.zoom);
        let pos = egui::pos2(rect.right() - 6.0, rect.top() + 6.0);
        painter.text(
            pos,
            egui::Align2::RIGHT_TOP,
            text,
            egui::FontId::proportional(12.0),
            egui::Color32::from_white_alpha(220),
        );
    }

    // Title + save button on one row; artist line below. Both truncate to
    // a single line so every cell has the same total height.
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        let resp = ui.small_button("💾");
        if resp.clicked() {
            if let Some(path) = save_one(&item.image, &item.title) {
                item.last_save = Some(std::time::Instant::now());
                log::info!("saved {}", path.display());
            }
        }
        resp.on_hover_text("Save this image to ./output/");
        ui.add(
            egui::Label::new(egui::RichText::new(&item.title).strong())
                .truncate(),
        );
    });
    ui.add(
        egui::Label::new(
            egui::RichText::new(format!(
                "{} — similarity {:.2}",
                item.artist, item.similarity
            ))
            .weak()
            .small(),
        )
        .truncate(),
    );

    // Flash "Saved" for 2 s after a successful save.
    if let Some(t) = item.last_save {
        let age = t.elapsed().as_secs_f32();
        if age < 2.0 {
            let alpha = ((2.0 - age) / 2.0 * 255.0) as u8;
            let pos = egui::pos2(rect.left() + 6.0, rect.top() + 6.0);
            ui.painter().text(
                pos,
                egui::Align2::LEFT_TOP,
                "✓ Saved",
                egui::FontId::proportional(14.0),
                egui::Color32::from_rgba_unmultiplied(120, 220, 120, alpha),
            );
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(80));
        } else {
            item.last_save = None;
        }
    }
}

/// Sanitize a string for use as a filename (safe characters only).
fn sanitize_filename(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == ' ' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    out.trim().replace(' ', "_")
}

/// Return a fresh `output/run_<YYYYMMDD_HHMMSS>/` directory, creating it.
fn output_run_dir() -> std::io::Result<std::path::PathBuf> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Rough local timestamp — we don't have chrono so use the epoch suffix.
    let dir = std::path::PathBuf::from("output").join(format!("run_{now}"));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Save a single result next to Export-all's run folders, under
/// `output/single/<title>_<epoch>.jpg`. Returns the path on success.
fn save_one(img: &RgbImage, title: &str) -> Option<std::path::PathBuf> {
    let dir = std::path::PathBuf::from("output").join("single");
    std::fs::create_dir_all(&dir).ok()?;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let path = dir.join(format!("{}_{ts}.jpg", sanitize_filename(title)));
    img.save(&path).ok()?;
    Some(path)
}

/// Open the directory in the platform's file manager (best-effort).
fn open_in_file_manager(path: &std::path::Path) {
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("explorer.exe")
            .arg(path)
            .spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(path).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = std::process::Command::new("xdg-open").arg(path).spawn();
    }
}

/// Resolve where the project root is — where `models/` and
/// `assets/paintings/` live. Checks (in order):
///   1. the current working directory
///   2. the directory containing the running executable
///   3. the executable's parent-of-parent (covers `target/release/` layout)
///   4. two levels up (covers `target/release/<target>/` for cross-compiled)
fn resolve_project_root() -> std::path::PathBuf {
    if std::path::Path::new("models").is_dir() {
        return std::env::current_dir().unwrap_or_else(|_| ".".into());
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.as_path();
        for _ in 0..4 {
            if let Some(parent) = dir.parent() {
                if parent.join("models").is_dir() {
                    return parent.to_path_buf();
                }
                dir = parent;
            } else {
                break;
            }
        }
    }
    // Fallback: stick with cwd; the pipeline will surface a clear error.
    std::env::current_dir().unwrap_or_else(|_| ".".into())
}

fn rgb_to_texture(ctx: &egui::Context, img: &RgbImage, name: &str) -> TextureHandle {
    let size = [img.width() as usize, img.height() as usize];
    let pixels: Vec<egui::Color32> = img
        .pixels()
        .map(|p| egui::Color32::from_rgb(p[0], p[1], p[2]))
        .collect();
    let color_image = ColorImage { size, pixels };
    ctx.load_texture(name, color_image, TextureOptions::LINEAR)
}
