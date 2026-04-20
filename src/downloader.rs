//! Background starter-gallery downloader.
//!
//! If `assets/paintings/` is empty or sparse on launch, this module spins up
//! a thread that:
//!   1. batches the curated list's File: titles against Wikimedia's API to
//!      resolve them to CDN thumbnail URLs;
//!   2. downloads each painting as JPEG + writes a sidecar `.json` with
//!      title/artist metadata;
//!   3. sends `DownloadMsg` progress to the UI at every step.
//!
//! All HTTP is paced at ~1 req/sec and honours Wikimedia's rate-limit
//! policy (https://meta.wikimedia.org/wiki/User-Agent_policy).

use crossbeam_channel::{bounded, Receiver, Sender};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

const UA: &str =
    "facenaissance/0.1 (+https://github.com/michelkluger/facenaissance; starter downloader)";
const API: &str = "https://commons.wikimedia.org/w/api.php";
const PACE: Duration = Duration::from_millis(1000);
/// Emit a Checkpoint after this many downloads so the gallery grows live.
const CHECKPOINT_EVERY: usize = 10;

/// Curated starter set — 40 well-known portraits that reliably exist on
/// Wikimedia Commons under these filenames. Each tuple is
/// `(slug, "File:...jpg", display_title, artist)`.
#[rustfmt::skip]
const ITEMS: &[(&str, &str, &str, &str)] = &[
    ("mona_lisa", "File:Mona Lisa, by Leonardo da Vinci, from C2RMF retouched.jpg", "Mona Lisa", "Leonardo da Vinci"),
    ("girl_with_a_pearl_earring", "File:1665 Girl with a Pearl Earring.jpg", "Girl with a Pearl Earring", "Johannes Vermeer"),
    ("lady_with_an_ermine", "File:The Lady with an Ermine.jpg", "Lady with an Ermine", "Leonardo da Vinci"),
    ("la_belle_ferronniere", "File:Leonardo da Vinci (attrib)- la Belle Ferroniere.jpg", "La Belle Ferronnière", "Leonardo da Vinci"),
    ("salvator_mundi", "File:Salvator Mundi by Leonardo da Vinci.jpg", "Salvator Mundi", "Leonardo da Vinci"),
    ("raphael_castiglione", "File:Baldassare Castiglione, by Raffaello Sanzio, from C2RMF retouched.jpg", "Portrait of Baldassare Castiglione", "Raphael"),
    ("raphael_fornarina", "File:Raffael - La Fornarina.jpeg", "La Fornarina", "Raphael"),
    ("raphael_self", "File:Raffaello Sanzio - Self-portrait - Google Art Project.jpg", "Self-Portrait", "Raphael"),
    ("van_gogh_self", "File:Vincent van Gogh - Self-Portrait - Google Art Project.jpg", "Self-Portrait", "Vincent van Gogh"),
    ("van_gogh_bandaged_ear", "File:Vincent Willem van Gogh 106.jpg", "Self-Portrait with Bandaged Ear", "Vincent van Gogh"),
    ("van_gogh_dr_gachet", "File:Portrait of Dr. Gachet.jpg", "Portrait of Dr. Gachet", "Vincent van Gogh"),
    ("rembrandt_self_portrait", "File:Rembrandt van Rijn - Self-Portrait - Google Art Project.jpg", "Self-Portrait", "Rembrandt van Rijn"),
    ("rembrandt_saskia", "File:Rembrandt Harmensz. van Rijn 085.jpg", "Saskia van Uylenburgh", "Rembrandt van Rijn"),
    ("vermeer_milkmaid", "File:Johannes Vermeer - Het melkmeisje - Google Art Project.jpg", "The Milkmaid", "Johannes Vermeer"),
    ("vermeer_lacemaker", "File:Johannes Vermeer - The lacemaker (c.1669-1671).jpg", "The Lacemaker", "Johannes Vermeer"),
    ("napoleon", "File:Jacques-Louis David - The Emperor Napoleon in His Study at the Tuileries - Google Art Project.jpg", "The Emperor Napoleon in His Study", "Jacques-Louis David"),
    ("david_marat", "File:Death of Marat by David.jpg", "The Death of Marat", "Jacques-Louis David"),
    ("botticelli_young_man", "File:Portrait of a Young Man by Sandro Botticelli - Louvre.jpg", "Portrait of a Young Man", "Sandro Botticelli"),
    ("botticelli_simonetta", "File:Sandro Botticelli 059.jpg", "Portrait of Simonetta Vespucci", "Sandro Botticelli"),
    ("botticelli_giuliano", "File:Sandro Botticelli 063.jpg", "Portrait of Giuliano de' Medici", "Sandro Botticelli"),
    ("caravaggio_bacchus", "File:Michelangelo Caravaggio 065.jpg", "Young Sick Bacchus", "Caravaggio"),
    ("caravaggio_narcissus", "File:Narcissus-Caravaggio (1594-96) edited.jpg", "Narcissus", "Caravaggio"),
    ("caravaggio_medusa", "File:Medusa by Carvaggio.jpg", "Medusa", "Caravaggio"),
    ("holbein_jane_seymour", "File:Hans Holbein the Younger - Jane Seymour, Queen of England - Google Art Project.jpg", "Jane Seymour, Queen of England", "Hans Holbein"),
    ("arnolfini_portrait", "File:Van Eyck - Arnolfini Portrait.jpg", "The Arnolfini Portrait", "Jan van Eyck"),
    ("durer_self_1500", "File:Albrecht Dürer - 1500 self-portrait (High resolution and detail).jpg", "Self-Portrait at 28", "Albrecht Dürer"),
    ("ingres_madame_moitessier", "File:Jean-Auguste-Dominique Ingres - Madame Moitessier - Google Art Project.jpg", "Madame Moitessier", "Jean-Auguste-Dominique Ingres"),
    ("ingres_caroline_riviere", "File:Jean Auguste Dominique Ingres 014.jpg", "Mademoiselle Caroline Rivière", "Jean-Auguste-Dominique Ingres"),
    ("klimt_adele", "File:Gustav Klimt 046.jpg", "Portrait of Adele Bloch-Bauer I", "Gustav Klimt"),
    ("vigee_marie_antoinette", "File:Marie Antoinette Adult.jpg", "Marie Antoinette", "Élisabeth Vigée Le Brun"),
    ("sargent_madame_x", "File:John Singer Sargent - Madame X - The Metropolitan Museum of Art.jpg", "Madame X", "John Singer Sargent"),
    ("sargent_carnation_lily", "File:Carnation, Lily, Lily, Rose, by John Singer Sargent, 1885-6.jpg", "Carnation, Lily, Lily, Rose", "John Singer Sargent"),
    ("whistler_mother", "File:Whistlers Mother high res.jpg", "Whistler's Mother", "James McNeill Whistler"),
    ("cezanne_self", "File:Paul Cézanne 157.jpg", "Self-Portrait", "Paul Cézanne"),
    ("courbet_despair", "File:Gustave Courbet - Le Désespéré.JPG", "The Desperate Man", "Gustave Courbet"),
    ("delacroix_self", "File:Eugène Ferdinand Victor Delacroix 019.jpg", "Self-Portrait", "Eugène Delacroix"),
    ("piero_federico", "File:Piero della Francesca 046.jpg", "Federico da Montefeltro", "Piero della Francesca"),
    ("memling_man_with_coin", "File:Hans Memling 050.jpg", "Portrait of a Man with a Roman Coin", "Hans Memling"),
    ("van_eyck_red_turban", "File:Portrait of a Man in a Red Turban (Jan van Eyck, 1433) cleaned.jpg", "Portrait of a Man in a Red Turban", "Jan van Eyck"),
    ("stuart_washington", "File:Gilbert Stuart - George Washington - Google Art Project (721059).jpg", "George Washington", "Gilbert Stuart"),
    // ---- second batch — broadens the gallery so top_n 64/128 reaches real variety ----
    ("van_gogh_postman", "File:Vincent van Gogh - Portrait of Postman Roulin - Google Art Project.jpg", "The Postman Joseph Roulin", "Vincent van Gogh"),
    ("rembrandt_broad_hat", "File:Rembrandt van Rijn - Self Portrait with a Broad-Brimmed Hat - WGA19206.jpg", "Self-Portrait in a Broad-Brimmed Hat", "Rembrandt van Rijn"),
    ("vermeer_girl_red_hat", "File:Jan Vermeer van Delft 009.jpg", "Girl with a Red Hat", "Johannes Vermeer"),
    ("velazquez_innocent_x", "File:Diego Velázquez - Portrait of Pope Innocent X - Google Art Project.jpg", "Portrait of Pope Innocent X", "Diego Velázquez"),
    ("velazquez_juan_pareja", "File:Juan de Pareja by Diego Velázquez.jpg", "Juan de Pareja", "Diego Velázquez"),
    ("velazquez_philip_iv", "File:Diego Velázquez 043.jpg", "Portrait of Philip IV", "Diego Velázquez"),
    ("velazquez_infanta_margarita", "File:Velazquez-lasmeninas01.jpg", "Infanta Margarita", "Diego Velázquez"),
    ("goya_self_spectacles", "File:Francisco de Goya y Lucientes - Self-Portrait with Spectacles - WGA10020.jpg", "Self-Portrait with Spectacles", "Francisco Goya"),
    ("goya_duchess_alba", "File:Alba Goya.jpg", "The Duchess of Alba", "Francisco Goya"),
    ("gainsborough_blue_boy", "File:Thomas Gainsborough - The Blue Boy - Google Art Project.jpg", "The Blue Boy", "Thomas Gainsborough"),
    ("gainsborough_mrs_siddons", "File:Thomas Gainsborough Lady Sarah Siddons.jpg", "Mrs. Siddons", "Thomas Gainsborough"),
    ("durer_self_1498", "File:Albrecht Dürer 070.jpg", "Self-Portrait at 26", "Albrecht Dürer"),
    ("hals_laughing_cavalier", "File:Frans Hals - The Laughing Cavalier - WGA11093.jpg", "The Laughing Cavalier", "Frans Hals"),
    ("hals_malle_babbe", "File:Frans Hals - Malle Babbe - Google Art Project.jpg", "Malle Babbe", "Frans Hals"),
    ("manet_berthe_morisot", "File:Berthe Morisot With a Bouquet of Violets.jpg", "Berthe Morisot with a Bouquet of Violets", "Édouard Manet"),
    ("manet_olympia", "File:Edouard Manet - Olympia - Google Art Project 3.jpg", "Olympia", "Édouard Manet"),
    ("renoir_jeanne_samary", "File:Jeanne Samary by Pierre-Auguste Renoir 1879.jpg", "Portrait of Jeanne Samary", "Pierre-Auguste Renoir"),
    ("caravaggio_fruit_basket", "File:Caravaggio - Boy with a Basket of Fruit.jpg", "Boy with a Basket of Fruit", "Caravaggio"),
    ("holbein_henry_viii", "File:Hans Holbein d. J. 074.jpg", "Portrait of Henry VIII", "Hans Holbein"),
    ("bronzino_lucrezia", "File:Bronzino - Lucrezia Panciatichi.jpg", "Lucrezia Panciatichi", "Bronzino"),
    ("ghirlandaio_old_man", "File:Domenico Ghirlandaio - Old Man with his Grandson - Google Art Project.jpg", "Old Man with his Grandson", "Domenico Ghirlandaio"),
    ("el_greco_nobleman", "File:Gentleman with his Hand on his Chest.jpg", "Gentleman with his Hand on his Chest", "El Greco"),
    ("reynolds_sarah_siddons", "File:Sir Joshua Reynolds - Sarah Siddons as the Tragic Muse - Google Art Project.jpg", "Sarah Siddons as the Tragic Muse", "Joshua Reynolds"),
    ("titian_man_glove", "File:Titian - Portrait of a Man with a Glove.jpg", "Man with a Glove", "Titian"),
    ("titian_pietro_aretino", "File:Pietro Aretino by Titian.jpeg", "Pietro Aretino", "Titian"),
    ("corot_self", "File:Camille Corot, Self-Portrait.jpg", "Self-Portrait", "Jean-Baptiste-Camille Corot"),
    ("raphael_madonna_chair", "File:Raphael - Madonna della seggiola.jpg", "Madonna della seggiola", "Raphael"),
    ("david_recamier", "File:Jacques-Louis David 016.jpg", "Portrait of Madame Récamier", "Jacques-Louis David"),
    ("rubens_susanna_lunden", "File:Peter Paul Rubens 104.jpg", "Susanna Lunden", "Peter Paul Rubens"),
    ("rubens_helena", "File:Peter Paul Rubens 105.jpg", "Hélène Fourment", "Peter Paul Rubens"),
    ("murillo_boys", "File:Bartolomé Esteban Perugino - Two Boys Eating a Melon and Grapes - Google Art Project.jpg", "Two Boys Eating Melon and Grapes", "Murillo"),
    ("van_dyck_charles_i", "File:Sir Anthony van Dyck - Charles I (1600-49) - Google Art Project.jpg", "Charles I", "Anthony van Dyck"),
    ("ingres_napoleon_throne", "File:Ingres, Napoleon on his Imperial throne.jpg", "Napoleon on his Imperial Throne", "Jean-Auguste-Dominique Ingres"),
    ("degas_absinthe", "File:Edgar Germain Hilaire Degas 064.jpg", "L'Absinthe", "Edgar Degas"),
    ("cassatt_cup_of_tea", "File:Mary Cassatt - The Cup of Tea - Google Art Project.jpg", "The Cup of Tea", "Mary Cassatt"),
    ("monet_camille", "File:Camille, or The Woman in the Green Dress - Claude Monet - Google Cultural Institute.jpg", "Camille, the Woman in the Green Dress", "Claude Monet"),
    ("millais_ophelia", "File:John Everett Millais - Ophelia - Google Art Project.jpg", "Ophelia", "John Everett Millais"),
    ("ingres_grande_odalisque", "File:Jean Auguste Dominique Ingres - La Grande Odalisque - Google Art Project.jpg", "La Grande Odalisque", "Jean-Auguste-Dominique Ingres"),
    ("tintoretto_self", "File:Jacopo Tintoretto - Self-Portrait - Google Art Project.jpg", "Self-Portrait", "Tintoretto"),
    ("eakins_agnew_clinic", "File:Thomas Eakins - Portrait of Dr. John H. Brinton - Google Art Project.jpg", "Portrait of Dr. John H. Brinton", "Thomas Eakins"),
    ("copley_mrs_smith", "File:John Singleton Copley - Mrs. James Smith (Elizabeth Murray) - Google Art Project.jpg", "Mrs. James Smith", "John Singleton Copley"),
    ("bellini_doge", "File:Gentile Bellini 003.jpg", "Portrait of a Doge", "Giovanni Bellini"),
    ("messina_salvator_mundi", "File:Antonello da Messina 054.jpg", "Portrait of a Man (Il Condottiere)", "Antonello da Messina"),
    ("klimt_judith", "File:Gustav Klimt 039.jpg", "Judith and the Head of Holofernes", "Gustav Klimt"),
    ("schiele_self", "File:Egon Schiele 122.jpg", "Self-Portrait with Physalis", "Egon Schiele"),
    ("repin_tolstoy", "File:Ilja Jefimowitsch Repin 004.jpg", "Portrait of Leo Tolstoy", "Ilya Repin"),
    ("fragonard_young_girl_reading", "File:Jean-Honoré Fragonard - A Young Girl Reading - Google Art Project.jpg", "A Young Girl Reading", "Jean-Honoré Fragonard"),
    ("david_bonaparte_great_st_bernard", "File:Jacques-Louis David 007.jpg", "Napoleon Crossing the Alps", "Jacques-Louis David"),
    ("caravaggio_david_goliath", "File:Caravaggio - David with the Head of Goliath - Google Art Project.jpg", "David with the Head of Goliath", "Caravaggio"),
    ("dali_self", "File:Dali - Self Portrait as Mona Lisa.jpg", "Self-Portrait as Mona Lisa", "Salvador Dalí"),
    ("kokoschka_self", "File:Kokoschka Selbstportrait.jpg", "Self-Portrait", "Oskar Kokoschka"),
    ("munch_self_hell", "File:Edvard Munch - Self-Portrait in Hell - Google Art Project.jpg", "Self-Portrait in Hell", "Edvard Munch"),
    ("munch_scream", "File:The Scream.jpg", "The Scream", "Edvard Munch"),
    ("matisse_self", "File:Henri Matisse Self-Portrait 1906.jpg", "Self-Portrait (1906)", "Henri Matisse"),
    ("gauguin_self", "File:Paul Gauguin - Self-portrait - Google Art Project.jpg", "Self-Portrait", "Paul Gauguin"),
    ("gauguin_tahitian_women", "File:Paul Gauguin 056.jpg", "Tahitian Women on the Beach", "Paul Gauguin"),
    ("cezanne_card_players", "File:Paul Cézanne, 1892-95, Les joueurs de carte (The Card Players), 60 x 73 cm, oil on canvas, Courtauld Institute of Art, London.jpg", "The Card Players", "Paul Cézanne"),
    ("bruegel_peasant_dance", "File:Pieter Bruegel the Elder - Peasant Dance.jpg", "Peasant Dance", "Pieter Bruegel the Elder"),
];

/// Messages from the downloader thread to the UI.
pub enum DownloadMsg {
    /// Heads-up while we're resolving File: titles to URLs.
    Resolving { total: usize },
    /// Progress: `done` files finished of `total`, currently working on
    /// `title`.
    Progress {
        done: usize,
        total: usize,
        title: String,
    },
    /// N new files have been written since the last Checkpoint — the UI
    /// should ask the worker to incrementally reindex so users can swap
    /// against the fresh paintings before the full download ends.
    Checkpoint { new_files_since_last: usize },
    /// All done — `new_files` is how many paintings were actually added
    /// this run (skipping ones that were already on disk).
    Done { new_files: usize },
    Error(String),
}

/// Handle for the UI to poll the downloader.
pub struct Downloader {
    pub rx: Receiver<DownloadMsg>,
}

/// Spawn the download thread. Returns immediately.
pub fn spawn(paintings_dir: PathBuf) -> Downloader {
    let (tx, rx) = bounded::<DownloadMsg>(64);
    thread::Builder::new()
        .name("downloader".into())
        .spawn(move || {
            if let Err(e) = run(tx.clone(), paintings_dir) {
                let _ = tx.send(DownloadMsg::Error(format!("{e:?}")));
            }
        })
        .expect("spawn downloader thread");
    Downloader { rx }
}

fn run(tx: Sender<DownloadMsg>, dir: PathBuf) -> anyhow::Result<()> {
    std::fs::create_dir_all(&dir)?;

    // Skip anything already on disk.
    let pending: Vec<&(&str, &str, &str, &str)> = ITEMS
        .iter()
        .filter(|(slug, ..)| !dir.join(format!("{slug}.jpg")).exists())
        .collect();

    if pending.is_empty() {
        let _ = tx.send(DownloadMsg::Done { new_files: 0 });
        return Ok(());
    }

    let total = pending.len();
    let _ = tx.send(DownloadMsg::Resolving { total });

    let agent = ureq::AgentBuilder::new()
        .user_agent(UA)
        .timeout(Duration::from_secs(60))
        .build();

    // Batch API lookups (15 titles per request).
    let mut url_for: std::collections::HashMap<&str, String> = Default::default();
    for chunk in pending.chunks(15) {
        let titles: Vec<&str> = chunk.iter().map(|e| e.1).collect();
        if let Ok(resolved) = resolve_titles(&agent, &titles) {
            for (slug, _file_title, _, _) in chunk {
                if let Some(u) = resolved.get(*_file_title) {
                    url_for.insert(*slug, u.clone());
                }
            }
        }
        thread::sleep(PACE);
    }

    // Download each painting in order, sending progress.
    let mut new_files = 0usize;
    let mut since_checkpoint = 0usize;
    for (i, (slug, _file_title, display, artist)) in pending.iter().enumerate() {
        let _ = tx.send(DownloadMsg::Progress {
            done: i,
            total,
            title: display.to_string(),
        });

        let Some(url) = url_for.get(*slug) else {
            log::warn!("no url resolved for {slug}");
            continue;
        };
        match download_with_retry(&agent, url) {
            Ok(bytes) => {
                let jpg_path = dir.join(format!("{slug}.jpg"));
                if std::fs::write(&jpg_path, &bytes).is_ok() {
                    // Sidecar metadata so paintings.rs picks up title + artist.
                    let meta = serde_json::json!({
                        "title": display,
                        "artist": artist,
                    });
                    let _ = std::fs::write(
                        dir.join(format!("{slug}.json")),
                        serde_json::to_vec(&meta)?,
                    );
                    new_files += 1;
                    since_checkpoint += 1;
                    if since_checkpoint >= CHECKPOINT_EVERY {
                        let _ = tx.send(DownloadMsg::Checkpoint {
                            new_files_since_last: since_checkpoint,
                        });
                        since_checkpoint = 0;
                    }
                }
            }
            Err(e) => log::warn!("download {slug}: {e}"),
        }
        thread::sleep(PACE);
    }

    if since_checkpoint > 0 {
        let _ = tx.send(DownloadMsg::Checkpoint {
            new_files_since_last: since_checkpoint,
        });
    }
    let _ = tx.send(DownloadMsg::Done { new_files });
    Ok(())
}

// ---- API / HTTP helpers ------------------------------------------------

#[derive(Deserialize)]
struct ApiResponse {
    query: Option<ApiQuery>,
}
#[derive(Deserialize)]
struct ApiQuery {
    #[serde(default)]
    normalized: Vec<ApiNormalized>,
    pages: std::collections::HashMap<String, ApiPage>,
}
#[derive(Deserialize)]
struct ApiNormalized {
    from: String,
    to: String,
}
#[derive(Deserialize)]
struct ApiPage {
    title: Option<String>,
    #[serde(default)]
    imageinfo: Vec<ApiImageInfo>,
}
#[derive(Deserialize)]
struct ApiImageInfo {
    thumburl: Option<String>,
    url: Option<String>,
}

fn resolve_titles(
    agent: &ureq::Agent,
    titles: &[&str],
) -> anyhow::Result<std::collections::HashMap<String, String>> {
    let resp: ApiResponse = agent
        .post(API)
        .send_form(&[
            ("action", "query"),
            ("format", "json"),
            ("prop", "imageinfo"),
            ("iiprop", "url"),
            ("iiurlwidth", "800"),
            ("titles", &titles.join("|")),
        ])?
        .into_json()?;

    let Some(q) = resp.query else {
        return Ok(Default::default());
    };
    // Map normalised-title → original-title so we can key by what we asked for.
    let norm_back: std::collections::HashMap<String, String> = q
        .normalized
        .iter()
        .map(|n| (n.to.clone(), n.from.clone()))
        .collect();

    let mut out = std::collections::HashMap::new();
    for page in q.pages.values() {
        let Some(title_norm) = page.title.as_ref() else {
            continue;
        };
        let orig = norm_back
            .get(title_norm)
            .cloned()
            .unwrap_or_else(|| title_norm.clone());
        if let Some(ii) = page.imageinfo.first() {
            if let Some(url) = ii.thumburl.clone().or_else(|| ii.url.clone()) {
                out.insert(orig, url);
            }
        }
    }
    Ok(out)
}

fn download_with_retry(agent: &ureq::Agent, url: &str) -> anyhow::Result<Vec<u8>> {
    // Mirrors the Python fetcher: 30s, 60s, 120s, 240s backoff on 429/5xx.
    for (i, wait) in [0u64, 30, 60, 120, 240].iter().enumerate() {
        if *wait > 0 {
            thread::sleep(Duration::from_secs(*wait));
        }
        match agent.get(url).call() {
            Ok(resp) => {
                let mut buf = Vec::with_capacity(512 * 1024);
                resp.into_reader().read_to_end(&mut buf)?;
                return Ok(buf);
            }
            Err(ureq::Error::Status(429, _)) => {
                log::warn!("429, backing off (attempt {})", i + 1);
                continue;
            }
            Err(ureq::Error::Status(code, _)) if (500..600).contains(&code) => {
                thread::sleep(Duration::from_secs(5 * (i as u64 + 1)));
                continue;
            }
            Err(e) => return Err(anyhow::anyhow!("{e}")),
        }
    }
    anyhow::bail!("download: exhausted retries")
}

use std::io::Read as _;

/// Paintings count threshold below which we auto-start the downloader.
/// A handful of curated entries gives a useful first-run experience.
pub const AUTOSTART_MIN_PAINTINGS: usize = 10;

/// How many `.jpg` files are in the paintings dir.
pub fn existing_count(paintings_dir: &Path) -> usize {
    std::fs::read_dir(paintings_dir)
        .map(|it| {
            it.flatten()
                .filter(|e| {
                    e.path()
                        .extension()
                        .and_then(|s| s.to_str())
                        .map(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg"))
                        .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0)
}
