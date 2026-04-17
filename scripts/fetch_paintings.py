"""Fetch a curated list of classical portraits from Wikimedia Commons
into assets/paintings/. Writes each as <slug>.jpg + <slug>.json metadata.

Run:   python scripts/fetch_paintings.py

Features:
  - Batched `action=query` imageinfo lookup.
  - Falls back to `generator=search` when a literal File: title 404s.
  - Skips files already present.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = "classic-me-demo/0.1 (+https://github.com/local/classic-me; fetch script)"
API = "https://commons.wikimedia.org/w/api.php"

# slug, "File:...jpg" title, display title, artist, optional search fallback
ITEMS: list[tuple[str, str, str, str, str]] = [
    # Originals (keep in list for idempotency with existing files)
    ("mona_lisa", "File:Mona Lisa, by Leonardo da Vinci, from C2RMF retouched.jpg", "Mona Lisa", "Leonardo da Vinci", "Mona Lisa painting"),
    ("girl_with_a_pearl_earring", "File:1665 Girl with a Pearl Earring.jpg", "Girl with a Pearl Earring", "Johannes Vermeer", "Vermeer Girl Pearl Earring"),
    # New Leonardo
    ("salvator_mundi", "File:Salvator Mundi by Leonardo da Vinci.jpg", "Salvator Mundi", "Leonardo da Vinci", "Salvator Mundi Leonardo"),
    ("la_belle_ferronniere", "File:Leonardo da Vinci (attrib)- la Belle Ferroniere.jpg", "La Belle Ferronnière", "Leonardo da Vinci", "La Belle Ferroniere Leonardo"),
    # Raphael
    ("raphael_castiglione", "File:Baldassare Castiglione, by Raffaello Sanzio, from C2RMF retouched.jpg", "Portrait of Baldassare Castiglione", "Raphael", "Castiglione Raphael portrait"),
    ("raphael_fornarina", "File:Raffael - La Fornarina.jpeg", "La Fornarina", "Raphael", "La Fornarina Raphael"),
    ("raphael_self", "File:Raffaello Sanzio - Self-portrait - Google Art Project.jpg", "Self-Portrait", "Raphael", "Raphael self-portrait"),
    # Titian
    ("titian_man_glove", "File:Titian - Portrait of a Man with a Glove.jpg", "Man with a Glove", "Titian", "Titian man with a glove"),
    ("titian_pietro_aretino", "File:Pietro Aretino by Titian.jpeg", "Pietro Aretino", "Titian", "Pietro Aretino Titian"),
    # Vermeer
    ("vermeer_girl_red_hat", "File:Jan Vermeer van Delft 009.jpg", "Girl with a Red Hat", "Johannes Vermeer", "Vermeer Girl Red Hat"),
    ("vermeer_milkmaid", "File:Johannes Vermeer - Het melkmeisje - Google Art Project.jpg", "The Milkmaid", "Johannes Vermeer", "Vermeer Milkmaid"),
    # Rembrandt
    ("rembrandt_self_portrait", "File:Rembrandt van Rijn - Self-Portrait - Google Art Project.jpg", "Self-Portrait", "Rembrandt van Rijn", "Rembrandt self-portrait Google Art"),
    ("rembrandt_broad_hat", "File:Rembrandt van Rijn - Self Portrait with a Broad-Brimmed Hat - WGA19206.jpg", "Self-Portrait in a Broad-Brimmed Hat", "Rembrandt van Rijn", "Rembrandt self portrait broad hat"),
    ("rembrandt_saskia", "File:Rembrandt Harmensz. van Rijn 085.jpg", "Saskia van Uylenburgh", "Rembrandt van Rijn", "Saskia Rembrandt"),
    # Van Gogh
    ("van_gogh_self", "File:Vincent van Gogh - Self-Portrait - Google Art Project.jpg", "Self-Portrait", "Vincent van Gogh", "Van Gogh self-portrait"),
    ("van_gogh_bandaged_ear", "File:Vincent Willem van Gogh 106.jpg", "Self-Portrait with Bandaged Ear", "Vincent van Gogh", "Van Gogh bandaged ear"),
    ("van_gogh_dr_gachet", "File:Portrait of Dr. Gachet.jpg", "Portrait of Dr. Gachet", "Vincent van Gogh", "Van Gogh Gachet"),
    ("van_gogh_postman", "File:Vincent van Gogh - Portrait of Postman Roulin - Google Art Project.jpg", "The Postman Joseph Roulin", "Vincent van Gogh", "Van Gogh Roulin postman"),
    # Botticelli
    ("botticelli_young_man", "File:Portrait of a Young Man by Sandro Botticelli - Louvre.jpg", "Portrait of a Young Man", "Sandro Botticelli", "Botticelli Young Man Louvre"),
    ("botticelli_simonetta", "File:Sandro Botticelli 059.jpg", "Portrait of Simonetta Vespucci", "Sandro Botticelli", "Simonetta Vespucci Botticelli"),
    ("botticelli_giuliano", "File:Sandro Botticelli 063.jpg", "Portrait of Giuliano de' Medici", "Sandro Botticelli", "Giuliano Medici Botticelli"),
    # Velázquez
    ("velazquez_innocent_x", "File:Diego Velázquez - Portrait of Pope Innocent X - Google Art Project.jpg", "Portrait of Pope Innocent X", "Diego Velázquez", "Velazquez Innocent X"),
    ("velazquez_juan_pareja", "File:Juan de Pareja by Diego Velázquez.jpg", "Juan de Pareja", "Diego Velázquez", "Juan de Pareja Velazquez"),
    ("velazquez_philip_iv", "File:Diego Velázquez 043.jpg", "Portrait of Philip IV of Spain in Armour", "Diego Velázquez", "Velazquez Philip IV"),
    ("velazquez_infanta_margarita", "File:Velazquez-lasmeninas01.jpg", "Infanta Margarita (Las Meninas detail)", "Diego Velázquez", "Las Meninas Infanta"),
    # Goya
    ("goya_self_spectacles", "File:Francisco de Goya y Lucientes - Self-Portrait with Spectacles - WGA10020.jpg", "Self-Portrait with Spectacles", "Francisco Goya", "Goya self-portrait spectacles"),
    ("goya_duchess_alba", "File:Alba Goya.jpg", "The Duchess of Alba", "Francisco Goya", "Duchess of Alba Goya"),
    # Ingres
    ("ingres_madame_moitessier", "File:Jean-Auguste-Dominique Ingres - Madame Moitessier - Google Art Project.jpg", "Madame Moitessier", "Jean-Auguste-Dominique Ingres", "Madame Moitessier Ingres"),
    ("ingres_caroline_riviere", "File:Jean Auguste Dominique Ingres 014.jpg", "Mademoiselle Caroline Rivière", "Jean-Auguste-Dominique Ingres", "Caroline Riviere Ingres"),
    # Gainsborough
    ("gainsborough_blue_boy", "File:Thomas Gainsborough - The Blue Boy - Google Art Project.jpg", "The Blue Boy", "Thomas Gainsborough", "Blue Boy Gainsborough"),
    ("gainsborough_mrs_siddons", "File:Thomas Gainsborough Lady Sarah Siddons.jpg", "Mrs. Siddons", "Thomas Gainsborough", "Mrs Siddons Gainsborough"),
    # Van Eyck
    ("arnolfini_portrait", "File:Van Eyck - Arnolfini Portrait.jpg", "The Arnolfini Portrait", "Jan van Eyck", "Arnolfini Portrait Van Eyck"),
    ("van_eyck_red_turban", "File:Portrait of a Man in a Red Turban (Jan van Eyck, 1433) cleaned.jpg", "Portrait of a Man (Self-Portrait?)", "Jan van Eyck", "Van Eyck red turban self portrait"),
    # Dürer
    ("durer_self_1500", "File:Albrecht Dürer - 1500 self-portrait (High resolution and detail).jpg", "Self-Portrait at 28", "Albrecht Dürer", "Durer self portrait 1500"),
    ("durer_self_1498", "File:Albrecht Dürer 070.jpg", "Self-Portrait at 26", "Albrecht Dürer", "Durer self portrait 1498"),
    # Hals
    ("hals_laughing_cavalier", "File:Frans Hals - The Laughing Cavalier - WGA11093.jpg", "The Laughing Cavalier", "Frans Hals", "Laughing Cavalier Hals"),
    ("hals_malle_babbe", "File:Frans Hals - Malle Babbe - Google Art Project.jpg", "Malle Babbe", "Frans Hals", "Malle Babbe Hals"),
    # Manet
    ("manet_berthe_morisot", "File:Berthe Morisot With a Bouquet of Violets.jpg", "Berthe Morisot with a Bouquet of Violets", "Édouard Manet", "Berthe Morisot bouquet Manet"),
    ("manet_olympia", "File:Edouard Manet - Olympia - Google Art Project 3.jpg", "Olympia", "Édouard Manet", "Manet Olympia"),
    # Renoir
    ("renoir_jeanne_samary", "File:Jeanne Samary by Pierre-Auguste Renoir 1879.jpg", "Portrait of Jeanne Samary", "Pierre-Auguste Renoir", "Jeanne Samary Renoir"),
    # Caravaggio
    ("caravaggio_bacchus", "File:Michelangelo Caravaggio 065.jpg", "Young Sick Bacchus", "Caravaggio", "Young sick Bacchus Caravaggio"),
    ("caravaggio_narcissus", "File:Narcissus-Caravaggio (1594-96) edited.jpg", "Narcissus", "Caravaggio", "Narcissus Caravaggio"),
    ("caravaggio_medusa", "File:Medusa by Carvaggio.jpg", "Medusa", "Caravaggio", "Medusa Caravaggio"),
    ("caravaggio_fruit_basket", "File:Caravaggio - Boy with a Basket of Fruit.jpg", "Boy with a Basket of Fruit", "Caravaggio", "Boy basket fruit Caravaggio"),
    # Holbein
    ("holbein_henry_viii", "File:Hans Holbein d. J. 074.jpg", "Portrait of Henry VIII", "Hans Holbein", "Holbein Henry VIII"),
    ("holbein_jane_seymour", "File:Hans Holbein the Younger - Jane Seymour, Queen of England - Google Art Project.jpg", "Jane Seymour", "Hans Holbein", "Holbein Jane Seymour"),
    # Klimt / Vigée / Sargent / Whistler / Cézanne / Courbet / Delacroix
    ("klimt_adele", "File:Gustav Klimt 046.jpg", "Portrait of Adele Bloch-Bauer I", "Gustav Klimt", "Klimt Adele Bloch-Bauer"),
    ("vigee_marie_antoinette", "File:Marie Antoinette Adult.jpg", "Marie Antoinette", "Élisabeth Vigée Le Brun", "Marie Antoinette Vigee Le Brun"),
    ("sargent_madame_x", "File:John Singer Sargent - Madame X - The Metropolitan Museum of Art.jpg", "Madame X", "John Singer Sargent", "Madame X Sargent"),
    ("whistler_mother", "File:Whistlers Mother high res.jpg", "Whistler's Mother", "James McNeill Whistler", "Whistler's Mother"),
    ("cezanne_self", "File:Paul Cézanne 157.jpg", "Self-Portrait", "Paul Cézanne", "Cezanne self-portrait"),
    ("courbet_despair", "File:Gustave Courbet - Le Désespéré.JPG", "The Desperate Man", "Gustave Courbet", "Courbet Desperate Man"),
    ("delacroix_self", "File:Eugène Ferdinand Victor Delacroix 019.jpg", "Self-Portrait", "Eugène Delacroix", "Delacroix self portrait"),
    # Rubens
    ("rubens_susanna_lunden", "File:Peter Paul Rubens 104.jpg", "Susanna Lunden", "Peter Paul Rubens", "Rubens Susanna Lunden"),
    ("rubens_helena", "File:Peter Paul Rubens 105.jpg", "Portrait of Hélène Fourment", "Peter Paul Rubens", "Rubens Helena Fourment"),
    # Bronzino / Memling / Piero / Ghirlandaio
    ("bronzino_lucrezia", "File:Bronzino - Lucrezia Panciatichi.jpg", "Lucrezia Panciatichi", "Bronzino", "Bronzino Lucrezia Panciatichi"),
    ("memling_man_with_coin", "File:Hans Memling 050.jpg", "Man with a Roman Coin", "Hans Memling", "Memling man with Roman coin"),
    ("piero_federico", "File:Piero della Francesca 046.jpg", "Federico da Montefeltro", "Piero della Francesca", "Piero della Francesca Federico"),
    ("ghirlandaio_old_man", "File:Domenico Ghirlandaio - Old Man with his Grandson - Google Art Project.jpg", "Old Man with his Grandson", "Domenico Ghirlandaio", "Ghirlandaio old man grandson"),
    # El Greco / Murillo / Zurbarán
    ("el_greco_nobleman", "File:Gentleman with his Hand on his Chest.jpg", "Gentleman with his Hand on his Chest", "El Greco", "El Greco gentleman hand chest"),
    ("murillo_boys", "File:Bartolomé Esteban Perugino - Two Boys Eating a Melon and Grapes - Google Art Project.jpg", "Two Boys Eating Melon and Grapes", "Murillo", "Murillo boys melon grapes"),
    # Reynolds / Copley / Stuart
    ("reynolds_sarah_siddons", "File:Sir Joshua Reynolds - Sarah Siddons as the Tragic Muse - Google Art Project.jpg", "Sarah Siddons as the Tragic Muse", "Joshua Reynolds", "Reynolds Sarah Siddons Tragic Muse"),
    ("stuart_washington", "File:Gilbert Stuart - George Washington - Google Art Project (721059).jpg", "George Washington (Athenaeum)", "Gilbert Stuart", "Gilbert Stuart Washington"),
    # Modigliani / Sargent extras / Sargent Gautreau
    ("sargent_carnation_lily", "File:Carnation, Lily, Lily, Rose, by John Singer Sargent, 1885-6.jpg", "Carnation, Lily, Lily, Rose", "John Singer Sargent", "Sargent Carnation Lily Rose"),
    # Corot / Chardin
    ("corot_self", "File:Camille Corot, Self-Portrait.jpg", "Self-Portrait", "Jean-Baptiste-Camille Corot", "Corot self-portrait"),
    # Vermeer extras
    ("vermeer_lacemaker", "File:Johannes Vermeer - The lacemaker (c.1669-1671).jpg", "The Lacemaker", "Johannes Vermeer", "Vermeer Lacemaker"),
    # Raphael Madonna (faces)
    ("raphael_madonna_chair", "File:Raphael - Madonna della seggiola.jpg", "Madonna della seggiola", "Raphael", "Madonna della seggiola Raphael"),
    # David extras
    ("david_marat", "File:Death of Marat by David.jpg", "Death of Marat", "Jacques-Louis David", "David Death of Marat"),
    ("david_recamier", "File:Jacques-Louis David 016.jpg", "Portrait of Madame Récamier", "Jacques-Louis David", "David Recamier"),
    # Original napoleon + lady with ermine are already in existing list
    ("napoleon", "File:Jacques-Louis David - The Emperor Napoleon in His Study at the Tuileries - Google Art Project.jpg", "The Emperor Napoleon in His Study", "Jacques-Louis David", "David Napoleon Tuileries"),
    ("lady_with_an_ermine", "File:The Lady with an Ermine.jpg", "Lady with an Ermine", "Leonardo da Vinci", "Lady with Ermine Leonardo"),
    ("self_portrait_van_gogh", "File:Vincent van Gogh - Self-Portrait - Google Art Project.jpg", "Self-Portrait (1887)", "Vincent van Gogh", "Van Gogh self-portrait"),
]


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def api_query_titles(titles: list[str]) -> dict:
    body = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": "800",
            "titles": "|".join(titles),
        }
    ).encode("utf-8")
    req = urllib.request.Request(API, data=body, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def api_search(query: str) -> str | None:
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query + " filetype:bitmap",
            "gsrnamespace": "6",
            "gsrlimit": "1",
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": "800",
        }
    )
    req = urllib.request.Request(f"{API}?{params}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        j = json.loads(r.read())
    for page in j.get("query", {}).get("pages", {}).values():
        if "imageinfo" in page:
            ii = page["imageinfo"][0]
            return ii.get("thumburl") or ii.get("url")
    return None


def resolve_urls(items) -> dict[str, str]:
    by_norm: dict[str, tuple[str, str]] = {}
    for slug, title, _, _, fallback in items:
        by_norm[title] = (slug, fallback)

    url_for: dict[str, str] = {}
    batch = 15
    titles = [t for _, t, _, _, _ in items]
    for i in range(0, len(titles), batch):
        chunk = titles[i : i + batch]
        try:
            j = api_query_titles(chunk)
        except Exception as e:
            print(f"[warn] batch query failed: {e}", file=sys.stderr)
            continue
        norm_back = {t: t for t in chunk}
        for n in j.get("query", {}).get("normalized", []):
            norm_back[n["to"]] = n["from"]
        for page in j.get("query", {}).get("pages", {}).values():
            t_norm = page.get("title", "")
            orig = norm_back.get(t_norm, t_norm)
            if orig in by_norm and "imageinfo" in page:
                ii = page["imageinfo"][0]
                url = ii.get("thumburl") or ii.get("url")
                if url:
                    url_for[by_norm[orig][0]] = url
        time.sleep(0.2)

    # For anything still missing, try search fallback.
    for slug, _, _, _, fallback in items:
        if slug in url_for or not fallback:
            continue
        try:
            url = api_search(fallback)
            if url:
                url_for[slug] = url
                print(f"[search-ok] {slug}  ({fallback})")
            time.sleep(0.3)
        except Exception as e:
            print(f"[search-fail] {slug}: {e}", file=sys.stderr)
    return url_for


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "assets" / "paintings"
    out_dir.mkdir(parents=True, exist_ok=True)

    url_for = resolve_urls(ITEMS)
    print(f"\nResolved {len(url_for)}/{len(ITEMS)} URLs. Downloading...\n")

    ok = 0
    for slug, _, disp, artist, _ in ITEMS:
        url = url_for.get(slug)
        jpg = out_dir / f"{slug}.jpg"
        meta = out_dir / f"{slug}.json"
        if jpg.exists() and meta.exists():
            print(f"[skip] {slug}")
            ok += 1
            continue
        if not url:
            print(f"[nourl] {slug}")
            continue
        try:
            data = get(url)
            jpg.write_bytes(data)
            meta.write_text(
                json.dumps({"title": disp, "artist": artist}, ensure_ascii=False)
            )
            print(f"[ok]   {slug}  ({len(data)//1024} KB)")
            ok += 1
            time.sleep(0.2)
        except Exception as e:
            print(f"[fail] {slug}: {e}", file=sys.stderr)

    print(f"\nDone. {ok}/{len(ITEMS)} paintings present in {out_dir}")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
