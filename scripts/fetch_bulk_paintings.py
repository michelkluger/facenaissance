"""Bulk-fetch portrait paintings from Wikimedia Commons categories.

Walks a set of high-signal portrait categories (Google Art Project portraits,
famous-museum portrait collections, self-portraits, century-grouped
portraits) and downloads each unique file as a 600-px thumbnail.

Existing files are skipped, so reruns are cheap. Metadata is derived from
the Commons filename.

Run:
    python scripts/fetch_bulk_paintings.py [TARGET_COUNT]
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = "facenaissance/0.1 (+https://github.com/michelkluger/facenaissance; bulk fetch)"
API = "https://commons.wikimedia.org/w/api.php"

# Prioritised list of portrait-heavy categories. We walk them in order and
# stop once we've collected TARGET unique files. Each category can easily
# have 500–3000 files, so we don't usually need many.
CATEGORIES = [
    "Portrait paintings in the Google Art Project",
    "Self-portraits by painters",
    "Portrait paintings in the Metropolitan Museum of Art",
    "Portrait paintings in the National Gallery, London",
    "Portrait paintings in the Louvre",
    "Portrait paintings in the Museo del Prado",
    "Portrait paintings in the Rijksmuseum Amsterdam",
    "Portrait paintings in the Uffizi Gallery",
    "Portrait paintings in the National Portrait Gallery, London",
    "17th-century portrait paintings",
    "18th-century portrait paintings",
    "19th-century portrait paintings",
    "16th-century portrait paintings",
    "15th-century portrait paintings",
    "Portrait paintings by Rembrandt van Rijn",
    "Portrait paintings by John Singer Sargent",
    "Portrait paintings by Anthony van Dyck",
    "Portrait paintings by Diego Velázquez",
    "Portrait paintings by Titian",
    "Portrait paintings by Thomas Gainsborough",
]

# Filenames containing any of these substrings are (almost always) not clean
# frontal portraits — group scenes, maps, architecture, crops of hands, etc.
BLOCK_SUBSTRINGS = (
    "group portrait",
    "family",
    "crucifixion",
    "adoration",
    "detail",
    "sketch",
    "study",
    "draft",
    "map",
    "architecture",
    "ceiling",
    "panorama",
    "battle",
    "landscape",
    "after-restoration comparison",
    "x-ray",
    "photograph",
    "framing",
    "drawing",
    "engraving",
    "etching",
    "lithograph",
    "diagram",
)


def _open_with_retry(req: urllib.request.Request, attempts: int = 5) -> bytes:
    """urlopen with exponential backoff. Wikimedia API limit is 500/hour
    for anonymous clients; upload.wikimedia.org tolerates ~1 req/s bursts
    before HAProxy 429s. Respects Retry-After when provided."""
    backoffs = (0, 30, 60, 120, 240)
    last_exc: Exception | None = None
    for i, base_wait in enumerate(backoffs[:attempts]):
        if base_wait:
            time.sleep(base_wait)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code == 429:
                retry_after = e.headers.get("Retry-After") if e.headers else None
                if retry_after and retry_after.isdigit():
                    time.sleep(min(int(retry_after), 300))
                print(f"  [429] backing off (attempt {i + 1})…", file=sys.stderr)
                continue
            if e.code in (500, 502, 503, 504):
                time.sleep(5 * (i + 1))
                continue
            raise
        except Exception as e:
            last_exc = e
            time.sleep(3)
    raise last_exc if last_exc else RuntimeError("http: exhausted retries")


def http_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return json.loads(_open_with_retry(req))


def http_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return _open_with_retry(req)
    raise last_exc if last_exc else RuntimeError("http_bytes: exhausted retries")


def walk_category(category: str, hard_limit: int) -> list[tuple[str, str]]:
    """Return (title, thumb_url) for files in a category (transitively)."""
    out: list[tuple[str, str]] = []
    cmcontinue = None
    while len(out) < hard_limit:
        params = {
            "action": "query",
            "format": "json",
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file",
            "gcmlimit": "50",
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": "600",
        }
        if cmcontinue:
            params["gcmcontinue"] = cmcontinue
        url = f"{API}?{urllib.parse.urlencode(params)}"
        try:
            j = http_json(url)
        except Exception as e:
            print(f"  [api-err] {category}: {e}", file=sys.stderr)
            break

        pages = j.get("query", {}).get("pages", {}) or {}
        for page in pages.values():
            title = page.get("title", "")
            ii = page.get("imageinfo") or []
            if not ii:
                continue
            url = ii[0].get("thumburl") or ii[0].get("url")
            if not url:
                continue
            if not any(title.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png")):
                continue
            low = title.lower()
            if any(b in low for b in BLOCK_SUBSTRINGS):
                continue
            out.append((title, url))

        cont = j.get("continue", {})
        cmcontinue = cont.get("gcmcontinue")
        if not cmcontinue:
            break
        time.sleep(1.0)
    return out


_slug_re = re.compile(r"[^a-zA-Z0-9]+")


def slug_from_title(title: str) -> str:
    # Strip "File:" prefix and extension.
    name = title
    if name.lower().startswith("file:"):
        name = name[5:]
    name = re.sub(r"\.(jpe?g|png)$", "", name, flags=re.IGNORECASE)
    s = _slug_re.sub("_", name).strip("_").lower()
    # Keep slugs short but recognisable.
    return s[:80]


def derive_meta(title: str) -> tuple[str, str]:
    """Best-effort (display_title, artist) from a Wikimedia File: title."""
    raw = title[5:] if title.lower().startswith("file:") else title
    raw = re.sub(r"\.(jpe?g|png)$", "", raw, flags=re.IGNORECASE)
    # Common patterns: "Artist - Title - Museum" or "Artist 001.jpg"
    parts = [p.strip() for p in raw.split(" - ") if p.strip()]
    if len(parts) >= 2:
        artist = parts[0]
        display = parts[1]
        if len(parts) >= 3 and any(tag in parts[-1].lower() for tag in ("google art", "wga", "nga", "museum")):
            pass  # drop trailing catalogue tag
        return display, artist
    if len(parts) == 1:
        return parts[0], "Unknown"
    return raw, "Unknown"


def main() -> int:
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "assets" / "paintings"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = {p.stem for p in out_dir.glob("*.jpg")}
    print(f"{len(existing)} paintings already on disk. Target: {target} total.")

    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for cat in CATEGORIES:
        if len(candidates) + len(existing) >= target:
            break
        print(f"\n==> {cat}")
        got = walk_category(cat, 600)
        new = 0
        for title, url in got:
            slug = slug_from_title(title)
            if not slug or slug in seen or slug in existing:
                continue
            seen.add(slug)
            candidates.append((title, url))
            new += 1
            if len(candidates) + len(existing) >= target:
                break
        print(f"   {len(got)} raw, {new} new unique")

    print(f"\nDownloading {len(candidates)} new paintings...\n")
    ok = 0
    fail = 0
    for title, url in candidates:
        slug = slug_from_title(title)
        jpg = out_dir / f"{slug}.jpg"
        meta = out_dir / f"{slug}.json"
        if jpg.exists():
            continue
        try:
            data = http_bytes(url)
            if len(data) < 20_000:
                # Tiny file = probably broken / icon.
                print(f"  [skip-tiny] {slug}  ({len(data)} B)", file=sys.stderr)
                continue
            jpg.write_bytes(data)
            disp, artist = derive_meta(title)
            meta.write_text(
                json.dumps({"title": disp, "artist": artist}, ensure_ascii=False)
            )
            ok += 1
            if ok % 25 == 0:
                print(f"  [{ok}] {slug}")
            # ~1 req/s matches Wikimedia's polite-client expectation; see
            # https://api.wikimedia.org/wiki/Rate_limits (500/hour anon) and
            # https://meta.wikimedia.org/wiki/User-Agent_policy.
            time.sleep(1.0)
        except Exception as e:
            fail += 1
            if fail < 10:
                print(f"  [fail] {slug}: {e}", file=sys.stderr)

    total = len(list(out_dir.glob("*.jpg")))
    print(f"\nDone. {ok} new, {fail} failed, {total} paintings on disk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
