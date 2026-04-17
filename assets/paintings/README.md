# Painting gallery

Drop classical portrait paintings here as `<name>.jpg` (or `.png`). On first
launch the app will detect the face in each, compute an ArcFace embedding,
and cache the result to `cache/paintings.json`. Delete that cache file if
you add/remove paintings.

Optional: alongside each image, place a `<name>.json` metadata file:

```json
{
  "title": "Mona Lisa",
  "artist": "Leonardo da Vinci"
}
```

## Recommended starter set

All of the following are in the public domain and easy to find on Wikipedia
or Wikimedia Commons. Rename the downloaded JPEG to the suggested filename.

| Filename | Painting | Artist |
|----------|----------|--------|
| `mona_lisa.jpg` | Mona Lisa | Leonardo da Vinci |
| `girl_with_a_pearl_earring.jpg` | Girl with a Pearl Earring | Johannes Vermeer |
| `self_portrait_van_gogh.jpg` | Self-Portrait with Bandaged Ear | Vincent van Gogh |
| `napoleon.jpg` | The Emperor Napoleon in His Study | Jacques-Louis David |
| `the_arnolfini_portrait.jpg` | The Arnolfini Portrait (groom's face) | Jan van Eyck |
| `young_man_holding_a_roundel.jpg` | Portrait of a Young Man Holding a Roundel | Sandro Botticelli |
| `portrait_of_a_man_in_red_chalk.jpg` | Portrait of a Man in Red Chalk | Leonardo da Vinci |
| `lady_with_an_ermine.jpg` | Lady with an Ermine | Leonardo da Vinci |

Paintings with a single, forward-facing face produce the best swaps.
Profile shots or paintings with tiny faces will often fail detection.
