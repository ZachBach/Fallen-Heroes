# images/

Every photograph on this page lives here, committed, and is referenced by path
from the tables at the top of the `<script data-dc-script>` block in
`index.html` — `HERO`, `PORTRAITS`, `OBJECTS`.

Dropping a file onto the live site does **not** work. `image-slot.js` gates
editing on `window.omelette.writeFile`, which only exists inside the design
tool, so on GitHub Pages every slot is read-only. That is why images are
committed rather than dropped.

## Hanging a photograph

1. Put the file here.
2. Fill in the matching `src` in `index.html`, plus `credit` / `creditHref` if
   the licence needs attribution.
3. `node verify.mjs` — it fails on any cross-origin request, which is the check
   that catches an accidentally pasted URL.

```js
const HERO = {
  src: './images/hero.webp',
  credit: 'Ministry of Defence of Ukraine, CC BY 2.0',
  creditHref: 'https://www.flickr.com/photos/ministryofdefenceua/',
};
```

## Rules

**Local paths only.** No hotlinking — not Getty, not Unsplash's CDN, not Drive.
Anything cross-origin breaks the page's no-external-request guarantee, and that
guarantee is the reason this page can promise it does not report a visitor's
IP to anyone. `verify.mjs` enforces it.

**Format and size.** WebP or AVIF. Portraits are 4:5, objects 1:1, the hero is
full-bleed. Roughly 1600px on the long edge is plenty — the page is 2.1 MB
without photographs and it would be a shame to triple that.

**Attribution.** Where a licence requires credit, set `credit` and
`creditHref`. For Unsplash this is not advisory: `image-slot.js` renders an
error tile *instead of* the photo if an Unsplash source has no credit, because
showing the photo uncredited is itself the terms violation.

## Licensing — read before buying anything

The repository is **public and Apache-2.0**, and anyone may fork it. That is
incompatible with most stock photography licences, which forbid
redistribution and sublicensing. A rights-managed or subscription image
committed here would be redistributed by every clone.

A time-limited licence is a second mismatch. The page describes itself as a
**permanent** memorial; a photograph licensed for three months is not
permanent, and continued use after it lapses is infringement.

So the images that fit are the ones that are free to redistribute:

- **Ukrainian government and military sources.** The Ministry of Defence and
  the Office of the President publish a great deal of photography under
  Creative Commons. Check the licence on the individual image — it varies —
  but this route is both legally clean and the right provenance for this
  memorial.
- **Wikimedia Commons**, filtered by licence.
- **Unsplash**, free with attribution, already enforced by the component.
- **Families and units directly**, which is what the rest of the page is built
  around.

Whatever the source, record where each file came from and under what licence
in this file, next to its name. A memorial that cannot say where its
photographs came from has a hole in exactly the place it should be strongest.

## Provenance

| file | subject | source | licence |
| ---- | ------- | ------ | ------- |
| `hero-defenders-vony-razom.jpg` | Defenders in the field — hero, room 01 | [Vony Razom on Unsplash](https://unsplash.com/photos/OMQB3qvCTq4) ([profile](https://unsplash.com/@vonyrazom)) | Unsplash Licence — free to use, credited on the page |
| `memorial-statue-moy-de-vitry.jpg` | People beside a statue — not currently placed | [Jonathan Ansel Moy de Vitry on Unsplash](https://unsplash.com/photos/VazH_1OSP9E) ([profile](https://unsplash.com/@jmdv)) | Unsplash Licence — free to use, credit if placed |

Both were downloaded at 1920px and committed. Nothing is hotlinked: the
Unsplash CDN never appears in a src, only in the credit links, which are
anchors and make no request until somebody clicks them.

`memorial-statue-moy-de-vitry.jpg` is 1920×2560 — portrait, so it suits a
portrait frame rather than the full-bleed hero. It is committed but not
placed. Delete it if it is not wanted.
