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
| `hero-maidan-flags.mp4` | The Field of Memory, Kyiv — hero, room 01 | [utopia 36 on Pexels](https://www.pexels.com/video/a-flag-and-flowers-are-placed-in-front-of-a-building-18550804/) | Pexels licence — free to use, credited on the page |
| `hero-maidan-flags.jpg` | Poster frame for the above | derived from the same video | as above |
| `memorial-statue-moy-de-vitry.jpg` | Motherland Monument, Kyiv — **not placed** | [Jonathan Ansel Moy de Vitry on Unsplash](https://unsplash.com/photos/VazH_1OSP9E) ([profile](https://unsplash.com/@jmdv)) | Unsplash Licence |

Nothing is hotlinked. The Pexels and Unsplash CDNs never appear in a `src`.

### Why the 4K master is not in this repo

The original is 3840×2160 at 26.7 Mbps with an audio track: **30.76 MB**. It
lives in `video-hero/`, which is gitignored. What ships is the 1080p
transcode, audio stripped: **3.06 MB**, a tenth of the size and
indistinguishable in a 62vh banner.

That is not tidiness. Git history is permanent — a 30 MB binary committed once
stays in every clone forever, even if deleted in the next commit, and this repo
is 2.9 MB without it. If the master is ever needed again it is on Pexels.

```
ffmpeg -i master.mp4 -vf scale=1920:-2 -c:v libx264 -preset slow -crf 28   -profile:v high -pix_fmt yuv420p -movflags +faststart -an out.mp4
```

`-an` strips audio: a memorial must not make noise, and browsers block
autoplay with sound anyway. `+faststart` moves the index to the front so it
starts playing before it has fully downloaded.

### A note on the statue photograph

`memorial-statue-moy-de-vitry.jpg` is committed but deliberately unplaced. It
shows the Motherland Monument with the **Soviet coat of arms** still on the
shield — Ukraine replaced it with the tryzub in August 2023, as part of
decommunisation during this war. Verify before using it anywhere: a hammer and
sickle at the head of a memorial to Ukrainians killed fighting Russia is not a
detail to discover after publishing.
