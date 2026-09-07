# Fallen-Heroes — Вічна пам'ять

A permanent, non-commercial memorial to the fallen defenders of Ukraine: a
church rendered live, filled with five hundred unlit votive candles, where a
visitor lights the first flame.

Static HTML. No build step, no backend, and **no external request of any kind**.

`github.com/ZachBach/Fallen-Heroes`

```bash
python -m http.server 8000     # then open http://localhost:8000/
node verify.mjs                # headless gate, both GPU backends
```

## What is in here

| file / dir          | role                                                        |
| ------------------- | ----------------------------------------------------------- |
| `index.html`        | the page — markup, styles, and the logic class               |
| `MemorialLight.js`  | the Hall of light: three.js `WebGPURenderer` + TSL           |
| `support.js`        | dc-runtime (generated — do not edit)                         |
| `image-slot.js`     | `<image-slot>` custom element (starter copy — do not edit)   |
| `vendor/`           | three.js, React, OrbitControls, fonts — see `INTEGRITY.txt`  |
| `verify.mjs`        | headless gate, both backends                                 |
| `.nojekyll`         | stops GitHub Pages running the page through Jekyll           |
| `LICENSE`           | Apache License 2.0 — covers the software                     |
| `NOTICE`            | what the licence does *not* cover; travels with every copy   |

2.1 MB total: 1.0 MB three.js, 552 KB fonts, 144 KB React, the rest page and
component code.

`MemorialLight` is a `.js`, not a `.jsx`, on purpose: it contains no JSX
syntax, so the dc-runtime loads it directly and never fetches
`@babel/standalone`. Keep it that way — one JSX tag in that file silently adds
a 3 MB download from unpkg.com.

## This page makes no external request

Not "almost none". None. `verify.mjs` fails if a single cross-origin request is
made, and it reports `0 external` on both backends.

Three separate things had to be vendored, because the obvious two were not all
of them:

- **three.js** came from `cdn.jsdelivr.net` — now `vendor/`, r0.184.0, minified.
- **Cormorant Garamond and IBM Plex Mono** came from `fonts.googleapis.com` /
  `fonts.gstatic.com` — now `vendor/fonts/`, in latin, latin-ext, cyrillic and
  cyrillic-ext. Cyrillic is not optional: the page is Ukrainian.
- **React, ReactDOM and Babel** came from `unpkg.com`, requested by
  `support.js` itself. React and ReactDOM now load from `vendor/` in `<head>`
  before `support.js`, whose loader short-circuits on
  `window.React && window.ReactDOM`. Babel is gone entirely.

It matters more here than on most pages. Visitors are grieving families, some
inside Ukraine, where `fonts.googleapis.com` is not reliably reachable in the
first place.

Every vendored file's provenance, sha384 and re-verification recipe is in
[`vendor/INTEGRITY.txt`](vendor/INTEGRITY.txt), each checked byte-identical
against upstream.

The import map lives in `<head>`, not in the `<helmet>`. A helmet is injected
at render time and the dc-runtime compiles the template twice; the copies
differ by whitespace, so the helmet's dedupe key misses and the browser rejects
the duplicate map as a conflict.

## The Hall of light

### The room

A nave with a seven-bay arcade down each side, ashlar walls, an apse and altar
dais, lancet windows the length of both walls and a rose window over the altar.
Nothing is modelled from a real building, and the page says so.

The glass is **blue and gold only**. Not a stylistic restraint — it is the
flag, and in a memorial to Ukraine's dead the window has no business being
anything else. It also happens to be near the maximum contrast the eye can be
given, which is what finally made the figure in it legible.

That figure is **Michael** — by convention the archangel who guards the dead,
and the figure on Kyiv's own coat of arms. He is a stylised icon assembled from
circles and ellipses, reproducing no particular painting and not offered as
one. Two things made him readable: he fills two thirds of the panel rather than
floating in the middle of it, and the whole silhouette is drawn **twice**, once
at size and once fractionally larger, so the difference between the two masks
becomes a lead outline. Figurative glass is always drawn with the figure cames
heavier than the field cames; without that, gold on blue dissolves into the
diaper behind it.

### Day and night

The room crosses between them on a timer (`cycleSeconds`, default 210). One
uniform drives everything that changes with the hour: the sky beyond the
windows, the colour the glass throws onto the stone, the direction the sun
swings, and whether the shafts falling through the nave are sunlight or
moonlight. It holds at each end and crosses quickly, because a church at noon
and a church at midnight are both worth sitting in and the permanent dusk
between them is not.

Night is genuinely dark but never black. A church lit only by the moon and a
rack of candles still shows you its walls, and one that does not is a black
rectangle for half of every cycle.

### The candles

Every candle starts **unlit**. A visitor arrives to a dark church full of
candles nobody has lit yet, and the first flame in it is theirs. The whole rack
stands there from the first frame; only the flames, halos and plume are gated
on the count.

A room holds **500** and no more. When it fills, the next candle opens the next
room, and only the room being looked at is ever built or drawn. That ceiling is
what buys the quality everywhere else: 500 candles all fit in the near field,
so every one carries full-detail geometry, and the whole million-particle
budget is shared between 500 wicks rather than thousands.

Layout is a golden-angle phyllotaxis spiral — the packing that keeps density
even, the same reason sunflowers use it. The *ordering* is separate: candles
are sorted by distance from the altar, so lighting begins at the steps and
spreads outward down the nave. Position is a pure function of the index and the
count lives in `localStorage`, so a flame lit last year is still burning, in
the same place, when its family comes back.

### A million particles

Smoke, sparks and ash are a GPU simulation: two storage buffers stepped by a
TSL compute shader every frame — position + life, velocity + seed. Buoyancy
strongest while a particle is hot, one octave of noise whose strength rises
with age so a plume leaves the wick narrow and opens out above it, drag, then
respawn at a candle drawn from the lit range.

On WebGPU that is a genuine compute pass at **1,000,000 particles**. three's
WebGL2 backend implements the same node through transform feedback, so one
piece of code drives both — only the count differs (150,000 there).

- **Density normalisation.** Per-particle brightness scales by
  `candles / particles drawn`, so light emitted *per candle* is constant
  whatever the count or the device tier. Without it the room is a bonfire at
  low counts and a wisp at high ones.
- **Adaptive draw range.** The simulation always steps the full field; the
  number *drawn* backs off below 28 fps and climbs back above 55, floor 60,000.
  Which particles get drawn is arbitrary, so the plume just thins.

Lifespans are 4–9 seconds with gentle buoyancy on purpose. Short lives plus a
strong rise is exactly what makes a particle system read as a firework, which
is the one thing this room must never look like.

### What the shaders do

Everything is procedural — no textures, no model files.

- **Wax** — a lathe profile with a concave molten pool, vertex drip-runs, and
  per-candle girth, burn-down and tint. The important part is subsurface
  scattering: light from the flame enters the top and travels down the pillar,
  brightest under the pool and at thin silhouette edges. Emissive rather than
  real transmission, which needs a scene pass per object.
- **Flame** — a Y-locked billboard with a `sin(π·y^0.72)` teardrop, turbulence
  advected upward and domain-warped, colour ramped by temperature, a blue
  Swan-band cone at the base, and the dark inner cone of unburnt vapour above
  the wick.
- **Votive holder** — a fluted glass cup, flutes as real geometry so the
  silhouette scallops and each rib catches the fresnel separately. Tinted
  cooler than the wax deliberately: a warm holder merges into the candle and
  disappears.
- **Stone** — coursed ashlar with the joints cut as darker lines rather than
  modelled, every course offset by half a block.

Flames are deliberately **not** an `InstancedMesh`. TSL's `billboarding()`
builds its basis from the mesh's world matrix and ignores instance matrices, so
every instanced flame collapses onto the mesh origin. They are one shared quad
buffer with per-candle attributes instead.

### Moving around

Orbit controls, with a **full 360°** turn — the view from behind the altar,
looking back down the church over every flame, is the best one in the room.

Zoom and pan are **off** on purpose: this canvas is a section inside a
scrolling page, and a control that ate the wheel would trap a visitor who was
only trying to reach the next room. Rotation is the whole interaction, and
auto-rotate stops the moment someone takes hold.

The nave's dimensions are set by the orbit, not the other way round. The camera
sits ~25 units from the rack and a full turn sweeps that in every direction, so
the arcade stands further out than 25 and the apse further back than the rack
plus 25. That is why the church is the size it is: it is what lets the camera
go all the way round without ever leaving the building. An earlier version
fenced the rotation to ±36° instead, which treated the symptom.

### Props

| prop           | default     | notes                                        |
| -------------- | ----------- | -------------------------------------------- |
| `roomSize`     | `500`       | 100–1000; changing it rebuilds the room       |
| `cycleSeconds` | `210`       | 30–900; one full turn of day and night        |
| `goldAccent`   | `"#C9A227"` | floor pool, flames, embers, rules             |
| `driftCamera`  | `true`      | slow auto-rotate until a visitor takes hold   |

### Backend

WebGPU where available, automatic WebGL2 fallback otherwise. The component
reports which one it actually got and the page prints that, rather than the
page claiming one. If three.js fails to load entirely the section renders flat
`#0d0e10` and the page still reads.

## Two traps worth knowing

**`backdrop-filter` over the canvas.** The sticky nav used to carry
`backdrop-filter: blur(10px)`. Blurring a backdrop containing a
continuously-updating canvas re-runs the blur every frame: measured on an Intel
UHD 630, removing it took the hall from **1 fps to 13**. Do not put it back.

**Coplanar geometry.** Everything standing on the floor is sunk below it —
walls to `-0.6`, pier bases to `-0.5`, the altar's upper step embedded in the
lower. Two faces at exactly the same depth give the buffer no way to break the
tie, so it picks differently per pixel and per frame; it shows as a dark line
that crawls when the camera moves. Keep new floor-standing geometry sunk.

## Verifying

```bash
node verify.mjs            # both backends
node verify.mjs webgpu     # one
```

Needs `puppeteer-core` (it will borrow an existing install, or `npm i
puppeteer-core`) and Chrome — set `CHROME_PATH` if it is not at the default
Windows location. Output goes to `.verify-out/`, which is gitignored.

It gates on: no console or page errors, the backend actually reached and
reported, **zero cross-origin requests**, the particle simulation running, a
peak frame rate above 20, and candles lighting, displaying and surviving a
reload, plus the room arithmetic.

Last run: `ALL CHECKS PASSED`, `0 external`, **60 fps peak** on both backends,
1,000,000 particles simulated and drawn on WebGPU.

Two traps if you extend the verifier. Its backend assertion must not match the
*pre-report* string "Rendered live · WebGPU with WebGL2 fallback" — a loose
regex there once let the check pass before the component had reported anything.
And frame rate is sampled as a **peak across a window**, because a single
reading lands in one of two ditches: shader compilation makes early frames
long, and headless Chrome throttles rAF to ~13 fps once it decides the page is
occluded, regardless of what it draws.

## Deployment

GitHub Pages, from this repo's `main` branch at the root. Deliberately separate
from `aureliusdynamic.com` — this is not a studio project and does not belong
on a business site.

It serves at **`https://zachbach.github.io/Fallen-Heroes/`**. Note that is a
*project* page, not the user page: `zachbach.github.io` with no path is served
by a repo that must itself be named `zachbach.github.io`. If you want this at
the bare domain, it has to move to that repo, and it would then displace
whatever else lives there.

Everything is relative-path and same-origin, so the subdirectory is fine as-is
— nothing needs a base href. To turn it on: **Settings → Pages → Source:
Deploy from a branch → `main` / `/ (root)`**.

## Lighting a candle

The count is stored in `localStorage` under `memorial.candles.lit` and is never
sent anywhere. Storage failures are swallowed — the room still works with site
data blocked, it just does not remember.

Note what this does and does not mean: the count is **per device**. It is not a
shared tally across visitors, and the page does not claim to be one.

## Image slots

Every photograph is an empty `<image-slot>`, filled by dropping a file on it.
Slot ids: `memorial-hero`, `portrait-1` … `portrait-8`, `obj-1` … `obj-3`.

Dropped images persist in a `.image-slots.state.json` sidecar **next to the
HTML**. There is none yet. Once slots are filled, that sidecar must travel with
the page or the images are gone.

## Content still needed

Every portrait, plaque and object is deliberately empty — the frames show `—`
and read "awaiting family submission". None of it is placeholder *text* to be
replaced with invented content: names and faces are only ever set from
documentation a family or unit provides. Also unset by design: the address,
telephone and curator's line under Visit, which the hosting institution fills.

## Licensing

**The software is Apache License 2.0 — free for anyone to use, fork, learn
from and build on, commercially or not, without asking.** See [LICENSE](LICENSE).
The candle field, the TSL compute particle system and the stained-glass shaders
are ordinary graphics work and there is no reason to hoard them.

**The memorial is not.** [NOTICE](NOTICE) sets out what the grant does not
reach, and Apache §4(d) makes that file travel with every copy and every
derivative:

- **Memorial content** — photographs, names, dates, units, plaque text,
  objects. Placed at the request of families and units, withdrawn at their
  request. Not the maintainer's to sub-license.
- **The name and identity.** Apache §6 grants no rights in names or marks.
  Build anything you like with the code; do not present it as *this* memorial.
- **The curatorial text** of the rooms.

Apache rather than MIT for two specific reasons, both of which matter for a
memorial: §6 withholds the name, so a hostile fork cannot pass itself off as
this one; and §4(b) requires modified files to carry prominent notice that they
were changed, so a defaced copy is obliged to say it has been altered.

The licence discriminates against no person, group or nationality and will not
be amended to. A licence is the wrong instrument for that — see the note in
NOTICE, and *Keeping this from being defaced* below for the mechanisms that
actually work.

Third-party components under `vendor/` carry their own licences, shipped
beside them: three.js and OrbitControls MIT, React and ReactDOM MIT, and both
font families SIL OFL 1.1.

## Keeping this from being defaced

A licence governs copying. It does not govern who can edit your repository, and
it will not stop vandalism — that is access control and platform moderation,
and it is configured on GitHub, not in a text file.

For a public memorial repo, in rough order of value:

1. **Branch protection on `main`.** Pages deploys from `main`, so protecting
   that branch protects the live site. Require a pull request, block force
   pushes and deletions.
2. **Interaction limits** — Settings → Moderation → Interaction limits.
   Restrict to prior contributors or existing users. This is the switch that
   actually stops drive-by comment spam, and it can be set permanently.
3. **Turn off what you are not using** — Wiki, Projects, and Discussions.
   Fewer surfaces to moderate. Consider whether Issues earns its keep.
4. **Require approval for all outside contributors' workflow runs**, so a
   hostile PR cannot execute anything.
5. **Report abuse to GitHub** for defacing forks, and use §6 of the licence
   where a fork trades on the name.

One thing to be clear-eyed about: **forking of a public repo cannot be
disabled** on a personal account. Anyone can take a copy. What you control is
that a copy may not carry this memorial's name, and must declare that it was
changed.
