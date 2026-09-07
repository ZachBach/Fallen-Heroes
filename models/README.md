# models/

The three figures at the crossing — Michael, the wolf, the dragon — are carved
in code, in `MemorialLight.js`. **If you have a real sculpted model, it goes
here and replaces the carved one.** Nothing about the room changes: same
plinths, same spotlights, same stone.

## Hanging a model

1. Export `.glb` and put it in this folder.
2. Fill in its entry in the `SCULPTURES` table near the top of the `SCULPTURE`
   section of `MemorialLight.js`:

```js
const SCULPTURES = {
  michael: { src: './models/michael.glb', height: 6.6, turn: 0, material: 'stone' },
  wolf:    null,
  dragon:  null,
};
```

3. `node verify.mjs` — it fails on any cross-origin request, which is the check
   that catches a model that still pulls a texture off the internet.

| field | meaning |
| --- | --- |
| `src` | path, always local. Never a URL. |
| `height` | metres, base to top. **The model is scaled to fit this**, so whatever units it was authored in do not matter. Michael 6.6, the beasts about 3.2. |
| `turn` | radians about Y, applied before anything else. `0` means the model already faces **+Z**, which is down the nave toward the visitor. Blender's default export puts −Y forward, so a Blender figure facing you in the viewport usually needs `turn: Math.PI`. |
| `material` | `'stone'` (default) repaints it in the church's limestone, which is almost always what you want. `'own'` keeps the glTF's own materials, and then the model must bring its textures with it — see below. |

`null`, the default, means "use the one carved in code" **and fetches
nothing**. That is deliberate: a hopeful path to a file that is not there is a
404 on every single load, and this page gates on having none.

If a model fails to load, the carved figure stays standing and a warning goes
to the console. A memorial should never render a hole where a statue was.

## What the loader does for you

Exported models arrive at every scale and origin there is. Rather than asking
you to get that right in the exporter, the loader measures the model's bounding
box and:

- scales it so its height matches `height`,
- centres it on the plinth in X and Z,
- stands its lowest point on the plinth top.

So a model authored in millimetres with its origin at the navel lands correctly.
What it cannot fix is which way the thing faces — that is `turn`.

## Blender

Export as **glTF 2.0 (.glb)**, binary, with:

- **+Y up** (the exporter's default, and what three.js expects)
- **Apply Modifiers** on
- **Compression** off — Draco needs a decoder this page does not vendor, and
  fetching one from a CDN would break the no-external-request guarantee. An
  uncompressed `.glb` of a stone figure is a couple of megabytes, which is fine.
- Materials: leave them. With `material: 'stone'` they are replaced anyway.
- **No cameras, no lights.** The room has its own.

Decimate first. This scene already draws a million particles and five hundred
candles; a statue does not need 800k triangles to read from thirty feet.
Something in the region of 30–80k is plenty, and a Decimate modifier at 0.1 on
a scan is usually invisible at this distance.

## Fusion 360 / CAD

Fusion does not export glTF. The route is **Fusion → OBJ or STEP → Blender →
glb**. In Blender, after importing: check the scale (CAD lands in millimetres,
so the model may be 1000× too big — `height` fixes that anyway), apply Shade
Smooth with an autosmooth angle around 30° so the facets from the CAD
tessellation do not read as facets, then export as above.

CAD tessellation gives very even triangles and no seams, which is fine for a
statue and terrible for a texture. Use `material: 'stone'` and it is moot.

## Photogrammetry and scans

The same route, with two extra steps: decimate hard, and **check the licence on
whatever you scanned**. A scan of a monument is a derivative of that monument.
Public-domain sculpture is safe; a sculpture by a living artist is not, and
neither is a museum's scan of an old one where the museum asserts rights in the
scan itself.

## Textures

If you use `material: 'own'`, every texture must be **embedded in the .glb**
(binary glTF does this by default; glTF-separate does not). An external `.png`
next to the model works too, since it is same-origin — but a texture URL
pointing anywhere off this site will fail `verify.mjs`, and rightly.

## Licensing

Whatever you put here is redistributed by every clone of this repository, which
is public and Apache-2.0 for its code. So a model here must be free to
redistribute. Record what it is and where it came from in the table below, the
same way `images/README.md` does — a memorial that cannot say where its
material came from has a hole in exactly the place it should be strongest.

| file | subject | source | licence |
| ---- | ------- | ------ | ------- |
| _(none yet — the figures are carved in code)_ | | | |
