/**
 * MemorialLight — three.js WebGPURenderer + TSL, written for a memorial room.
 *
 * A field of votive candles on dark stone, the smoke and sparks rising off
 * them simulated on the GPU, and one slow shaft of light. Every surface and
 * every particle is a TSL node graph, so it compiles to WGSL on WebGPU and
 * falls back to WebGL2 where WebGPU is absent.
 *
 * Four properties govern the design:
 *
 *   PERSISTENT AND IN PLACE. Candle i sits at a fixed point on a golden-angle
 *   phyllotaxis spiral. The field only ever extends OUTWARD — lighting the
 *   five-thousandth candle does not move the first. Position is a pure
 *   function of the index and the count lives in localStorage, so a flame lit
 *   last year is still burning, in the same place, when its family returns.
 *
 *   CAPPED AT 5000. Deliberately. It is enough to fill the room to the
 *   horizon, and fixing it lets every buffer be allocated once at startup —
 *   no growth path, no reallocation, no frame where the room is rebuilt.
 *   Lighting a candle costs one number: instanced meshes bump `.count`, the
 *   flame quad soup bumps its draw range.
 *
 *   A MILLION PARTICLES. Smoke, sparks and ash are a GPU simulation — two
 *   storage buffers stepped by a TSL compute shader every frame. On WebGPU
 *   that is a real compute pass; three's WebGL2 backend implements the same
 *   node through transform feedback, so ONE piece of code drives both, and
 *   only the count differs.
 *
 *   BUILT ONCE. The scene is constructed on mount and never torn down by a
 *   prop change. Every prop reaches the render loop through a ref or a
 *   uniform. Rebuilding a room of candles because someone lit one would put
 *   out every other flame for a frame, which is precisely the wrong thing for
 *   this page to do.
 *
 * Requires the import map for "three", "three/webgpu" and "three/tsl" (see the
 * page head — all three resolve to ./vendor/, no external request).
 *
 * Props: candles, gold, drift, onReady
 */
const { useEffect, useRef } = React;

/* Golden angle. Successive candles land 137.5° apart, the packing that keeps
 * density even as the spiral grows — the same reason sunflowers use it. Any
 * rational fraction of a turn would form spokes and leave gaps. */
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
const SPACING = 0.52;          // world units between neighbouring candles
/* Positional jitter, per axis. Large enough to break up the spiral's arms — a
 * bare phyllotaxis lattice shows unmistakable curved rows once the field is a
 * few thousand deep and you are looking across it at a grazing angle — and
 * small enough that two votives never intersect. The glass is 0.268 across and
 * candles scale to 1.05, so the widest is 0.281; a bound of 0.11 leaves a
 * worst case of 0.52 - 2(0.11) = 0.30 between neighbours. */
const JITTER = 0.11;
const WAX_H = 0.30;            // full-height candle, before per-candle burn-down
const IGNITE_SECONDS = 1.6;    // how long a newly lit flame takes to come up

/* One room holds 500 candles and no more. When it fills, the next candle opens
 * the next room, and only the room being looked at is ever built or drawn.
 *
 * That ceiling is what buys the quality everywhere else. Five hundred candles
 * all fit in the near field, so every one of them can carry full-detail
 * geometry with no level-of-detail tiers, and the whole million-particle
 * budget is shared between 500 wicks instead of thousands — about two thousand
 * particles per flame. A single unbounded field had to spend both of those to
 * stay afloat. */
const ROOM_SIZE = 500;

/* Particle budget. WebGPU runs a genuine compute pass and takes a million
 * comfortably; the WebGL2 path runs the same node through transform feedback,
 * which is markedly slower per element, so it gets a smaller field of larger,
 * brighter particles that reads the same at a glance. */
const PARTICLES_WEBGPU = 1000000;
const PARTICLES_WEBGL = 150000;
/* Floor for the adaptive draw range — the plume thins on a slow machine but
 * never disappears. */
const PARTICLES_FLOOR = 60000;

/* Deterministic per-candle randomness. Candle i must look the same on every
 * visit and on every device — a memorial that reshuffles itself is a toy — so
 * nothing here may come from Math.random(). (Particle seeds are a different
 * matter and do use Math.random; see the note where they are filled.) */
function rand(i, salt) {
  const s = Math.sin((i + 1) * 127.1 + salt * 311.7) * 43758.5453;
  return s - Math.floor(s);
}

/* One room's candle positions, in LIGHTING order.
 *
 * The layout is the phyllotaxis spiral; the ordering is not. Candles are
 * sorted by distance from a point just in front of the field's centre, so
 * lighting proceeds from the near edge of the nave outward and the candle a
 * visitor lights appears in front of them rather than somewhere behind their
 * shoulder. Nothing about it is camera-dependent at click time — it cannot be,
 * because the same candle has to be in the same place when a family comes back
 * — so it is computed once against the default view and then fixed. */
function buildRoom(n) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const r = SPACING * Math.sqrt(i + 0.5);
    const a = i * GOLDEN_ANGLE;
    pts.push([
      Math.cos(a) * r + (rand(i, 1) * 2 - 1) * JITTER,
      Math.sin(a) * r + (rand(i, 2) * 2 - 1) * JITTER,
    ]);
  }
  // Sort focus sits on the ALTAR side of the disc, so lighting begins at the
  // steps and spreads outward down the nave. (It used to sit on the near side,
  // which lit from the viewer backwards.) Negative z is toward the apse.
  const fz = -SPACING * Math.sqrt(n) * 0.62;
  pts.sort((p, q) => (p[0] * p[0] + (p[1] - fz) ** 2) - (q[0] * q[0] + (q[1] - fz) ** 2));
  return pts;
}

/* How far down a candle has burned, and how thick it was poured. Both fixed
 * per index, so a candle never un-burns and never changes shape. Girth stays
 * inside 0.93–1.05 so the overlap bound above holds. */
const candleBurn = (i) => 0.42 + rand(i, 3) * 0.58;
const candleGirth = (i) => 0.93 + rand(i, 5) * 0.12;
/* Where candle i's flame sits: just above its melted pool. */
const flameBase = (i) => WAX_H * candleBurn(i) * 0.962 + 0.028;

const fieldRadius = (n) => SPACING * Math.sqrt(Math.max(n, 1));

/* CPU-side smoothstep. Used to hold the day/night clock at each end of its
 * travel and cross between them quickly, rather than spending the whole cycle
 * in a permanent dusk. */
function smoothstep01(x, a, b) {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

function MemorialLight({
  lit = 0, roomSize = ROOM_SIZE, gold = '#C9A227', drift = true,
  cycleSeconds = 210, onReady = null,
}) {
  const mountRef = useRef(null);
  // Live channels into the render loop. Props are pushed through these rather
  // than through the effect's dependency array, so no prop change rebuilds the
  // scene — the one exception is roomSize, which changes the geometry itself.
  const targetRef = useRef(lit);
  const driftRef = useRef(drift);
  const cycleRef = useRef(cycleSeconds);
  const apiRef = useRef(null);

  useEffect(() => {
    targetRef.current = Math.max(0, Math.min(roomSize, Math.floor(lit)));
  }, [lit, roomSize]);
  useEffect(() => { driftRef.current = drift; }, [drift]);
  useEffect(() => { cycleRef.current = Math.max(20, cycleSeconds); }, [cycleSeconds]);
  useEffect(() => { apiRef.current?.setGold(gold); }, [gold]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    let disposed = false;
    let stop = () => {};

    (async () => {
      let THREE, TSL;
      try {
        THREE = await import('three');
        try { TSL = await import('three/tsl'); } catch (e) { TSL = THREE; }
        if (!TSL || !TSL.vec3) throw new Error('TSL entry unavailable');
      } catch (err) {
        console.warn('[MemorialLight] three.js/TSL unavailable', err);
        return;
      }
      if (disposed) return;

      const {
        vec3, vec4, float, int, uniform, mix, clamp, smoothstep, oneMinus,
        positionLocal, positionWorld, normalWorld, cameraPosition, cameraViewMatrix,
        mx_noise_float, mx_noise_vec3, mx_fractal_noise_float, uv, attribute,
        instanceIndex, storage, Fn, If, pow, fract, dot, normalize,
        abs: tabs, sin: tsin, cos: tcos, max: tmax, min: tmin, floor: tfloor,
        step: tstep, length: tlen, atan: tatan, exp: texp,
      } = TSL;

      /* This room. Positions are built once, in lighting order, and every
       * buffer below is sized to exactly this many candles. */
      const N = Math.max(60, Math.min(1000, Math.floor(roomSize)));
      const ROOM = buildRoom(N);
      /* The rack stands back down the nave, not in the camera's lap. Centred on
       * the origin it reached to within five units of the viewer, so the front
       * row filled the bottom of the frame and fell behind the footer bar.
       * Pushed back, the whole field reads at once and there is floor in front
       * of it — which is also how a real rack is placed, with room to stand.
       *
       * Applied AFTER buildRoom's sort, so it slides the field without
       * disturbing the outward-from-the-altar lighting order. At -13 the far
       * edge of the disc reaches the altar dais at -24.6, which is what puts
       * the first flames at the steps and raises the whole rack in frame. */
      const FIELD_Z = -13.0;
      const candlePosition = (i) => [ROOM[i][0], ROOM[i][1] + FIELD_Z];

      /* ---------------------------------------------------------- uniforms */
      const uTime = uniform(0);
      const uDt = uniform(0.016);
      // Wrapped clock for hashing. Feeding raw elapsed seconds into sin() drifts
      // into float32 mush after a few minutes; this stays in [0,1).
      const uHashT = uniform(0);
      const uGold = uniform(new THREE.Color(gold));
      // Fractional candle count: integer part is how many are fully lit, the
      // fraction is how far the newest one has ignited. One number drives both
      // the field size and the ignition animation.
      const uLit = uniform(0);
      const uLitCount = uniform(1);   // max(1, ceil(uLit)) — respawn range
      const uNewest = uniform(0);     // index of the candle currently igniting
      const uBurst = uniform(0);      // 0..1 spark-burst envelope, decays after a click
      const uFieldR = uniform(fieldRadius(N));
      /* Plume density normaliser. A million particles shared between 260
       * candles is ~3800 each; shared between 5000 it is 200. Without this the
       * room would be a bonfire at low counts and a wisp at high ones. Scaling
       * per-particle brightness by (candles / particles drawn) makes the light
       * emitted PER CANDLE constant, so the plume looks the same whatever the
       * count and whatever the device tier. */
      const uPGain = uniform(0.05);
      /* Day/night. 0 is deep night, 1 is full daylight, and the room crosses
       * between them on a slow timer. Everything that changes with the hour
       * reads this one uniform: the sky beyond the windows, what colour the
       * glass throws onto the stone, and whether the shafts falling through
       * the nave are sunlight or moonlight. */
      const uDay = uniform(0);
      const uSunDir = uniform(new THREE.Vector3(0.4, 0.7, -0.6));

      /* --------------------------------------------------------- renderer */
      const renderer = new THREE.WebGPURenderer({ antialias: true, alpha: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setSize(mount.clientWidth || 1200, mount.clientHeight || 600, false);
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.05;
      Object.assign(renderer.domElement.style, { width: '100%', height: '100%', display: 'block' });
      mount.appendChild(renderer.domElement);
      try { await renderer.init(); } catch (err) { console.warn('[MemorialLight] init failed', err); return; }
      if (disposed) { renderer.dispose(); return; }

      // Orbit controls are optional: if the module fails to load, the room
      // still renders and simply keeps its slow automatic drift.
      let OrbitControls = null;
      try {
        ({ OrbitControls } = await import('./vendor/OrbitControls.js'));
      } catch (err) {
        console.warn('[MemorialLight] OrbitControls unavailable, drift only', err);
      }
      if (disposed) { renderer.dispose(); return; }

      const isWebGPU = !!renderer.backend?.isWebGPUBackend;
      const backend = isWebGPU ? 'WebGPU' : 'WebGL2';
      const P_COUNT = isWebGPU ? PARTICLES_WEBGPU : PARTICLES_WEBGL;
      const P_FLOOR = Math.min(PARTICLES_FLOOR, P_COUNT);

      const scene = new THREE.Scene();
      scene.fog = new THREE.FogExp2(0x0d0e10, 0.021);
      const sky = new THREE.Color(0x080a10);
      scene.background = sky;
      /* Framed to hold the rack in the lower half and the windows in the
       * upper. A camera down at candle height sees a beautiful floor and no
       * church at all, which rather wastes the church. */
      const camera = new THREE.PerspectiveCamera(44, 16 / 9, 0.1, 800);
      camera.position.set(0, 5.8, 11.5);

      /* Orbit controls, deliberately constrained. Zoom and pan are OFF: this
       * canvas is a section inside a scrolling page, and a control that eats
       * the wheel would trap a visitor who was only trying to reach the next
       * room. Rotation is the whole interaction — walk around the rack, look
       * up at the windows — and the polar limit stops the camera dropping
       * through the floor. */
      let controls = null;
      let userTook = false;
      if (OrbitControls) {
        controls = new OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 1.6, FIELD_Z);
        controls.enableDamping = true;
        controls.dampingFactor = 0.055;
        controls.enableZoom = false;
        controls.enablePan = false;
        controls.rotateSpeed = 0.42;
        controls.minPolarAngle = 0.22;
        controls.maxPolarAngle = 1.48;   // just above the floor plane
        /* A FULL turn, deliberately — no azimuth fence at all.
         *
         * The earlier version clamped to ±36°, because with the camera 21 units
         * out from a target near the west end, swinging further put the viewer
         * through a wall and looking at the back of it: black screen, nothing
         * outside the building to see. The fence treated the symptom.
         *
         * The cause was the radius. Now that the orbit is centred on the rack
         * itself the radius is about 13, and 13 units in ANY direction from
         * (0, -13) stays inside an arcade at ±16.6 and walls at ±30. So the
         * camera can go all the way round without ever leaving the nave — and
         * the view from behind the altar, looking back down the length of the
         * church over every flame, is the best one in the room. Fencing that
         * off to avoid a bug would have been the wrong trade. */
        controls.autoRotate = drift;
        controls.autoRotateSpeed = 0.22;
        // Once a visitor takes hold, the room stops moving on its own and
        // stays where they left it.
        controls.addEventListener('start', () => { userTook = true; });
        controls.update();
      }

      const hemi = new THREE.HemisphereLight(0x35424f, 0x08090a, 0.45);
      scene.add(hemi);
      /* The light coming through the windows, as an actual light rather than
       * only as the additive shafts. Without it the shafts hang in the air
       * over stone that never brightens, and the church reads as a painted
       * backdrop. It swings with the sun angle, so the nave lights from one
       * side and then the other across the cycle. */
      const sunLight = new THREE.DirectionalLight(0xfff0dc, 1.0);
      sunLight.position.set(14, 20, -8);
      scene.add(sunLight);
      // One representative warm light for the near field. The candles do not
      // each get a real light — thousands of point lights is not a thing any
      // renderer will do — so the warmth on the wax is emissive, computed per
      // fragment from its own flame. See the SSS block below.
      const warm = new THREE.PointLight(new THREE.Color(gold).getHex(), 22, 34, 2.1);
      warm.position.set(0, 1.5, FIELD_Z + 3.4);
      scene.add(warm);

      /* ------------------------------------------------------ TSL helpers */
      // Small-magnitude hash. Inputs are kept in roughly [0,1000] so float32
      // sin() stays well-conditioned — the reason particle seeds are uploaded
      // from the CPU rather than derived from a large instance index.
      const hashF = (x) => fract(tsin(x).mul(43758.5453));
      const goldVec = () => vec3(uGold.r, uGold.g, uGold.b);

      // Per-candle ignition ramp, read straight off the fractional count:
      //   idx <  floor(uLit) -> 1     (fully burning)
      //   idx == floor(uLit) -> 0..1  (the one just lit, coming up)
      //   idx >  floor(uLit) -> 0     (not lit; not drawn either)
      const igniteFor = (idxFloat) => clamp(uLit.sub(idxFloat), float(0), float(1));

      const viewDir = () => normalize(positionWorld.sub(cameraPosition));
      const fresnel = (p) => pow(oneMinus(tabs(dot(normalWorld, viewDir()))), p);

      /* Flicker. A real flame does not pulse on one sine — it wanders on a slow
       * envelope with a faster tremor over it, and goes still for seconds at a
       * time. Two noise rates, biased toward steady so the room reads calm
       * rather than like a fire. */
      const flicker = (seed) => {
        const slow = mx_noise_float(vec3(seed.mul(7.7), uTime.mul(0.65), float(0.0)));
        const fast = mx_noise_float(vec3(seed.mul(3.1), uTime.mul(2.9), float(11.3)));
        return clamp(slow.mul(0.16).add(fast.mul(0.07)).add(0.93), float(0.55), float(1.5));
      };

      /* ============================================================ FLOOR */
      const floorMat = new THREE.MeshStandardNodeMaterial({ metalness: 0.04 });
      {
        const r = tlen(vec3(positionWorld.x, float(0), positionWorld.z.sub(FIELD_Z)));
        // No way to sum thousands of real lights per fragment, so the field is
        // treated as what it physically is at a distance: one disc emitter.
        // Fitted rather than derived — the honest name is a cheat, and it is
        // the same cheat every renderer makes.
        const pool = oneMinus(smoothstep(float(0.0), uFieldR.mul(1.45).add(2.5), r));
        const near = oneMinus(smoothstep(float(0.0), uFieldR.mul(0.55).add(1.2), r));

        const coarse = mx_noise_float(positionWorld.mul(0.33)).mul(0.5).add(0.5);
        const grain = mx_noise_float(positionWorld.mul(8.5)).mul(0.5).add(0.5);
        const fine = mx_noise_float(positionWorld.mul(31.0)).mul(0.5).add(0.5);

        const stone = mix(vec3(0.020, 0.021, 0.024), vec3(0.052, 0.054, 0.060), coarse);
        floorMat.colorNode = stone
          .add(grain.sub(0.5).mul(0.024))
          .add(fine.sub(0.5).mul(0.012))
          .add(goldVec().mul(pool.mul(pool).mul(0.30)))
          .add(goldVec().mul(near.mul(near).mul(0.22)));
        // Polished under the candles, drier further out — the wet look is what
        // makes pooled light read as reflection rather than paint.
        floorMat.roughnessNode = clamp(
          grain.mul(0.22).add(0.62).sub(pool.mul(0.34)),
          float(0.08), float(0.95),
        );
      }
      const floor = new THREE.Mesh(new THREE.PlaneGeometry(900, 900), floorMat);
      floor.rotation.x = -Math.PI / 2;
      scene.add(floor);

      /* ====================================================== CANDLE WAX */
      /* Lathe profile, bottom to top: a straight-sided pillar, a softened
       * shoulder, and a concave pool of molten wax dipping to the wick. The
       * pool is the part that matters — it is where the light comes through,
       * and a flat-topped cylinder reads as a peg immediately. */
      const waxProfile = [
        [0.000, 0.000], [0.086, 0.000], [0.090, 0.014], [0.090, 0.780],
        [0.0895, 0.870], [0.0885, 0.930], [0.0855, 0.968], [0.0790, 0.992],
        [0.0680, 0.998], [0.0520, 0.980], [0.0330, 0.964], [0.0150, 0.957],
        [0.0050, 0.960], [0.000, 0.962],
      ];
      // Coarse profile for the far tier: same silhouette, a third of the rings.
      const waxProfileLow = [
        [0.000, 0.000], [0.088, 0.000], [0.090, 0.800], [0.086, 0.960],
        [0.068, 0.998], [0.030, 0.966], [0.000, 0.960],
      ];
      const lathe = (profile, seg) => new THREE.LatheGeometry(
        profile.map(([x, y]) => new THREE.Vector2(x, y * WAX_H)), seg,
      );

      const waxMat = new THREE.MeshStandardNodeMaterial();
      {
        const idx = float(instanceIndex);
        const seed = hashF(idx.mul(0.7331).add(0.5));
        const fl = flicker(seed);
        const ig = igniteFor(idx);
        const h = clamp(positionLocal.y.div(WAX_H), float(0), float(1));

        // Vertical drip runs. Poured wax is not a smooth cylinder; it carries
        // faint runs from the pour. Amplitude is low, so geometry normals stay
        // honest without recomputing them.
        const ang = tatan(positionLocal.z, positionLocal.x);
        const drip = mx_noise_float(vec3(ang.mul(2.4), positionLocal.y.mul(7.0), seed.mul(19.0)));
        const runs = tsin(ang.mul(11.0).add(seed.mul(6.283))).mul(0.5).add(0.5);
        waxMat.positionNode = positionLocal.add(
          vec3(positionLocal.x, float(0), positionLocal.z)
            .mul(drip.mul(0.05).add(runs.mul(0.022)).mul(smoothstep(float(0.05), float(0.5), h))),
        );

        // Unbleached wax is warm off-white with real batch variation; a field
        // of identical white cylinders looks printed.
        const waxTint = mix(vec3(0.86, 0.82, 0.73), vec3(0.94, 0.90, 0.83), hashF(idx.mul(0.311).add(3.3)));
        const grain = mx_noise_float(positionLocal.mul(46.0)).mul(0.5).add(0.5);
        waxMat.colorNode = waxTint.mul(grain.mul(0.10).add(0.95));

        const molten = smoothstep(float(0.86), float(0.99), h);
        waxMat.roughnessNode = clamp(
          grain.mul(0.16).add(0.52).sub(molten.mul(0.40)),
          float(0.06), float(0.95),
        );

        /* Subsurface scattering — the single thing separating wax from
         * plastic. Light from the flame enters the top and travels down
         * through the pillar, so it glows from within: brightest just under
         * the pool, and at the thin silhouette edges where less wax stands
         * between the flame and the eye. Emissive rather than transmission on
         * purpose — real transmission needs a scene pass per object, which at
         * thousands of candles is not affordable. */
        const depth = pow(h, float(2.9));
        const edge = fresnel(float(2.4));
        const deep = texp(h.sub(1.0).mul(4.5));
        const sss = depth.mul(0.85).add(deep.mul(0.30)).add(edge.mul(depth).mul(1.5));
        const sssColor = mix(vec3(1.00, 0.62, 0.26), goldVec(), float(0.35));
        const poolGlow = molten.mul(0.9).mul(mix(vec3(1.0, 0.78, 0.45), goldVec(), float(0.2)));
        waxMat.emissiveNode = sssColor.mul(sss).add(poolGlow).mul(fl).mul(ig).mul(0.9);
      }

      /* ========================================== VOTIVE HOLDER (glass) */
      /* A fluted votive glass. The flutes are real geometry, not a texture:
       * the profile is lathed and then displaced radially by a cosine in the
       * azimuth, so the silhouette scallops and each rib catches the fresnel
       * separately. That per-rib break-up is what stops a field of glasses
       * reading as a field of grey tubes.
       *
       * Additive rather than transmissive. Real refraction needs the
       * framebuffer sampled per instance, which at 5000 holders is out; on a
       * dark ground what actually reads as glass is the bright rim, the warm
       * light trapped in the wall, and the ring where the base pools it. */
      /* Heights are in units of WAX_H, like the wax profile — lathe() scales
       * them. Authoring these in world units instead is what once made every
       * holder render at a third of its height, a saucer with the candle
       * standing proud of it.
       *
       * The rim tops out at 1.06 WAX_H: a little above the tallest unburned
       * candle, so a fresh flame clears the glass while a burned-down one sits
       * down inside it, which is how a rack of votives actually looks. */
      const glassProfile = [
        [0.086, 0.000], [0.116, 0.000], [0.122, 0.022], [0.124, 0.078],
        [0.120, 0.168], [0.1195, 0.392], [0.121, 0.672], [0.124, 0.868],
        [0.129, 0.986], [0.133, 1.042], [0.1315, 1.060],
      ];
      const glassProfileLow = [
        [0.086, 0.000], [0.120, 0.028], [0.120, 0.561], [0.126, 0.925],
        [0.133, 1.043], [0.1315, 1.060],
      ];

      const glassMat = new THREE.MeshBasicNodeMaterial({
        transparent: true, depthWrite: false, side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
      });
      const GLASS_TOP = WAX_H * 1.06;   // must match the glass profile top above
      {
        const idx = float(instanceIndex);
        const seed = hashF(idx.mul(0.7331).add(0.5));
        const fl = flicker(seed);
        const ig = igniteFor(idx);
        const h = clamp(positionLocal.y.div(GLASS_TOP), float(0), float(1));
        const ang = tatan(positionLocal.z, positionLocal.x);

        // 16 flutes, fading out at the lip where votive glass is usually
        // smooth, and gone at the base where it thickens into the foot.
        const flute = tcos(ang.mul(16.0)).mul(0.5).add(0.5);
        const fluteMask = smoothstep(float(0.04), float(0.22), h)
          .mul(oneMinus(smoothstep(float(0.80), float(0.97), h)));
        // 0.06, not more: the widest glass is then 0.133 * 1.06 * 1.05 girth =
        // 0.148 radius, so neighbours stay inside the 0.30 clearance the
        // JITTER bound guarantees.
        glassMat.positionNode = positionLocal.add(
          vec3(positionLocal.x, float(0), positionLocal.z).mul(flute.mul(fluteMask).mul(0.06)),
        );

        const rim = fresnel(float(2.2));
        /* The cup has to read as a distinct object standing in front of the
         * wax, not as a glow around it. Four separate cues do that: a standing
         * wall you can see all the way up, light trapped low where the glass
         * meets the flame, a hard bright lip that states where the glass ENDS,
         * and a foot ring. The lip matters most — without a top edge the eye
         * reads the glass as a puddle at the base. */
        const wall = smoothstep(float(1.02), float(0.90), h).mul(0.13);
        const caught = oneMinus(smoothstep(float(0.0), float(0.95), h)).mul(0.26);
        const lip = smoothstep(float(0.90), float(0.995), h).mul(0.95);
        const foot = smoothstep(float(0.10), float(0.015), h).mul(0.26);
        // Each rib picks up its own highlight, so the glass reads as faceted.
        const ribSpec = pow(flute, float(3.0)).mul(fluteMask).mul(rim).mul(0.9);

        const body = rim.mul(0.50).add(wall).add(caught).add(lip).add(foot).add(ribSpec);
        /* Cooler than the wax on purpose. Glass and candle are near enough in
         * value that a warm glass merges into the wax and the holder vanishes;
         * soda-lime really is faintly green, and leaning on that is what
         * separates the two materials. */
        const glassTint = mix(vec3(0.60, 0.74, 0.72), goldVec(), float(0.34));
        glassMat.colorNode = vec4(glassTint.mul(body).mul(fl).mul(ig).mul(0.95), body.mul(ig));
      }

      /* ============================================================ WICK */
      const wickGeo = new THREE.CylinderGeometry(0.0045, 0.0075, 0.030, 6, 1);
      const wickMat = new THREE.MeshStandardNodeMaterial({ roughness: 0.9 });
      {
        const idx = float(instanceIndex);
        const fl = flicker(hashF(idx.mul(0.7331).add(0.5)));
        const ig = igniteFor(idx);
        const t = clamp(uv().y, float(0), float(1));
        wickMat.colorNode = mix(vec3(0.05, 0.04, 0.038), vec3(0.14, 0.11, 0.09), t);
        wickMat.emissiveNode = mix(vec3(1.0, 0.34, 0.06), vec3(1.0, 0.72, 0.34), t)
          .mul(smoothstep(float(0.35), float(1.0), t)).mul(fl).mul(ig).mul(1.5);
      }

      /* ================================================= FLAME QUAD SOUP */
      /* Flames are NOT an InstancedMesh. TSL's billboarding() builds its basis
       * from the mesh's world matrix and knows nothing about instance
       * matrices, so every instanced flame would collapse onto the mesh
       * origin. Instead each flame is a quad in one shared buffer carrying its
       * candle's centre in `position` and its corner in `aCorner`, and the
       * vertex node billboards by hand off the camera basis. Growing the field
       * is then a setDrawRange call. */
      const makeQuadSoup = (lift) => {
        const g = new THREE.BufferGeometry();
        const n = N;
        const centers = new Float32Array(n * 6 * 3);
        const corners = new Float32Array(n * 6 * 2);
        const seeds = new Float32Array(n * 6);
        const idxs = new Float32Array(n * 6);
        const CX = [-0.5, 0.5, 0.5, -0.5, 0.5, -0.5];
        const CY = [0, 0, 1, 0, 1, 1];   // y = 0 is the wick
        for (let i = 0; i < n; i++) {
          const [x, z] = candlePosition(i);
          const y = flameBase(i) + lift;
          for (let v = 0; v < 6; v++) {
            const o = i * 6 + v;
            centers[o * 3] = x; centers[o * 3 + 1] = y; centers[o * 3 + 2] = z;
            corners[o * 2] = CX[v]; corners[o * 2 + 1] = CY[v];
            seeds[o] = rand(i, 7);
            idxs[o] = i;
          }
        }
        g.setAttribute('position', new THREE.BufferAttribute(centers, 3));
        g.setAttribute('aCorner', new THREE.BufferAttribute(corners, 2));
        g.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 1));
        g.setAttribute('aIdx', new THREE.BufferAttribute(idxs, 1));
        g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, FIELD_Z), fieldRadius(N) + 4);
        return g;
      };

      // Y-locked billboard: flames turn to face the camera but never tip over,
      // because a flame's "up" is gravity's, not the viewer's.
      const billboardPosition = (wNode, hNode, leanNode) => {
        const corner = attribute('aCorner', 'vec2');
        const camRight = vec3(cameraViewMatrix[0][0], cameraViewMatrix[1][0], cameraViewMatrix[2][0]);
        const right = normalize(vec3(camRight.x, float(0), camRight.z));
        // Lean shears the quad with height, so the flame bends in a draught
        // instead of sliding sideways as a rigid card.
        const shear = leanNode ? leanNode.mul(corner.y).mul(corner.y) : float(0);
        return positionLocal
          .add(right.mul(corner.x.mul(wNode).add(shear)))
          .add(vec3(float(0), corner.y.mul(hNode), float(0)));
      };

      /* -------- the flame */
      const flameMat = new THREE.MeshBasicNodeMaterial({
        transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
      });
      const FLAME_W = 0.125, FLAME_H = 0.250;
      {
        const seed = attribute('aSeed', 'float');
        const idx = attribute('aIdx', 'float');
        const ig = igniteFor(idx);
        const fl = flicker(seed);

        // A flame coming up starts short and grows; height leads brightness,
        // which is what a wick actually does.
        const grow = pow(ig, float(0.65));
        // Per-candle size. Wick length and draught differ candle to candle;
        // identically sized flames across a field is the tell.
        const size = seed.mul(0.34).add(0.83);
        // Slow draught, shared direction but per-candle phase.
        const lean = mx_noise_float(vec3(seed.mul(13.0), uTime.mul(0.31), float(5.0))).mul(0.030);
        flameMat.positionNode = billboardPosition(
          float(FLAME_W).mul(size).mul(grow.mul(0.35).add(0.65)),
          float(FLAME_H).mul(size).mul(grow).mul(fl.mul(0.16).add(0.88)),
          lean.mul(grow),
        );

        const c = attribute('aCorner', 'vec2');
        const x0 = c.x;
        const y = clamp(c.y, float(0.0001), float(1));

        /* Turbulence, advected upward. Sampling along -y*speed makes the
         * pattern travel up the flame the way hot gas does, rather than
         * boiling in place. Domain-warped: a slow noise displaces the lookup
         * of a faster one, which is what gives real flame edges their folded,
         * licking quality instead of a uniform wobble. */
        const warp = mx_noise_float(vec3(x0.mul(2.0), y.mul(1.3).sub(uTime.mul(0.7)), seed.mul(9.0)));
        const nz = mx_fractal_noise_float(
          vec3(
            x0.mul(5.0).add(warp.mul(0.6)),
            y.mul(2.6).sub(uTime.mul(1.55)).add(warp.mul(0.4)),
            seed.mul(23.0),
          ),
          3, 2.0, 0.55, 1.0,
        );
        const nz2 = mx_noise_float(vec3(seed.mul(51.0), uTime.mul(0.9), y.mul(1.4)));
        // The flame is pinned at the wick and free at the tip.
        const waver = nz.mul(0.055).add(nz2.mul(0.03)).mul(smoothstep(float(0.06), float(0.95), y));
        const x = x0.add(waver);

        /* Teardrop half-width: zero at the wick, widest a third of the way up,
         * drawn to a point at the tip. sin(pi * y^0.72) gives that asymmetry;
         * the outer pow softens the shoulder. */
        const hw = pow(tsin(pow(y, float(0.72)).mul(Math.PI)), float(0.62)).mul(0.34);
        const d = clamp(oneMinus(tabs(x).div(tmax(hw, float(0.0001)))), float(0), float(1));

        const cool = oneMinus(smoothstep(float(0.35), float(1.0), y)).mul(0.55).add(0.45);
        const T = clamp(pow(d, float(1.35)).mul(cool).mul(fl), float(0), float(1));

        /* The dark inner cone. Directly above the wick sits vaporised wax that
         * has not reached combustion yet — genuinely dark, and the detail most
         * fake flames miss. A small lens, low and central. */
        const cone = smoothstep(float(0.01), float(0.09), y)
          .mul(oneMinus(smoothstep(float(0.10), float(0.30), y)))
          .mul(smoothstep(float(0.40), float(0.95), d));

        /* Colour by temperature. A candle runs roughly 1000–1400 °C: deep red
         * at the cool edge, orange through the body, amber-white in the core.
         * The base is different physics — a thin blue Swan-band cone where
         * oxygen is plentiful and combustion is complete. */
        let col = mix(vec3(0.42, 0.055, 0.010), vec3(1.00, 0.30, 0.035), smoothstep(float(0.02), float(0.30), T));
        col = mix(col, vec3(1.00, 0.58, 0.12), smoothstep(float(0.28), float(0.58), T));
        col = mix(col, vec3(1.00, 0.82, 0.42), smoothstep(float(0.55), float(0.80), T));
        col = mix(col, vec3(1.00, 0.95, 0.80), smoothstep(float(0.78), float(0.96), T));
        const blue = oneMinus(smoothstep(float(0.015), float(0.16), y)).mul(smoothstep(float(0.25), float(0.9), d));
        col = mix(col, vec3(0.30, 0.52, 1.00), blue.mul(0.7));
        col = mix(col, goldVec(), smoothstep(float(0.30), float(0.62), T).mul(0.10));

        /* Outer envelope: the dim luminous shell of burning gas just outside
         * the bright body. Without it a flame ends on a hard edge and reads as
         * a decal. */
        const shell = pow(clamp(oneMinus(tabs(x).div(tmax(hw.mul(1.55), float(0.0001)))), float(0), float(1)), float(2.2));
        const envelope = shell.mul(oneMinus(d)).mul(0.30).mul(cool);

        // Ignition flares brighter than the steady flame before settling.
        const flare = oneMinus(ig).mul(ig).mul(4.0).add(1.0);
        const intensity = T.mul(oneMinus(cone.mul(0.62))).add(envelope).mul(ig).mul(flare);
        flameMat.colorNode = vec4(col.mul(intensity).mul(2.35), intensity);
      }

      /* -------- halo: light in the air around each flame */
      const haloMat = new THREE.MeshBasicNodeMaterial({
        transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
      });
      const HALO_W = 0.95, HALO_H = 0.95;
      {
        const seed = attribute('aSeed', 'float');
        const idx = attribute('aIdx', 'float');
        const ig = igniteFor(idx);
        const fl = flicker(seed);
        haloMat.positionNode = billboardPosition(
          float(HALO_W).mul(fl.mul(0.10).add(0.94)),
          float(HALO_H).mul(fl.mul(0.10).add(0.94)),
          null,
        );
        const c = attribute('aCorner', 'vec2');
        const p = vec3(c.x, c.y.sub(0.42), float(0));
        const r = clamp(tlen(p).mul(2.15), float(0), float(1));
        // Two lobes: a tight core and a wide faint bloom. This stands in for a
        // post-processing bloom pass, which at this candle count costs more
        // than it returns — and an additive sprite is what the glow physically
        // is anyway (scattering in the air).
        const core = pow(oneMinus(r), float(3.4)).mul(0.42);
        const wide = pow(oneMinus(r), float(1.5)).mul(0.09);
        const a = core.add(wide).mul(fl).mul(ig);
        const tint = mix(vec3(1.0, 0.68, 0.30), goldVec(), float(0.45));
        haloMat.colorNode = vec4(tint.mul(a), a);
      }

      /* ============================================================ FIELD */
      /* Fixed allocation at N. Lighting a candle only ever changes a
       * count and a draw range — no buffer is ever reallocated. */
      /* Level of detail, by how many candles are lit — which is also how far
       * the camera has pulled back, since the two are locked together.
       *
       * This is not an optimisation for its own sake. A full field at close
       * detail is roughly eleven million triangles, nearly all of them
       * sub-pixel, and sub-pixel triangles are the pathological case for a
       * GPU: every one costs a whole 2x2 quad of shading however little of it
       * you can see. Measured, that field ran at 5 fps on WebGPU and 4 on
       * WebGL2 — with the particles proven innocent, since the smaller WebGL2
       * plume stalled just as hard. At the far tier a candle is about ten
       * pixels tall and the coarse geometry is indistinguishable. */
      const LODS = [
        { upTo: 500, wax: [waxProfile, 28], glass: [glassProfile, 48], wick: 6 },
        { upTo: 1600, wax: [waxProfile, 14], glass: [glassProfile, 24], wick: 5 },
        { upTo: Infinity, wax: [waxProfileLow, 8], glass: [glassProfileLow, 10], wick: 3 },
      ].map((l) => ({
        upTo: l.upTo,
        wax: lathe(l.wax[0], l.wax[1]),
        glass: lathe(l.glass[0], l.glass[1]),
        wick: new THREE.CylinderGeometry(0.0045, 0.0075, 0.030, l.wick, 1),
      }));
      const lodFor = (n) => LODS.find((l) => n <= l.upTo) || LODS[LODS.length - 1];
      let lodNow = null;

      const group = new THREE.Group();
      const wax = new THREE.InstancedMesh(LODS[0].wax, waxMat, N);
      const glass = new THREE.InstancedMesh(LODS[0].glass, glassMat, N);
      const wick = new THREE.InstancedMesh(LODS[0].wick, wickMat, N);
      wax.frustumCulled = glass.frustumCulled = wick.frustumCulled = false;
      {
        const m = new THREE.Matrix4();
        const q = new THREE.Quaternion();
        const pos = new THREE.Vector3();
        const scl = new THREE.Vector3();
        const axis = new THREE.Vector3(0, 1, 0);
        for (let i = 0; i < N; i++) {
          const [x, z] = candlePosition(i);
          const burn = candleBurn(i);
          const girth = candleGirth(i);
          // Turned a little, so drip runs and flutes do not line up across the
          // field.
          q.setFromAxisAngle(axis, rand(i, 4) * Math.PI * 2);
          wax.setMatrixAt(i, m.compose(pos.set(x, 0, z), q, scl.set(girth, burn, girth)));
          glass.setMatrixAt(i, m.compose(pos.set(x, 0, z), q, scl.set(girth, 1, girth)));
          wick.setMatrixAt(i, m.compose(pos.set(x, WAX_H * burn * 0.962 + 0.013, z), q, scl.set(1, 1, 1)));
        }
        wax.instanceMatrix.needsUpdate = true;
        glass.instanceMatrix.needsUpdate = true;
        wick.instanceMatrix.needsUpdate = true;
      }
      const flames = new THREE.Mesh(makeQuadSoup(0), flameMat);
      const halos = new THREE.Mesh(makeQuadSoup(0.02), haloMat);
      flames.frustumCulled = halos.frustumCulled = false;
      halos.renderOrder = 1;
      flames.renderOrder = 2;
      glass.renderOrder = 3;
      group.add(wax, wick, glass, halos, flames);
      scene.add(group);

      /* ================================================ GPU PARTICLE FIELD */
      /* Two storage buffers stepped by a compute shader:
       *   pos = vec4(x, y, z, life)      life falls 1 -> 0, then respawns
       *   vel = vec4(vx, vy, vz, seed)   seed is fixed for the particle's life
       *
       * A third, read-only buffer holds every candle's flame position, because
       * the shader cannot recompute it: the CPU's candlePosition() runs in
       * float64 and its sin() of a large argument does not survive float32 on
       * the GPU. Uploading 5000 vec4s once is exact and costs 80 KB. */
      const candleAttr = new THREE.StorageBufferAttribute(N, 4);
      for (let i = 0; i < N; i++) {
        const [x, z] = candlePosition(i);
        candleAttr.array[i * 4] = x;
        candleAttr.array[i * 4 + 1] = flameBase(i);
        candleAttr.array[i * 4 + 2] = z;
        candleAttr.array[i * 4 + 3] = candleGirth(i);
      }
      candleAttr.needsUpdate = true;
      const candleBuf = storage(candleAttr, 'vec4', N);

      const posAttr = new THREE.StorageBufferAttribute(P_COUNT, 4);
      const velAttr = new THREE.StorageBufferAttribute(P_COUNT, 4);
      /* Seeds come from Math.random, not from the deterministic rand() above.
       * Candles must be identical on every visit; smoke must not be. Seeding
       * from the instance index instead would need a hash of a number up to a
       * million, and float32 sin() of that is visibly banded. */
      for (let i = 0; i < P_COUNT; i++) {
        velAttr.array[i * 4 + 3] = Math.random();
        posAttr.array[i * 4 + 3] = Math.random();   // staggered initial life
      }
      posAttr.needsUpdate = true;
      velAttr.needsUpdate = true;
      const posBuf = storage(posAttr, 'vec4', P_COUNT);
      const velBuf = storage(velAttr, 'vec4', P_COUNT);

      /* Respawn: place the particle at a candle drawn from the currently lit
       * range, with a velocity that is mostly up. A candle that was just lit
       * throws a real spark burst — uBurst decays over about a second after
       * each click, and only the igniting candle sees it. */
      const respawn = (pos, vel, life, seed) => {
        const r1 = hashF(seed.mul(311.7).add(uHashT));
        const r2 = hashF(seed.mul(571.3).add(uHashT).add(1.7));
        const r3 = hashF(seed.mul(149.1).add(uHashT).add(3.1));
        const ci = tfloor(r1.mul(uLitCount)).toVar();
        const c = candleBuf.element(int(ci));
        const ang = r2.mul(6.28318);
        const rad = r3.mul(0.016).add(0.004);
        const isNew = tstep(tabs(ci.sub(uNewest)), float(0.5));
        const kick = uBurst.mul(isNew).mul(2.6).add(0.14);
        pos.assign(vec3(
          c.x.add(tcos(ang).mul(rad)),
          c.y.add(r3.mul(0.018)),
          c.z.add(tsin(ang).mul(rad)),
        ));
        vel.assign(vec3(
          tcos(ang).mul(kick).mul(0.26),
          r2.mul(0.09).add(0.09).mul(kick.mul(0.5).add(0.82)),
          tsin(ang).mul(kick).mul(0.26),
        ));
        life.assign(float(1));
      };

      const simulate = Fn(() => {
        const P = posBuf.element(instanceIndex);
        const V = velBuf.element(instanceIndex);
        const pos = P.xyz.toVar();
        const life = P.w.toVar();
        const vel = V.xyz.toVar();
        const seed = V.w.toVar();

        /* Long lifespans, roughly 4 s to 9 s. Smoke off a candle does not
         * sprint away and vanish; it hangs, spreads and thins. Short lives
         * plus a strong rise is what makes a particle system read as a
         * firework, which is the one thing this room must never look like. */
        life.subAssign(uDt.mul(seed.mul(0.11).add(0.12)));

        If(life.lessThanEqual(float(0)), () => {
          respawn(pos, vel, life, seed);
        });

        // Gentle buoyancy, strongest while the particle is still hot.
        vel.assign(vel.add(vec3(float(0), uDt.mul(life.mul(0.42).add(0.09)), float(0))));
        /* Turbulence. One octave, because this runs a million times a frame —
         * and because smoke off a candle is dominated by one scale anyway.
         * Sampling along -time makes the field drift upward with the plume.
         * Strength rises as the particle ages, so a plume leaves the wick
         * narrow and opens out above it, which is what a real one does. */
        const n = mx_noise_vec3(pos.mul(1.1).add(vec3(float(0), uTime.mul(-0.40), uTime.mul(0.09))));
        const spread = oneMinus(life).mul(0.95).add(0.12);
        vel.assign(vel.add(n.mul(vec3(1.5, 0.55, 1.5)).mul(uDt.mul(spread))));
        // Drag, clamped so a long frame cannot invert the velocity.
        vel.assign(vel.mul(oneMinus(tmin(uDt.mul(1.7), float(0.85)))));
        pos.assign(pos.add(vel.mul(uDt)));

        P.assign(vec4(pos, life));
        V.assign(vec4(vel, seed));
      })().compute(P_COUNT);

      /* Particles render as points. On WebGPU that is native point-list
       * topology — one pixel each, no size control, which for a million
       * particles is exactly right: the plume reads as luminous dust rather
       * than as a swarm of discs, and there is no overdraw. WebGL2 does honour
       * gl_PointSize, so the smaller field there gets slightly larger points
       * to land at a similar density on screen. */
      const pGeo = new THREE.BufferGeometry();
      pGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(P_COUNT * 3), 3));
      pGeo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 5, FIELD_Z), fieldRadius(N) + 20);
      const pMat = new THREE.PointsNodeMaterial({
        transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      });
      {
        const P = posBuf.toAttribute();
        pMat.positionNode = P.xyz;
        const life = P.w;
        const height = P.y;

        /* Colour by age, which for a spark is colour by temperature. A
         * particle leaves the wick white-hot, cools through amber to a dull
         * ember, and ends as cold ash that only catches what light is in the
         * room. */
        const hot = pow(life, float(3.0));
        let col = mix(vec3(0.16, 0.13, 0.12), vec3(1.00, 0.34, 0.06), smoothstep(float(0.10), float(0.45), life));
        col = mix(col, vec3(1.00, 0.66, 0.22), smoothstep(float(0.45), float(0.72), life));
        col = mix(col, vec3(1.00, 0.93, 0.78), smoothstep(float(0.80), float(0.97), life));
        col = mix(col, goldVec(), float(0.12));

        // Fade in as it leaves the wick, out as it cools and climbs. Without
        // the leading fade every particle pops into existence at full
        // brightness right at the flame.
        const born = smoothstep(float(1.0), float(0.93), life);
        const cooled = smoothstep(float(0.0), float(0.35), life);
        const climb = oneMinus(smoothstep(float(2.0), float(9.0), height));
        const a = born.mul(cooled).mul(climb).mul(hot.mul(0.55).add(0.05)).mul(uPGain);

        pMat.colorNode = vec4(col.mul(a.mul(1.6)), a);
        if (!isWebGPU) pMat.sizeNode = float(1.7);
      }
      const particles = new THREE.Points(pGeo, pMat);
      particles.frustumCulled = false;
      particles.renderOrder = 0;
      scene.add(particles);

      let computeOk = true;
      try {
        await renderer.computeAsync(simulate);
      } catch (err) {
        // A backend that cannot run the node at all: drop the plume rather
        // than the room. Everything else on this page still renders.
        computeOk = false;
        particles.visible = false;
        console.warn('[MemorialLight] particle compute unavailable, continuing without it', err);
      }
      if (disposed) { renderer.dispose(); return; }

      try { onReady && onReady({ backend, particles: computeOk ? P_COUNT : 0 }); } catch (e) { /* the room still works */ }

      /* ============================================================ CHURCH */
      /* The room the candles stand in: a nave with an arcade down each side,
       * a wall of lancet windows above it, and an apse with a rose window at
       * the east end. Nothing is modelled from a real building and the page
       * says so — this is a rendered room, and the honesty note in the footer
       * covers it.
       *
       * Everything here is stone or glass and nothing here moves, so it is all
       * ordinary geometry with node materials. The whole church is under a
       * thousand triangles of walls plus a handful of lathed columns; the
       * budget still belongs to the candles. */
      /* Nave dimensions are set by the ORBIT, not the other way round. The
       * camera sits about 19 units from the rack so the whole field reads at
       * once, and a full turn sweeps that 19 in every direction — so the
       * arcade has to stand further out than 19, and the apse further back
       * than the rack plus 19. Widening the room is what buys a camera far
       * enough back to see it, without fencing the rotation. */
      const NAVE_W = 27.0;      // centre to arcade
      const NAVE_L = 44;        // centre to each end
      const WALL_H = 30;
      const BAYS = 7;           // arcade bays per side
      const BAY_Z = (NAVE_L * 2) / BAYS;

      const church = new THREE.Group();
      scene.add(church);

      /* -------- stone: walls, arcade, apse */
      const stoneMat = new THREE.MeshStandardNodeMaterial({ roughness: 0.86, metalness: 0.0 });
      {
        // Ashlar blocks — coursed masonry, with the joints cut as darker
        // lines rather than modelled. Blocks are wider than they are tall,
        // and every course is offset by half a block, which is the single
        // cue that separates masonry from a flat grey wall.
        const p = positionWorld;
        const course = p.y.mul(1.35);
        const row = tfloor(course);
        const along = mix(p.x, p.z, tstep(float(0.5), tabs(normalWorld.x)));
        const block = along.mul(0.42).add(fract(row.mul(0.5)).mul(0.5));
        const jx = tabs(fract(block).sub(0.5)).mul(2.0);
        const jy = tabs(fract(course).sub(0.5)).mul(2.0);
        const joint = smoothstep(float(0.86), float(0.99), tmax(jx, jy));

        const grain = mx_noise_float(p.mul(1.4)).mul(0.5).add(0.5);
        const fine = mx_noise_float(p.mul(11.0)).mul(0.5).add(0.5);
        // Cool limestone that warms toward the candles and toward daylight.
        const base = mix(vec3(0.085, 0.083, 0.079), vec3(0.140, 0.134, 0.122), grain);
        const lowGlow = oneMinus(smoothstep(float(0.0), float(9.0), p.y)).mul(0.5);
        stoneMat.colorNode = base
          .add(fine.sub(0.5).mul(0.02))
          .sub(joint.mul(0.045))
          .add(goldVec().mul(lowGlow.mul(0.14)))
          .mul(uDay.mul(0.85).add(0.85));
        stoneMat.roughnessNode = clamp(fine.mul(0.2).add(0.78).add(joint.mul(0.1)), float(0.3), float(1));
      }

      const addBox = (w, h, d, x, y, z, ry) => {
        const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), stoneMat);
        m.position.set(x, y, z);
        if (ry) m.rotation.y = ry;
        church.add(m);
        return m;
      };
      /* SINK exists to stop z-fighting, and it is not optional.
       *
       * Everything standing on the floor used to have its underside at exactly
       * y = 0 — the same plane as the floor itself. Two coplanar faces give the
       * depth buffer no way to decide which is in front, so the GPU picks
       * differently from pixel to pixel and from frame to frame. It shows as a
       * thin dark line along the base of every wall, and because the tie breaks
       * differently as the camera moves, the line crawls the moment you orbit.
       *
       * The fix is to give the buffer something to decide with: everything that
       * meets the floor is pushed BELOW it, so the join is a solid intersection
       * rather than a tie. Nothing here is visible from inside the church, so
       * sinking costs nothing. */
      const SINK = 0.6;

      // Side walls, west (entrance) wall, and the apse wall behind the altar.
      // Each is SINK taller than the room and dropped by SINK, so it spans
      // -SINK..WALL_H and its underside is buried.
      const wall = (w, d, x, z) =>
        addBox(w, WALL_H + SINK, d, x, (WALL_H - SINK) / 2, z);
      wall(0.9, NAVE_L * 2, -NAVE_W - 1.4, 0);
      wall(0.9, NAVE_L * 2, NAVE_W + 1.4, 0);
      wall(NAVE_W * 2 + 3.8, 0.9, 0, -NAVE_L);
      wall(NAVE_W * 2 + 3.8, 0.9, 0, NAVE_L);

      /* A low altar dais at the east end, so the space has a direction. Two
       * steps, and they have the same problem twice over: the lower one meets
       * the floor, and the upper one used to sit exactly on top of the lower.
       * Both are sunk, and the upper one is deep enough to be embedded in the
       * lower rather than balanced on it. */
      /* Far enough back that the rack clears it. At 13.2 the dais front edge
       * landed at z = -23.0 while the field reached -24.63, so the outermost
       * ring and a half of candles stood INSIDE the step with only their
       * flames above it. Note the altar is placed relative to FIELD_Z, so
       * sliding the field would not have helped — the overlap was in the
       * offset. Field radius 11.63 + half the 6.4 dais depth needs 14.83
       * minimum; 16.0 leaves a metre of stone between the two. */
      const ALTAR_Z = FIELD_Z - 16.0;
      addBox(17, 0.34 + SINK, 6.4, 0, 0.17 - SINK / 2, ALTAR_Z);
      addBox(12.5, 0.48, 4.4, 0, 0.44, ALTAR_Z + 0.5);

      /* -------- the arcade: piers with a moulded base and capital */
      const pierProfile = [
        [0.00, -0.50], [0.86, -0.50], [0.92, 0.10], [0.78, 0.30], [0.70, 0.55],
        [0.66, 1.00], [0.66, 8.60], [0.72, 8.85], [0.86, 9.05], [0.94, 9.30],
        [0.80, 9.55], [0.74, 9.75], [0.00, 9.75],
      ].map(([x, y]) => new THREE.Vector2(x, y));
      const pierGeo = new THREE.LatheGeometry(pierProfile, 16);
      const PIER_SCALE = new THREE.Vector3(1.35, 1.62, 1.35);
      const piers = new THREE.InstancedMesh(pierGeo, stoneMat, BAYS * 2 + 2);
      {
        const m = new THREE.Matrix4();
        const q = new THREE.Quaternion();
        const s = PIER_SCALE;
        let k = 0;
        for (let side = -1; side <= 1; side += 2) {
          for (let i = 0; i <= BAYS; i++) {
            const z = -NAVE_L + i * BAY_Z;
            piers.setMatrixAt(k++, m.compose(
              new THREE.Vector3(side * NAVE_W, 0, z), q, s,
            ));
          }
        }
        piers.count = k;
        piers.instanceMatrix.needsUpdate = true;
      }
      church.add(piers);

      /* -------- stained glass */
      /* Leaded glass, built rather than drawn: the panel is divided into cells
       * on a brick grid, each cell takes a colour from a small liturgical
       * palette, and the lead comes from the distance to the cell edges. Real
       * leaded windows ARE geometric like this — the leading follows the
       * cames, not a noise field — so building it this way costs less and
       * looks more like the thing.
       *
       * The central lancets carry a standing winged figure: Michael, by
       * convention the archangel who guards the dead. It is a silhouette
       * assembled from circles and ellipses, not a reproduction of any
       * painting — a stylised icon, which is what a window at this distance
       * reads as anyway. */
      /* Two colours only: blue and gold. Not a stylistic restraint — it is the
       * flag, and in a memorial to Ukraine's dead the window has no business
       * being anything else. It also happens to be the strongest pairing
       * available: gold on deep blue is close to the maximum contrast the eye
       * can be given, which is what finally made the figure legible. */
      const GLASS_BLUE = vec3(0.030, 0.075, 0.300);
      const GLASS_BLUE_MID = vec3(0.055, 0.135, 0.440);
      const GLASS_BLUE_PALE = vec3(0.110, 0.230, 0.600);
      const GLASS_GOLD = vec3(0.880, 0.620, 0.090);
      const GLASS_GOLD_PALE = vec3(0.980, 0.830, 0.330);

      // Field glass: three depths of blue, with gold reserved for the border,
      // the tracery accents and the figure itself.
      const glassPalette = (idx) => {
        let c = GLASS_BLUE;
        c = mix(c, GLASS_BLUE_MID, tstep(float(0.5), idx));
        c = mix(c, GLASS_BLUE_PALE, tstep(float(1.5), idx));
        c = mix(c, GLASS_BLUE_MID, tstep(float(2.5), idx));
        c = mix(c, GLASS_GOLD, tstep(float(3.5), idx));   // one pane in five
        return c;
      };

      /* Saturation is the whole point of stained glass, and ACES desaturates
       * anything it has to compress. Pushing chroma before the panel is scaled
       * up is what keeps a deep blue deep instead of turning it into pale
       * lavender the moment the sun is behind it. */
      const punch = (c, amount) => {
        const lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
        return mix(vec3(lum, lum, lum), c, float(amount));
      };

      /* Michael, in panel space (x centred on 0, y running 0..1).
       *
       * By convention the archangel who weighs and guards the dead, and the
       * figure on Kyiv's own coat of arms — which is why he is the one
       * standing in these lights.
       *
       * Two things had to change before he read as anything at all. He is much
       * larger, filling two thirds of the panel rather than floating in the
       * middle of it; and the whole silhouette is drawn twice, once at size
       * and once fractionally larger, so the difference between the two gives
       * a lead outline. That outline is not decoration — figurative glass is
       * ALWAYS drawn with the figure cames heavier than the field cames, and
       * without it a gold figure on blue dissolves into the diaper behind it.
       *
       * A stylised icon assembled from circles and ellipses. It reproduces no
       * particular painting and is not offered as one. */
      const archangel = (x, y, g) => {
        const gg = float(g);
        const circle = (cx, cy, r) => oneMinus(smoothstep(
          float(r).add(gg).sub(0.005), float(r).add(gg),
          tlen(vec3(x.sub(cx), y.sub(cy), float(0))),
        ));
        const halo = circle(0, 0.775, 0.098);
        const head = circle(0, 0.770, 0.056);

        // Robe: a column that widens toward the hem, as vestments do.
        const w = float(0.132).sub(y.sub(0.22).mul(0.118)).max(float(0.056)).add(gg);
        const robe = oneMinus(smoothstep(w.sub(0.006), w, tabs(x)))
          .mul(smoothstep(float(0.200).sub(gg), float(0.220).sub(gg), y))
          .mul(oneMinus(smoothstep(float(0.702).add(gg), float(0.722).add(gg), y)));

        // Wings, swept up and out from the shoulders.
        const wing = (s) => {
          const wx = x.sub(float(s).mul(0.208)).div(float(0.152).add(gg));
          const wy = y.sub(0.605).div(float(0.258).add(gg));
          return oneMinus(smoothstep(float(0.962), float(1.0), tlen(vec3(wx, wy, float(0)))));
        };
        const wings = tmax(wing(-1), wing(1));

        // The sword, held upright at his right hand.
        const blade = oneMinus(smoothstep(float(0.012).add(gg), float(0.017).add(gg), tabs(x.sub(0.188))))
          .mul(smoothstep(float(0.248).sub(gg), float(0.262).sub(gg), y))
          .mul(oneMinus(smoothstep(float(0.740).add(gg), float(0.756).add(gg), y)));
        const guard = oneMinus(smoothstep(float(0.046).add(gg), float(0.052).add(gg), tabs(x.sub(0.188))))
          .mul(oneMinus(smoothstep(float(0.013).add(gg), float(0.018).add(gg), tabs(y.sub(0.652)))));
        const sword = tmax(blade, guard);

        const all = tmax(tmax(tmax(halo, head), tmax(robe, wings)), sword);
        return { halo, head, robe, wings, sword, all };
      };

      const glassMatWindow = new THREE.MeshBasicNodeMaterial({ side: THREE.DoubleSide });
      {
        const u = uv();
        // Pointed-arch mask: the panel is a rectangle below the springing and
        // two circle arcs above it, which is what makes it a lancet rather
        // than a picture frame.
        const ax = u.x.sub(0.5);
        const spring = float(0.62);
        const arcR = float(0.62);
        const inArch = oneMinus(smoothstep(
          arcR.sub(0.01), arcR,
          tlen(vec3(tabs(ax).add(0.12), u.y.sub(spring).max(float(0)), float(0))),
        ));
        const inBody = oneMinus(smoothstep(float(0.485), float(0.5), tabs(ax)));
        const panel = tmin(inBody, mix(float(1), inArch, tstep(spring, u.y)))
          .mul(smoothstep(float(0.0), float(0.012), u.y));

        /* The field: small leaded panes in courses, offset like brickwork.
         * Kept FINE and almost entirely blue so it reads as texture behind the
         * figure. The earlier version randomised five colours across big cells,
         * which produced confetti — every pane competing with the figure
         * instead of sitting behind it. */
        const cy = u.y.mul(22.0);
        const rowN = tfloor(cy);
        const cx = u.x.mul(9.0).add(fract(rowN.mul(0.5)).mul(0.5));
        const cellId = fract(tsin(tfloor(cx).mul(12.9898).add(rowN.mul(78.233))).mul(43758.5453));
        const lead = smoothstep(float(0.78), float(0.95),
          tmax(tabs(fract(cx).sub(0.5)).mul(2.0), tabs(fract(cy).sub(0.5)).mul(2.0)));

        let col = glassPalette(tfloor(cellId.mul(5.0)));

        // Gold border band, as almost every lancet has.
        const border = smoothstep(float(0.395), float(0.435), tabs(ax));
        col = mix(col, mix(GLASS_GOLD, GLASS_GOLD_PALE, fract(cy.mul(0.25))), border);

        // Field leading first, so the figure is not cut up by the diaper.
        col = col.mul(oneMinus(lead.mul(0.88)));

        /* The figure, drawn over the field, with its own heavier lead. */
        const fig = archangel(ax, u.y, 0);
        const figOut = archangel(ax, u.y, 0.017);
        const outline = clamp(figOut.all.sub(fig.all), float(0), float(1));

        col = mix(col, GLASS_GOLD, fig.wings.mul(0.94));
        col = mix(col, GLASS_GOLD_PALE.mul(0.86), fig.robe.mul(0.96));
        col = mix(col, GLASS_GOLD_PALE, fig.sword);
        col = mix(col, GLASS_GOLD_PALE, fig.head);
        col = mix(col, GLASS_GOLD, fig.halo.mul(oneMinus(fig.head)));
        // Heavy came around the whole silhouette — near-black, like real lead.
        col = mix(col, vec3(0.012, 0.016, 0.030), outline.mul(0.95));

        /* Day and night are not the same light through the same glass. By day
         * the sun is behind the panel and it becomes a light source; at night
         * only the moon and the candles below reach it, and it barely glows.
         *
         * The day multiplier is 2.6, not the 9.0 this started at. Past roughly
         * 3 the ACES curve compresses the bright channel and drags every
         * colour toward white — which is precisely how a blue-and-gold window
         * turns into pastel wallpaper. Chroma is pushed BEFORE the scale so
         * what compression remains has something to bite on. */
        const dayLit = punch(col, 1.35).mul(2.6);
        const nightLit = punch(col, 1.20).mul(0.42).add(vec3(0.020, 0.030, 0.070));
        glassMatWindow.colorNode = vec4(
          mix(nightLit, dayLit, uDay).mul(panel),
          panel,
        );
      }

      /* -------- the rose window */
      /* A rose is not a lancet and cannot borrow its shader: the geometry is
       * radial, so the cells, the leading and the tracery all run in polar
       * coordinates — a ring of petals around a central medallion, divided by
       * stone spokes. Running the lancet's brick grid on a square panel is
       * what made it look like a test card. */
      const roseMat = new THREE.MeshBasicNodeMaterial({
        transparent: true, side: THREE.DoubleSide,
      });
      {
        const p = uv().sub(0.5).mul(2.0);           // -1..1
        const r = tlen(vec3(p.x, p.y, float(0)));
        const ang = tatan(p.y, p.x);
        const disc = oneMinus(smoothstep(float(0.965), float(1.0), r));

        const SPOKES = 12;
        const spokeF = ang.div(Math.PI * 2).mul(SPOKES).add(100.0);
        const ring = tfloor(r.mul(3.4));
        const cellId = fract(tsin(tfloor(spokeF).mul(12.9898).add(ring.mul(41.31))).mul(43758.5453));
        let col = glassPalette(tfloor(cellId.mul(5.0)));

        // Alternating courses: blue field, gold every other ring, so the rose
        // reads as the flag turning around a gold centre.
        col = mix(col, GLASS_GOLD, tstep(float(0.5), fract(ring.mul(0.5))).mul(0.75));
        // Central medallion, the eye of the window.
        const medallion = oneMinus(smoothstep(float(0.145), float(0.165), r));
        col = mix(col, GLASS_GOLD_PALE, medallion);

        // Stone tracery: the radial spokes and the concentric rings between
        // the courses of glass.
        const spokeLead = smoothstep(float(0.80), float(0.97), tabs(fract(spokeF).sub(0.5)).mul(2.0));
        const ringLead = smoothstep(float(0.80), float(0.97), tabs(fract(r.mul(3.4)).sub(0.5)).mul(2.0));
        const outerRim = smoothstep(float(0.86), float(0.94), r);
        const lead = tmax(tmax(spokeLead, ringLead), outerRim);
        col = col.mul(oneMinus(lead.mul(0.94)));

        const dayLit = punch(col, 1.35).mul(2.8);
        const nightLit = punch(col, 1.20).mul(0.44).add(vec3(0.020, 0.030, 0.070));
        roseMat.colorNode = vec4(mix(nightLit, dayLit, uDay).mul(disc), disc);
      }

      const WIN_W = 4.3, WIN_H = 12.5, WIN_Y = 13.8;
      const winGeo = new THREE.PlaneGeometry(WIN_W, WIN_H);
      const windows = [];
      for (let side = -1; side <= 1; side += 2) {
        for (let i = 0; i < BAYS; i++) {
          const z = -NAVE_L + BAY_Z * (i + 0.5);
          const w = new THREE.Mesh(winGeo, glassMatWindow);
          /* On the INNER face of the wall, not inside it. The wall box is
           * 0.9 thick and centred on NAVE_W + 1.4, so its inner face is at
           * NAVE_W + 0.95; a window placed on the wall's centre line is
           * buried in solid stone and renders as nothing at all. */
          w.position.set(side * (NAVE_W + 0.93), WIN_Y, z);
          w.rotation.y = side * -Math.PI / 2;
          church.add(w);
          windows.push({ x: w.position.x, z, side });
        }
      }
      // The rose window over the altar, on the inner face of the apse wall.
      const rose = new THREE.Mesh(new THREE.PlaneGeometry(13.5, 13.5), roseMat);
      rose.position.set(0, 19.0, -NAVE_L + 0.48);
      church.add(rose);

      /* ------------------------------------------- the carved shields */
      /* Heraldry along the arcade, one shield to a bay, carved in low relief.
       *
       * Ukrainian units carry extraordinary insignia, and the meanings are
       * worth keeping: the Special Operations wolf comes from the old belief
       * that Cossacks could turn into wolves and catch arrows; military
       * intelligence took an owl, because owls eat bats; a rocket artillery
       * regiment flies a dragon whose fire is its missiles; the naval special
       * operations centre uses a seahorse for its adaptability to water; the
       * LGBTI soldiers took a unicorn, which is said to die rather than be
       * taken; a Zakarpattia unit chose the sun over wheat, for soldiers who
       * can still bring light. The trident over the altar is the state arms.
       *
       * WHAT THESE ARE NOT: copies of anyone's chevron. The actual patches are
       * artwork, designed by named artists and owned by them, and this
       * repository is Apache-2.0 — anything embedded here gets redistributed
       * by every fork. So what is carved is the MOTIF, not the badge: a wolf,
       * an owl, a dragon. Those are old heraldic and folk forms belonging to
       * nobody, and each is drawn here from scratch as a silhouette. A unit
       * that wants its own mark on this wall can send it, like everything else
       * on this page. */
      const shieldMat = new THREE.MeshStandardNodeMaterial({
        roughness: 0.82, metalness: 0.0, transparent: true, side: THREE.DoubleSide,
      });
      {
        const u = uv();
        const P = vec3(u.x.sub(0.5).mul(2.0), u.y.sub(0.5).mul(2.0), float(0));  // -1..1
        const px = P.x, py = P.y;

        // --- primitives. Everything below is built from these three.
        const disc = (cx, cy, r) => oneMinus(smoothstep(
          float(r).sub(0.012), float(r), tlen(vec3(px.sub(cx), py.sub(cy), float(0)))));
        const oval = (cx, cy, rx, ry) => oneMinus(smoothstep(
          float(0.985), float(1.0),
          tlen(vec3(px.sub(cx).div(rx), py.sub(cy).div(ry), float(0)))));
        const bar = (cx, cy, hw, hh) =>
          oneMinus(smoothstep(float(hw).sub(0.012), float(hw), tabs(px.sub(cx))))
            .mul(oneMinus(smoothstep(float(hh).sub(0.012), float(hh), tabs(py.sub(cy)))));

        /* --- the six motifs, each an original silhouette ---------------- */

        // Tryzub. Three prongs on a shaft — the state arms, drawn to the
        // proportions anyone would recognise rather than traced from a file.
        const tryzub = tmax(tmax(
          tmax(bar(0, 0.10, 0.075, 0.62), bar(-0.34, 0.26, 0.070, 0.42)),
          tmax(bar(0.34, 0.26, 0.070, 0.42), bar(0, -0.52, 0.16, 0.075))),
          tmax(bar(-0.205, 0.64, 0.205, 0.062), bar(0.205, 0.64, 0.205, 0.062)));

        // Owl — intelligence. Wide skull, huge facing eyes, ear tufts.
        const owlEyes = tmax(disc(-0.20, 0.20, 0.155), disc(0.20, 0.20, 0.155));
        const owl = tmax(
          tmax(oval(0, -0.02, 0.46, 0.62),
               tmax(bar(-0.32, 0.52, 0.085, 0.20), bar(0.32, 0.52, 0.085, 0.20))),
          float(0)).sub(owlEyes.mul(0.85))
          .add(tmax(disc(-0.20, 0.20, 0.062), disc(0.20, 0.20, 0.062)))
          .add(bar(0, 0.02, 0.05, 0.10)).clamp(0, 1);

        // Wolf — the Cossack who turns into one. Broad head, tapering muzzle,
        // upright ears.
        const wolf = tmax(
          tmax(oval(0, 0.12, 0.46, 0.42), oval(0, -0.38, 0.20, 0.34)),
          tmax(bar(-0.34, 0.52, 0.105, 0.24), bar(0.34, 0.52, 0.105, 0.24)))
          .sub(tmax(disc(-0.17, 0.16, 0.055), disc(0.17, 0.16, 0.055)).mul(0.9))
          .clamp(0, 1);

        // Dragon — rocket artillery, whose fire is its missiles. Serpent head
        // with a swept horn and a breath of flame.
        const dragon = tmax(
          tmax(oval(-0.05, 0.16, 0.40, 0.30), oval(0.30, -0.02, 0.26, 0.16)),
          tmax(bar(-0.26, 0.50, 0.075, 0.22),
               tmax(oval(0.62, -0.10, 0.16, 0.075), oval(0.84, -0.16, 0.10, 0.05))))
          .sub(disc(0.06, 0.20, 0.055).mul(0.9)).clamp(0, 1);

        // Seahorse — naval special operations, for adapting to the water.
        const seahorse = tmax(tmax(
          oval(-0.02, 0.42, 0.20, 0.26), oval(-0.30, 0.34, 0.16, 0.10)),
          tmax(oval(0.06, 0.02, 0.17, 0.30), oval(-0.06, -0.36, 0.15, 0.22)))
          .add(oval(-0.22, -0.56, 0.13, 0.10)).clamp(0, 1);

        // Unicorn — said to die rather than be taken. Head in profile, horn
        // raised.
        const unicorn = tmax(tmax(
          oval(-0.04, 0.10, 0.34, 0.42), oval(0.16, -0.32, 0.18, 0.30)),
          tmax(bar(-0.30, 0.52, 0.085, 0.22),
               oval(0.10, 0.66, 0.055, 0.32)))
          .sub(disc(0.06, 0.16, 0.05).mul(0.9)).clamp(0, 1);

        // Sun over wheat — soldiers who can still bring light.
        const rays = tabs(fract(tatan(py.sub(0.22), px).mul(6.0 / Math.PI)).sub(0.5)).mul(2.0);
        const sun = disc(0, 0.22, 0.30)
          .add(smoothstep(float(0.55), float(0.95), rays)
            .mul(oneMinus(smoothstep(float(0.30), float(0.58), tlen(vec3(px, py.sub(0.22), float(0)))))));
        const wheat = tmax(tmax(bar(0, -0.55, 0.045, 0.34), bar(-0.28, -0.58, 0.040, 0.28)),
                           bar(0.28, -0.58, 0.040, 0.28));
        const sunwheat = tmax(sun, wheat).clamp(0, 1);

        /* --- pick one per shield ---------------------------------------- */
        const which = tfloor(fract(float(instanceIndex).mul(1.0 / 6.0).add(0.001)).mul(6.0));
        let motif = tryzub;
        motif = mix(motif, owl, tstep(float(0.5), which));
        motif = mix(motif, wolf, tstep(float(1.5), which));
        motif = mix(motif, dragon, tstep(float(2.5), which));
        motif = mix(motif, seahorse, tstep(float(3.5), which));
        motif = mix(motif, unicorn, tstep(float(4.5), which));
        motif = mix(motif, sunwheat, tstep(float(5.5), which));

        /* --- the shield itself, and the carving ------------------------- */
        // Heraldic shape: square shoulders, sides drawn in, a point at the
        // bottom. The classic form, and it reads at a distance.
        const taper = smoothstep(float(0.35), float(-1.0), py).mul(0.55);
        /* The outline twice, at two sizes. The difference between them is the
         * raised bezel around the edge, and without it the plate has no border
         * at all — the gilded device just floats on the masonry with nothing
         * to say it is mounted on anything. A real shield reads because of its
         * rim before it reads because of its charge. */
        const shieldAt = (hw, top) =>
          oneMinus(smoothstep(float(hw).sub(taper).sub(0.02), float(hw).sub(taper), tabs(px)))
            .mul(oneMinus(smoothstep(float(top).sub(0.03), float(top), py)))
            .mul(oneMinus(smoothstep(float(top).sub(0.03), float(top).add(0.06), py.negate())));
        const shield = shieldAt(0.82, 0.92);
        const border = clamp(shield.sub(shieldAt(0.70, 0.82)), float(0), float(1));

        /* Relief, not paint. The motif is treated as a raised surface and lit
         * from the upper left: the horizontal gradient of the mask becomes a
         * fake normal, so the carving catches light on one edge and shadows on
         * the other. It is a cheat — there is no geometry here at all — but it
         * is the cheat that makes a flat panel read as cut stone. */
        const lift = motif.sub(0.5).mul(2.0).clamp(0, 1);
        const edgeL = motif.sub(oneMinus(smoothstep(float(0.0), float(0.03), px.add(0.012))).mul(0));
        const bevel = smoothstep(float(0.0), float(0.42), motif)
          .mul(oneMinus(smoothstep(float(0.58), float(1.0), motif)));
        const rim = smoothstep(float(0.86), float(1.0), tabs(px).add(tabs(py)).mul(0.62));

        const stoneBase = mix(vec3(0.180, 0.172, 0.156), vec3(0.245, 0.234, 0.210),
          mx_noise_float(positionWorld.mul(3.0)).mul(0.5).add(0.5));
        // The device itself is gilded — old shields were, and it ties the wall
        // to the glass and the flames.
        const gilt = mix(GLASS_GOLD.mul(0.95), GLASS_GOLD_PALE.mul(1.15), bevel);
        let col = mix(stoneBase, gilt, lift.mul(0.92));
        col = col.add(bevel.mul(0.30));                       // catch light on the cut edge
        col = col.sub(rim.mul(0.04));
        // Bezel: a lighter band of dressed stone all the way round, so the
        // plate is an object on the wall rather than a stain in it.
        col = mix(col, vec3(0.34, 0.325, 0.295).add(goldVec().mul(0.10)), border.mul(0.92));
        // Warm from the candles below, cool from the windows above.
        col = col.mul(uDay.mul(0.45).add(0.80))
          .add(goldVec().mul(oneMinus(smoothstep(float(0.0), float(1.0), py)).mul(0.05)));

        shieldMat.colorNode = vec4(col, shield);
        shieldMat.roughnessNode = clamp(oneMinus(lift.mul(0.45)).mul(0.9), float(0.18), float(1));
        shieldMat.emissiveNode = gilt.mul(lift).mul(0.22);
      }

      const shieldGeo = new THREE.PlaneGeometry(4.6, 5.4);
      const shields = new THREE.InstancedMesh(shieldGeo, shieldMat, BAYS * 2);
      {
        const m = new THREE.Matrix4();
        const q = new THREE.Quaternion();
        const axis = new THREE.Vector3(0, 1, 0);
        const one = new THREE.Vector3(1, 1, 1);
        let k = 0;
        for (let side = -1; side <= 1; side += 2) {
          for (let i = 0; i < BAYS; i++) {
            const z = -NAVE_L + BAY_Z * (i + 0.5);
            q.setFromAxisAngle(axis, side * -Math.PI / 2);
            shields.setMatrixAt(k++, m.compose(
              new THREE.Vector3(side * (NAVE_W + 0.90), 6.6, z), q, one));
          }
        }
        shields.count = k;
        shields.instanceMatrix.needsUpdate = true;
      }
      shields.frustumCulled = false;
      church.add(shields);

      /* -------- what comes through the windows */
      /* One shaft per window, as an additive card hung in the air on the axis
       * the light travels. This is the cheap way to do volumetric light and it
       * is the right way here: the alternative is a raymarch per pixel, and
       * every one of those pixels would be spent on air rather than on the
       * candles the room is actually for.
       *
       * By day these are sunlight, warmed and tinted by the glass they pass
       * through. At night they are the moon doing the same job in blue, and
       * much fainter — but present, because a memorial that goes black for
       * half its cycle is not one anybody would sit in. */
      const shaftMat = new THREE.MeshBasicNodeMaterial({
        transparent: true, depthWrite: false, side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
      });
      {
        const u = uv();
        const across = oneMinus(clamp(u.x.sub(0.5).abs().mul(2.15), float(0), float(1)));
        const down = smoothstep(float(0.0), float(0.09), u.y)
          .mul(oneMinus(smoothstep(float(0.16), float(0.96), u.y)));
        // Dust turning in the beam, so it does not read as a flat card.
        const motes = mx_noise_float(vec3(u.x.mul(6.0), u.y.mul(2.6).sub(uTime.mul(0.06)), float(3.0)))
          .mul(0.5).add(0.72);
        const sun = vec3(1.00, 0.86, 0.62);
        const moon = vec3(0.52, 0.64, 1.00);
        const tint = mix(moon, sun, uDay);
        const strength = mix(float(0.020), float(0.075), uDay);
        shaftMat.colorNode = vec4(tint, across.mul(down).mul(motes).mul(strength));
      }
      /* Each beam runs FROM its window TO a patch of floor, and is built from
       * that pair of points rather than positioned by eye.
       *
       * The previous version placed the card at `w.x * 0.50` — half the
       * distance to the wall — as a 30-unit plane centred near the vault. It
       * never touched a window, which is exactly why the light read as falling
       * out of the roof.
       *
       * Two crossed planes per beam, not one. A single quad vanishes when the
       * camera lines up with its edge, and the orbit goes all the way round,
       * so at two points in every turn the light would simply switch off. */
      const shaftGeo = new THREE.PlaneGeometry(6.4, 1);   // unit length; scaled per beam
      const UP = new THREE.Vector3(0, 1, 0);
      for (const w of windows) {
        const from = new THREE.Vector3(w.x, WIN_Y + 3.0, w.z);
        // Lands inboard and a little down-nave, so the beams rake across the
        // floor around the rack instead of dropping straight down the wall.
        const to = new THREE.Vector3(w.side * 5.5, 0, w.z + 10.0);
        const dir = new THREE.Vector3().subVectors(to, from);
        const len = dir.length();
        dir.normalize();
        for (const roll of [0, Math.PI / 2]) {
          const s = new THREE.Mesh(shaftGeo, shaftMat);
          s.quaternion.setFromUnitVectors(UP, dir);
          s.rotateY(roll);                       // local Y is now the beam axis
          s.position.copy(from).addScaledVector(dir, len / 2);
          s.scale.set(1, len, 1);
          s.renderOrder = 0;
          church.add(s);
        }
      }

      /* ========================================================= STATUES */
      /* Michael at the crossing, a wolf and a dragon flanking him.
       *
       * These are real solids, not the flat silhouettes the shields use. The
       * orbit goes all the way round, and a cutout standing in the middle of
       * the nave reads as painted cardboard the instant you pass it. So: a
       * robe is a profile of revolution — the same lathe trick as the candle
       * wax — with head, wings, sword and beast built from solids on top.
       *
       * The beasts are deliberately blocky. Quadrupeds do not lathe, and
       * carved stone animals on a plinth ARE blocky; the stylisation is what
       * the material would give you anyway. */
      const statueMat = new THREE.MeshStandardNodeMaterial({ roughness: 0.74, metalness: 0.0 });
      {
        const p = positionWorld;
        const grain = mx_noise_float(p.mul(2.6)).mul(0.5).add(0.5);
        const fine = mx_noise_float(p.mul(14.0)).mul(0.5).add(0.5);
        // Pale limestone, warmer than the walls so the figures lift off them.
        const base = mix(vec3(0.200, 0.190, 0.170), vec3(0.278, 0.265, 0.238), grain);
        // Candlelight reaches the lower half and dies out going up, which is
        // what actually happens to anything standing over a rack of candles.
        const fromBelow = oneMinus(smoothstep(float(0.4), float(6.0), p.y)).mul(0.7);
        statueMat.colorNode = base.add(fine.sub(0.5).mul(0.03))
          .add(goldVec().mul(fromBelow.mul(0.30)))
          .mul(uDay.mul(0.34).add(0.72));
        statueMat.roughnessNode = clamp(fine.mul(0.22).add(0.68), float(0.3), float(1));
        statueMat.emissiveNode = goldVec().mul(fromBelow.mul(0.05));
      }

      const statues = new THREE.Group();
      church.add(statues);
      const part = (geo, x, y, z, rot, scl) => {
        const m = new THREE.Mesh(geo, statueMat);
        m.position.set(x, y, z);
        if (rot) m.rotation.set(rot[0], rot[1], rot[2]);
        if (scl) m.scale.set(scl[0], scl[1], scl[2]);
        statues.add(m);
        return m;
      };
      const box = (w, h, d) => new THREE.BoxGeometry(w, h, d);
      const ball = (r) => new THREE.SphereGeometry(r, 18, 12);

      // Plinths. Sunk below their step, for the same z-fighting reason as
      // everything else that meets a surface.
      const plinth = (x, y, z, w, h) => {
        part(box(w, h + 0.5, w), x, y + h / 2 - 0.25, z);
        part(box(w * 1.18, 0.18, w * 1.18), x, y + h, z);      // cap moulding
      };

      const DAIS_TOP = 0.68, STEP_TOP = 0.34;
      const MZ = ALTAR_Z + 1.4;        // Michael, just forward of the altar

      /* -------- Michael */
      {
        const H = 1.7;                                  // plinth height
        plinth(0, DAIS_TOP, MZ, 2.5, H);
        const base = DAIS_TOP + H + 0.09;

        // Robe: hem wide, drawn in to the shoulders.
        /* Flared hem, drawn in at the waist, out again across the chest. A
         * straight taper lathes into a traffic cone — the waist is the single
         * profile point that makes it read as a figure under cloth. */
        const robe = [
          [0.00, 0.00], [0.90, 0.00], [0.96, 0.035], [0.88, 0.14], [0.70, 0.34],
          [0.58, 0.50], [0.54, 0.60], [0.58, 0.70], [0.60, 0.80],
          [0.54, 0.90], [0.40, 0.97], [0.00, 1.00],
        ].map(([x, y]) => new THREE.Vector2(x * 1.16, y * 3.7));
        part(new THREE.LatheGeometry(robe, 24), 0, base, MZ);

        const shoulder = base + 3.7;
        part(ball(0.36), 0, shoulder + 0.36, MZ);                       // head
        part(new THREE.TorusGeometry(0.56, 0.055, 8, 24),
             0, shoulder + 0.44, MZ - 0.30, [0.30, 0, 0]);              // halo

        // Arms: right raised with the sword, left lowered holding the scales
        // of judgement, which is how he is nearly always shown.
        part(box(0.22, 1.5, 0.22), 0.62, shoulder - 0.32, MZ, [0, 0, -0.42]);
        part(box(0.22, 1.4, 0.22), -0.60, shoulder - 0.50, MZ, [0, 0, 0.16]);

        // The sword, upright.
        part(box(0.13, 3.0, 0.13), 1.02, shoulder + 1.05, MZ, [0, 0, -0.08]);
        part(box(0.62, 0.13, 0.16), 0.95, shoulder - 0.32, MZ);         // crossguard
        part(box(0.17, 0.30, 0.17), 0.93, shoulder - 0.56, MZ);         // grip

        /* Wings. Flattened tapered solids rather than planes: a wing IS thin,
         * so a thin box is the honest primitive, and it still has a silhouette
         * from the side where a plane would vanish. */
        /* A fan of tapered feathers per side, radiating from one point at the
         * shoulder and rising above the head. The first attempt was two big
         * slabs set at an angle, which from the floor of the nave read
         * unmistakably as a windmill: the giveaway is that real wings radiate
         * from a single joint and sweep UP, they do not stick out sideways. */
        for (const s of [-1, 1]) {
          const sx = s * 0.46, sy = shoulder + 0.12, sz = MZ - 0.40;
          /* Wide enough to OVERLAP. Feathers 0.46 deep at 18 degrees apart
           * leave gaps, and a wing with gaps in it is a rake. Overlapping
           * them merges the fan into one mass that still shows its edges. */
          const feathers = [
            [2.30, 0.82, 0.06], [2.65, 0.80, 0.32], [2.80, 0.76, 0.58],
            [2.60, 0.68, 0.84], [2.15, 0.58, 1.10], [1.60, 0.46, 1.34],
          ];
          for (const [len, ht, tilt] of feathers) {
            const a = s > 0 ? tilt : Math.PI - tilt;
            part(box(len, ht, 0.12),
                 sx + Math.cos(a) * len * 0.5,
                 sy + Math.sin(a) * len * 0.5,
                 sz, [0, s * 0.20, a]);
          }
        }
      }

      /* -------- the wolf, and the dragon */
      const BEAST_X = 7.4;
      {
        // Wolf, sitting: the Cossack who could turn into one and catch arrows.
        const H = 1.15;
        plinth(-BEAST_X, STEP_TOP, MZ + 0.6, 2.0, H);
        const b = STEP_TOP + H + 0.09;
        part(ball(0.78), -BEAST_X, b + 0.80, MZ + 0.6, null, [1, 1.15, 0.85]);   // haunches
        part(box(0.62, 1.5, 0.62), -BEAST_X, b + 1.55, MZ + 0.35);                // chest
        part(ball(0.44), -BEAST_X, b + 2.42, MZ + 0.28, null, [1, 0.95, 1.15]);   // skull
        part(box(0.30, 0.30, 0.62), -BEAST_X, b + 2.30, MZ - 0.14);               // muzzle
        for (const s of [-1, 1]) {
          part(box(0.20, 0.42, 0.14), -BEAST_X + s * 0.26, b + 2.80, MZ + 0.34,
               [0, 0, s * 0.18]);                                                 // ears
          part(box(0.24, 1.1, 0.24), -BEAST_X + s * 0.36, b + 0.72, MZ - 0.10);   // forelegs
        }
        part(box(0.26, 0.26, 1.1), -BEAST_X, b + 0.42, MZ + 1.20, [0.5, 0, 0]);   // tail
      }
      {
        // Dragon, couchant with the head raised: rocket artillery, whose fire
        // is its missiles.
        const H = 1.15;
        plinth(BEAST_X, STEP_TOP, MZ + 0.6, 2.0, H);
        const b = STEP_TOP + H + 0.09;
        part(ball(0.85), BEAST_X, b + 0.72, MZ + 0.7, null, [1, 0.85, 1.25]);     // coiled body
        part(box(0.46, 1.7, 0.46), BEAST_X, b + 1.55, MZ + 0.20, [0.30, 0, 0]);   // neck
        part(box(0.44, 0.40, 0.92), BEAST_X, b + 2.40, MZ - 0.42);                // head
        part(box(0.20, 0.18, 0.40), BEAST_X, b + 2.34, MZ - 1.02);                // snout
        for (const s of [-1, 1]) {
          part(box(0.14, 0.44, 0.14), BEAST_X + s * 0.17, b + 2.74, MZ - 0.26,
               [0, 0, s * 0.26]);                                                 // horns
          // Wings, half-furled.
          for (const [len, tilt] of [[1.55, 0.22], [1.75, 0.52], [1.60, 0.82], [1.25, 1.08]]) {
            const a = s > 0 ? tilt : Math.PI - tilt;
            part(box(len, 0.56, 0.11),
                 BEAST_X + s * 0.38 + Math.cos(a) * len * 0.5,
                 b + 1.45 + Math.sin(a) * len * 0.5,
                 MZ + 0.62, [0, s * 0.26, a]);
          }
        }
        part(box(0.24, 0.24, 1.5), BEAST_X, b + 0.44, MZ + 1.55, [0.42, 0, 0]);   // tail
      }

      /* -------- light falling on each of them */
      /* A spot from the vault onto every figure, and a matching visible beam.
       * Two separate things doing two separate jobs: the SpotLight is what
       * actually lifts the stone out of the dark, the crossed cards are what
       * you see hanging in the air. Neither alone reads as a shaft of light
       * landing on a statue. */
      const statueLights = [];
      for (const [x, reach] of [[0, 1.30], [-BEAST_X, 1.0], [BEAST_X, 1.0]]) {
        const sp = new THREE.SpotLight(0xffe9cf, 0, 42, 0.40, 0.92, 1.0);
        sp.position.set(x * 0.55, 21, MZ + 5.0);
        sp.target.position.set(x, 2.2, MZ);
        church.add(sp, sp.target);
        statueLights.push({ light: sp, reach });

        const from = new THREE.Vector3(x * 0.55, 20.5, MZ + 5.0);
        const to = new THREE.Vector3(x, 0.6, MZ);
        const dir = new THREE.Vector3().subVectors(to, from);
        const len = dir.length();
        dir.normalize();
        for (const roll of [0, Math.PI / 2]) {
          const s = new THREE.Mesh(new THREE.PlaneGeometry(3.4 * reach, 1), shaftMat);
          s.quaternion.setFromUnitVectors(UP, dir);
          s.rotateY(roll);
          s.position.copy(from).addScaledVector(dir, len / 2);
          s.scale.set(1, len, 1);
          church.add(s);
        }
      }

      /* ------------------------------------------------------------ loop */
      const resize = () => {
        const w = mount.clientWidth, h = mount.clientHeight;
        if (!w || !h) return;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h, false);
      };
      resize();
      const ro = new ResizeObserver(resize);
      ro.observe(mount);

      /* The whole rack stands there from the first frame — five hundred
       * candles, every one of them UNLIT. Only the flames, halos and plume are
       * gated on the count.
       *
       * That is the substance of the room, not a detail of it. A visitor
       * arrives to a dark church full of candles nobody has lit yet, and the
       * first flame in it is theirs. The wax, glass and wick shaders already
       * multiply their warm terms by igniteFor(index), so an unlit candle
       * needs nothing special: it simply renders with no flame of its own,
       * catching only what light the room gives it. */
      const applyCount = (litF) => {
        const shown = Math.min(Math.ceil(litF), N);
        uLit.value = litF;
        uLitCount.value = Math.max(1, shown);
        uNewest.value = Math.floor(litF);

        wax.count = glass.count = wick.count = N;      // always the full rack
        const lod = lodFor(N);
        if (lod !== lodNow) {
          lodNow = lod;
          wax.geometry = lod.wax;
          glass.geometry = lod.glass;
          wick.geometry = lod.wick;
        }

        flames.geometry.setDrawRange(0, shown * 6);
        halos.geometry.setDrawRange(0, shown * 6);
        // Nothing lit yet is the normal opening state, but a zero-vertex draw
        // makes WebGPU warn every frame — so hide rather than draw empty.
        const anyLit = shown > 0;
        flames.visible = halos.visible = anyLit;
        if (computeOk) particles.visible = anyLit;
      };

      // The room comes up from dark rather than snapping to a full field.
      // Arriving somewhere quiet should feel like arriving.
      let litF = 0;
      let prevWhole = 0;
      // Particles simulated vs particles drawn. See the adaptive block below.
      let pDrawn = P_COUNT;
      const t0 = performance.now();
      let tPrev = t0;
      let frames = 0, fpsT = t0, fps = 0;

      renderer.setAnimationLoop(() => {
        const now = performance.now();
        const t = (now - t0) / 1000;
        // Clamped, so a backgrounded tab does not resume by fast-forwarding
        // every ignition and every particle at once.
        const dt = Math.min((now - tPrev) / 1000, 0.05);
        tPrev = now;
        uTime.value = t;
        uDt.value = dt;
        uHashT.value = (t * 0.37) % 1;

        const target = targetRef.current;
        if (litF !== target) {
          const gap = target - litF;
          // One candle per IGNITE_SECONDS when a visitor lights one; far
          // faster when filling in a stored field on load, which would
          // otherwise take an hour at ceremony speed.
          const rate = Math.max(1 / IGNITE_SECONDS, Math.abs(gap) * 1.6);
          const stepped = litF + Math.sign(gap) * rate * dt;
          litF = gap > 0 ? Math.min(stepped, target) : Math.max(stepped, target);
          uFieldR.value = fieldRadius(litF);
        }
        // A candle finishing its ignition throws sparks.
        const whole = Math.floor(litF);
        if (whole > prevWhole) uBurst.value = 1;
        prevWhole = whole;
        uBurst.value *= Math.exp(-dt * 2.6);
        applyCount(litF);
        // Constant light per candle, whatever the count or the device tier.
        uPGain.value = Math.min(1, (140 * Math.max(litF, 1)) / pDrawn);

        if (computeOk) {
          try { renderer.compute(simulate); } catch (e) { computeOk = false; particles.visible = false; }
        }

        /* Day/night. A full turn of the clock takes cycleSeconds; the room
         * spends most of it settled at day or at night and crosses between
         * them quickly, because a church at noon and a church at midnight are
         * both worth sitting in and the half-lit state between is not. */
        const phase = (t / Math.max(20, cycleRef.current)) % 1;
        const raw = 0.5 - 0.5 * Math.cos(phase * Math.PI * 2);
        uDay.value = smoothstep01(raw, 0.30, 0.70);
        // The sun swings across the south windows and the moon takes the same
        // path at night, which is what makes the shafts move across the floor.
        const sunA = phase * Math.PI * 2;
        uSunDir.value.set(Math.cos(sunA) * 0.75, Math.abs(Math.sin(sunA)) * 0.62 + 0.22, -0.55).normalize();

        const d = uDay.value;
        sky.setRGB(0.028 + d * 0.075, 0.034 + d * 0.095, 0.052 + d * 0.115);
        hemi.color.setRGB(0.10 + d * 0.14, 0.13 + d * 0.16, 0.20 + d * 0.13);
        // Night is genuinely dark but never black: a church lit only by the
        // moon and a rack of candles still shows you its walls, and one that
        // does not is a black rectangle for half of every cycle.
        hemi.intensity = 0.42 + d * 0.46;
        sunLight.color.setRGB(0.42 + d * 0.58, 0.52 + d * 0.42, 0.95 - d * 0.09);
        sunLight.intensity = 0.22 + d * 1.25;
        sunLight.position.copy(uSunDir.value).multiplyScalar(30);
        /* The statue spots stay lit around the clock — they are what the room
         * does for its figures, not what the weather does. They lift with the
         * day so the stone does not go flat at noon, and never drop to nothing
         * at night, because a spotlit statue in a dark church is the whole
         * effect. */
        for (const sl of statueLights) sl.light.intensity = (330 + d * 260) * sl.reach;
        scene.fog.color.copy(sky);
        scene.fog.density = 0.020 - d * 0.007;

        // The camera is the visitor's now. Orbit controls own it; the slow
        // auto-rotate is only there until someone takes hold.
        if (controls) {
          controls.autoRotate = driftRef.current && !userTook;
          controls.update();
          /* Hard fence. The azimuth limits above should make this unreachable,
           * but a clamp costs three comparisons and guarantees the promise:
           * the camera stays inside the nave, so a visitor can never end up
           * staring at the outside of a wall with nothing behind it. */
          const P = camera.position;
          P.x = Math.max(-25.4, Math.min(25.4, P.x));
          P.z = Math.max(-NAVE_L + 3.0, Math.min(NAVE_L - 4.0, P.z));
          P.y = Math.max(1.15, Math.min(WALL_H - 5.0, P.y));
        } else {
          const sway = driftRef.current ? Math.sin(t * 0.06) * 1.1 : 0;
          camera.position.set(sway, 5.8, 11.5);
          camera.lookAt(0, 1.6, FIELD_Z);
        }

        warm.intensity = (20 + Math.sin(t * 1.9) * 2.0 + Math.sin(t * 4.3) * 1.0)
          * Math.min(1, litF / 40 + 0.05);

        renderer.render(scene, camera);

        frames++;
        if (now - fpsT >= 1000) {
          fps = Math.round((frames * 1000) / (now - fpsT));
          frames = 0; fpsT = now;

          /* Adaptive draw range. The simulation always steps the full field —
           * it is a fixed cost and a cheap one — but rasterising a million
           * additive points is not free on integrated graphics, so the number
           * actually DRAWN backs off when frames get long and climbs again
           * when they do not. Which particles get drawn is arbitrary, so
           * there is no artefact from moving the line; the plume just thins.
           * The first few seconds are skipped because shader compilation and
           * the first buffer upload make them unrepresentatively slow. */
          if (computeOk && t > 4) {
            if (fps < 28 && pDrawn > P_FLOOR) {
              pDrawn = Math.max(P_FLOOR, Math.round(pDrawn * 0.6));
              pGeo.setDrawRange(0, pDrawn);
            } else if (fps > 55 && pDrawn < P_COUNT) {
              pDrawn = Math.min(P_COUNT, Math.round(pDrawn * 1.4));
              pGeo.setDrawRange(0, pDrawn);
            }
          }

          // Read by the headless verifier, and handy from a console.
          window.__memorialStats = {
            fps, backend, particles: computeOk ? P_COUNT : 0, drawn: computeOk ? pDrawn : 0,
            candles: Math.ceil(litF), roomSize: N, lod: LODS.indexOf(lodNow),
          };
        }
      });

      stop = () => {
        renderer.setAnimationLoop(null);
        ro.disconnect();
        if (controls) controls.dispose();
        scene.traverse((o) => {
          if (o.geometry) o.geometry.dispose();
          if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((mm) => mm.dispose());
        });
        // The LOD tiers not currently attached are not reachable by traverse.
        LODS.forEach((l) => { l.wax.dispose(); l.glass.dispose(); l.wick.dispose(); });
        renderer.dispose();
        if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
      };

      apiRef.current = {
        setGold: (hex) => { try { uGold.value.set(hex); } catch (e) { /* ignore */ } },
      };
    })();

    return () => { disposed = true; apiRef.current = null; stop(); };
    // Deliberately empty: the scene is built once. Props reach it through the
    // refs and uniforms above. See the header note.
  }, [roomSize]);

  return React.createElement('div', {
    ref: mountRef,
    style: { position: 'absolute', inset: 0, overflow: 'hidden', background: '#0d0e10' },
  });
}

if (typeof module !== 'undefined') module.exports = { MemorialLight };
if (typeof window !== 'undefined') window.MemorialLight = MemorialLight;
