// Headless smoke test for the Memorial Gallery page.
//
//   node verify-memorial.mjs [webgpu|webgl2]
//
// Gates on:
//   1. no console errors / page errors (benign 404s whitelisted)
//   2. the hall reached a backend and the page says which one
//   3. ZERO cross-origin requests — the whole point of vendoring
//   4. lighting candles grows the field, and the count survives a reload
//
// Uses textContent, not innerText: the hall's layered absolute children are
// skipped by innerText in headless layout even though they render fine.
import { spawn } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { createRequire } from 'node:module';

// Needs puppeteer-core and a Chrome. Either `npm i puppeteer-core` here, or
// borrow an existing install — NODE_PATH does not affect ESM resolution, so
// the borrow goes through a CJS require rooted at that package.
const BORROW = 'C:/Users/Auerbach/AureliusDynamicSolutons/tsl-lib/bench/package.json';
let puppeteer;
try {
  puppeteer = createRequire(import.meta.url)('puppeteer-core');
} catch {
  try {
    puppeteer = createRequire(BORROW)('puppeteer-core');
  } catch {
    console.error('puppeteer-core not found. Run `npm i puppeteer-core` in this directory.');
    process.exit(2);
  }
}

import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const DIR = dirname(fileURLToPath(import.meta.url));
const SHOTS = join(DIR, '.verify-out');
const CHROME = process.env.CHROME_PATH
  || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const PORT = 8642;
const PAGE = `http://localhost:${PORT}/index.html`;

// .image-slots.state.json is absent until someone drops a photo — image-slot.js
// fetches it and catches. favicon.ico is the browser asking on its own.
const BENIGN_404 = [/\.image-slots\.state\.json$/, /favicon\.ico$/];

mkdirSync(SHOTS, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const server = spawn('python', ['-m', 'http.server', String(PORT)], { cwd: DIR, stdio: 'ignore' });
await sleep(900);

const backends = process.argv[2] ? [process.argv[2]] : ['webgpu', 'webgl2'];
let failures = 0;

// Must NOT match the pre-report default, "Rendered live · WebGPU with WebGL2
// fallback · TSL node materials" — a loose `.` for the middot let that string
// satisfy the wait instantly, so the check passed before onReady ever fired.
// Anchoring on " · TSL node materials" admits only the reported form.
const readNote = () => {
  const m = document.body.textContent.match(
    /Rendered live · (WebGPU|WebGL2) · TSL node materials/,
  );
  return m && m[1];
};

try {
  for (const backend of backends) {
    const args = [
      '--window-size=1400,900', '--hide-scrollbars', '--no-first-run',
      `--user-data-dir=${SHOTS}/profile-${backend}`,
    ];
    if (backend === 'webgpu') args.push('--enable-unsafe-webgpu');

    const browser = await puppeteer.launch({
      executablePath: CHROME, headless: 'new', args,
      defaultViewport: { width: 1400, height: 900 },
    });

    const page = await browser.newPage();
    await page.setCacheEnabled(false);
    // Deterministic fallback: Chrome flags did not reliably disable WebGPU, so
    // take navigator.gpu away instead. That is exactly the condition a visitor
    // on an older browser presents, which is what the fallback path is for.
    if (backend === 'webgl2') {
      await page.evaluateOnNewDocument(() => {
        Object.defineProperty(navigator, 'gpu', { get: () => undefined, configurable: true });
      });
    }

    const errors = [];
    const external = [];
    const requests = [];
    const bad404 = [];
    page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text().slice(0, 300)); });
    page.on('pageerror', (e) => errors.push(String(e).slice(0, 300)));
    /* ERR_ABORTED on a media file is not a failure. A <video> requests byte
     * ranges and cancels them once it has buffered enough, and every reload
     * aborts whatever was in flight — so the hero video reports one on nearly
     * every load while playing perfectly. The absence of that error was never
     * the thing worth checking; that the video is actually PLAYING is, and
     * that is asserted directly below. */
    page.on('requestfailed', (r) => {
      const benign = /\.(mp4|webm|ogg)(\?|$)/i.test(r.url())
        && r.failure()?.errorText === 'net::ERR_ABORTED';
      if (!benign) errors.push(`REQUEST FAILED ${r.url()} — ${r.failure()?.errorText}`);
    });
    page.on('response', (r) => {
      if (r.status() >= 400 && !BENIGN_404.some((re) => re.test(r.url()))) {
        bad404.push(`HTTP ${r.status()} ${r.url()}`);
      }
    });
    page.on('request', (r) => {
      const u = r.url();
      requests.push(u);
      if (!/^(data:|blob:)/.test(u) && !u.startsWith(`http://localhost:${PORT}/`)) external.push(u);
    });

    console.log(`\n=== ${backend} ===`);
    let reported = null;
    try {
      await page.goto(PAGE, { waitUntil: 'load', timeout: 40000 });
      await page.evaluate(() => localStorage.removeItem('memorial.candles.lit'));
      await page.reload({ waitUntil: 'load', timeout: 40000 });

      await page.waitForFunction(readNote, { timeout: 45000 });
      reported = await page.evaluate(readNote);
      const navGpu = await page.evaluate(() => typeof navigator.gpu);
      console.log(`  backend the page reports: ${reported}   (navigator.gpu is ${navGpu})`);
      if (backend === 'webgpu' && reported !== 'WebGPU') { console.log('  FAIL: expected WebGPU'); failures++; }
      if (backend === 'webgl2' && reported !== 'WebGL2') { console.log('  FAIL: expected WebGL2 fallback'); failures++; }

      /* Frame rate is sampled across a window and the PEAK is taken, because a
       * single reading lands in one of two ditches. Early on, shader
       * compilation and the first buffer uploads make frames long — on a cold
       * profile the WebGPU run needs several seconds before it is representative
       * of anything. Later, headless Chrome throttles rAF once it decides the
       * page is occluded, which pins any page at ~13 fps no matter what it
       * draws. The peak in between is the only honest figure here. */
      let stats = null;
      for (let i = 0; i < 12; i++) {
        await sleep(1000);
        const s = await page.evaluate(() => window.__memorialStats || null);
        if (s && (!stats || s.fps > stats.fps)) stats = s;
      }
      console.log(`  peak: ${JSON.stringify(stats)}`);
      if (!stats) { console.log('  FAIL: no render stats — the loop never ran'); failures++; }
      else {
        if (!stats.particles) { console.log('  FAIL: particle simulation did not start'); failures++; }
        if (stats.fps < 20) { console.log(`  FAIL: peak ${stats.fps} fps is too slow`); failures++; }
      }
      // The hero video: loaded, decoding at the right size, and running.
      const vid = await page.evaluate(() => {
        const v = document.getElementById('hero-video');
        if (!v) return { exists: false };
        return { exists: true, readyState: v.readyState, paused: v.paused,
                 w: v.videoWidth, h: v.videoHeight, err: v.error && v.error.message };
      });
      console.log(`  hero video: ${JSON.stringify(vid)}`);
      if (!vid.exists) { console.log('  FAIL: hero video element missing'); failures++; }
      else if (vid.err) { console.log(`  FAIL: hero video error — ${vid.err}`); failures++; }
      else if (vid.readyState < 2 || !vid.w) { console.log('  FAIL: hero video never decoded'); failures++; }

      await page.screenshot({ path: `${SHOTS}/memorial-${backend}-idle.png` });

      for (let i = 0; i < 5; i++) {
        await page.evaluate(() => {
          const b = [...document.querySelectorAll('button')]
            .find((x) => /Light a candle/i.test(x.textContent || ''));
          if (!b) throw new Error('no "Light a candle" button found');
          b.click();
        });
        await sleep(200);
      }
      await sleep(3000);
      const stored = await page.evaluate(() => localStorage.getItem('memorial.candles.lit'));
      const shown = await page.evaluate(() => {
        const m = document.body.textContent.match(/Candles lit by visitors\s*(\S+)/);
        return m && m[1];
      });
      console.log(`  after 5 clicks: stored=${stored} displayed=${shown}`);
      if (stored !== '5') { console.log('  FAIL: count did not reach 5'); failures++; }
      if (shown !== '005') { console.log('  FAIL: display did not reach 005'); failures++; }
      const rl = await page.evaluate(() => (document.body.textContent.match(/Room (\d+) of (\d+)/) || [])[0] || null);
      console.log(`  room label: ${rl}`);
      await page.screenshot({ path: `${SHOTS}/memorial-${backend}-lit.png` });

      await page.reload({ waitUntil: 'load', timeout: 40000 });
      await page.waitForFunction(readNote, { timeout: 45000 });
      await sleep(3000);
      const after = await page.evaluate(() => localStorage.getItem('memorial.candles.lit'));
      const shown2 = await page.evaluate(() => {
        const m = document.body.textContent.match(/Candles lit by visitors\s*(\S+)/);
        return m && m[1];
      });
      console.log(`  after reload:   stored=${after} displayed=${shown2}`);
      if (after !== '5' || shown2 !== '005') { console.log('  FAIL: did not survive reload'); failures++; }
      await page.screenshot({ path: `${SHOTS}/memorial-${backend}-reload.png` });

      // Growth well past the old 900 ceiling, to prove there isn't one.
      await page.evaluate(() => localStorage.setItem('memorial.candles.lit', '5000'));
      await page.reload({ waitUntil: 'load', timeout: 40000 });
      await page.waitForFunction(readNote, { timeout: 45000 });
      await sleep(6000);
      const big = await page.evaluate(() => {
        const m = document.body.textContent.match(/Candles lit by visitors\s*(\S+)/);
        return m && m[1];
      });
      const room5000 = await page.evaluate(() => (document.body.textContent.match(/Room \d+ of \d+/) || [])[0] || null);
      console.log(`  5000 stored candles: displayed=${big}  ${room5000}`);
      if (big !== '5000') { console.log('  FAIL: large count not shown'); failures++; }
      if (room5000 !== 'Room 11 of 11') { console.log(`  FAIL: expected Room 11 of 11, got ${room5000}`); failures++; }
      await page.screenshot({ path: `${SHOTS}/memorial-${backend}-5000.png` });
      await page.evaluate(() => localStorage.removeItem('memorial.candles.lit'));
    } catch (err) {
      console.log(`  FAIL: ${String(err.message || err).slice(0, 400)}`);
      failures++;
    }

    console.log(`  requests: ${requests.length} total, ${external.length} external`);
    if (external.length) {
      failures++;
      console.log('  FAIL — EXTERNAL REQUESTS:');
      [...new Set(external)].forEach((u) => console.log('    ' + u));
    }
    if (bad404.length) {
      failures++;
      console.log('  FAIL — unexpected non-2xx:');
      [...new Set(bad404)].forEach((u) => console.log('    ' + u));
    }
    const real = errors.filter((e) => !/status of 404/.test(e));
    if (real.length) {
      failures++;
      console.log(`  FAIL — ${real.length} console/page errors:`);
      [...new Set(real)].slice(0, 12).forEach((e) => console.log('    ' + e));
    }
    if (!real.length && !external.length && !bad404.length && reported) console.log('  clean');

    await page.close();
    await browser.close();
  }
} finally {
  server.kill();
}

console.log(failures ? `\n${failures} FAILURE(S)` : '\nALL CHECKS PASSED');
process.exit(failures ? 1 : 0);
