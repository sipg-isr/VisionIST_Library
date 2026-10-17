'use strict';
/* The phone page. No framework, no inline script, no third-party request:
   everything it loads comes from this box (see the Content-Security-Policy).

   Flow: read the pairing token from the URL fragment -> load the session
   config -> show the camera (or the phone's own camera app when a live
   preview is not allowed) -> capture, label, send -> long-poll the results
   feed and draw whatever the program pushes back. */
(() => {
  const $ = (id) => document.getElementById(id);
  const el = (tag, props = {}, ...kids) => {
    const node = Object.assign(document.createElement(tag), props);
    for (const k of kids) node.append(k);
    return node;
  };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const store = (() => {                     // storage can throw (private mode)
    const get = (area, k) => { try { return window[area].getItem(k); } catch (e) { return null; } };
    const set = (area, k, v) => { try { window[area].setItem(k, v); } catch (e) { /* ignore */ } };
    return { get, set };
  })();

  const S = {
    token: null, cfg: null, cv: 0, limits: {}, facing: 'environment',
    stream: null, shots: [], labels: new Set(), feedAfter: 0, sending: false,
    sensorsOn: false, latest: {}, cards: new Map(), ended: false, gen: 0,
    streaming: false, streamGen: 0, sent: 0, liveSeq: 0, liveUrl: null,
  };

  // ---- identity of this phone (a random id kept in the browser) ----------
  let deviceId = store.get('localStorage', 'phone_device');
  if (!deviceId) {
    deviceId = Array.from(crypto.getRandomValues(new Uint8Array(8)), (b) => b.toString(16).padStart(2, '0')).join('');
    store.set('localStorage', 'phone_device', deviceId);
  }
  const deviceName = /iPhone|iPad/.test(navigator.userAgent) ? 'iPhone'
    : /Android/.test(navigator.userAgent) ? 'Android' : 'browser';

  // ---- API ----------------------------------------------------------------
  class AuthError extends Error {}
  async function api(path, opts = {}) {
    const res = await fetch('./api/' + path, {
      ...opts, cache: 'no-store',
      headers: { ...(opts.headers || {}), Authorization: 'Bearer ' + S.token },
    });
    if (res.status === 401 || res.status === 429) {
      const body = await res.json().catch(() => ({}));
      throw Object.assign(new AuthError(body.error || 'not authorised'), { status: res.status });
    }
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.error || ('HTTP ' + res.status));
    }
    return res;
  }
  const apiJson = async (path, opts) => (await api(path, opts)).json();
  const idq = () => 'device=' + encodeURIComponent(deviceId) + '&name=' + encodeURIComponent(deviceName);

  // ---- status line --------------------------------------------------------
  function say(text, kind = '') { const m = $('msg'); m.textContent = text; m.dataset.kind = kind; }
  function pill(text, state) { const p = $('pill'); p.textContent = text; p.dataset.state = state; }

  // ---- pairing ------------------------------------------------------------
  function readToken() {
    const m = /[#&]t=([A-Za-z0-9_-]+)/.exec(location.hash);
    if (m) {
      store.set('sessionStorage', 'phone_token', m[1]);
      history.replaceState(null, '', location.pathname + location.search);   // token off the address bar
    }
    return store.get('sessionStorage', 'phone_token');
  }

  function showPairing(message) {
    $('capture').hidden = true; $('results').hidden = true; $('pair').hidden = false;
    if (message) pill(message, 'bad');
    setStream(false);
    stopCamera();
  }

  $('pairForm').addEventListener('submit', (e) => {
    e.preventDefault();
    const t = $('pairToken').value.trim();
    if (!/^[A-Za-z0-9_-]{8,}$/.test(t)) return;
    store.set('sessionStorage', 'phone_token', t);
    $('pairToken').value = '';
    start();
  });

  // ---- config -> controls -------------------------------------------------
  async function loadConfig() {
    const r = await apiJson('config?' + idq());
    S.cfg = r.config; S.cv = r.config_version; S.limits = r.limits;
    renderConfig();
  }

  function renderConfig() {
    const c = S.cfg;
    $('title').textContent = c.title;
    document.title = c.title;
    for (const k of Array.from(S.labels)) if (!c.labels.includes(k) && !c.custom_labels) S.labels.delete(k);
    const box = $('labels'); box.replaceChildren();
    const names = Array.from(new Set([...c.labels, ...S.labels]));
    for (const name of names) box.append(chip(name));
    $('labelsHead').hidden = !(names.length || c.custom_labels);
    $('customForm').hidden = !c.custom_labels;
    $('note').hidden = !c.note;
    $('sensorBtn').hidden = !(c.sensors.length) || S.sensorsOn;
    if (S.shots.length > shotLimit()) { S.shots.length = shotLimit(); renderShots(); }
    updateSend();
  }

  function chip(name) {
    const b = el('button', { type: 'button', className: 'chip', textContent: name });
    b.setAttribute('aria-pressed', S.labels.has(name) ? 'true' : 'false');
    b.addEventListener('click', () => {
      if (S.labels.has(name)) S.labels.delete(name); else S.labels.add(name);
      b.setAttribute('aria-pressed', S.labels.has(name) ? 'true' : 'false');
      updateSend();
    });
    return b;
  }

  $('customForm').addEventListener('submit', (e) => {
    e.preventDefault();
    const v = $('customLabel').value.replace(/\s+/g, ' ').trim().slice(0, 64);
    if (!v) return;
    S.labels.add(v); $('customLabel').value = '';
    renderConfig();
  });

  // ---- camera -------------------------------------------------------------
  function stopCamera() {
    if (S.stream) for (const t of S.stream.getTracks()) t.stop();
    S.stream = null;
    $('cam').srcObject = null;
  }

  function noCamera(why) {
    setStream(false);
    stopCamera();
    $('nocamText').textContent = why;
    $('nocam').hidden = false; $('shutter').hidden = true; $('flip').hidden = true;
    $('stream').hidden = true;                       // a stream needs the live camera
  }

  async function startCamera() {
    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      noCamera('A live preview needs https. Use the "Camera app" button: it opens your phone\'s own camera.');
      return;
    }
    stopCamera();
    try {
      S.stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: S.facing }, width: { ideal: 1920 }, height: { ideal: 1080 } },
      });
      const v = $('cam');
      v.srcObject = S.stream;
      await v.play().catch(() => {});
      $('nocam').hidden = true; $('shutter').hidden = false; $('stream').hidden = false;
      const devices = await navigator.mediaDevices.enumerateDevices().catch(() => []);
      $('flip').hidden = devices.filter((d) => d.kind === 'videoinput').length < 2;
      if (navigator.wakeLock) navigator.wakeLock.request('screen').catch(() => {});
    } catch (err) {
      noCamera(err && err.name === 'NotAllowedError'
        ? 'Camera permission was denied. Allow it in the browser settings, or use the "Camera app" button.'
        : 'No usable camera: ' + (err && err.message ? err.message : err) + '. Use the "Camera app" button.');
    }
  }

  $('flip').addEventListener('click', () => {
    S.facing = S.facing === 'environment' ? 'user' : 'environment';
    startCamera();
  });
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && S.cfg && !S.ended && !S.stream && $('shutter').hidden === false) startCamera();
  });

  // ---- capture ------------------------------------------------------------
  const shotLimit = () => (S.cfg && S.cfg.mode === 'burst' ? S.cfg.burst_max : 1);

  const scaled = (w, h, side) => {
    const k = side > 0 ? Math.min(1, side / Math.max(w, h)) : 1;
    return [Math.max(1, Math.round(w * k)), Math.max(1, Math.round(h * k))];
  };

  async function toJpeg(source, w, h, side = S.cfg.max_side, quality = S.cfg.jpeg_quality) {
    const c = el('canvas');
    [c.width, c.height] = scaled(w, h, side);
    c.getContext('2d').drawImage(source, 0, 0, c.width, c.height);
    return new Promise((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error('could not encode the photo'))), 'image/jpeg', quality));
  }

  function addShot(blob) {
    const limit = shotLimit();
    if (limit === 1) { for (const s of S.shots) URL.revokeObjectURL(s.url); S.shots = []; }
    if (S.shots.length >= limit) { say('At most ' + limit + ' photos per submission.', 'bad'); return false; }
    S.shots.push({ blob, url: URL.createObjectURL(blob), sensors: snapshot() });
    renderShots(); updateSend(); say('');
    if (navigator.vibrate) navigator.vibrate(15);
    return true;
  }

  function renderShots() {
    const box = $('shots'); box.replaceChildren();
    S.shots.forEach((s, i) => {
      const rm = el('button', { type: 'button', textContent: '×' });
      rm.setAttribute('aria-label', 'Remove photo ' + (i + 1));
      rm.addEventListener('click', () => {
        URL.revokeObjectURL(s.url); S.shots.splice(i, 1); renderShots(); updateSend();
      });
      box.append(el('div', { className: 'shot' }, el('img', { src: s.url, alt: 'Photo ' + (i + 1) }), rm));
    });
  }

  $('shutter').addEventListener('click', async () => {
    const v = $('cam');
    if (!v.videoWidth) return;
    try { addShot(await toJpeg(v, v.videoWidth, v.videoHeight)); } catch (e) { say(e.message, 'bad'); }
  });

  $('file').addEventListener('change', async (e) => {
    const files = Array.from(e.target.files || []);
    e.target.value = '';
    for (const f of files) {
      try {
        let blob;
        try {
          const bmp = await createImageBitmap(f, { imageOrientation: 'from-image' });
          blob = await toJpeg(bmp, bmp.width, bmp.height);
          bmp.close && bmp.close();
        } catch (err) {
          if (!/^image\/(jpeg|png|webp)$/.test(f.type)) throw new Error('this browser could not read that photo (' + (f.type || 'unknown type') + ')');
          blob = f;                         // already a format the box takes
        }
        if (!addShot(blob)) break;
      } catch (err) { say(err.message, 'bad'); }
    }
  });

  // ---- sensors ------------------------------------------------------------
  const num = (x) => (typeof x === 'number' && Number.isFinite(x) ? x : undefined);
  const pack = (o) => { const r = {}; for (const [k, v] of Object.entries(o)) if (num(v) !== undefined) r[k] = v; return r; };

  function snapshot() {
    if (!S.sensorsOn) return null;
    const out = {};
    for (const g of S.cfg.sensors) if (S.latest[g]) out[g] = { ...S.latest[g] };
    return Object.keys(out).length ? out : null;
  }

  $('sensorBtn').addEventListener('click', async () => {
    const want = S.cfg.sensors;
    try {
      for (const Ev of [window.DeviceMotionEvent, window.DeviceOrientationEvent]) {
        if (Ev && typeof Ev.requestPermission === 'function') {          // iOS: needs this tap
          if ((await Ev.requestPermission()) !== 'granted') throw new Error('motion access was refused');
        }
      }
      // Desktop browsers fire these events with null angles: no real reading, so no entry.
      if (want.includes('orientation')) window.addEventListener('deviceorientation', (e) => {
        if (num(e.alpha) === undefined && num(e.beta) === undefined && num(e.gamma) === undefined) return;
        S.latest.orientation = pack({ alpha: e.alpha, beta: e.beta, gamma: e.gamma, absolute: e.absolute ? 1 : 0, t: Date.now() / 1000 });
      });
      if (want.includes('motion')) window.addEventListener('devicemotion', (e) => {
        const a = e.accelerationIncludingGravity || {}, r = e.rotationRate || {};
        if ([a.x, a.y, a.z, r.alpha, r.beta, r.gamma].every((v) => num(v) === undefined)) return;
        S.latest.motion = pack({ ax: a.x, ay: a.y, az: a.z, rx: r.alpha, ry: r.beta, rz: r.gamma, t: Date.now() / 1000 });
      });
      if (want.includes('lidar')) {
        // A web page cannot read LiDAR: iOS Safari has no WebXR and ARKit depth is native-only.
        // Say so in the data (available: 0) instead of leaving the program to guess; a native
        // client posts real depth to /api/frame (see README).
        let webxr = false;
        try { webxr = !!(navigator.xr && await navigator.xr.isSessionSupported('immersive-ar')); } catch (e) { /* none */ }
        S.latest.lidar = { available: 0, webxr: webxr ? 1 : 0 };
      }
      if (want.includes('location')) {
        if (!navigator.geolocation) throw new Error('no location on this browser');
        navigator.geolocation.watchPosition((p) => {
          const c = p.coords;
          S.latest.location = pack({ lat: c.latitude, lon: c.longitude, alt: c.altitude, acc: c.accuracy, alt_acc: c.altitudeAccuracy, heading: c.heading, speed: c.speed, t: p.timestamp / 1000 });
        }, (err) => say('Location: ' + err.message, 'bad'), { enableHighAccuracy: true, maximumAge: 5000 });
      }
      S.sensorsOn = true; $('sensorBtn').hidden = true;
      say(want.includes('lidar') ? 'Sensors on. LiDAR cannot be read from a web page on this device.' : 'Sensors on: each photo carries them.', 'ok');
    } catch (err) { say(err.message, 'bad'); }
  });

  // ---- send ---------------------------------------------------------------
  function updateSend() {
    const need = S.cfg && S.cfg.require_label && S.labels.size === 0;
    $('send').disabled = S.sending || S.shots.length === 0 || need;
  }

  $('send').addEventListener('click', async () => {
    if (S.sending || !S.shots.length) return;
    S.sending = true; updateSend(); say('Sending…');
    try {
      const fd = new FormData();
      S.shots.forEach((s, i) => fd.append('image', s.blob, 'shot' + i + '.jpg'));
      fd.append('meta', JSON.stringify({
        labels: Array.from(S.labels), note: S.cfg.note ? $('note').value : '',
        sensors: S.shots.map((s) => s.sensors),
        device: { id: deviceId, name: deviceName }, client_t: Date.now() / 1000,
      }));
      const r = await apiJson('submit', { method: 'POST', body: fd });
      say('Sent as #' + r.seq + '.', 'ok');
      for (const s of S.shots) URL.revokeObjectURL(s.url);
      S.shots = []; S.labels.clear(); $('note').value = '';
      renderShots(); renderConfig();
    } catch (err) {
      if (err instanceof AuthError) endSession(err.message); else say(err.message, 'bad');
    } finally { S.sending = false; updateSend(); }
  });

  // ---- stream -------------------------------------------------------------
  // Frames go out one at a time (the next is captured only after the box has the
  // previous one), so a slow link lowers the rate instead of queueing frames.
  async function streamLoop(gen) {
    const alive = () => S.streaming && gen === S.streamGen && !S.ended;
    let failures = 0, shown = 0;
    while (alive()) {
      const t0 = performance.now();
      try {
        const v = $('cam');
        if (v.videoWidth && v.readyState >= 2 && document.visibilityState === 'visible') {
          const [w, h] = scaled(v.videoWidth, v.videoHeight, S.cfg.stream_max_side);
          const blob = await toJpeg(v, v.videoWidth, v.videoHeight, S.cfg.stream_max_side, S.cfg.stream_quality);
          if (!alive()) return;
          const fd = new FormData();
          fd.append('image', blob, 'frame.jpg');
          fd.append('meta', JSON.stringify({
            labels: Array.from(S.labels), note: S.cfg.note ? $('note').value : '',
            sensors: [snapshot()], size: [w, h],
            device: { id: deviceId, name: deviceName }, client_t: Date.now() / 1000,
          }));
          await api('frame', { method: 'POST', body: fd });
          failures = 0; S.sent += 1;
          if (performance.now() - shown > 1000) { shown = performance.now(); say('Streaming: ' + S.sent + ' frames sent.', 'ok'); }
        }
      } catch (err) {
        if (!alive()) return;
        if (err instanceof AuthError) { endSession(err.message); return; }
        failures += 1; say('Stream: ' + err.message, 'bad');
        await sleep(Math.min(5000, 500 * failures));
      }
      const wait = 1000 / (S.cfg.stream_fps || 5) - (performance.now() - t0);
      if (wait > 0) await sleep(wait);
    }
  }

  function setStream(on) {
    on = !!on && !!S.stream && !S.ended;
    if (on === S.streaming) return;
    S.streaming = on;
    const gen = ++S.streamGen;                     // a stopped loop notices and exits
    $('stream').setAttribute('aria-pressed', on ? 'true' : 'false');
    if (on) { S.sent = 0; say('Streaming\u2026', 'ok'); streamLoop(gen); } else say('Stream stopped.');
  }

  $('stream').addEventListener('click', () => setStream(!S.streaming));

  // ---- results ------------------------------------------------------------
  async function mediaUrl(id, kind) {
    const blob = await (await api('media/' + id + '/' + kind)).blob();
    return URL.createObjectURL(blob);
  }

  function renderCard(c) {
    const when = new Date(c.t * 1000).toLocaleTimeString();
    const node = el('article', { className: 'card' },
      el('header', {}, el('span', { textContent: c.ref_seq ? 'Re #' + c.ref_seq : 'Result' }), el('time', { textContent: when })));
    if (c.title) node.append(el('h3', { textContent: c.title }));
    if (c.text) node.append(el('pre', { textContent: c.text }));
    const urls = [];
    if (c.image) {
      const img = el('img', { alt: c.title || 'result image' });
      node.append(img);
      mediaUrl(c.id, 'image').then((u) => { urls.push(u); img.src = u; }).catch(() => { img.replaceWith(el('p', { className: 'muted', textContent: 'image not available' })); });
    }
    if (c.glb) {
      const holder = el('div', { className: 'viewer' }, el('span', { className: 'muted', textContent: 'Loading 3D…' }));
      node.append(holder);
      show3d(c, holder, urls);
    }
    return { node, urls, id: c.id };
  }

  async function show3d(c, holder, urls) {
    let url;
    try { url = await mediaUrl(c.id, 'glb'); urls.push(url); } catch (e) { holder.textContent = '3D model not available'; return; }
    try {
      const mod = await import(new URL('assets/vendor/viewer.js', document.baseURI).href);
      holder.replaceChildren();
      mod.mount(holder, url);
    } catch (e) {
      holder.replaceChildren(el('span', {}, 'This build has no 3D viewer. ',
        el('a', { href: url, download: 'result-' + c.id + '.glb', textContent: 'Download the model' })));
    }
  }

  // The live panel: ONE fixed place that always shows the newest result, replaced
  // whole (text, title and image together) so a stream of results never moves the page.
  function clearLive() {
    S.liveSeq += 1;
    if (S.liveUrl) URL.revokeObjectURL(S.liveUrl);
    S.liveUrl = null;
    $('liveImg').removeAttribute('src'); $('liveImgBox').hidden = true;
    $('liveRef').textContent = 'Live result'; $('liveTime').textContent = '';
    $('liveTitle').hidden = true;
    $('liveText').textContent = 'Waiting for the program\u2026';
  }

  async function showLive(c) {
    const id = ++S.liveSeq;                        // a newer result retires an older, slower one
    let url = null;
    if (c.image) { try { url = await mediaUrl(c.id, 'image'); } catch (e) { url = null; } }
    if (id !== S.liveSeq) { if (url) URL.revokeObjectURL(url); return; }
    $('liveRef').textContent = c.ref_seq ? 'Re #' + c.ref_seq : 'Live result';
    $('liveTime').textContent = new Date(c.t * 1000).toLocaleTimeString();
    const h = $('liveTitle'); h.textContent = c.title || ''; h.hidden = !c.title;
    $('liveText').textContent = c.text || (c.glb && !url ? '3D model: open it under Results.' : '');
    const old = S.liveUrl;
    if (url) $('liveImg').src = url; else $('liveImg').removeAttribute('src');
    $('liveImgBox').hidden = !url;
    S.liveUrl = url;
    if (old) URL.revokeObjectURL(old);
  }

  function addCards(cards) {
    showLive(cards[cards.length - 1]);
    if (S.streaming) { $('empty').hidden = true; return; }   // while streaming only the live panel updates
    for (const c of cards) {
      if (S.cards.has(c.id)) continue;
      const r = renderCard(c);
      S.cards.set(c.id, r);
      $('cards').prepend(r.node);
    }
    while (S.cards.size > 20) {                    // keep the page light
      const [id, r] = S.cards.entries().next().value;
      r.node.remove(); for (const u of r.urls) URL.revokeObjectURL(u);
      S.cards.delete(id);
    }
    $('empty').hidden = S.cards.size > 0;
  }

  function programStatus(age) {
    if (age === null || age === undefined) pill('waiting for the program', 'warn');
    else if (age < 30) pill('program connected', 'ok');
    else pill('program idle ' + Math.round(age) + ' s', 'warn');
  }

  function endSession(message) {
    S.ended = true;
    setStream(false);
    stopCamera();
    store.set('sessionStorage', 'phone_token', '');
    showPairing(null);
    pill(message || 'session ended', 'bad');
  }

  async function feedLoop(gen) {
    let failures = 0;
    while (!S.ended && gen === S.gen) {
      try {
        const r = await apiJson('feed?after=' + S.feedAfter + '&wait=15&cv=' + S.cv + '&' + idq());
        if (gen !== S.gen) return;                 // re-paired meanwhile
        failures = 0;
        S.feedAfter = r.last;
        programStatus(r.program_age_s);
        if (r.cards.length) addCards(r.cards);
        if (r.config_version !== S.cv) await loadConfig();
      } catch (err) {
        if (gen !== S.gen) return;
        if (err instanceof AuthError) { endSession(err.status === 429 ? 'too many attempts, wait a few minutes' : 'session ended or token invalid'); return; }
        failures += 1; pill('connection lost', 'bad');
        await sleep(Math.min(10000, 1000 * failures));
      }
    }
  }

  // ---- start --------------------------------------------------------------
  function resetUi() {
    for (const s of S.shots) URL.revokeObjectURL(s.url);
    for (const r of S.cards.values()) { r.node.remove(); for (const u of r.urls) URL.revokeObjectURL(u); }
    S.shots = []; S.cards.clear(); S.labels.clear(); S.feedAfter = 0; S.cfg = null;
    clearLive();
    renderShots(); $('empty').hidden = false; $('note').value = ''; say('');
  }

  async function start() {
    const gen = ++S.gen;                           // a newer start() retires an older feed loop
    const previous = S.token;
    S.token = readToken();
    S.ended = false;
    if (!S.token) { showPairing('not paired'); return; }
    if (previous && previous !== S.token) resetUi();
    try {
      await loadConfig();
    } catch (err) {
      if (err instanceof AuthError) { S.token = null; showPairing(err.status === 429 ? 'too many attempts, wait a few minutes' : 'token not accepted'); return; }
      showPairing('cannot reach the box'); return;
    }
    $('pair').hidden = true; $('capture').hidden = false; $('results').hidden = false;
    if (gen !== S.gen) return;
    pill('connected', 'ok');
    await startCamera();
    feedLoop(gen);
  }

  // A pairing link opened in a tab that already shows this page only changes
  // the #fragment: no reload, so pick the new token up here.
  window.addEventListener('hashchange', () => { if (/[#&]t=/.test(location.hash)) start(); });

  start();
})();
