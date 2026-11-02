const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const cameraSource = fs.readFileSync(path.join(__dirname, '../public/js/camera_preview.js'), 'utf8');

class FakeTrack extends EventTarget {
  constructor() {
    super();
    this.kind = 'video';
    this.readyState = 'live';
    this.stopCount = 0;
  }

  stop() {
    this.stopCount += 1;
    this.readyState = 'ended';
  }
}

class FakeVideo extends EventTarget {
  constructor({ playError = null, initiallyReady = true } = {}) {
    super();
    this.playError = playError;
    this.videoWidth = initiallyReady ? 640 : 0;
    this.videoHeight = initiallyReady ? 480 : 0;
    this.readyState = initiallyReady ? 2 : 0;
    this.srcObject = null;
  }

  async play() {
    if (this.playError) throw this.playError;
    this.videoWidth = 640;
    this.videoHeight = 480;
    this.readyState = 2;
    this.dispatchEvent(new Event('loadedmetadata'));
    this.dispatchEvent(new Event('canplay'));
  }
}

function createCamera({ getUserMedia, video = new FakeVideo() }) {
  const window = { setTimeout, clearTimeout };
  const context = vm.createContext({
    window,
    navigator: { mediaDevices: { getUserMedia } },
    AbortController,
  });
  vm.runInContext(cameraSource, context);
  const calls = [];
  const container = { classList: { add: (value) => calls.push(['add', value]), remove: (value) => calls.push(['remove', value]) } };
  const placeholder = { setAttribute: (...args) => calls.push(['attribute', ...args]), removeAttribute: (name) => calls.push(['remove-attribute', name]) };
  const status = { set innerHTML(value) { calls.push(['status', value]); }, set textContent(value) { calls.push(['status-text', value]); } };
  const footnote = { set textContent(value) { calls.push(['footnote', value]); } };
  const badge = { hidden: true };
  const button = { textContent: '' };
  return {
    camera: window.BYAKUCamera,
    calls,
    video,
    container,
    placeholder,
    status,
    footnote,
    badge,
    button,
    start: () => window.BYAKUCamera.start(video, container, placeholder, status, footnote, badge, button),
    stop: () => window.BYAKUCamera.stop(video, container, placeholder, status, footnote, badge, button),
  };
}

test('camera opens after permission, waits for video readiness, and activates preview', async () => {
  const track = new FakeTrack();
  let requestedConstraints;
  const harness = createCamera({
    getUserMedia: async (constraints) => {
      requestedConstraints = constraints;
      return { getTracks: () => [track] };
    },
    video: new FakeVideo({ initiallyReady: false }),
  });

  const stream = await harness.start();
  assert.equal(stream, harness.camera.stream);
  assert.equal(harness.video.srcObject, stream);
  assert.equal(harness.video.muted, true);
  assert.equal(harness.video.autoplay, true);
  assert.equal(harness.video.playsInline, true);
  assert.equal(harness.badge.hidden, false);
  assert.ok(harness.calls.some(([action, value]) => action === 'add' && value === 'has-stream'));
  assert.equal(requestedConstraints.audio, false);
  assert.deepEqual(
    { facingMode: requestedConstraints.video.facingMode, width: requestedConstraints.video.width.ideal, height: requestedConstraints.video.height.ideal },
    { facingMode: 'user', width: 640, height: 480 },
  );
});

test('camera denial returns a useful permission error without activating preview', async () => {
  const harness = createCamera({
    getUserMedia: async () => {
      const error = new Error('Permission denied');
      error.name = 'NotAllowedError';
      throw error;
    },
  });

  await assert.rejects(harness.start(), /Camera permission was denied/);
  assert.equal(harness.camera.stream, null);
  assert.equal(harness.badge.hidden, true);
});

test('missing and busy cameras return specific errors', async (t) => {
  for (const [name, expected] of [
    ['NotFoundError', /No camera device was found/],
    ['NotReadableError', /camera is in use/],
  ]) {
    await t.test(name, async () => {
      const harness = createCamera({
        getUserMedia: async () => {
          const error = new Error(name);
          error.name = name;
          throw error;
        },
      });
      await assert.rejects(harness.start(), expected);
    });
  }
});

test('camera playback failure releases the stream and reports the failure', async () => {
  const track = new FakeTrack();
  const harness = createCamera({
    getUserMedia: async () => ({ getTracks: () => [track] }),
    video: new FakeVideo({ playError: new Error('Playback was blocked') }),
  });

  await assert.rejects(harness.start(), /Playback was blocked/);
  assert.equal(track.stopCount, 1);
  assert.equal(harness.video.srcObject, null);
  assert.equal(harness.camera.stream, null);
});

test('stopping releases every media track and clears preview state', async () => {
  const tracks = [new FakeTrack(), new FakeTrack()];
  const stream = { getTracks: () => tracks };
  const harness = createCamera({ getUserMedia: async () => stream });

  await harness.start();
  harness.stop();
  assert.deepEqual(tracks.map((track) => track.stopCount), [1, 1]);
  assert.equal(harness.video.srcObject, null);
  assert.equal(harness.camera.stream, null);
  assert.equal(harness.badge.hidden, true);
  assert.equal(harness.button.textContent, 'Enable camera preview');
});

test('stopping while permission is pending prevents a late stream from remaining active', async () => {
  const track = new FakeTrack();
  let resolveStream;
  const harness = createCamera({
    getUserMedia: () => new Promise((resolve) => { resolveStream = resolve; }),
  });

  const startup = harness.start();
  harness.stop();
  resolveStream({ getTracks: () => [track] });
  assert.equal(await startup, null);
  assert.equal(track.stopCount, 1);
  assert.equal(harness.camera.stream, null);
});
