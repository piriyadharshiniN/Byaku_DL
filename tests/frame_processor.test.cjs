if (typeof CustomEvent === 'undefined') {
  global.CustomEvent = class CustomEvent extends Event {
    constructor(type, options = {}) {
      super(type, options);
      this.detail = options.detail;
    }
  };
}
const test = require('node:test');
const assert = require('node:assert/strict');

const pendingBlobs = [];
const intervals = new Map();
let nextIntervalId = 0;
global.window = {
  devicePixelRatio: 1,
  setInterval: (callback, delay) => {
    const id = ++nextIntervalId;
    intervals.set(id, { callback, delay });
    return id;
  },
  clearInterval: (id) => intervals.delete(id),
  setTimeout,
  clearTimeout,
};
global.document = {
  createElement: () => ({
    width: 0,
    height: 0,
    getContext: () => ({ drawImage() {} }),
    toBlob: (callback) => pendingBlobs.push(callback),
  }),
};
global.HTMLMediaElement = { HAVE_CURRENT_DATA: 2 };
global.FileReader = class {
  readAsDataURL() {
    this.result = 'data:image/jpeg;base64,ZmFrZQ==';
    this.onload();
  }
};

require('../public/js/browser_websocket_client.js');

class FakeConnection extends EventTarget {
  constructor() {
    super();
    this.sent = [];
  }

  send(message) {
    this.sent.push(message);
    return true;
  }

  setState() {}

  emit(message) {
    this.dispatchEvent(new CustomEvent('update', { detail: message }));
  }
}

function makeProcessor() {
  const connection = new FakeConnection();
  const video = { readyState: 2, videoWidth: 320, videoHeight: 240 };
  const processor = new window.BYAKUFrameProcessor(connection, video);
  processor.start();
  return { connection, processor };
}

async function settleCapture() {
  const callback = pendingBlobs.shift();
  assert.ok(callback, 'expected an encoded frame to be waiting');
  callback({ size: 1 });
  await new Promise((resolve) => setImmediate(resolve));
}

test('frame processor allows one in-flight frame and never queues stale captures', async () => {
  const { connection, processor } = makeProcessor();

  const firstCapture = processor.capture();
  await processor.capture();
  assert.equal(pendingBlobs.length, 1);
  await settleCapture();
  await firstCapture;
  assert.equal(connection.sent.filter((item) => item.type === 'frame').length, 1);
  assert.equal(processor.busy, true);

  await processor.capture();
  assert.equal(pendingBlobs.length, 0);
  assert.equal(connection.sent.filter((item) => item.type === 'frame').length, 1);
  processor.stop();
});

test('scene scanning starts immediately and schedules non-overlapping captures every five seconds', async () => {
  const { connection, processor } = makeProcessor();
  const interval = intervals.get(processor.timer);

  assert.equal(interval.delay, 5000);
  assert.equal(pendingBlobs.length, 1);
  interval.callback();
  assert.equal(pendingBlobs.length, 1);

  await settleCapture();
  const firstFrame = connection.sent.find((item) => item.type === 'frame');
  interval.callback();
  assert.equal(pendingBlobs.length, 0);
  connection.emit({ type: 'frame_result', frame_id: firstFrame.frame_id });

  interval.callback();
  assert.equal(pendingBlobs.length, 1);
  await settleCapture();
  assert.equal(connection.sent.filter((item) => item.type === 'frame').length, 2);
  processor.stop();
});

test('a question waits for a frame captured after the question was submitted', async () => {
  const { connection, processor } = makeProcessor();

  const captureBeforeQuestion = processor.capture();
  assert.equal(processor.sendQuery('Where is the bottle?'), true);
  await settleCapture();
  await captureBeforeQuestion;
  const firstFrame = connection.sent.find((item) => item.type === 'frame');
  connection.emit({ type: 'frame_result', frame_id: firstFrame.frame_id });
  assert.equal(connection.sent.some((item) => item.type === 'query'), false);

  const freshCapture = processor.capture();
  await settleCapture();
  await freshCapture;
  const secondFrame = connection.sent.filter((item) => item.type === 'frame')[1];
  connection.emit({ type: 'frame_result', frame_id: secondFrame.frame_id });
  assert.deepEqual(
    connection.sent.filter((item) => item.type === 'query').map((item) => item.text),
    ['Where is the bottle?'],
  );
  processor.stop();
});

test('a cold-start watchdog warns without releasing the only in-flight frame', async () => {
  const { connection, processor } = makeProcessor();
  const errors = [];
  processor.addEventListener('result', (event) => errors.push(event.detail));
  let watchdogCallback;
  const setTimeoutOriginal = window.setTimeout;
  const clearTimeoutOriginal = window.clearTimeout;
  window.setTimeout = (callback, delay) => {
    if (delay === processor.frameTimeoutMs) {
      watchdogCallback = callback;
      return 123;
    }
    return setTimeoutOriginal(callback, delay);
  };
  window.clearTimeout = (timer) => {
    if (timer !== 123) clearTimeoutOriginal(timer);
  };

  const capture = processor.capture();
  await settleCapture();
  await capture;
  const frameId = processor.pendingFrameId;
  assert.equal(processor.frameTimeoutMs, 90000);
  assert.equal(typeof watchdogCallback, 'function');
  watchdogCallback();

  assert.equal(processor.busy, true);
  assert.equal(processor.pendingFrameId, frameId);
  assert.equal(errors[0].code, 'frame_slow');
  assert.equal(connection.sent.filter((item) => item.type === 'frame').length, 1);
  processor.stop();
  window.setTimeout = setTimeoutOriginal;
  window.clearTimeout = clearTimeoutOriginal;
});

test('frame result retains the exact encoded camera image for human review', async () => {
  const { connection, processor } = makeProcessor();
  const results = [];
  processor.addEventListener('result', (event) => results.push(event.detail));

  const capture = processor.capture();
  await settleCapture();
  await capture;
  const frame = connection.sent.find((item) => item.type === 'frame');
  assert.equal(frame.data, 'ZmFrZQ==');
  connection.emit({ type: 'frame_result', frame_id: frame.frame_id });
  assert.equal(results[0].client_frame_data_url, 'data:image/jpeg;base64,ZmFrZQ==');
  processor.stop();
});

test('stopping during frame encoding prevents a late send', async () => {
  const { connection, processor } = makeProcessor();
  const capture = processor.capture();
  processor.stop();
  await settleCapture();
  await capture;

  assert.equal(connection.sent.filter((item) => item.type === 'frame').length, 0);
});

test('bounding boxes show persistent ID, direction and categorical depth in matching colors', () => {
  const labels = [];
  const strokeColors = [];
  const context = {
    setTransform() {},
    clearRect() {},
    strokeRect() { strokeColors.push(this.strokeStyle); },
    measureText: (text) => ({ width: text.length * 7 }),
    fillRect() {},
    fillText: (text) => labels.push(text),
  };
  const canvas = {
    width: 0,
    height: 0,
    clientWidth: 300,
    clientHeight: 300,
    getContext: () => context,
  };

  window.BYAKUFrameProcessor.drawOverlay(
    canvas,
    { width: 100, height: 100 },
    [
      { class_name: 'person', track_id: 3, horizontal_position: 'right', depth_category: 'NEAR', bounding_box: [70, 10, 90, 40] },
      { class_name: 'chair', track_id: 4, horizontal_position: 'center', depth_category: 'MID-RANGE', bounding_box: [40, 10, 60, 40] },
      { class_name: 'bottle', track_id: 5, horizontal_position: 'left', depth_category: 'FAR', bounding_box: [10, 10, 30, 40] },
    ],
  );

  assert.deepEqual(labels, [
    'person | ID 3 | RIGHT | NEAR',
    'chair | ID 4 | CENTER | MID-RANGE',
    'bottle | ID 5 | LEFT | FAR',
  ]);
  assert.deepEqual(strokeColors, ['#ef4444', '#facc15', '#3b82f6']);
});
