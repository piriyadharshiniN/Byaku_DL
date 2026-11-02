const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const interfaceSource = fs.readFileSync(path.join(__dirname, '../public/js/assistant_interface.js'), 'utf8');

class FakeElement {
  constructor() {
    this.listeners = new Map();
    this.attributes = new Map();
    this.classList = {
      add() {},
      remove() {},
      toggle() {},
    };
    this.dataset = {};
    this.parentElement = { classList: this.classList };
    this.hidden = false;
    this.textContent = '';
  }

  addEventListener(type, listener) {
    this.listeners.set(type, listener);
  }

  setAttribute(name, value) {
    this.attributes.set(name, value);
  }
}

test('camera startup failure remains specific in the assistant status', async () => {
  const selectors = [
    '#assistant-panel',
    '#state-label',
    '#assistant-detail',
    '#transcript-line',
    '#toast',
    '#camera-video',
    '#camera-view',
    '#camera-overlay',
    '#camera-placeholder',
    '#camera-status',
    '#camera-footnote',
    '#preview-badge',
    '#camera-panel',
    '#assistant-greeting-fallback',
    '#start-listening',
    '#stop-assistant',
    '#text-query-form',
    '#text-query-input',
    '#assistant-action-label',
  ];
  const elements = new Map(selectors.map((selector) => [selector, new FakeElement()]));
  let recognition;
  let cameraStartCalls = 0;
  const window = {
    setTimeout,
    clearTimeout,
    addEventListener() {},
    SpeechRecognition: class {
      constructor() { recognition = this; }
      start() { this.onstart?.(); }
      stop() {}
    },
    BYAKUConnection: class extends EventTarget {},
    BYAKUCamera: {
      start: async () => {
        cameraStartCalls += 1;
        throw new Error('The camera is in use by another app or cannot be opened.');
      },
      get stream() { return null; },
      stop() {},
    },
    BYAKUSpeechController: class {
      constructor({ onState }) { this.onState = onState; }
      speak(text) {
        this.onState('started', text);
        this.onState('ended', text);
        return true;
      }
      cancel() {}
    },
    BYAKULiveNarrationScheduler: class { stop() {} },
  };
  const context = vm.createContext({
    window,
    document: { querySelector: (selector) => elements.get(selector) || null },
    performance,
    setTimeout,
    clearTimeout,
  });
  vm.runInContext(interfaceSource, context);

  elements.get('#start-listening').listeners.get('click')();
  assert.equal(cameraStartCalls, 0);
  recognition.onresult({ results: [[{ transcript: 'START' }]] });
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(cameraStartCalls, 1);
  assert.equal(elements.get('#camera-status').textContent, 'Camera unavailable');
  assert.equal(elements.get('#assistant-detail').textContent, 'The camera is in use by another app or cannot be opened.');
  assert.equal(elements.get('#toast').textContent, 'The camera is in use by another app or cannot be opened.');
});

test('spoken START activates scanning and voice questions are sent while the camera stays active', async () => {
  const elements = new Map(
    [
      '#assistant-panel', '#state-label', '#assistant-detail', '#transcript-line', '#toast',
      '#camera-video', '#camera-view', '#camera-overlay', '#camera-placeholder',
      '#camera-status', '#camera-footnote', '#preview-badge', '#camera-panel',
      '#assistant-greeting-fallback',
      '#start-listening', '#stop-assistant', '#text-query-form', '#text-query-input',
      '#assistant-action-label',
    ].map((selector) => [selector, new FakeElement()]),
  );
  let recognition;
  let cameraStartCalls = 0;
  const sentQueries = [];
  const window = {
    setTimeout,
    clearTimeout,
    addEventListener() {},
    SpeechRecognition: class {
      constructor() { recognition = this; }
      start() { this.onstart?.(); }
      stop() {}
    },
    BYAKUConnection: class extends EventTarget {
      async resume() {}
      close() {}
    },
    BYAKUCamera: {
      start: async () => { cameraStartCalls += 1; return {}; },
      get stream() { return null; },
      stop() {},
    },
    BYAKUFrameProcessor: class extends EventTarget {
      start() {}
      stop() {}
      sendQuery(query) { sentQueries.push(query); return true; }
      static drawOverlay() {}
    },
    BYAKUSpeechController: class {
      constructor({ onState }) { this.onState = onState; }
      speak(text) {
        this.onState('started', text);
        this.onState('ended', text);
        return true;
      }
      cancel() {}
    },
    BYAKULiveNarrationScheduler: class {
      start() {}
      stop() {}
      update() {}
    },
  };
  const context = vm.createContext({
    window,
    document: { querySelector: (selector) => elements.get(selector) || null },
    performance,
    setTimeout,
    clearTimeout,
  });
  vm.runInContext(interfaceSource, context);

  elements.get('#start-listening').listeners.get('click')();
  assert.equal(cameraStartCalls, 0);
  recognition.onresult({ results: [[{ transcript: 'START' }]] });
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(cameraStartCalls, 1);
  assert.equal(elements.get('#state-label').textContent, 'Listening');
  recognition.onresult({ results: [[{ transcript: 'Where is the bottle?' }]] });

  assert.deepEqual(sentQueries, ['Where is the bottle?']);
  assert.equal(elements.get('#state-label').textContent, 'Processing');
});

test('assistant speaks one greeting on entry and leaves camera and microphone off', () => {
  const elements = new Map(
    [
      '#assistant-panel', '#state-label', '#assistant-detail', '#transcript-line', '#toast',
      '#camera-video', '#camera-view', '#camera-overlay', '#camera-placeholder',
      '#camera-status', '#camera-footnote', '#preview-badge', '#camera-panel',
      '#start-listening', '#stop-assistant', '#text-query-form', '#text-query-input',
      '#assistant-action-label', '#assistant-greeting-fallback',
    ].map((selector) => [selector, new FakeElement()]),
  );
  const greetings = [];
  let cameraStarts = 0;
  let microphoneStarts = 0;
  const window = {
    setTimeout,
    clearTimeout,
    addEventListener() {},
    SpeechRecognition: class {
      start() { microphoneStarts += 1; this.onstart?.(); }
      stop() {}
    },
    BYAKUConnection: class extends EventTarget {},
    BYAKUCamera: {
      start: async () => { cameraStarts += 1; return {}; },
      get stream() { return null; },
      stop() {},
    },
    BYAKUSpeechController: class {
      constructor({ onState }) { this.onState = onState; }
      speak(text) {
        greetings.push(text);
        this.onState('started', text);
        this.onState('ended', text);
        return true;
      }
      cancel() {}
    },
    BYAKULiveNarrationScheduler: class { stop() {} },
  };
  const context = vm.createContext({
    window,
    document: { querySelector: (selector) => elements.get(selector) || null },
    performance,
    setTimeout,
    clearTimeout,
  });
  vm.runInContext(interfaceSource, context);

  assert.deepEqual(greetings, ['Your assistant is ready.']);
  assert.equal(cameraStarts, 0);
  assert.equal(microphoneStarts, 0);
  assert.equal(elements.get('#state-label').textContent, 'Idle');
  assert.equal(elements.get('#assistant-greeting-fallback').hidden, true);
});
