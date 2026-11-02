const test = require('node:test');
const assert = require('node:assert/strict');

global.window = {
  setTimeout,
  clearTimeout,
};

const { BYAKUSpeechController, BYAKULiveNarrationScheduler } = require('../public/js/assistant_speech.js');

class FakeUtterance {
  constructor(text) {
    this.text = text;
    this.voice = null;
  }
}

class FakeSynthesis {
  constructor() {
    this.paused = false;
    this.voices = [
      { lang: 'en-GB', name: 'UK voice' },
      { lang: 'en-US', name: 'US voice' },
    ];
    this.utterances = [];
    this.cancelCount = 0;
    this.resumeCount = 0;
  }

  getVoices() { return this.voices; }

  speak(utterance) {
    this.utterances.push(utterance);
    this.speaking = true;
    utterance.onstart?.();
  }

  cancel() {
    this.cancelCount += 1;
    this.speaking = false;
  }

  resume() {
    this.resumeCount += 1;
    this.paused = false;
  }
}

function createFakeClock() {
  let now = 0;
  let id = 0;
  const timers = new Map();
  return {
    now: () => now,
    setTimeout(fn, delay) {
      const timerId = ++id;
      timers.set(timerId, { fn, due: now + delay });
      return timerId;
    },
    clearTimeout(timerId) { timers.delete(timerId); },
    advance(ms) {
      const target = now + ms;
      while (true) {
        const next = [...timers.entries()]
          .filter(([, timer]) => timer.due <= target)
          .sort((left, right) => left[1].due - right[1].due)[0];
        if (!next) break;
        const [timerId, timer] = next;
        timers.delete(timerId);
        now = timer.due;
        timer.fn();
      }
      now = target;
    },
    get pendingCount() { return timers.size; },
  };
}

test('speech waits for actual start, chooses an available English voice, and surfaces errors', async () => {
  const synthesis = new FakeSynthesis();
  const states = [];
  const speech = new BYAKUSpeechController({
    synthesis,
    Utterance: FakeUtterance,
    onState: (state, message) => states.push([state, message]),
  });

  assert.equal(speech.speak('A bottle is on the left.'), true);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(synthesis.utterances[0].voice.name, 'US voice');
  assert.ok(states.some(([state]) => state === 'started'));
  synthesis.utterances[0].onerror({ error: 'audio-busy' });
  assert.deepEqual(states.at(-1), ['error', 'Browser speech synthesis failed (audio-busy).']);
});

test('question speech replaces routine speech but routine speech never overlaps', async () => {
  const synthesis = new FakeSynthesis();
  const speech = new BYAKUSpeechController({ synthesis, Utterance: FakeUtterance });

  assert.equal(speech.speak('Routine scene update'), true);
  assert.equal(speech.speak('Another routine update'), false);
  assert.equal(speech.speak('Where is the bottle? The bottle is on the left.', { priority: 'question' }), true);
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(synthesis.cancelCount, 1);
  assert.equal(synthesis.utterances.length, 2);
  assert.equal(synthesis.utterances[1].text, 'Where is the bottle?');
  assert.equal(synthesis.utterances[0].onend, null);
  synthesis.utterances[1].onend();
  assert.equal(synthesis.utterances[2].text, 'The bottle is on the left.');
});

test('long narration speaks sentence segments in order with intelligible defaults', async () => {
  const synthesis = new FakeSynthesis();
  const states = [];
  const speech = new BYAKUSpeechController({
    synthesis,
    Utterance: FakeUtterance,
    onState: (state) => states.push(state),
  });
  const narration = `First sentence. ${'A long description '.repeat(18)}continues here. Final sentence.`;

  assert.equal(speech.speak(narration), true);
  await new Promise((resolve) => setImmediate(resolve));
  for (let index = 0; index < synthesis.utterances.length || speech.current; index += 1) {
    const utterance = synthesis.utterances[index];
    assert.ok(utterance.text.length <= 240);
    assert.equal(utterance.rate, 0.95);
    assert.equal(utterance.pitch, 1);
    assert.equal(utterance.volume, 0.95);
    utterance.onend();
  }
  assert.equal(speech.current, null);
  assert.deepEqual(states.filter((state) => state === 'started' || state === 'ended'), ['started', 'ended']);
  assert.match(synthesis.utterances[0].text, /^First sentence/);
  assert.match(synthesis.utterances.at(-1).text, /Final sentence\.$/);
});

test('cancelling speech prevents queued narration segments from starting', async () => {
  const synthesis = new FakeSynthesis();
  const states = [];
  const speech = new BYAKUSpeechController({
    synthesis,
    Utterance: FakeUtterance,
    onState: (state) => states.push(state),
  });
  speech.speak('First sentence. Second sentence.');
  await new Promise((resolve) => setImmediate(resolve));
  speech.cancel();
  synthesis.utterances[0].onend?.();
  assert.equal(synthesis.utterances.length, 1);
  assert.equal(speech.current, null);
  assert.equal(states.at(-1), 'cancelled');
});

test('live narration speaks the first and changed scene without repeating unchanged text', () => {
  const clock = createFakeClock();
  const spoken = [];
  const scheduler = new BYAKULiveNarrationScheduler({
    speak: (text) => { spoken.push([clock.now(), text]); return true; },
    now: clock.now,
    setTimeoutFn: clock.setTimeout,
    clearTimeoutFn: clock.clearTimeout,
  });

  scheduler.start();
  scheduler.update('A chair is visible on the left.', 0);
  assert.deepEqual(spoken, [[0, 'A chair is visible on the left.']]);
  clock.advance(4999);
  assert.equal(spoken.length, 1);
  scheduler.update('A chair is visible on the left.', 4999);
  clock.advance(1);
  assert.equal(spoken.length, 1);
  scheduler.update('A bottle is now visible in the center.', 5001);
  clock.advance(5000);
  assert.deepEqual(spoken[1], [10000, 'A bottle is now visible in the center.']);
});

test('stale narration is skipped and stopping cancels its scheduled timer', () => {
  const clock = createFakeClock();
  const spoken = [];
  const stale = [];
  const scheduler = new BYAKULiveNarrationScheduler({
    speak: (text) => { spoken.push(text); return true; },
    onStale: (age) => stale.push(age),
    now: clock.now,
    setTimeoutFn: clock.setTimeout,
    clearTimeoutFn: clock.clearTimeout,
    maxFrameAgeMs: 1000,
  });

  scheduler.start();
  scheduler.update('A person is visible.', 0);
  clock.advance(5000);
  assert.deepEqual(spoken, ['A person is visible.']);
  assert.deepEqual(stale, [5000]);
  scheduler.stop();
  assert.equal(clock.pendingCount, 0);
  clock.advance(10000);
  assert.equal(spoken.length, 1);
});
