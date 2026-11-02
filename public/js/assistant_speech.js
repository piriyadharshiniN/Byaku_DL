(() => {
  function splitNarration(text, maxLength = 240) {
    const sentenceParts = typeof Intl.Segmenter === 'function'
      ? [...new Intl.Segmenter(undefined, { granularity: 'sentence' }).segment(text)].map(({ segment }) => segment.trim())
      : (text.match(/[^.!?]+[.!?]+["')\]]*|[^.!?]+$/g) || [text]).map((part) => part.trim());
    const segments = [];
    for (const sentence of sentenceParts.filter(Boolean)) {
      let remaining = sentence;
      while (remaining.length > maxLength) {
        let splitAt = remaining.lastIndexOf(' ', maxLength);
        if (splitAt < 1) splitAt = maxLength;
        segments.push(remaining.slice(0, splitAt).trim());
        remaining = remaining.slice(splitAt).trim();
      }
      if (remaining) segments.push(remaining);
    }
    return segments;
  }

  class BYAKUSpeechController {
    constructor({
      synthesis = window.speechSynthesis,
      Utterance = window.SpeechSynthesisUtterance,
      onState = () => {},
      voiceTimeoutMs = 1200,
    } = {}) {
      this.synthesis = synthesis;
      this.Utterance = Utterance;
      this.onState = onState;
      this.voiceTimeoutMs = voiceTimeoutMs;
      this.current = null;
      this.sequence = 0;
      this.segments = [];
      this.segmentIndex = 0;
      this.voice = null;
    }

    speak(text, { priority = 'routine' } = {}) {
      if (typeof text !== 'string' || !text.trim()) {
        this.onState('error', 'No speech text is available.');
        return false;
      }
      if (!this.synthesis || typeof this.Utterance !== 'function') {
        this.onState('error', 'Browser speech synthesis is unavailable.');
        return false;
      }
      if (this.current) {
        if (priority !== 'question') return false;
        this.cancel();
      }

      const sequence = ++this.sequence;
      const narration = text.trim();
      this.segments = splitNarration(narration);
      this.segmentIndex = 0;
      this.current = { text: narration };
      this.onState('pending', narration);
      void this._start(sequence, narration);
      return true;
    }

    async _start(sequence, narration) {
      try {
        let voices = this.synthesis.getVoices();
        if (!voices.length && this.voiceTimeoutMs > 0) {
          voices = await this._waitForVoices();
        }
        if (this.sequence !== sequence) return;
        if (!voices.length) {
          this.current = null;
          this.segments = [];
          this.onState('error', 'No browser speech voice is available. Check the browser speech settings.');
          return;
        }
        this.voice = voices.find((voice) => /^en-US\b/i.test(voice.lang))
          || voices.find((voice) => /^en\b/i.test(voice.lang))
          || voices.find((voice) => voice.default)
          || voices[0];
        if (this.synthesis.paused) this.synthesis.resume();
        this._speakNext(sequence, narration);
      } catch (error) {
        if (this.sequence !== sequence) return;
        this.current = null;
        this.segments = [];
        this.onState('error', `Browser speech synthesis could not start: ${error?.message || error}`);
      }
    }

    _speakNext(sequence, narration) {
      if (this.sequence !== sequence) return;
      const utterance = new this.Utterance(this.segments[this.segmentIndex]);
      utterance.voice = this.voice;
      utterance.rate = 0.95;
      utterance.pitch = 1;
      utterance.volume = 0.95;
      this.current = utterance;
      utterance.onstart = () => {
        if (this.sequence === sequence && this.segmentIndex === 0) {
          this.onState('started', narration);
        }
      };
      utterance.onend = () => {
        if (this.sequence !== sequence) return;
        this.segmentIndex += 1;
        if (this.segmentIndex < this.segments.length) {
          this._speakNext(sequence, narration);
          return;
        }
        this.current = null;
        this.segments = [];
        this.onState('ended', narration);
      };
      utterance.onerror = (event) => {
        if (this.sequence !== sequence) return;
        this.current = null;
        this.segments = [];
        this.onState('error', `Browser speech synthesis failed (${event?.error || 'unknown error'}).`);
      };
      try {
        this.synthesis.speak(utterance);
      } catch (error) {
        if (this.sequence !== sequence) return;
        this.current = null;
        this.segments = [];
        this.onState('error', `Browser speech synthesis could not continue: ${error?.message || error}`);
      }
    }

    _waitForVoices() {
      return new Promise((resolve) => {
        let settled = false;
        const finish = () => {
          if (settled) return;
          settled = true;
          window.clearTimeout(timeout);
          this.synthesis.removeEventListener?.('voiceschanged', finish);
          resolve(this.synthesis.getVoices());
        };
        const timeout = window.setTimeout(finish, this.voiceTimeoutMs);
        this.synthesis.addEventListener?.('voiceschanged', finish, { once: true });
      });
    }

    cancel() {
      this.sequence += 1;
      if (this.current) {
        this.current.onstart = null;
        this.current.onend = null;
        this.current.onerror = null;
      }
      this.current = null;
      this.segments = [];
      this.segmentIndex = 0;
      this.synthesis?.cancel();
      this.onState('cancelled', '');
    }
  }

  class BYAKULiveNarrationScheduler {
    constructor({
      speak,
      shouldSpeak = () => true,
      onStale = () => {},
      now = () => performance.now(),
      setTimeoutFn = window.setTimeout.bind(window),
      clearTimeoutFn = window.clearTimeout.bind(window),
      intervalMs = 5000,
      maxFrameAgeMs = 10000,
    }) {
      this.speak = speak;
      this.shouldSpeak = shouldSpeak;
      this.onStale = onStale;
      this.now = now;
      this.setTimeoutFn = setTimeoutFn;
      this.clearTimeoutFn = clearTimeoutFn;
      this.intervalMs = intervalMs;
      this.maxFrameAgeMs = maxFrameAgeMs;
      this.active = false;
      this.latestText = '';
      this.latestAt = 0;
      this.lastSpokenText = '';
      this.firstDescriptionSent = false;
      this.nextDue = 0;
      this.timer = null;
    }

    start() {
      this.active = true;
    }

    update(text, receivedAt = this.now()) {
      if (!this.active || typeof text !== 'string' || !text.trim()) return;
      this.latestText = text.trim();
      this.latestAt = receivedAt;
      if (!this.firstDescriptionSent) {
        this.firstDescriptionSent = true;
        this.nextDue = this.now() + this.intervalMs;
        this._speakLatest();
        this._schedule();
      }
    }

    _speakLatest() {
      if (!this.latestText || !this.shouldSpeak()) return false;
      const age = this.now() - this.latestAt;
      if (age > this.maxFrameAgeMs) {
        this.onStale(age);
        return false;
      }
      if (this.latestText === this.lastSpokenText) return false;
      const text = this.latestText;
      if (!this.speak(text, { priority: 'routine' })) return false;
      this.lastSpokenText = this.latestText;
      return true;
    }

    _schedule() {
      if (!this.active) return;
      const delay = Math.max(0, this.nextDue - this.now());
      this.timer = this.setTimeoutFn(() => {
        this.timer = null;
        if (!this.active) return;
        this._speakLatest();
        const now = this.now();
        do {
          this.nextDue += this.intervalMs;
        } while (this.nextDue <= now);
        this._schedule();
      }, delay);
    }

    stop() {
      this.active = false;
      if (this.timer !== null) this.clearTimeoutFn(this.timer);
      this.timer = null;
      this.latestText = '';
      this.latestAt = 0;
      this.lastSpokenText = '';
      this.firstDescriptionSent = false;
      this.nextDue = 0;
    }
  }

  window.BYAKUSpeechController = BYAKUSpeechController;
  window.BYAKULiveNarrationScheduler = BYAKULiveNarrationScheduler;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { BYAKUSpeechController, BYAKULiveNarrationScheduler };
  }
})();
