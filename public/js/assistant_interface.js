(() => {
  const panel = document.querySelector('#assistant-panel');
  const stateLabel = document.querySelector('#state-label');
  const detail = document.querySelector('#assistant-detail');
  const transcriptLine = document.querySelector('#transcript-line');
  const toast = document.querySelector('#toast');
  const video = document.querySelector('#camera-video');
  const cameraView = document.querySelector('#camera-view');
  const overlay = document.querySelector('#camera-overlay');
  const placeholder = document.querySelector('#camera-placeholder');
  const cameraStatus = document.querySelector('#camera-status');
  const cameraFootnote = document.querySelector('#camera-footnote');
  const previewBadge = document.querySelector('#preview-badge');
  const cameraPanel = document.querySelector('#camera-panel');
  const greetingFallback = document.querySelector('#assistant-greeting-fallback');
  const startButton = document.querySelector('#start-listening');
  const stopButton = document.querySelector('#stop-assistant');
  const queryForm = document.querySelector('#text-query-form');
  const queryInput = document.querySelector('#text-query-input');
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const backend = new window.BYAKUConnection();
  let recognition = null;
  let frameProcessor = null;
  let questionPending = false;
  let assistantActive = false;
  let cameraStartPromise = null;
  let cameraStartToken = 0;
  let hasLastFrame = false;
  let toastTimer = null;
  let state = 'idle';
  let greetingPending = false;
  let greetingAttempted = false;

  function setState(next, message = '') {
    state = next;
    panel.dataset.state = next;
    const content = {
      idle: ['AI Assistant', 'Tap Start Assistant to begin live scene analysis.', 'Idle'],
      listening: ['Listening...', 'Say “START” or ask a question', 'Listening'],
      camera_active: ['Camera Active', 'Live scene analysis is running.', 'Camera Active'],
      processing: ['Processing...', 'Starting local camera preview', 'Processing'],
      speaking: ['Speaking', 'Current scene narration', 'Speaking'],
      speech_complete: ['Speech complete', 'Narration finished. Live scene analysis is running.', 'Speech complete'],
      speech_error: ['Speech Error', 'Read the displayed response or check browser speech settings.', 'Speech Error'],
    }[next];
    detail.textContent = message || content?.[1] || 'Assistant is unavailable.';
    stateLabel.textContent = content?.[2] || 'Error';
    const actionLabel = document.querySelector('#assistant-action-label');
    const isActive = assistantActive;
    if (actionLabel) actionLabel.textContent = isActive ? 'Assistant Active · Tap to Ask' : 'Start Assistant';
    startButton.classList.toggle('is-listening', next === 'listening');
    startButton.setAttribute('aria-pressed', String(isActive));
    startButton.setAttribute(
      'aria-label',
      isActive
        ? 'Ask BYAKU a voice question. The camera session is active.'
        : 'Start Assistant. Request camera access and connect to live scene analysis.',
    );
  }

  function showToast(message) {
    toast.textContent = message;
    toast.classList.add('is-visible');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('is-visible'), 4200);
  }

  const speech = new window.BYAKUSpeechController({
    onState: (speechState, message) => {
      if (speechState === 'started') {
        if (greetingPending) {
          greetingFallback.hidden = true;
          return;
        }
        if (recognition) {
          try { recognition.stop(); } catch { /* Recognition may already be stopped. */ }
        }
        setState('speaking', message);
      } else if (speechState === 'ended' || speechState === 'cancelled') {
        if (greetingPending) {
          greetingPending = false;
          greetingFallback.hidden = true;
          setState('idle', 'Your assistant is ready. Start BYAKU when you are ready.');
          return;
        }
        if (speechState === 'ended' && questionPending) questionPending = false;
        if (assistantActive && !questionPending) {
          setState(speechState === 'ended' ? 'speech_complete' : 'camera_active');
          window.setTimeout(startListening, 350);
        }
      } else if (speechState === 'error') {
        if (greetingPending) {
          greetingPending = false;
          greetingFallback.hidden = false;
          setState('idle', 'Your assistant is ready. Tap the greeting button if automatic speech was blocked.');
          return;
        }
        questionPending = false;
        setState('speech_error', message);
        showToast(message);
      }
    },
  });

  function retryAssistantGreeting() {
    greetingPending = true;
    if (!speech.speak('Your assistant is ready.')) {
      greetingPending = false;
      greetingFallback.hidden = false;
    }
  }

  greetingFallback.addEventListener('click', retryAssistantGreeting);
  const narration = new window.BYAKULiveNarrationScheduler({
    speak: (text, options) => speech.speak(text, options),
    shouldSpeak: () => assistantActive && !questionPending,
    onStale: (age) => {
      cameraFootnote.textContent = `Latest scene is ${Math.round(age / 1000)} seconds old; waiting for a fresh frame.`;
    },
  });

  async function enablePreview() {
    try {
      const stream = await window.BYAKUCamera.start(
        video, cameraView, placeholder, cameraStatus, cameraFootnote, previewBadge, null,
      );
      return Boolean(stream);
    } catch (error) {
      const message = error.message || 'Unable to access the camera. Check browser permissions.';
      cameraStatus.textContent = 'Camera unavailable';
      cameraFootnote.textContent = message;
      showToast(message);
      return false;
    }
  }

  function handleBackendMessage(message) {
    if (message.type === 'error') {
      if (message.stage === 'query') {
        questionPending = false;
        setState('camera_active', message.message || 'The question could not be processed.');
        if (assistantActive) window.setTimeout(startListening, 350);
      } else if (message.code !== 'frame_slow' && assistantActive) {
        setState('camera_active', 'The camera is active; a frame could not be processed.');
      }
      showToast(`${message.stage || 'Backend'}: ${message.message || 'Request failed'}`);
      return;
    }
    if (message.type === 'status') {
      const cameraState = message.stage_status?.camera;
      if (cameraState === 'connected') {
        cameraStatus.innerHTML = '<span class="status-dot" aria-hidden="true"></span>AI processing';
      } else if (cameraState === 'disconnected') {
        const retainedFrame = message.last_frame_available || hasLastFrame;
        cameraStatus.textContent = 'Camera disconnected';
        cameraFootnote.textContent = retainedFrame
          ? 'Camera disconnected · last processed frame remains available'
          : 'Camera disconnected · no processed frame is available';
      }
      return;
    }
    if (message.type === 'query_result') {
      const answer = typeof message.answer === 'string' ? message.answer : '';
      transcriptLine.textContent = answer ? `BYAKU: ${answer}` : 'BYAKU could not answer from the current scene.';
      const narrationText = message.narration || answer;
      if (typeof narrationText !== 'string' || !narrationText.trim()) {
        questionPending = false;
        setState('speech_error', 'The backend did not return a spoken answer. Read the displayed result.');
        showToast('No speech text was returned for this question.');
        return;
      }
      speech.speak(narrationText, { priority: 'question' });
      return;
    }
    if (message.type !== 'frame_result') return;

    hasLastFrame = true;
    window.BYAKUFrameProcessor.drawOverlay(overlay, message.frame_size, message.objects);
    const objects = message.objects || [];
    cameraStatus.innerHTML = `<span class="status-dot" aria-hidden="true"></span>${message.stage_status?.detection === 'running' ? 'Camera Active' : 'Detection unavailable'}`;
    const processingMs = Number.isFinite(message.processing_ms) ? `${message.processing_ms} ms processing` : 'processing time unavailable';
    const frameAgeMs = Number.isFinite(message.client_frame_age_ms) ? ` · ${message.client_frame_age_ms} ms frame age` : '';
    cameraFootnote.textContent = `${objects.length} tracked object${objects.length === 1 ? '' : 's'} · ${processingMs}${frameAgeMs}`;
    narration.update(message.scene_narration, performance.now());
  }

  backend.addEventListener('disconnected', () => {
    if (!assistantActive) return;
    cameraStatus.textContent = 'Backend disconnected';
    cameraFootnote.textContent = hasLastFrame
      ? 'Backend disconnected · last processed frame remains available'
      : 'Backend disconnected · no processed frame is available';
    showToast('The backend connection was lost. BYAKU will try to reconnect.');
  });

  async function startCameraSession(resumeListening = true) {
    if (frameProcessor) return true;
    if (cameraStartPromise) return cameraStartPromise;
    const startToken = ++cameraStartToken;
    cameraStartPromise = (async () => {
      cameraPanel.hidden = false;
      cameraPanel.parentElement.classList.add('camera-active');
      setState('processing', 'Opening camera and connecting to the AI backend.');
      const started = await enablePreview();
      if (!started) {
        setState('idle', cameraFootnote.textContent || 'Camera startup was cancelled.');
        return false;
      }
      if (startToken !== cameraStartToken) {
        window.BYAKUCamera.stop(video, cameraView, placeholder, cameraStatus, cameraFootnote, previewBadge, null);
        return false;
      }
      try {
        await backend.resume();
        if (startToken !== cameraStartToken) {
          backend.close();
          window.BYAKUCamera.stop(video, cameraView, placeholder, cameraStatus, cameraFootnote, previewBadge, null);
          return false;
        }
        frameProcessor = new window.BYAKUFrameProcessor(backend, video);
        frameProcessor.addEventListener('result', (event) => handleBackendMessage(event.detail));
        frameProcessor.start();
        assistantActive = true;
        stopButton.hidden = false;
        narration.start();
        setState(
          state === 'listening' ? 'listening' : 'camera_active',
          state === 'listening' ? 'Camera active. Ask your question.' : 'Live scene analysis is running.',
        );
        cameraStatus.innerHTML = '<span class="status-dot" aria-hidden="true"></span>Connecting to backend';
        if (resumeListening) window.setTimeout(startListening, 500);
        return true;
      } catch (error) {
        frameProcessor?.stop();
        frameProcessor = null;
        stopButton.hidden = true;
        backend.close();
        if (window.BYAKUCamera.stream) {
          window.BYAKUCamera.stop(video, cameraView, placeholder, cameraStatus, cameraFootnote, previewBadge, null);
        }
        setState('idle', 'The camera was stopped because the Python backend could not connect.');
        showToast(error.message || 'Could not connect to the Python backend.');
        return false;
      }
    })();
    try {
      return await cameraStartPromise;
    } finally {
      cameraStartPromise = null;
    }
  }

  async function onTranscript(transcript) {
    const normalized = transcript.trim();
    transcriptLine.textContent = `Heard: “${normalized}”`;
    const isStart = /^start[.!?\s]*$/i.test(normalized);
    if (isStart) {
      if (!assistantActive) {
        void startCameraSession(false).then((started) => {
          if (started) startListening();
        });
      } else {
        setState('camera_active', 'Camera active. Ask your question.');
        startListening();
      }
      return;
    }

    if (!assistantActive || !frameProcessor) {
      setState('idle', 'Say “START” to activate the camera before asking a live-scene question.');
      showToast('Say “START” to begin live scene analysis.');
      return;
    }

    questionPending = true;
    setState('processing', 'Sending your question to the scene reasoning module.');
    if (!frameProcessor?.sendQuery(normalized)) {
      questionPending = false;
      setState('idle', 'The backend connection is unavailable.');
      showToast('Could not send the query to the Python backend.');
    }
  }

  function startListening() {
    if (!Recognition) {
      setState(
        assistantActive ? 'camera_active' : 'idle',
        assistantActive
          ? 'Live scene analysis is running. Voice recognition is unavailable; type your question below.'
          : 'Voice recognition is unavailable. Type your question below or try Chrome or Edge.',
      );
      showToast('Voice recognition is unavailable in this browser. Use Chrome or Edge, or type your question below.');
      return;
    }
    if (recognition) {
      try { recognition.stop(); } catch { /* Already stopped. */ }
    }
    recognition = new Recognition();
    recognition.lang = 'en-US';
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onstart = () => {
      setState('listening');
      transcriptLine.textContent = '';
    };
    recognition.onresult = (event) => onTranscript(event.results[0][0].transcript);
    recognition.onerror = (event) => {
      setState(
        assistantActive ? 'camera_active' : 'idle',
        assistantActive ? 'Live scene analysis is running.' : '',
      );
      const messages = {
        'audio-capture': 'The browser could not capture microphone audio. Check that your microphone is connected and not in use.',
        'language-not-supported': 'English speech recognition is unavailable in this browser. Try Chrome or Edge, or type below.',
        'network': 'The browser speech-recognition service is unavailable. Try Chrome or Edge, or type your question below.',
        'no-speech': 'No speech transcript arrived from the browser. Try Chrome or Edge, check its speech-recognition service, or type below.',
        'not-allowed': 'Microphone permission was denied. Allow microphone access for this site, then try again.',
        'service-not-allowed': 'The browser blocked its speech-recognition service. Try Chrome or Edge, or type your question below.',
      };
      const message = messages[event.error] || `Voice recognition failed (${event.error || 'unknown error'}). You can type your question below.`;
      showToast(message);
    };
    recognition.onend = () => {
      if (state === 'listening') {
        if (assistantActive) setState('camera_active');
        else setState('idle');
      }
    };
    try {
      recognition.start();
    } catch (error) {
      setState('idle');
      showToast(`Voice recognition could not start${error?.message ? `: ${error.message}` : ''}. You can type below.`);
    }
  }

  queryForm.addEventListener('submit', (event) => {
    event.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;
    queryInput.value = '';
    void onTranscript(query);
  });

  function stopAssistant() {
    cameraStartToken += 1;
    assistantActive = false;
    questionPending = false;
    narration.stop();
    speech.cancel();
    if (recognition) {
      try { recognition.stop(); } catch { /* Recognition may already be stopped. */ }
      recognition = null;
    }
    frameProcessor?.stop();
    frameProcessor = null;
    stopButton.hidden = true;
    backend.close();
    if (window.BYAKUCamera.stream) {
      window.BYAKUCamera.stop(video, cameraView, placeholder, cameraStatus, cameraFootnote, previewBadge, null);
    }
    cameraPanel.hidden = false;
    cameraPanel.parentElement.classList.remove('camera-active');
    setState('idle', 'Assistant stopped.');
  }

  startButton.addEventListener('click', () => {
    if (!assistantActive) {
      setState('listening', 'Say “START” to begin live scene analysis.');
      startListening();
      return;
    }
    if (!questionPending && !['processing', 'speaking'].includes(state)) startListening();
  });
  stopButton.addEventListener('click', stopAssistant);
  window.addEventListener('pagehide', stopAssistant);

  if (!greetingAttempted) {
    greetingAttempted = true;
    retryAssistantGreeting();
  }
})();
