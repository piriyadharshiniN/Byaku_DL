(() => {
  let activeStream = null;
  let startSequence = 0;
  let startupController = null;
  let activeTrackEnded = null;

  function getIdealVideoConstraints(preferredDeviceId = null) {
    const base = {
      facingMode: 'user',
      width: { ideal: 640 },
      height: { ideal: 480 },
    };
    if (preferredDeviceId) {
      return { ...base, deviceId: { exact: preferredDeviceId } };
    }
    return base;
  }

  function cameraError(error) {
    if (error?.name === 'NotAllowedError' || error?.name === 'PermissionDeniedError') {
      return new Error('Camera permission was denied. Allow camera access for this site and try again.');
    }
    if (error?.name === 'NotFoundError' || error?.name === 'DevicesNotFoundError') {
      return new Error('No camera device was found on this computer.');
    }
    if (error?.name === 'NotReadableError' || error?.name === 'TrackStartError') {
      return new Error('The camera is in use by another app or cannot be opened.');
    }
    if (error?.name === 'OverconstrainedError' || error?.name === 'ConstraintNotSatisfiedError') {
      return new Error('The selected camera does not support the requested video settings.');
    }
    if (error?.name === 'SecurityError' || error?.name === 'TypeError') {
      return new Error('Camera access requires a secure page (HTTPS or localhost) and browser permission.');
    }
    return new Error(error?.message || 'Unable to start the camera preview.');
  }

  function waitForVideo(video, stream, signal) {
    return new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => finish(new Error('The camera stream started but did not produce video.')), 10000);
      const tracks = stream.getTracks();
      let playbackStarted = false;
      const ready = () => video.videoWidth > 0
        && video.videoHeight > 0
        && video.readyState >= 2
        && tracks.some((track) => track.readyState === 'live');
      const cleanup = () => {
        window.clearTimeout(timeout);
        video.removeEventListener('loadedmetadata', checkReady);
        video.removeEventListener('canplay', checkReady);
        video.removeEventListener('error', onError);
        signal.removeEventListener('abort', onAbort);
        tracks.forEach((track) => track.removeEventListener?.('ended', onEnded));
      };
      const finish = (error) => {
        cleanup();
        if (error) reject(error);
        else resolve();
      };
      const checkReady = () => {
        if (playbackStarted && ready()) finish();
      };
      const onError = () => finish(new Error('The browser could not play the camera video.'));
      const onEnded = () => finish(new Error('The camera stopped before video became available.'));
      const onAbort = () => finish(new Error('Camera startup was cancelled.'));

      video.addEventListener('loadedmetadata', checkReady);
      video.addEventListener('canplay', checkReady);
      video.addEventListener('error', onError);
      signal.addEventListener('abort', onAbort, { once: true });
      tracks.forEach((track) => track.addEventListener?.('ended', onEnded, { once: true }));
      if (signal.aborted) {
        onAbort();
        return;
      }
      try {
        Promise.resolve(video.play()).then(() => {
          playbackStarted = true;
          checkReady();
        }, (error) => finish(new Error(`The browser could not start camera video playback: ${error?.message || error}`)));
      } catch (error) {
        finish(error);
      }
    });
  }

  async function getCameraDevices() {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((device) => device.kind === 'videoinput');
  }

  async function start(video, container, placeholder, status, footnote, badge, button, options = {}) {
    const deviceId = options.deviceId || null;
    const preferredFacing = options.facingMode || 'user';
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('Camera access is not available in this browser.');
    }
    if (!video) throw new Error('The camera preview element is missing from this page.');

    stop(video, container, placeholder, status, footnote, badge, button);
    const sequence = ++startSequence;
    startupController = new AbortController();
    const controller = startupController;
    let stream;

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: deviceId
          ? { deviceId: { exact: deviceId }, width: { ideal: 640 }, height: { ideal: 480 } }
          : { facingMode: preferredFacing, width: { ideal: 640 }, height: { ideal: 480 } },
      });
      if (sequence !== startSequence) {
        stream.getTracks().forEach((track) => track.stop());
        return null;
      }
      activeStream = stream;
      video.muted = true;
      video.playsInline = true;
      video.autoplay = true;
      video.srcObject = stream;

      await waitForVideo(video, stream, controller.signal);
      if (sequence !== startSequence) {
        stream.getTracks().forEach((track) => track.stop());
        return null;
      }
      startupController = null;
    } catch (error) {
      if (sequence !== startSequence) {
        stream?.getTracks().forEach((track) => track.stop());
        return null;
      }
      startupController = null;
      stop(video, container, placeholder, status, footnote, badge, button);
      throw cameraError(error);
    }

    activeTrackEnded = () => {
      if (activeStream !== stream || sequence !== startSequence) return;
      stop(video, container, placeholder, status, footnote, badge, button);
      if (status) status.textContent = 'Camera disconnected';
      if (footnote) footnote.textContent = 'The camera stopped. Start the assistant again to reconnect.';
    };
    stream.getTracks().forEach((track) => track.addEventListener?.('ended', activeTrackEnded, { once: true }));
    if (container) container.classList.add('has-stream');
    if (placeholder) placeholder.setAttribute('aria-hidden', 'true');
    if (status) status.innerHTML = '<span class="status-dot" aria-hidden="true"></span>Camera preview active';
    if (footnote) footnote.textContent = 'Browser camera stream';
    if (badge) badge.hidden = false;
    if (button) button.textContent = 'Stop camera preview';

    return activeStream;
  }

  function stop(video, container, placeholder, status, footnote, badge, button) {
    startSequence += 1;
    startupController?.abort();
    startupController = null;
    const stream = activeStream;
    activeStream = null;
    if (stream && activeTrackEnded) {
      stream.getTracks().forEach((track) => track.removeEventListener?.('ended', activeTrackEnded));
    }
    activeTrackEnded = null;
    stream?.getTracks().forEach((track) => track.stop());
    if (video) video.srcObject = null;
    if (container) container.classList.remove('has-stream');
    if (placeholder) placeholder.removeAttribute('aria-hidden');
    if (status) status.innerHTML = '<span class="status-dot status-dot-muted" aria-hidden="true"></span>Camera Ready';
    if (footnote) footnote.textContent = 'No camera stream';
    if (badge) badge.hidden = true;
    if (button) button.textContent = 'Enable camera preview';
  }

  window.BYAKUCamera = {
    start,
    stop,
    getCameraDevices,
    get stream() { return activeStream; },
    getIdealVideoConstraints,
  };
})();
