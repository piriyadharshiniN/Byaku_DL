(() => {
  const video = document.querySelector('#dash-camera-video');
  const overlay = document.querySelector('#dash-camera-overlay');
  const container = document.querySelector('#dash-camera-view');
  const placeholder = document.querySelector('#dash-camera-placeholder');
  const cameraStatus = document.querySelector('#dash-camera-status');
  const detail = document.querySelector('#dash-camera-detail');
  const button = document.querySelector('#dash-camera-toggle');
  const pauseButton = document.querySelector('#dash-camera-pause');
  const resumeButton = document.querySelector('#dash-camera-resume');
  const reviewCanvas = document.querySelector('#review-frame-canvas');
  const connection = new window.BYAKUConnection();
  let frameProcessor = null;
  let liveSessionId = null;
  let reviewSessionId = null;
  let currentReviewFrame = null;
  let currentFrameMessage = null;
  let currentDetections = [];
  let selectedDetection = null;
  let reviewHistory = [];
  let reviewImageToken = 0;
  const reviewSubmissionIds = new Map();
  let selectedGraphNodeId = null;
  let missedDrawMode = false;
  let missedDragStart = null;
  let missedBBox = null;
  let latestNarration = '';
  let latestDepthMessage = null;

  const speechController = new window.BYAKUSpeechController({
    onState(state, text) {
      const status = document.querySelector('#speech-playback-status');
      const speak = document.querySelector('#narration-speak');
      const stop = document.querySelector('#narration-stop');
      const labels = {
        pending: 'Speech queued',
        started: 'Speaking',
        ended: 'Speech playback completed',
        cancelled: 'Speech stopped',
        error: text || 'Speech playback failed',
      };
      if (status) status.textContent = labels[state] || 'Playback not started';
      if (speak) speak.disabled = !latestNarration || state === 'pending' || state === 'started';
      if (stop) stop.disabled = state !== 'pending' && state !== 'started';
    },
  });

  function statusText(value) {
    return String(value || 'unavailable').replaceAll('_', ' ');
  }

  function setText(selector, value) {
    const element = document.querySelector(selector);
    if (element) element.textContent = statusText(value);
  }

  function formatRelativeDepth(value) {
    return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(2) : 'Unavailable';
  }

  function formatPosition(object) {
    return [object.horizontal_position, object.vertical_position]
      .filter((value) => typeof value === 'string' && value)
      .join(', ') || 'Unavailable';
  }

  function detectionKey(object) {
    if (object?.track_id !== null && object?.track_id !== undefined) return `track:${object.track_id}`;
    return `${object?.model_class_name || object?.class_name}:${(object?.bounding_box || []).join(',')}`;
  }

  function setReviewStatus(message, isError = false) {
    const status = document.querySelector('#review-feedback-status');
    if (!status) return;
    status.textContent = message;
    status.dataset.error = String(isError);
  }

  function populateCorrectionFields(object) {
    const label = document.querySelector('#review-corrected-label');
    const values = object?.bbox && typeof object.bbox === 'object'
      ? [object.bbox.x1, object.bbox.y1, object.bbox.x2, object.bbox.y2]
      : object?.bounding_box;
    if (label) label.value = object?.model_class_name || object?.class_name || '';
    if (!Array.isArray(values)) return;
    ['#review-bbox-x1', '#review-bbox-y1', '#review-bbox-x2', '#review-bbox-y2']
      .forEach((selector, index) => {
        const field = document.querySelector(selector);
        if (field) field.value = String(values[index]);
      });
  }

  function drawReviewFrame(frameUrl, frameSize, objects) {
    const canvas = document.querySelector('#review-frame-canvas');
    const empty = document.querySelector('#review-frame-empty');
    if (!canvas || !empty || typeof frameUrl !== 'string') return;
    const token = ++reviewImageToken;
    const image = new Image();
    image.onload = () => {
      if (token !== reviewImageToken) return;
      const context = canvas.getContext('2d');
      if (!context) return;
      const width = Number(frameSize?.width) || image.naturalWidth;
      const height = Number(frameSize?.height) || image.naturalHeight;
      canvas.width = width;
      canvas.height = height;
      context.drawImage(image, 0, 0, width, height);
      const selectedKey = selectedDetection ? detectionKey(selectedDetection) : null;
      for (const object of objects) {
        const box = object.bounding_box;
        if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite)) continue;
        const [x1, y1, x2, y2] = box;
        context.lineWidth = Math.max(2, width / 320);
        context.strokeStyle = selectedKey === detectionKey(object) ? '#ffe26b' : '#56f0bc';
        context.strokeRect(x1, y1, x2 - x1, y2 - y1);
        const confidence = Number.isFinite(Number(object.confidence))
          ? ` ${(Number(object.confidence) * 100).toFixed(0)}%`
          : '';
        const track = object.track_id === null || object.track_id === undefined ? '' : ` #${object.track_id}`;
        const remembered = object.remembered_correction?.corrected_name;
        const predicted = object.model_class_name || object.class_name || 'object';
        const label = `${predicted}${remembered ? ` → ${remembered}` : ''}${track}${confidence}`;
        context.font = `600 ${Math.max(12, width / 48)}px sans-serif`;
        const labelWidth = context.measureText(label).width + 10;
        const labelTop = Math.max(0, y1 - 24);
        context.fillStyle = 'rgba(25, 52, 61, 0.88)';
        context.fillRect(x1, labelTop, labelWidth, 22);
        context.fillStyle = '#ffffff';
        context.fillText(label, x1 + 5, labelTop + 16);
      }
      if (missedBBox) {
        context.lineWidth = Math.max(2, width / 240);
        context.setLineDash([8, 5]);
        context.strokeStyle = '#ff8b6b';
        context.strokeRect(
          missedBBox.x1,
          missedBBox.y1,
          missedBBox.x2 - missedBBox.x1,
          missedBBox.y2 - missedBBox.y1,
        );
        context.setLineDash([]);
      }
      empty.hidden = true;
      canvas.hidden = false;
    };
    image.onerror = () => {
      if (token !== reviewImageToken) return;
      canvas.hidden = true;
      empty.hidden = false;
      empty.textContent = 'The latest camera frame could not be displayed.';
    };
    image.src = frameUrl;
  }

  function renderReviewDetections() {
    const list = document.querySelector('#review-detection-list');
    const selectedLabel = document.querySelector('#review-selected');
    const confirmButton = document.querySelector('#review-confirm');
    const incorrectButton = document.querySelector('#review-incorrect');
    const correctionForm = document.querySelector('#review-correction-form');
    const falsePositiveButton = document.querySelector('#review-false-positive');
    const bboxButton = document.querySelector('#review-bbox-edit');
    const missedButton = document.querySelector('#review-missed-start');
    if (!list || !selectedLabel || !confirmButton || !incorrectButton || !correctionForm) return;
    const knownClasses = [...new Set(currentDetections
      .flatMap((object) => [object.model_class_name, object.class_name])
      .filter((name) => typeof name === 'string' && name.trim()))].sort();
    const classList = document.querySelector('#review-known-classes');
    if (classList) {
      classList.replaceChildren();
      for (const className of knownClasses) {
        const option = document.createElement('option');
        option.value = className;
        classList.append(option);
      }
    }
    list.replaceChildren();
    if (!currentDetections.length) {
      const emptyItem = document.createElement('li');
      emptyItem.className = 'review-empty';
      emptyItem.textContent = 'No current detections';
      list.append(emptyItem);
    } else {
      for (const object of currentDetections) {
        const item = document.createElement('li');
        const select = document.createElement('button');
        select.type = 'button';
        select.className = 'review-detection-option';
        select.classList.toggle('is-selected', selectedDetection && detectionKey(selectedDetection) === detectionKey(object));
        const track = object.track_id === null || object.track_id === undefined ? 'untracked' : `track #${object.track_id}`;
        const confidence = Number.isFinite(Number(object.confidence))
          ? `${(Number(object.confidence) * 100).toFixed(1)}%`
          : 'confidence unavailable';
        const remembered = object.remembered_correction?.corrected_name;
        const predicted = object.model_class_name || object.class_name || 'Unknown';
        select.textContent = `${predicted}${remembered ? ` → remembered: ${remembered}` : ''} · ${track} · ${confidence}`;
        select.addEventListener('click', () => {
          selectedDetection = object;
          populateCorrectionFields(object);
          renderReviewDetections();
          drawReviewFrame(currentReviewFrame?.dataUrl, currentReviewFrame?.size, currentDetections);
        });
        item.append(select);
        list.append(item);
      }
    }
    if (selectedDetection) {
      const track = selectedDetection.track_id === null || selectedDetection.track_id === undefined
        ? 'untracked'
        : `track #${selectedDetection.track_id}`;
      const predictedName = selectedDetection.model_class_name || selectedDetection.class_name || 'Unknown';
      const currentRecord = reviewHistory.find((record) =>
        record.frame_id === String(currentReviewFrame?.id)
        && detectionKey(record.original_prediction) === detectionKey(selectedDetection)
        && record.review_status !== 'undone');
      const reviewLabel = currentRecord
        ? currentRecord.review_status === 'corrected' ? 'Corrected'
          : currentRecord.review_status === 'rejected' ? 'Rejected'
            : 'Accepted'
        : 'Pending review';
      selectedLabel.textContent = `Selected: ${predictedName}${selectedDetection.remembered_correction ? ` → remembered: ${selectedDetection.class_name}` : ''} · ${track} · ${reviewLabel}`;
      const predicted = document.querySelector('#review-predicted-name');
      if (predicted) predicted.textContent = predictedName;
    } else {
      selectedLabel.textContent = 'Select a detection from the list to review it.';
    }
    const hasSelection = Boolean(selectedDetection);
    const alreadyReviewed = hasSelection && reviewHistory.some((record) =>
      record.frame_id === String(currentReviewFrame?.id)
      && detectionKey(record.original_prediction) === detectionKey(selectedDetection)
      && record.review_status !== 'undone');
    confirmButton.disabled = !hasSelection || alreadyReviewed;
    incorrectButton.disabled = !hasSelection || alreadyReviewed;
    if (falsePositiveButton) falsePositiveButton.disabled = !hasSelection || alreadyReviewed;
    if (bboxButton) bboxButton.disabled = !hasSelection || alreadyReviewed;
    if (missedButton) missedButton.disabled = !currentReviewFrame?.dataUrl;
    correctionForm.hidden = !hasSelection || correctionForm.dataset.open !== 'true';
    document.querySelector('#review-question').textContent = alreadyReviewed
      ? 'This detection has already been reviewed.'
      : 'Is this object detection correct?';
    if (alreadyReviewed) {
      const record = reviewHistory.find((item) =>
        item.frame_id === String(currentReviewFrame?.id)
        && detectionKey(item.original_prediction) === detectionKey(selectedDetection)
        && item.review_status !== 'undone');
      setReviewStatus(record?.human_corrected_name
        ? `Saved: model predicted “${selectedDetection.class_name}”; corrected to “${record.human_corrected_name}”.`
        : 'Confirmation saved for this detection.');
    }
  }

  function renderReviewHistory() {
    const list = document.querySelector('#review-history');
    if (!list) return;
    list.replaceChildren();
    if (!reviewHistory.length) {
      const empty = document.createElement('li');
      empty.textContent = 'No saved review actions.';
      list.append(empty);
      return;
    }
    for (const record of reviewHistory) {
      const item = document.createElement('li');
      const time = new Date(record.timestamp).toLocaleString();
      const label = record.original_prediction?.class_name || 'Unknown object';
      const corrected = record.human_corrected_name
        ? ` → ${record.human_corrected_name}`
        : record.review_status === 'confirmed' || record.review_status === 'accepted'
          ? ' · confirmed correct'
          : '';
      const source = record.source || record.frame_reference?.source || 'live';
      const sourceLabel = source === 'upload_video' || source === 'video'
        ? '[Video Upload]'
        : source === 'upload_image' || source === 'image'
          ? '[Image Upload]'
          : '[Live Camera]';
      const summary = document.createElement('span');
      const feedbackType = record.feedback_type ? ` · ${record.feedback_type.replaceAll('_', ' ')}` : '';
      const frame = record.frame_id || record.frame_number;
      const reviewState = record.review_status === 'confirmed' ? 'Accepted'
        : record.review_status === 'corrected' ? 'Corrected'
          : record.review_status === 'rejected' ? 'Rejected'
            : record.review_status === 'undone' ? 'Undone'
              : 'Pending review';
      summary.textContent = `${sourceLabel} ${time} · ${label}${record.track_id === null || record.track_id === undefined ? '' : ` #${record.track_id}`}${frame == null ? '' : ` · frame ${frame}`}${corrected}${feedbackType} · ${reviewState}`;
      item.append(summary);
      if (record.review_status !== 'undone') {
        const undo = document.createElement('button');
        undo.type = 'button';
        undo.className = 'text-button review-undo-item';
        undo.textContent = 'Undo';
        undo.addEventListener('click', () => undoReview(record.feedback_id));
        item.append(' ', undo);
      }
      list.append(item);
    }
    renderReviewDetections();
  }

  async function loadReviewHistory() {
    if (!reviewSessionId) {
      reviewHistory = [];
      renderReviewHistory();
      return;
    }
    const response = await fetch(`/api/feedback?limit=100&session_id=${encodeURIComponent(reviewSessionId)}`, { cache: 'no-store' });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || `Feedback history returned ${response.status}`);
    reviewHistory = Array.isArray(body.feedback) ? body.feedback : [];
    renderReviewHistory();
  }
  if (typeof window !== 'undefined') {
    window.byakuReloadReviewHistory = loadReviewHistory;
  }

  function refreshReviewScene(message) {
    if (message.session_id && message.session_id !== liveSessionId) {
      liveSessionId = message.session_id;
      reviewSessionId = message.session_id;
      loadReviewHistory().catch((error) => setReviewStatus(`Could not load this session's review history: ${error.message}`, true));
    }
    const objects = Array.isArray(message.objects) ? message.objects : [];
    const previousKey = selectedDetection ? detectionKey(selectedDetection) : null;
    const frameUrl = message.client_frame_data_url || message.frame_preview || message.last_frame_preview;
    const frameId = message.frame_id ?? message.frame_number ?? null;
    currentReviewFrame = {
      id: frameId,
      number: message.frame_number ?? null,
      sessionId: message.session_id || liveSessionId,
      size: message.frame_size,
      dataUrl: frameUrl || (currentReviewFrame?.id === frameId ? currentReviewFrame.dataUrl : null),
      reviewFrameToken: message.review_frame_token ?? null,
    };
    currentFrameMessage = message;
    currentDetections = objects;
    if (previousKey) {
      selectedDetection = objects.find((object) => detectionKey(object) === previousKey) || null;
    }
    const frameLabel = document.querySelector('#review-frame-label');
    if (frameLabel) {
      const cameraConnected = message.stage_status?.camera === 'connected';
      frameLabel.textContent = currentReviewFrame.id === null
        ? 'Waiting for a live frame'
        : `${cameraConnected ? 'Live frame' : 'Last processed frame'} · ${cameraConnected ? 'camera active' : 'camera disconnected'} · ${objects.length} detections`;
    }
    const empty = document.querySelector('#review-frame-empty');
    if (!currentReviewFrame.dataUrl && empty) {
      empty.hidden = false;
      empty.textContent = 'A saved detection is available, but its image is not in this browser session. Start the dashboard camera preview to review a live frame.';
      document.querySelector('#review-frame-canvas').hidden = true;
    } else {
      drawReviewFrame(currentReviewFrame.dataUrl, currentReviewFrame.size, objects);
    }
    renderReviewDetections();
  }

  function stableSubmissionId(correctionType, correctedValue) {
    const key = JSON.stringify([
      reviewSessionId,
      currentReviewFrame?.id,
      detectionKey(selectedDetection),
      correctionType,
      correctedValue,
    ]);
    if (!reviewSubmissionIds.has(key)) {
      reviewSubmissionIds.set(key, window.crypto?.randomUUID?.() || `${reviewSessionId}-${reviewSubmissionIds.size}`);
    }
    return reviewSubmissionIds.get(key);
  }

  function editedBoundingBox() {
    const selectors = ['#review-bbox-x1', '#review-bbox-y1', '#review-bbox-x2', '#review-bbox-y2'];
    const values = selectors.map((selector) => document.querySelector(selector)?.value.trim() || '');
    if (values.every((value) => !value)) return null;
    if (values.some((value) => value === '')) throw new Error('Enter all four coordinates for a corrected bounding box.');
    const [x1, y1, x2, y2] = values.map(Number);
    if (![x1, y1, x2, y2].every(Number.isFinite) || x2 <= x1 || y2 <= y1) {
      throw new Error('Corrected bounding box coordinates must be valid and have positive width and height.');
    }
    const { width, height } = currentReviewFrame?.size || {};
    if (x1 < 0 || y1 < 0 || x2 > width || y2 > height) {
      throw new Error(`Corrected box must stay inside the ${width}×${height} frame.`);
    }
    return { x1, y1, x2, y2 };
  }

  async function saveReview(feedbackType, correctedValue, reviewStatus, options = {}) {
    const detection = Object.prototype.hasOwnProperty.call(options, 'detection') ? options.detection : selectedDetection;
    const sessionId = currentReviewFrame?.sessionId || reviewSessionId;
    if (!currentReviewFrame || !sessionId) {
      setReviewStatus('Start the live demonstration and wait for a processed frame before reviewing.', true);
      return false;
    }
    if (!detection && feedbackType !== 'missed_object') return false;
    const originalBox = detection?.bbox && typeof detection.bbox === 'object'
      ? detection.bbox
      : detection?.bounding_box
        ? Object.fromEntries(['x1', 'y1', 'x2', 'y2'].map((key, index) => [key, detection.bounding_box[index]]))
        : options.bbox;
    if (!originalBox) {
      setReviewStatus('This review needs a frame bounding box.', true);
      return false;
    }
    const reviewedDetection = detection
      ? {
          ...detection,
          class_name: detection.model_class_name || detection.class_name,
          bbox: originalBox,
        }
      : {
          class_name: 'missed object',
          label: 'missed object',
          confidence: 0,
          bbox: originalBox,
          bounding_box: [originalBox.x1, originalBox.y1, originalBox.x2, originalBox.y2],
        };
    const correctedBox = options.correctedBBox || null;
    const controls = document.querySelectorAll('#review-confirm, #review-incorrect, #review-save');
    controls.forEach((control) => { control.disabled = true; });
    let saved = false;
    const payload = {
      session_id: sessionId,
      frame_id: currentReviewFrame.id === null ? null : String(currentReviewFrame.id),
      frame_number: currentReviewFrame.number,
      review_frame_token: currentReviewFrame.reviewFrameToken,
      image_dimensions: currentReviewFrame.size || null,
      track_id: detection?.track_id ?? null,
      original_prediction: reviewedDetection,
      corrected_value: correctedValue,
      human_corrected_name: ['class_correction', 'missed_object'].includes(feedbackType) ? correctedValue : null,
      corrected_bbox: correctedBox,
      correction_type: feedbackType,
      review_status: reviewStatus,
      input_source: 'live_camera',
      submission_id: stableSubmissionId(feedbackType, correctedValue),
    };
    try {
      const response = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || `Feedback save returned ${response.status}`);
      document.querySelector('#review-correction-form').dataset.open = 'false';
      document.querySelector('#review-correction-form').reset();
      setReviewStatus(feedbackType === 'class_correction'
        ? `Saved: model predicted “${reviewedDetection.class_name}”; corrected to “${correctedValue}”.`
        : feedbackType === 'false_positive'
          ? 'False positive saved to persistent review history.'
          : feedbackType === 'missed_object'
            ? `Missed “${correctedValue}” saved with its annotated frame region.`
            : correctedBox
              ? 'Corrected box saved to persistent review history.'
              : 'Confirmation saved to persistent review history.');
      await loadReviewHistory();
      saved = true;
    } catch (error) {
      setReviewStatus(`Could not save review: ${error.message}`, true);
    } finally {
      renderReviewDetections();
    }
    return saved;
  }

  async function submitCorrection(event) {
    event.preventDefault();
    const correctedName = document.querySelector('#review-corrected-label').value.trim();
    if (!correctedName) {
      setReviewStatus('A correct object name is required before saving.', true);
      return;
    }
    let correctedBBox;
    try {
      correctedBBox = editedBoundingBox();
    } catch (error) {
      setReviewStatus(error.message, true);
      return;
    }
    const original = selectedDetection?.bbox && typeof selectedDetection.bbox === 'object'
      ? selectedDetection.bbox
      : Object.fromEntries(['x1', 'y1', 'x2', 'y2'].map((key, index) => [key, selectedDetection.bounding_box[index]]));
    const bboxChanged = correctedBBox && ['x1', 'y1', 'x2', 'y2'].some((key) => correctedBBox[key] !== Number(original[key]));
    const labelChanged = correctedName !== (selectedDetection?.model_class_name || selectedDetection?.class_name);
    const feedbackType = labelChanged && bboxChanged
      ? 'class_and_bbox_correction'
      : bboxChanged ? 'bbox_correction' : 'class_correction';
    await saveReview(feedbackType, correctedName, 'corrected', {
      correctedBBox: bboxChanged ? correctedBBox : null,
    });
  }

  async function undoReview(feedbackId) {
    try {
      const response = await fetch(`/api/feedback/${encodeURIComponent(feedbackId)}/undo`, { method: 'POST' });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || `Undo returned ${response.status}`);
      setReviewStatus('Last saved review action was marked undone; history is preserved.');
      await loadReviewHistory();
    } catch (error) {
      setReviewStatus(`Could not undo review action: ${error.message}`, true);
    }
  }

  function updatePipeline(status) {
    const values = status?.stage_status
      ? { ...status, ...status.stage_status }
      : status || {};
    const keys = {
      camera: 'camera',
      detection: 'detection',
      tracking: 'tracking',
      depth: 'depth',
      spatial: 'spatial',
      scene: 'scene_graph',
      memory: 'memory',
      narration: 'narration',
      review: 'review',
    };
    document.querySelectorAll('.pipeline-track li[data-module]').forEach((item) => {
      const stage = keys[item.dataset.module];
      const value = stage === 'review'
        ? values.review || (values.backend === 'connected' ? 'available' : 'unavailable')
        : values[stage] || 'unavailable';
      item.dataset.status = ['ready', 'connected'].includes(value) ? 'running' : value;
      const node = item.querySelector('.pipeline-node');
      let detailLabel = node.querySelector('[data-stage-status]');
      if (!detailLabel) {
        detailLabel = document.createElement('small');
        detailLabel.dataset.stageStatus = '';
        node.append(detailLabel);
      }
      detailLabel.textContent = statusText(value).toUpperCase();
      const stageTime = values.stage_timings_ms?.[stage];
      node.title = `${statusText(value)}${Number.isFinite(stageTime) ? ` · ${stageTime.toFixed(1)} ms` : ''}`;
    });
    const modelStatus = values.model_status || {};
    const yoloStatus = document.querySelector('#detection-model-caption');
    if (yoloStatus && modelStatus.yolo) {
      yoloStatus.textContent = modelStatus.yolo.loaded
        ? `${modelStatus.yolo.model || 'YOLO model'} loaded via ${modelStatus.yolo.backend || 'model backend'} on ${values.device || 'CPU'}.`
        : `YOLO model unavailable: ${modelStatus.yolo.error || 'model weights are missing.'}`;
    }
    const depthModelStatus = document.querySelector('#depth-model-caption');
    if (depthModelStatus && modelStatus.depth) {
      depthModelStatus.textContent = modelStatus.depth.loaded
        ? `${modelStatus.depth.model || 'Depth Anything V2'} loaded via ${modelStatus.depth.backend || 'model backend'}. Actual inference is reported for each processed frame.`
        : `Depth Anything V2 unavailable: ${modelStatus.depth.error || 'model not loaded.'} Depth stays UNKNOWN; no substitute is used.`;
    }

    const cameraState = values.camera;
    setText('#system-camera', cameraState || 'unavailable');
    if (typeof values.last_frame_available === 'boolean') {
      const frameStatus = document.querySelector('#system-last-frame');
      if (frameStatus) {
        frameStatus.textContent = values.last_frame_available
          ? cameraState === 'connected' ? 'Available · live' : 'Available · retained'
          : 'None available';
      }
      if (!values.last_frame_available && cameraState !== 'connected') {
        detail.textContent = 'Camera is off · no processed frame is available';
        cameraStatus.innerHTML = '<span class="status-dot status-dot-muted" aria-hidden="true"></span>Camera disconnected · no retained frame';
      }
    }
    setText('#system-detection', values.detection);
    setText('#system-tracking', values.tracking);
    setText('#system-depth', values.depth);
    setText('#system-tts', values.tts);
    setText('#system-latency', Number.isFinite(values.processing_ms) ? `${values.processing_ms.toFixed(1)} ms` : '—');
    setText('#system-initialization', Number.isFinite(values.initialization_ms) ? `${values.initialization_ms.toFixed(1)} ms` : '—');
    setText('#system-timing-source', values.timing_source || 'not reported');

    const connectionPill = document.querySelector('.dashboard-head-status .connection-pill span:last-child');
    if (connectionPill) connectionPill.textContent = values.backend === 'connected' ? 'Python backend connected' : 'Python backend disconnected';
    const sidebarStatus = document.querySelector('.dashboard-sidebar-status strong');
    if (sidebarStatus) sidebarStatus.textContent = values.backend === 'connected' ? 'Backend connected' : 'Backend disconnected';
    const sidebarDetail = document.querySelector('.dashboard-sidebar-status small');
    if (sidebarDetail) {
      sidebarDetail.textContent = cameraState === 'connected'
        ? 'Camera connected'
        : values.last_frame_available ? 'Camera off · last frame retained' : 'Camera disconnected';
    }
    const overall = document.querySelector('.backend-state');
    if (overall) {
      overall.textContent = values.backend === 'connected'
        ? 'Live backend status'
        : 'Backend disconnected';
    }
    if (cameraState) {
      cameraStatus.innerHTML = cameraState === 'connected'
        ? '<span class="status-dot" aria-hidden="true"></span>Camera connected'
        : `<span class="status-dot status-dot-muted" aria-hidden="true"></span>${values.last_frame_available ? 'Camera disconnected · last frame retained' : 'Camera disconnected'}`;
    }
  }

  function replaceRows(selector, rows, values, emptyText) {
    const body = document.querySelector(selector);
    if (!body) return;
    body.replaceChildren();
    if (!rows.length) {
      const row = document.createElement('tr');
      row.className = 'empty-row';
      const cell = document.createElement('td');
      cell.colSpan = values.length;
      cell.textContent = emptyText;
      row.append(cell);
      body.append(row);
      return;
    }
    for (const item of rows) {
      const row = document.createElement('tr');
      for (const value of values(item)) {
        const cell = document.createElement('td');
        cell.textContent = String(value ?? '—');
        row.append(cell);
      }
      body.append(row);
    }
  }

  function renderQuery(query, reasoning, narration) {
    const badge = document.querySelector('#query .data-tag');
    const strong = document.querySelector('.query-empty strong');
    const description = document.querySelector('.query-empty p');
    const object = reasoning?.objects?.[0];
    if (badge) badge.textContent = query ? 'Answered' : 'No query received';
    if (strong) strong.textContent = narration || 'Waiting for a question';
    if (description) description.textContent = query ? `“${query}”` : 'Query, target position, relative depth category, and spoken answer will appear here.';
    setText('.query-result-list div:nth-child(1) dd', object?.class_name || '—');
    setText('.query-result-list div:nth-child(2) dd', object ? formatPosition(object) : '—');
    setText('.query-result-list div:nth-child(3) dd', object ? formatRelativeDepth(object.relative_depth_value) : '—');
  }

  function renderNarration(text, shouldSpeak) {
    const narrationText = document.querySelector('#narration-text');
    const narrationStatus = document.querySelector('#narration-status');
    const available = typeof text === 'string' && text.trim().length > 0;
    latestNarration = available ? text.trim() : '';
    if (narrationText) narrationText.textContent = available
      ? latestNarration
      : 'No narration text was returned for the latest processed frame.';
    const speakButton = document.querySelector('#narration-speak');
    if (speakButton) speakButton.disabled = !available;
    if (narrationStatus) narrationStatus.textContent = !available
      ? 'No narration text'
      : shouldSpeak === false ? 'Scene unchanged · narration suppressed' : 'Scene changed · text ready';
  }

  function drawDepthSelection() {
    const image = document.querySelector('#depth-preview');
    const canvas = document.querySelector('#depth-selection-overlay');
    const box = selectedDetection?.bounding_box;
    if (!image || !canvas || image.hidden || !box || !Array.isArray(box)) {
      if (canvas) canvas.hidden = true;
      return;
    }
    const draw = () => {
      if (!image.naturalWidth || !image.naturalHeight) return;
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const context = canvas.getContext('2d');
      if (!context) return;
      const [x1, y1, x2, y2] = box.map(Number);
      if (![x1, y1, x2, y2].every(Number.isFinite)) return;
      context.fillStyle = 'rgba(255,255,255,.2)';
      context.fillRect(x1, y1, x2 - x1, y2 - y1);
      context.lineWidth = Math.max(2, image.naturalWidth / 300);
      context.strokeStyle = '#fff';
      context.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const sampleWidth = (x2 - x1) * 0.5;
      const sampleHeight = (y2 - y1) * 0.5;
      const sampleX = x1 + sampleWidth * 0.5;
      const sampleY = y1 + sampleHeight * 0.5;
      context.fillStyle = 'rgba(255,255,255,.4)';
      context.fillRect(sampleX, sampleY, sampleWidth, sampleHeight);
      canvas.hidden = false;
    };
    if (image.complete) draw();
    else image.addEventListener('load', draw, { once: true });
  }

  function renderMemory(message) {
    const list = document.querySelector('#scene-memory-list');
    const frameBadge = document.querySelector('#memory-frame');
    if (!list) return;
    const memory = Array.isArray(message.scene_memory) ? message.scene_memory : [];
    if (frameBadge) frameBadge.textContent = `Frame ${message.frame_number ?? '—'} · ${memory.length} remembered`;
    list.replaceChildren();
    if (!memory.length) {
      const item = document.createElement('li');
      item.textContent = 'No objects are currently remembered.';
      list.append(item);
    } else {
      for (const object of memory) {
        const item = document.createElement('li');
        const age = object.missed_frames ? ` · missed ${object.missed_frames} frame(s)` : ' · visible now';
        item.textContent = `${object.class_name || object.label || 'Object'}${object.track_id == null ? '' : ` #${object.track_id}`} · ${formatPosition(object)} · ${object.depth_category || 'UNKNOWN'}${age} · last seen frame ${object.last_seen_frame ?? '—'}`;
        list.append(item);
      }
    }
    const changes = document.querySelector('#scene-change-list');
    if (changes) {
      changes.replaceChildren();
      const events = Array.isArray(message.track_events) ? message.track_events : [];
      const sceneChanges = Array.isArray(message.scene_changes) ? message.scene_changes : [];
      const combined = [...events, ...sceneChanges].slice(-8);
      if (!combined.length) {
        const item = document.createElement('li');
        item.textContent = 'No new, moved, lost, or reacquired objects reported yet.';
        changes.append(item);
      } else {
        for (const change of combined) {
          const item = document.createElement('li');
          item.textContent = `${change.state || change.type || 'update'}${change.track_id == null ? '' : ` · track #${change.track_id}`}${change.label ? ` · ${change.label}` : ''}${change.from_position ? ` · ${change.from_position} → ${change.to_position}` : ''}`;
          changes.append(item);
        }
      }
    }
  }

  function renderGraph(graph) {
    const svg = document.querySelector('#scene-graph-svg');
    const empty = document.querySelector('#scene-graph-empty');
    const inspector = document.querySelector('#graph-selection');
    const relationList = document.querySelector('#graph-relationships');
    const badge = document.querySelector('#scene-graph .data-tag');
    if (!svg || !empty || !inspector || !relationList || !badge) return;

    const nodes = Array.isArray(graph?.nodes) ? graph.nodes : [];
    const edges = Array.isArray(graph?.edges) ? graph.edges : [];
    const nodeIds = new Set(nodes.map((node) => String(node.object_id ?? node.id ?? '')));
    const positionedNodes = new Map();
    const centerX = 300;
    const centerY = 104;
    const radius = Math.min(175, 42 + nodes.length * 12);
    nodes.forEach((node, index) => {
      const angle = nodes.length === 1 ? 0 : (Math.PI * 2 * index / nodes.length) - Math.PI / 2;
      positionedNodes.set(String(node.object_id ?? node.id ?? ''), {
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * Math.min(62, radius * 0.55),
      });
    });
    if (!nodes.some((node) => String(node.object_id ?? node.id ?? '') === selectedGraphNodeId)) {
      selectedGraphNodeId = String(nodes[0]?.object_id ?? nodes[0]?.id ?? '');
    }

    svg.replaceChildren();
    relationList.replaceChildren();
    empty.hidden = nodes.length > 0;
    badge.textContent = `${nodes.length} nodes · ${edges.length} edges`;
    if (!nodes.length) {
      selectedGraphNodeId = null;
      inspector.textContent = 'No scene-graph nodes were returned for the latest frame.';
      return;
    }

    const edgeLayer = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    for (const edge of edges) {
      const sourceId = String(edge.source ?? '');
      const targetId = String(edge.target ?? '');
      const source = positionedNodes.get(sourceId);
      const target = positionedNodes.get(targetId);
      if (!source || !target || !nodeIds.has(sourceId) || !nodeIds.has(targetId)) continue;
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('class', 'graph-edge');
      line.setAttribute('x1', String(source.x));
      line.setAttribute('y1', String(source.y));
      line.setAttribute('x2', String(target.x));
      line.setAttribute('y2', String(target.y));
      edgeLayer.append(line);
      const relation = edge.relationship || edge.relation;
      if (typeof relation === 'string' && relation) {
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('class', 'graph-edge-label');
        label.setAttribute('x', String((source.x + target.x) / 2));
        label.setAttribute('y', String((source.y + target.y) / 2 - 5));
        label.setAttribute('text-anchor', 'middle');
        label.textContent = relation.replaceAll('_', ' ');
        edgeLayer.append(label);
      }
    }
    svg.append(edgeLayer);

    for (const [index, node] of nodes.entries()) {
      const nodeId = String(node.object_id ?? node.id ?? '');
      if (!nodeId) continue;
      const position = positionedNodes.get(nodeId);
      const label = String(node.class_name || node.model_class_name || node.label || 'Object');
      const track = node.track_id === null || node.track_id === undefined ? '' : ` #${node.track_id}`;
      const displayLabel = `${label.slice(0, 16)}${track}`;
      const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      group.setAttribute('class', 'graph-object-node');
      group.setAttribute('data-node-id', nodeId);
      group.setAttribute('tabindex', '0');
      group.setAttribute('role', 'button');
      group.setAttribute('aria-label', `${label}${track}; select to inspect`);
      if (nodeId === selectedGraphNodeId) group.classList.add('is-selected');
      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('x', String(position.x - 52));
      rect.setAttribute('y', String(position.y - 18));
      rect.setAttribute('width', '104');
      rect.setAttribute('height', '36');
      rect.setAttribute('rx', '10');
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', String(position.x));
      text.setAttribute('y', String(position.y + 4));
      text.setAttribute('text-anchor', 'middle');
      text.textContent = displayLabel;
      group.append(rect, text);
      const selectNode = () => {
        selectedGraphNodeId = nodeId;
        const selectedObject = currentDetections.find((object) =>
          String(object.track_id ?? object.id) === String(node.track_id ?? node.object_id ?? node.id),
        );
        if (selectedObject) {
          selectedDetection = selectedObject;
          populateCorrectionFields(selectedObject);
          renderReviewDetections();
          drawReviewFrame(currentReviewFrame?.dataUrl, currentReviewFrame?.size, currentDetections);
          drawDepthSelection();
        }
        renderGraph(graph);
      };
      group.addEventListener('click', selectNode);
      group.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          selectNode();
        }
      });
      svg.append(group);
      if (index === 0 && !nodes.some((item) => String(item.object_id ?? item.id ?? '') === selectedGraphNodeId)) {
        selectedGraphNodeId = nodeId;
      }
    }

    const selected = nodes.find((node) => String(node.object_id ?? node.id ?? '') === selectedGraphNodeId) || nodes[0];
    if (selected) {
      selectedGraphNodeId = String(selected.object_id ?? selected.id ?? '');
      const selectedName = selected.class_name || selected.model_class_name || selected.label || 'Object';
      const track = selected.track_id === null || selected.track_id === undefined ? 'not tracked' : `track #${selected.track_id}`;
      const confidence = Number.isFinite(Number(selected.confidence)) ? `${(Number(selected.confidence) * 100).toFixed(1)}% confidence` : 'confidence unavailable';
      inspector.textContent = `${selectedName} · ${track} · ${confidence} · position ${formatPosition(selected)} · relative depth ${formatRelativeDepth(selected.relative_depth_value)}`;
      for (const edge of edges.filter((item) => String(item.source ?? '') === selectedGraphNodeId || String(item.target ?? '') === selectedGraphNodeId)) {
        const item = document.createElement('li');
        const source = nodes.find((node) => String(node.object_id ?? node.id ?? '') === String(edge.source ?? ''));
        const target = nodes.find((node) => String(node.object_id ?? node.id ?? '') === String(edge.target ?? ''));
        item.textContent = `${source?.class_name || source?.label || edge.source} ${edge.relationship || edge.relation || 'relates to'} ${target?.class_name || target?.label || edge.target}`;
        relationList.append(item);
      }
    }
  }

  function renderFrame(message) {
    const objects = Array.isArray(message.objects) ? message.objects : [];
    const detections = Array.isArray(message.detections) ? message.detections : objects;
    const tracks = Array.isArray(message.tracks)
      ? message.tracks
      : objects.filter((object) => object.track_id !== null && object.track_id !== undefined);

    refreshReviewScene(message);
    window.BYAKUFrameProcessor.drawOverlay(overlay, message.frame_size, objects);
    const cameraConnected = message.stage_status?.camera === 'connected';
    detail.textContent = `${cameraConnected ? 'Live frame' : 'Retained frame'} ${message.frame_number ?? ''} · ${detections.length} detections · ${tracks.length} tracked · ${message.processing_ms ?? '—'} ms total`;
    cameraStatus.innerHTML = cameraConnected
      ? '<span class="status-dot" aria-hidden="true"></span>Camera connected · processing'
      : '<span class="status-dot status-dot-muted" aria-hidden="true"></span>Camera disconnected · last frame available';
    const cameraError = document.querySelector('#dash-camera-error');
    if (cameraError) cameraError.textContent = message.depth_error
      ? `Frame processed. Depth Anything V2 did not produce a usable depth map: ${message.depth_error}`
      : `Frame ${message.frame_number ?? '—'} processed by the loaded models. Review uses persistent, session-scoped feedback.`;
    updatePipeline({
      ...(message.stage_status || {}),
      backend: 'connected',
      last_frame_available: true,
      processing_ms: message.processing_ms,
      initialization_ms: message.initialization_ms,
      stage_timings_ms: message.stage_timings_ms,
    });
    renderNarration(
      message.scene_narration
      || message.scene_summary_speech?.spoken_text
      || message.scene_description?.summary
      || message.narration,
      message.should_speak,
    );
    renderMemory(message);

    const count = document.querySelector('#detection-count');
    if (count) count.textContent = `${detections.length} detection${detections.length === 1 ? '' : 's'} · ${tracks.length} tracked`;
    replaceRows('#detection-rows', objects, (item) => [
      `${item.class_name || item.label || 'Unknown'} · ${Number.isFinite(Number(item.confidence)) ? `${(Number(item.confidence) * 100).toFixed(1)}%` : '—'}`,
      item.track_id !== null && item.track_id !== undefined ? `#${item.track_id} · ${item.track_status || 'tracked'}` : 'untracked',
      item.movement_direction === 'unknown'
        ? 'Not enough frames'
        : `${item.movement_direction || 'unknown'}${item.velocity_px_per_frame ? ` · ${item.velocity_px_per_frame.x}, ${item.velocity_px_per_frame.y} px/frame` : ''}`,
      formatPosition(item),
      item.depth_category || 'UNKNOWN',
      Array.isArray(item.bounding_box) ? item.bounding_box.map((value) => Math.round(value)).join(', ') : '—',
    ], 'No detections received');
    const trackerDetail = document.querySelector('#tracking');
    if (trackerDetail) {
      const eventSummary = (message.track_events || [])
        .map((event) => `${event.state}${event.track_id == null ? '' : ` #${event.track_id}`}`)
        .join(', ');
      trackerDetail.textContent = `${tracks.length} active tracked objects · ${eventSummary || 'no lifecycle changes'} · movement values are image pixels per sampled frame, not physical speed.`;
    }

    const preview = document.querySelector('#depth-preview');
    const depthStatus = document.querySelector('#depth-status');
    const depthMessage = document.querySelector('#depth-message');
    const sourcePreview = document.querySelector('#depth-source-preview');
    const sourceEmpty = document.querySelector('.depth-image-empty');
    latestDepthMessage = message;
    if (currentReviewFrame?.dataUrl && sourcePreview) {
      sourcePreview.src = currentReviewFrame.dataUrl;
      sourcePreview.hidden = false;
      if (sourceEmpty) sourceEmpty.hidden = true;
    }
    if (message.depth_preview && preview) {
      preview.src = `data:image/jpeg;base64,${message.depth_preview}`;
      preview.hidden = false;
      if (depthStatus) depthStatus.textContent = `Inference succeeded · ${message.depth_inference_ms ?? '—'} ms`;
      if (depthMessage) depthMessage.hidden = true;
      drawDepthSelection();
    } else {
      preview.hidden = true;
      const overlayCanvas = document.querySelector('#depth-selection-overlay');
      if (overlayCanvas) overlayCanvas.hidden = true;
      if (depthStatus) depthStatus.textContent = message.depth_model?.loaded ? 'Inference unavailable' : 'Model unavailable';
      if (depthMessage) {
        depthMessage.hidden = false;
        depthMessage.textContent = message.depth_error || message.depth_model?.error || 'No depth output was returned. Object categories remain UNKNOWN.';
      }
    }
    const depthModelCaption = document.querySelector('#depth-model-caption');
    if (depthModelCaption && message.depth_model) {
      depthModelCaption.textContent = message.depth_model.loaded
        ? `${message.depth_model.model || 'Depth Anything V2'} · ${message.depth_model.backend || 'inference backend'} · ${message.depth_inference_ms ?? '—'} ms on this frame.`
        : `Model unavailable: ${message.depth_model.error || 'pretrained model could not be loaded.'}`;
    }

    renderGraph(message.graph);
  }

  function renderQueryResult(message) {
    const question = document.querySelector('#query-recognized');
    const answer = document.querySelector('#query-answer');
    const evidence = document.querySelector('#query-evidence');
    if (question) question.textContent = `Recognized question: “${message.text || ''}”`;
    if (answer) answer.textContent = message.answer || 'No answer was returned.';
    const askButton = document.querySelector('#scene-question-form button[type="submit"]');
    if (askButton) askButton.disabled = !liveSessionId;
    const speakAnswer = document.querySelector('#query-speak-answer');
    if (speakAnswer) speakAnswer.disabled = !message.answer;
    if (evidence) {
      evidence.replaceChildren();
      const objects = Array.isArray(message.scene_memory) ? message.scene_memory : [];
      if (!objects.length) {
        const item = document.createElement('li');
        item.textContent = `Frame ${message.frame_number ?? '—'}: no objects in the active session memory.`;
        evidence.append(item);
      } else {
        for (const object of objects.slice(0, 8)) {
          const item = document.createElement('li');
          item.textContent = `${object.class_name || object.label || 'Object'} · ${formatPosition(object)} · ${object.depth_category || 'UNKNOWN'} · ${object.status || 'remembered'}`;
          evidence.append(item);
        }
      }
    }
  }

  async function refreshStatus() {
    try {
      const response = await fetch('/api/status', { cache: 'no-store' });
      if (!response.ok) throw new Error(`Backend status returned ${response.status}`);
      const status = await response.json();
      updatePipeline(status);
      if (status.last_frame) {
        renderFrame({
          ...status.last_frame,
          frame_preview: status.last_frame_preview,
          last_frame_available: status.last_frame_available,
        });
      }
      if (status.last_query) renderQuery(status.last_query, status.last_reasoning, status.last_narration);
    } catch (error) {
      updatePipeline({ backend: 'disconnected', camera: 'disconnected' });
      const cameraError = document.querySelector('#dash-camera-error');
      if (cameraError) cameraError.textContent = `Backend status check failed: ${error.message}`;
    }
  }

  connection.addEventListener('statechange', (event) => {
    const state = event.detail?.state;
    const cameraError = document.querySelector('#dash-camera-error');
    if (state === 'idle' || state === 'error') {
      if (liveSessionId && cameraError) cameraError.textContent = 'Backend connection lost. Reconnecting automatically; frame capture will resume when the connection returns.';
      if (state === 'error') cameraStatus.innerHTML = '<span class="status-dot" aria-hidden="true"></span>Backend connection error';
    } else if (state === 'connected' && cameraError && frameProcessor) {
      cameraError.textContent = 'Backend connected. Waiting for the next live frame.';
    }
  });

  async function startCamera() {
    try {
      await window.BYAKUCamera.start(video, container, placeholder, cameraStatus, detail, null, button);
      await connection.connect();
      frameProcessor = new window.BYAKUFrameProcessor(connection, video);
      frameProcessor.addEventListener('result', (event) => {
        const message = event.detail;
        if (message.type === 'camera_started') {
          liveSessionId = message.session_id || liveSessionId;
          reviewSessionId = liveSessionId || reviewSessionId;
          if (liveSessionId) loadReviewHistory().catch((error) => setReviewStatus(`Could not load this session's review history: ${error.message}`, true));
          const askButton = document.querySelector('#scene-question-form button[type="submit"]');
          if (askButton) askButton.disabled = false;
          cameraStatus.innerHTML = '<span class="status-dot" aria-hidden="true"></span>Camera connected';
        } else if (message.type === 'camera_stopped') {
          liveSessionId = null;
          const askButton = document.querySelector('#scene-question-form button[type="submit"]');
          if (askButton) askButton.disabled = true;
        } else if (message.type === 'frame_result') renderFrame(message);
        else if (message.type === 'query_result') renderQueryResult(message);
        else if (message.type === 'error') {
          const messageText = `${message.stage || 'backend'}: ${message.message || 'The request failed.'}`;
          detail.textContent = messageText;
          const cameraError = document.querySelector('#dash-camera-error');
          if (cameraError) cameraError.textContent = `${messageText}${message.recoverable ? ' Retrying on the next captured frame.' : ''}`;
          cameraStatus.innerHTML = '<span class="status-dot" aria-hidden="true"></span>Processing error';
        }
      });
      frameProcessor.start();
      button.textContent = 'Stop live demo';
      button.disabled = false;
      pauseButton.disabled = false;
      resumeButton.disabled = true;
    } catch (error) {
      const message = error.message || 'Python backend unavailable';
      detail.textContent = message;
      const cameraError = document.querySelector('#dash-camera-error');
      if (cameraError) cameraError.textContent = `Live demonstration could not start: ${message}`;
      cameraStatus.innerHTML = '<span class="status-dot" aria-hidden="true"></span>Backend unavailable';
      button.textContent = 'Start live demo';
      frameProcessor?.stop();
      frameProcessor = null;
      connection.close();
      if (window.BYAKUCamera.stream) {
        window.BYAKUCamera.stop(video, container, placeholder, cameraStatus, detail, null, button);
        button.textContent = 'Start live demo';
      }
    }
  }

  button.addEventListener('click', async () => {
    if (frameProcessor) {
      frameProcessor.stop();
      frameProcessor = null;
      connection.close();
      window.BYAKUCamera.stop(video, container, placeholder, cameraStatus, detail, null, button);
      liveSessionId = null;
      pauseButton.disabled = true;
      resumeButton.disabled = true;
      button.textContent = 'Start live demo';
      const askButton = document.querySelector('#scene-question-form button[type="submit"]');
      if (askButton) askButton.disabled = true;
      loadReviewHistory().catch((error) => setReviewStatus(`Could not refresh review history: ${error.message}`, true));
      updatePipeline({ backend: 'disconnected', camera: 'disconnected' });
      return;
    }
    await startCamera();
  });

  pauseButton?.addEventListener('click', () => {
    frameProcessor?.pause();
    pauseButton.disabled = true;
    resumeButton.disabled = false;
    cameraStatus.innerHTML = '<span class="status-dot status-dot-muted" aria-hidden="true"></span>Camera connected · paused';
    detail.textContent = 'Live preview retained · frame capture paused';
  });
  resumeButton?.addEventListener('click', () => {
    frameProcessor?.resume();
    pauseButton.disabled = false;
    resumeButton.disabled = true;
    cameraStatus.innerHTML = '<span class="status-dot" aria-hidden="true"></span>Camera connected · resuming';
  });

  document.querySelector('#review-correction-form')?.addEventListener('submit', submitCorrection);
  document.querySelector('#review-confirm')?.addEventListener('click', () => {
    if (selectedDetection) saveReview('detection_review', selectedDetection.class_name, 'confirmed');
  });
  document.querySelector('#review-incorrect')?.addEventListener('click', () => {
    if (!selectedDetection) return;
    const form = document.querySelector('#review-correction-form');
    populateCorrectionFields(selectedDetection);
    form.dataset.open = 'true';
    form.hidden = false;
    document.querySelector('#review-corrected-label').focus();
    setReviewStatus('Enter the correct object name. Nothing is saved until you confirm Save Correction.');
  });
  document.querySelector('#review-false-positive')?.addEventListener('click', () => {
    if (selectedDetection) saveReview('false_positive', null, 'rejected');
  });
  document.querySelector('#review-bbox-edit')?.addEventListener('click', () => {
    if (!selectedDetection) return;
    populateCorrectionFields(selectedDetection);
    const form = document.querySelector('#review-correction-form');
    form.dataset.open = 'true';
    form.hidden = false;
    document.querySelector('#review-corrected-label').focus();
    setReviewStatus('Adjust the object class and/or box coordinates, then save the correction.');
  });
  document.querySelector('#review-missed-start')?.addEventListener('click', () => {
    if (!currentReviewFrame?.dataUrl) {
      setReviewStatus('A current camera frame is required to mark a missed object.', true);
      return;
    }
    missedBBox = null;
    missedDrawMode = true;
    document.querySelector('#review-correction-form').hidden = true;
    document.querySelector('#review-correction-form').dataset.open = 'false';
    document.querySelector('#review-missed-form').hidden = false;
    document.querySelector('#review-missed-label').focus();
    document.querySelector('#review-missed-box-status').textContent = 'Drag over the missed object in the frame to mark its box.';
    reviewCanvas.classList.add('is-drawing-box');
    setReviewStatus('Drag on the frame to mark the missed object bounding box.');
  });
  reviewCanvas?.addEventListener('pointerdown', (event) => {
    if (!missedDrawMode || !reviewCanvas.width || !reviewCanvas.height) return;
    const rect = reviewCanvas.getBoundingClientRect();
    missedDragStart = {
      x: (event.clientX - rect.left) * reviewCanvas.width / rect.width,
      y: (event.clientY - rect.top) * reviewCanvas.height / rect.height,
    };
    reviewCanvas.setPointerCapture(event.pointerId);
    event.preventDefault();
  });
  reviewCanvas?.addEventListener('pointermove', (event) => {
    if (!missedDrawMode || !missedDragStart) return;
    const rect = reviewCanvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * reviewCanvas.width / rect.width;
    const y = (event.clientY - rect.top) * reviewCanvas.height / rect.height;
    missedBBox = {
      x1: Math.max(0, Math.min(missedDragStart.x, x)),
      y1: Math.max(0, Math.min(missedDragStart.y, y)),
      x2: Math.min(reviewCanvas.width, Math.max(missedDragStart.x, x)),
      y2: Math.min(reviewCanvas.height, Math.max(missedDragStart.y, y)),
    };
    drawReviewFrame(currentReviewFrame?.dataUrl, currentReviewFrame?.size, currentDetections);
  });
  reviewCanvas?.addEventListener('pointerup', () => {
    if (!missedDrawMode || !missedBBox) return;
    missedDragStart = null;
    const valid = missedBBox.x2 - missedBBox.x1 >= 10 && missedBBox.y2 - missedBBox.y1 >= 10;
    if (!valid) missedBBox = null;
    document.querySelector('#review-missed-box-status').textContent = valid
      ? `Region: ${missedBBox.x1.toFixed(0)}, ${missedBBox.y1.toFixed(0)} to ${missedBBox.x2.toFixed(0)}, ${missedBBox.y2.toFixed(0)}`
      : 'Box must be at least 10 × 10 pixels. Drag a larger region.';
    drawReviewFrame(currentReviewFrame?.dataUrl, currentReviewFrame?.size, currentDetections);
  });
  document.querySelector('#review-missed-form')?.addEventListener('submit', async (event) => {
    event.preventDefault();
    const label = document.querySelector('#review-missed-label').value.trim();
    if (!missedBBox) {
      setReviewStatus('Draw a bounding box around the missed object before saving.', true);
      return;
    }
    if (!label) {
      setReviewStatus('Enter the missed object class before saving.', true);
      return;
    }
    const saved = await saveReview('missed_object', label, 'corrected', {
      detection: null,
      bbox: missedBBox,
    });
    if (!saved) return;
    missedDrawMode = false;
    missedBBox = null;
    reviewCanvas.classList.remove('is-drawing-box');
    document.querySelector('#review-missed-form').reset();
    document.querySelector('#review-missed-form').hidden = true;
    drawReviewFrame(currentReviewFrame?.dataUrl, currentReviewFrame?.size, currentDetections);
  });
  document.querySelector('#review-missed-cancel')?.addEventListener('click', () => {
    missedDrawMode = false;
    missedDragStart = null;
    missedBBox = null;
    reviewCanvas.classList.remove('is-drawing-box');
    document.querySelector('#review-missed-form').reset();
    document.querySelector('#review-missed-form').hidden = true;
    drawReviewFrame(currentReviewFrame?.dataUrl, currentReviewFrame?.size, currentDetections);
    setReviewStatus('Missed-object review cancelled; no record was saved.');
  });
  document.querySelector('#review-cancel')?.addEventListener('click', () => {
    const form = document.querySelector('#review-correction-form');
    form.dataset.open = 'false';
    form.reset();
    renderReviewDetections();
    setReviewStatus('Correction cancelled; no review record was saved.');
  });
  document.querySelector('#review-history-refresh')?.addEventListener('click', () => {
    loadReviewHistory().catch((error) => setReviewStatus(`Could not load review history: ${error.message}`, true));
  });
  document.querySelector('#scene-question-form')?.addEventListener('submit', (event) => {
    event.preventDefault();
    const input = document.querySelector('#scene-question');
    const question = input.value.trim();
    if (!question) return;
    if (!liveSessionId || !frameProcessor) {
      document.querySelector('#query-answer').textContent = 'Start the live demonstration to ask a question grounded in its scene memory.';
      return;
    }
    document.querySelector('#query-recognized').textContent = `Recognized question: “${question}”`;
    document.querySelector('#query-answer').textContent = 'Checking the active session scene memory…';
    const askButton = document.querySelector('#scene-question-form button[type="submit"]');
    askButton.disabled = true;
    if (!connection.send({ type: 'query', text: question })) {
      askButton.disabled = false;
      document.querySelector('#query-answer').textContent = 'Question was not sent: the backend WebSocket is disconnected.';
    }
  });
  document.querySelector('#scene-question-voice')?.addEventListener('click', () => {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) {
      document.querySelector('#query-answer').textContent = 'Voice recognition is unavailable in this browser. Type your question instead.';
      return;
    }
    const recognition = new Recognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    const voiceButton = document.querySelector('#scene-question-voice');
    voiceButton.disabled = true;
    document.querySelector('#query-recognized').textContent = 'Listening for a question…';
    recognition.addEventListener('result', (event) => {
      const transcript = event.results?.[0]?.[0]?.transcript?.trim();
      if (!transcript) {
        document.querySelector('#query-answer').textContent = 'No spoken question was recognized.';
        return;
      }
      document.querySelector('#scene-question').value = transcript;
      document.querySelector('#scene-question-form').requestSubmit();
    });
    recognition.addEventListener('error', (event) => {
      document.querySelector('#query-answer').textContent = `Voice recognition failed: ${event.error || 'unknown error'}. You can type the question instead.`;
    });
    recognition.addEventListener('end', () => { voiceButton.disabled = false; });
    try {
      recognition.start();
    } catch (error) {
      voiceButton.disabled = false;
      document.querySelector('#query-answer').textContent = `Voice recognition could not start: ${error.message}`;
    }
  });
  document.querySelector('#narration-speak')?.addEventListener('click', () => speechController.speak(latestNarration));
  document.querySelector('#query-speak-answer')?.addEventListener('click', () => {
    const answer = document.querySelector('#query-answer')?.textContent;
    speechController.speak(answer);
  });
  document.querySelector('#narration-stop')?.addEventListener('click', () => speechController.cancel());

  document.querySelectorAll('.nav-item').forEach((link) => {
    link.addEventListener('click', () => {
      document.querySelector('.nav-item.is-current')?.classList.remove('is-current');
      link.classList.add('is-current');
      document.querySelector('.dashboard-sidebar')?.classList.remove('is-menu-open');
      document.querySelector('#workflow-menu-toggle')?.setAttribute('aria-expanded', 'false');
    });
  });

  const menuToggle = document.querySelector('#workflow-menu-toggle');
  menuToggle?.addEventListener('click', () => {
    const expanded = menuToggle.getAttribute('aria-expanded') === 'true';
    menuToggle.setAttribute('aria-expanded', String(!expanded));
    menuToggle.setAttribute('aria-label', expanded ? 'Open project navigation' : 'Close project navigation');
    document.querySelector('.dashboard-sidebar')?.classList.toggle('is-menu-open', !expanded);
  });

  refreshStatus();
  const browserSpeechStatus = document.querySelector('#system-browser-speech');
  if (browserSpeechStatus) {
    browserSpeechStatus.textContent = window.speechSynthesis && typeof window.SpeechSynthesisUtterance === 'function'
      ? 'API present · playback not reported'
      : 'API unavailable';
  }
  loadReviewHistory().catch((error) => setReviewStatus(`Could not load review history: ${error.message}`, true));
  window.setInterval(refreshStatus, 5000);
  window.addEventListener('pagehide', () => {
    frameProcessor?.stop();
    connection.close();
    if (window.BYAKUCamera.stream) {
      window.BYAKUCamera.stop(video, container, placeholder, cameraStatus, detail, null, button);
    }
  });
})();