(() => {
  const IMAGE_EXTENSIONS = new Set(['jpg', 'jpeg', 'png']);
  const VIDEO_EXTENSIONS = new Set(['mp4', 'mov', 'avi', 'mkv', 'webm']);
  const IMAGE_MAX_BYTES = 12 * 1024 * 1024;
  const VIDEO_MAX_BYTES = 100 * 1024 * 1024;

  class BYAKUWorkflowUploadController {
    constructor({
      documentRef = document,
      fetchFn = window.fetch.bind(window),
      URLRef = URL,
      SpeechController = window.BYAKUSpeechController,
      FormDataClass = window.FormData,
      windowRef = window,
    } = {}) {
      this.document = documentRef;
      this.fetch = fetchFn;
      this.URL = URLRef;
      this.FormData = FormDataClass;
      this.elements = {
        form: documentRef.querySelector('#workflow-upload-form'),
        file: documentRef.querySelector('#workflow-upload-file'),
        process: documentRef.querySelector('#workflow-upload-process'),
        remove: documentRef.querySelector('#workflow-upload-remove'),
        status: documentRef.querySelector('#workflow-upload-status-text'),
        loading: documentRef.querySelector('#workflow-upload-loading'),
        preview: documentRef.querySelector('#workflow-upload-preview'),
        filename: documentRef.querySelector('#workflow-upload-filename'),
        imagePreview: documentRef.querySelector('#workflow-upload-image-preview'),
        videoPreview: documentRef.querySelector('#workflow-upload-video-preview'),
        results: documentRef.querySelector('#workflow-upload-results'),
        annotated: documentRef.querySelector('#workflow-upload-annotated'),
        description: documentRef.querySelector('#workflow-upload-description'),
        objects: documentRef.querySelector('#workflow-upload-objects'),
        samples: documentRef.querySelector('#workflow-upload-samples'),
        speak: documentRef.querySelector('#workflow-upload-speak'),
      };
      this.selectedFile = null;
      this.previewUrl = null;
      this.description = '';
      this.processing = false;
      this.speech = SpeechController
        ? new SpeechController()
        : { speak: () => false, cancel: () => {} };

      this.elements.file.addEventListener('change', () => this.selectFile(this.elements.file.files?.[0] || null));
      this.elements.remove.addEventListener('click', () => this.clearSelection());
      this.elements.form.addEventListener('submit', (event) => {
        event.preventDefault();
        void this.processSelection();
      });
      this.elements.speak.addEventListener('click', () => {
        if (this.description) this.speech.speak(this.description);
      });
      windowRef.addEventListener('pagehide', () => {
        this.releasePreview();
        this.speech.cancel();
      }, { once: true });
    }

    setStatus(message, { loading = false, error = false } = {}) {
      this.elements.status.textContent = message;
      this.elements.status.dataset.error = String(error);
      this.elements.loading.hidden = !loading;
    }

    releasePreview() {
      if (this.previewUrl) this.URL.revokeObjectURL(this.previewUrl);
      this.previewUrl = null;
      this.elements.imagePreview.removeAttribute('src');
      this.elements.videoPreview.removeAttribute('src');
      this.elements.videoPreview.load?.();
    }

    clearSelection() {
      if (this.processing) return;
      this.selectedFile = null;
      this.releasePreview();
      this.elements.file.value = '';
      this.elements.preview.hidden = true;
      this.elements.remove.hidden = true;
      this.elements.process.disabled = true;
      this.elements.results.hidden = true;
      this.description = '';
      this.setStatus('Select a JPG, JPEG, PNG, MP4, MOV, AVI, MKV, or WebM file. Maximum: 12 MB per image and 100 MB per video.');
    }

    selectFile(file) {
      if (this.processing) return;
      this.releasePreview();
      this.selectedFile = null;
      this.elements.results.hidden = true;
      this.description = '';
      this.elements.speak.disabled = true;
      if (!file) {
        this.clearSelection();
        return;
      }
      const extension = file.name.split('.').pop()?.toLowerCase();
      const isImage = IMAGE_EXTENSIONS.has(extension);
      const isVideo = VIDEO_EXTENSIONS.has(extension);
      if (file.size <= 0) {
        this.setStatus('The selected file is empty.', { error: true });
      } else if (!isImage && !isVideo) {
        this.setStatus('Unsupported format. Choose a JPG, JPEG, PNG, MP4, MOV, AVI, MKV, or WebM file.', { error: true });
      } else if (file.size > (isImage ? IMAGE_MAX_BYTES : VIDEO_MAX_BYTES)) {
        this.setStatus(`The selected ${isImage ? 'image' : 'video'} exceeds the ${isImage ? '12 MB' : '100 MB'} limit.`, { error: true });
      } else {
        this.selectedFile = file;
        this.previewUrl = this.URL.createObjectURL(file);
        this.elements.filename.textContent = file.name;
        this.elements.imagePreview.hidden = !isImage;
        this.elements.videoPreview.hidden = !isVideo;
        if (isImage) this.elements.imagePreview.src = this.previewUrl;
        else this.elements.videoPreview.src = this.previewUrl;
        this.elements.preview.hidden = false;
        this.elements.remove.hidden = false;
        this.elements.process.disabled = false;
        this.setStatus(`${isImage ? 'Image' : 'Video'} selected. Choose Process selected file to start analysis.`);
        return;
      }
      this.elements.preview.hidden = true;
      this.elements.remove.hidden = false;
      this.elements.process.disabled = true;
    }

    async processSelection() {
      if (!this.selectedFile || this.processing) return;
      const extension = this.selectedFile.name.split('.').pop()?.toLowerCase();
      const mediaType = IMAGE_EXTENSIONS.has(extension) ? 'image' : 'video';
      const formData = new this.FormData();
      formData.append('upload', this.selectedFile, this.selectedFile.name);
      this.processing = true;
      this.elements.file.disabled = true;
      this.elements.process.disabled = true;
      this.elements.remove.disabled = true;
      this.elements.results.hidden = true;
      this.setStatus(`Processing ${mediaType} with BYAKU vision models. This may take a while.`, { loading: true });
      try {
        const response = await this.fetch(`/api/upload/${mediaType}`, { method: 'POST', body: formData });
        let payload;
        try {
          payload = await response.json();
        } catch {
          throw new Error('The backend returned an unreadable processing response.');
        }
        if (!payload || typeof payload !== 'object') {
          throw new Error('The backend returned an unexpected processing response.');
        }
        if (!response.ok) {
          throw new Error(payload.message || payload.detail || payload.error || 'The upload could not be processed.');
        }
        this.renderResult(payload);
        const processed = payload.frames_processed ?? (payload.media_type === 'image' ? 1 : 0);
        const failed = payload.frames_failed ?? 0;
        const detCount = payload.detection_count ?? (payload.objects ? payload.objects.length : 0);
        const specificWarnings = (payload.warnings && payload.warnings.length > 0)
          ? ` (${payload.warnings.join(' · ')})`
          : '';
        const statusSummary = payload.media_type === 'video'
          ? `Video analysis complete: ${processed} frame${processed === 1 ? '' : 's'} successfully analyzed${failed > 0 ? `, ${failed} failed` : ''} · ${detCount} detections${specificWarnings}.`
          : `Image analysis complete: ${processed} frame analyzed · ${detCount} detections${specificWarnings}.`;
        this.setStatus(statusSummary, { error: payload.status === 'error' });
      } catch (error) {
        this.elements.results.hidden = true;
        this.setStatus(error.message || 'The upload could not be processed. Try another supported file.', { error: true });
      } finally {
        this.processing = false;
        this.elements.file.disabled = false;
        this.elements.process.disabled = !this.selectedFile;
        this.elements.remove.disabled = false;
      }
    }

    renderResult(result) {
      this.description = typeof result.scene_description === 'string' ? result.scene_description : '';
      this.elements.description.textContent = this.description || 'A scene description is not available for this result.';
      
      this.elements.objects.replaceChildren();
      for (const item of result.objects || []) {
        const row = this.document.createElement('li');
        const track = item.track_id === null || item.track_id === undefined ? 'ID unavailable' : `ID ${item.track_id}`;
        const position = item.horizontal_position || 'position unavailable';
        const depth = item.depth_category || 'UNKNOWN';
        row.textContent = `${item.class_name || 'Object'} · ${track} · ${position} · ${depth}`;
        if (result.media_type === 'image') {
          const reviewItem = this.renderDetectionReviewItem(item, { sample: null, result });
          row.append(reviewItem);
        }
        this.elements.objects.append(row);
      }

      this.elements.samples.replaceChildren();
      const samples = result.samples || [];
      for (const sample of samples) {
          const card = this.document.createElement('div');
          card.className = 'upload-sample-card';
          if (sample.status === 'decode_error') {
            card.classList.add('upload-sample-error');
            const meta = this.document.createElement('p');
            meta.className = 'upload-sample-meta';
            meta.textContent = `Frame ${sample.frame_number ?? '—'}: Decode error · Frame could not be decoded.`;
            const noteErr = this.document.createElement('p');
            noteErr.className = 'upload-sample-error-msg';
            noteErr.textContent = 'Source frame failed to decode and was skipped. Other frames were analyzed normally.';
            card.append(meta, noteErr);
            this.elements.samples.append(card);
            continue;
          }

          let overlayCanvas = null;
          if (typeof sample.annotated_image === 'string' && sample.annotated_image.length > 0) {
            const wrap = this.document.createElement('div');
            wrap.className = 'upload-sample-frame-wrap';
            
            const img = this.document.createElement('img');
            img.className = 'upload-sample-annotated';
            img.alt = `Annotated frame ${sample.frame_number ?? '—'}`;
            const imgSrc = `data:image/jpeg;base64,${sample.annotated_image}`;
            img.src = imgSrc;
            img.title = 'Click to inspect full-size';
            img.addEventListener('click', () => {
              const cap = `Frame ${sample.frame_number ?? '—'}${Number.isFinite(sample.timestamp_seconds) ? ` · ${sample.timestamp_seconds.toFixed(2)}s` : ''} · ${sample.objects?.length || 0} detections`;
              this.openLightbox(imgSrc, cap);
            });
            wrap.append(img);

            overlayCanvas = this.document.createElement('canvas');
            overlayCanvas.className = 'upload-sample-overlay';
            wrap.append(overlayCanvas);

            card.append(wrap);
          }

          const meta = this.document.createElement('p');
          meta.className = 'upload-sample-meta';
          const seconds = Number.isFinite(sample.timestamp_seconds)
            ? ` · ${sample.timestamp_seconds.toFixed(1)} s`
            : '';
          const detected = (sample.objects || []).map((item) => item.class_name).filter(Boolean);
          meta.textContent = `Frame ${sample.frame_number ?? '—'}${seconds}: ${detected.join(', ') || 'no detections'}`;
          card.append(meta);

          const detList = this.document.createElement('ul');
          detList.className = 'upload-sample-detection-list';
          const sampleObjs = sample.objects || [];
          for (const item of sampleObjs) {
            const reviewItem = this.renderDetectionReviewItem(item, {
              sample,
              result,
              onSelect: (selectedObj, selectedEl) => {
                detList.querySelectorAll('.upload-detection-item').forEach((el) => el.classList.remove('is-selected'));
                selectedEl?.classList.add('is-selected');
                this.highlightDetectionBox(overlayCanvas, sample.annotated_size, selectedObj);
              },
            });
            detList.append(reviewItem);
          }
          card.append(detList);

          this.elements.samples.append(card);
        }

      this.elements.samples.hidden = samples.length === 0;
      this.elements.annotated.hidden = result.media_type !== 'image' || typeof result.annotated_image !== 'string';
      if (!this.elements.annotated.hidden) {
        const imgSrc = `data:image/jpeg;base64,${result.annotated_image}`;
        this.elements.annotated.src = imgSrc;
        this.elements.annotated.title = 'Click to inspect full-size';
        this.elements.annotated.onclick = () => {
          this.openLightbox(imgSrc, `Uploaded Image · ${result.objects?.length || 0} detections`);
        };
      } else {
        this.elements.annotated.removeAttribute('src');
        this.elements.annotated.onclick = null;
      }
      this.elements.speak.disabled = !this.description;
      this.elements.results.hidden = false;
    }

    renderDetectionReviewItem(item, { sample, result, onSelect }) {
      const row = this.document.createElement('li');
      row.className = 'upload-detection-item';

      const header = this.document.createElement('div');
      header.className = 'upload-detection-header';

      const title = this.document.createElement('span');
      title.className = 'upload-detection-title';
      const track = item.track_id !== null && item.track_id !== undefined ? ` [ID: #${item.track_id}]` : (result.media_type === 'video' ? ' [Untracked]' : '');
      const conf = Number.isFinite(item.confidence) ? ` (${(item.confidence * 100).toFixed(0)}%)` : '';
      const pos = item.horizontal_position || 'UNKNOWN';
      const depth = item.depth_category || 'UNKNOWN';
      title.textContent = `${item.class_name || 'Object'}${conf}${track} · ${pos} · ${depth}`;

      const badge = this.document.createElement('span');
      badge.className = 'review-badge review-badge-pending';
      badge.textContent = 'Pending';

      header.append(title, badge);
      row.append(header);

      const actions = this.document.createElement('div');
      actions.className = 'upload-detection-actions';

      const btnConfirm = this.document.createElement('button');
      btnConfirm.type = 'button';
      btnConfirm.className = 'upload-review-btn btn-confirm';
      btnConfirm.textContent = '✓ Correct';

      const btnCorrect = this.document.createElement('button');
      btnCorrect.type = 'button';
      btnCorrect.className = 'upload-review-btn btn-correct';
      btnCorrect.textContent = '✎ Rename';

      actions.append(btnConfirm, btnCorrect);
      row.append(actions);

      const form = this.document.createElement('form');
      form.className = 'upload-correction-form';
      form.hidden = true;
      const input = this.document.createElement('input');
      input.type = 'text';
      input.className = 'upload-correction-input';
      input.value = item.class_name || '';
      input.placeholder = 'Correct object class';
      input.required = true;
      const btnSave = this.document.createElement('button');
      btnSave.type = 'submit';
      btnSave.className = 'upload-review-btn btn-save-correction';
      btnSave.textContent = 'Save';
      const btnCancel = this.document.createElement('button');
      btnCancel.type = 'button';
      btnCancel.className = 'upload-review-btn btn-cancel-correction';
      btnCancel.textContent = 'Cancel';
      form.append(input, btnSave, btnCancel);
      row.append(form);

      row.addEventListener('click', (e) => {
        if (e.target.tagName === 'BUTTON' || e.target.tagName === 'INPUT') return;
        if (typeof onSelect === 'function') onSelect(item, row);
      });

      btnConfirm.addEventListener('click', async (e) => {
        e.stopPropagation();
        btnConfirm.disabled = true;
        btnCorrect.disabled = true;
        try {
          await this.submitReview({
            source: result.media_type === 'video' ? 'upload_video' : 'upload_image',
            session_id: sample?.session_id || result.session_id,
            frame_number: sample?.frame_number ?? 1,
            frame_id: `${result.session_id}:frame-${sample?.frame_number ?? 1}`,
            review_frame_token: sample?.review_frame_token || result.review_frame_token,
            detection_id: item.detection_id,
            track_id: item.track_id ?? null,
            original_prediction: {
              class_name: item.class_name,
              confidence: item.confidence,
              horizontal_position: item.horizontal_position,
              depth_category: item.depth_category,
              bounding_box: item.bounding_box,
            },
            correction_type: 'confirm',
            review_status: 'confirmed',
          });
          badge.className = 'review-badge review-badge-accepted';
          badge.textContent = 'Accepted';
          actions.hidden = true;
        } catch (err) {
          btnConfirm.disabled = false;
          btnCorrect.disabled = false;
        }
      });

      btnCorrect.addEventListener('click', (e) => {
        e.stopPropagation();
        form.hidden = !form.hidden;
        if (!form.hidden) input.focus?.();
      });
      btnCancel.addEventListener('click', (e) => {
        e.stopPropagation();
        form.hidden = true;
      });

      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        const correctedName = input.value.trim();
        if (!correctedName) return;
        btnSave.disabled = true;
        try {
          await this.submitReview({
            source: result.media_type === 'video' ? 'upload_video' : 'upload_image',
            session_id: sample?.session_id || result.session_id,
            frame_number: sample?.frame_number ?? 1,
            frame_id: `${result.session_id}:frame-${sample?.frame_number ?? 1}`,
            review_frame_token: sample?.review_frame_token || result.review_frame_token,
            detection_id: item.detection_id,
            track_id: item.track_id ?? null,
            original_prediction: {
              class_name: item.class_name,
              confidence: item.confidence,
              horizontal_position: item.horizontal_position,
              depth_category: item.depth_category,
              bounding_box: item.bounding_box,
            },
            correction_type: 'class_correction',
            review_status: 'corrected',
            human_corrected_name: correctedName,
          });
          badge.className = 'review-badge review-badge-corrected';
          badge.textContent = `Corrected: ${item.class_name} → ${correctedName}`;
          form.hidden = true;
          actions.hidden = true;
        } catch (err) {
          btnSave.disabled = false;
        }
      });

      return row;
    }

    highlightDetectionBox(canvas, size, item) {
      if (!canvas || !canvas.getContext) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      const width = canvas.clientWidth || (Array.isArray(size) ? size[0] : size?.width) || 640;
      const height = canvas.clientHeight || (Array.isArray(size) ? size[1] : size?.height) || 360;
      canvas.width = width;
      canvas.height = height;
      ctx.clearRect(0, 0, width, height);

      const box = item?.bounding_box;
      if (!Array.isArray(box) || box.length !== 4) return;
      const srcW = (Array.isArray(size) ? size[0] : size?.width) || width;
      const srcH = (Array.isArray(size) ? size[1] : size?.height) || height;
      const scaleX = width / srcW;
      const scaleY = height / srcH;
      const x1 = box[0] * scaleX;
      const y1 = box[1] * scaleY;
      const boxW = (box[2] - box[0]) * scaleX;
      const boxH = (box[3] - box[1]) * scaleY;

      ctx.fillStyle = 'rgba(255, 226, 107, 0.25)';
      ctx.fillRect(x1, y1, boxW, boxH);
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#ffe26b';
      ctx.strokeRect(x1, y1, boxW, boxH);
    }

    async submitReview(reviewPayload) {
      const response = await this.fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reviewPayload),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || `Feedback submission failed: ${response.status}`);
      }
      if (typeof window !== 'undefined' && typeof window.byakuReloadReviewHistory === 'function') {
        window.byakuReloadReviewHistory().catch(() => {});
      }
      return data;
    }

    openLightbox(src, caption) {
      let lightbox = this.document.querySelector('#byaku-lightbox');
      if (!lightbox) {
        lightbox = this.document.createElement('div');
        lightbox.id = 'byaku-lightbox';
        lightbox.className = 'byaku-lightbox';
        const closeBtn = this.document.createElement('button');
        closeBtn.className = 'byaku-lightbox-close';
        closeBtn.textContent = '✕';
        closeBtn.setAttribute('aria-label', 'Close enlarged image');
        const img = this.document.createElement('img');
        img.className = 'byaku-lightbox-img';
        img.alt = 'Enlarged frame view';
        const cap = this.document.createElement('p');
        cap.className = 'byaku-lightbox-caption';
        lightbox.append(closeBtn, img, cap);
        this.document.body?.append(lightbox);
        const close = () => { lightbox.hidden = true; };
        closeBtn.addEventListener('click', close);
        lightbox.addEventListener('click', (e) => { if (e.target === lightbox) close(); });
        this.document.addEventListener?.('keydown', (e) => { if (e.key === 'Escape') close(); });
      }
      const img = lightbox.querySelector('.byaku-lightbox-img') || lightbox.querySelector('img');
      const cap = lightbox.querySelector('.byaku-lightbox-caption') || lightbox.querySelector('p');
      if (img) img.src = src;
      if (cap) cap.textContent = caption || '';
      lightbox.hidden = false;
    }
  }

  if (typeof window !== 'undefined' && typeof document !== 'undefined' && document.querySelector('#workflow-upload-form')) {
    window.byakuWorkflowUpload = new BYAKUWorkflowUploadController();
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { BYAKUWorkflowUploadController, IMAGE_MAX_BYTES, VIDEO_MAX_BYTES };
  }
})();
