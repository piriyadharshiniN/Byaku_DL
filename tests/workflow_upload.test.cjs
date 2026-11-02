const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { BYAKUWorkflowUploadController, IMAGE_MAX_BYTES, VIDEO_MAX_BYTES } = require('../public/js/workflow_upload.js');

const SELECTORS = [
  '#workflow-upload-form', '#workflow-upload-file', '#workflow-upload-process',
  '#workflow-upload-remove', '#workflow-upload-status-text', '#workflow-upload-loading',
  '#workflow-upload-preview', '#workflow-upload-filename', '#workflow-upload-image-preview',
  '#workflow-upload-video-preview', '#workflow-upload-results', '#workflow-upload-annotated',
  '#workflow-upload-description', '#workflow-upload-objects', '#workflow-upload-samples',
  '#workflow-upload-speak',
];

class FakeElement {
  constructor() {
    this.listeners = new Map();
    this.children = [];
    this.dataset = {};
    this.hidden = false;
    this.disabled = false;
    this.value = '';
    this.files = [];
    this._textContent = undefined;
  }

  get textContent() {
    if (this._textContent !== undefined) return this._textContent;
    return this.children.map((c) => (typeof c === 'string' ? c : (c.textContent || ''))).join('');
  }
  set textContent(val) {
    this._textContent = val;
  }

  addEventListener(type, listener) { this.listeners.set(type, listener); }
  append(...children) {
    for (const child of children) {
      this.children.push(child);
    }
  }
  replaceChildren(...children) {
    this.children = [];
    if (children.length > 0) this.append(...children);
  }
  removeAttribute(name) { delete this[name]; }
  setAttribute(name, value) { this[name] = value; }
  querySelector(selector) {
    for (const child of this.children) {
      if (child.className && selector.startsWith('.') && child.className.includes(selector.slice(1))) return child;
      if (child.querySelector) {
        const found = child.querySelector(selector);
        if (found) return found;
      }
    }
    return null;
  }
  querySelectorAll(selector) {
    const matches = [];
    for (const child of this.children) {
      if (child.className && selector.startsWith('.') && child.className.includes(selector.slice(1))) matches.push(child);
      if (child.querySelectorAll) matches.push(...child.querySelectorAll(selector));
    }
    return matches;
  }
  classList = {
    add: (...cls) => {
      this.className = [...new Set([...(this.className || '').split(' ').filter(Boolean), ...cls])].join(' ');
    },
    remove: (...cls) => {
      this.className = (this.className || '').split(' ').filter((c) => !cls.includes(c)).join(' ');
    },
    contains: (cls) => (this.className || '').split(' ').includes(cls),
  };
  getContext() {
    return {
      clearRect() {},
      strokeRect() {},
      fillRect() {},
      drawImage() {},
      measureText: () => ({ width: 50 }),
      fillText() {},
    };
  }
  load() {}
}

class FakeFormData {
  constructor() { this.entries = []; }
  append(...entry) { this.entries.push(entry); }
}

function createController({ fetchFn, spoken = [] }) {
  const elements = new Map(SELECTORS.map((selector) => [selector, new FakeElement()]));
  const revoked = [];
  const windowRef = { addEventListener() {} };
  const speech = class {
    speak(text) { spoken.push(text); return true; }
    cancel() {}
  };
  const controller = new BYAKUWorkflowUploadController({
    documentRef: {
      querySelector: (selector) => elements.get(selector),
      createElement: () => new FakeElement(),
    },
    fetchFn,
    URLRef: {
      createObjectURL: (file) => `blob:${file.name}`,
      revokeObjectURL: (url) => revoked.push(url),
    },
    SpeechController: speech,
    FormDataClass: FakeFormData,
    windowRef,
  });
  return { controller, elements, revoked };
}

test('workflow dashboard documents its pipeline and preserves the live assistant', () => {
  const dashboard = fs.readFileSync(path.join(__dirname, '../public/dashboard.html'), 'utf8');
  const assistant = fs.readFileSync(path.join(__dirname, '../public/assistant.html'), 'utf8');
  const uploadScript = fs.readFileSync(path.join(__dirname, '../public/js/workflow_upload.js'), 'utf8');

  assert.match(dashboard, /class="pipeline-panel/);
  assert.match(dashboard, /id="pipeline-title"/);
  assert.match(dashboard, /id="scene-graph"/);
  assert.match(dashboard, /id="query"/);
  assert.doesNotMatch(dashboard, /id="settings"/);
  assert.doesNotMatch(dashboard, /id="events"|Important scene events/);
  assert.deepEqual(
    Array.from(dashboard.matchAll(/<li data-module="([^"]+)">/g), (match) => match[1]),
    ['camera', 'detection', 'tracking', 'depth', 'spatial', 'scene', 'memory', 'narration', 'review'],
  );
  assert.match(dashboard, /id="workflow-upload-form"/);
  assert.match(dashboard, /id="upload-analysis"/);
  assert.match(dashboard, /id="detection"/);
  assert.match(dashboard, /id="human-review"/);
  assert.match(dashboard, /id="live-view"/);
  assert.match(dashboard, /assistant_speech\.js/);
  assert.match(assistant, /id="start-listening"/);
  assert.doesNotMatch(uploadScript, /BYAKUCamera|mediaDevices\.getUserMedia/);
});

test('welcome, assistant, and workflow pages use the shared BYAKU logo asset', () => {
  const pages = [
    '../public/index.html',
    '../public/welcome/index.html',
    '../public/assistant.html',
    '../public/dashboard.html',
  ];

  for (const relativePath of pages) {
    const markup = fs.readFileSync(path.join(__dirname, relativePath), 'utf8');
    assert.match(markup, /\/assets\/byaku-logo\.png/);
    assert.match(markup, /\/css\/brand-logo\.css/);
  }
  assert.equal(fs.existsSync(path.join(__dirname, '../public/assets/byaku-logo.png')), true);
});

test('valid image upload renders annotated detections and speaks only on request', async () => {
  let requestUrl;
  let requestBody;
  const spoken = [];
  const { controller, elements } = createController({
    spoken,
    fetchFn: async (url, options) => {
      requestUrl = url;
      requestBody = options.body;
      return {
        ok: true,
        json: async () => ({
          media_type: 'image',
          status: 'success',
          annotated_image: 'YW5ub3RhdGVk',
          scene_description: 'A bottle is on the right and appears far.',
          objects: [{ class_name: 'bottle', track_id: 4, horizontal_position: 'right', depth_category: 'FAR' }],
        }),
      };
    },
  });
  controller.selectFile({ name: 'scene.png', size: 1024 });
  await controller.processSelection();

  assert.equal(requestUrl, '/api/upload/image');
  assert.deepEqual(requestBody.entries[0].slice(0, 2), ['upload', { name: 'scene.png', size: 1024 }]);
  assert.equal(elements.get('#workflow-upload-preview').hidden, false);
  assert.equal(elements.get('#workflow-upload-results').hidden, false);
  assert.equal(elements.get('#workflow-upload-annotated').src, 'data:image/jpeg;base64,YW5ub3RhdGVk');
  assert.equal(elements.get('#workflow-upload-description').textContent, 'A bottle is on the right and appears far.');
  assert.equal(elements.get('#workflow-upload-objects').children[0].textContent, 'bottle · ID 4 · right · FAR');
  assert.deepEqual(spoken, []);
  elements.get('#workflow-upload-speak').listeners.get('click')();
  assert.deepEqual(spoken, ['A bottle is on the right and appears far.']);
  assert.equal(IMAGE_MAX_BYTES, 12 * 1024 * 1024);
});

test('valid video upload shows sampled scene results and offers the overall description', async () => {
  let requestUrl;
  const spoken = [];
  const { controller, elements } = createController({
    spoken,
    fetchFn: async (url) => {
      requestUrl = url;
      return {
        ok: true,
        json: async () => ({
          media_type: 'video',
          status: 'success',
          scene_description: 'Across 2 sampled video frames, A person is on the left. Later, A chair is in the center.',
          objects: [{ class_name: 'person', track_id: 8, horizontal_position: 'left', depth_category: 'UNKNOWN' }],
          samples: [{ frame_number: 1, timestamp_seconds: 0, objects: [{ class_name: 'person' }] }],
        }),
      };
    },
  });
  controller.selectFile({ name: 'walk.mp4', size: 2048 });
  await controller.processSelection();

  assert.equal(requestUrl, '/api/upload/video');
  assert.equal(elements.get('#workflow-upload-video-preview').hidden, false);
  assert.equal(elements.get('#workflow-upload-samples').hidden, false);
  assert.match(elements.get('#workflow-upload-samples').children[0].textContent, /Frame 1 · 0\.0 s: person/);
  elements.get('#workflow-upload-speak').listeners.get('click')();
  assert.equal(spoken.length, 1);
  assert.equal(VIDEO_MAX_BYTES, 100 * 1024 * 1024);
});

test('unsupported and oversized files are rejected before upload', async () => {
  let requests = 0;
  const { controller, elements } = createController({
    fetchFn: async () => { requests += 1; throw new Error('should not upload'); },
  });
  controller.selectFile({ name: 'model.exe', size: 1 });
  assert.equal(elements.get('#workflow-upload-process').disabled, true);
  assert.match(elements.get('#workflow-upload-status-text').textContent, /Unsupported format/);
  controller.selectFile({ name: 'large.png', size: IMAGE_MAX_BYTES + 1 });
  assert.equal(elements.get('#workflow-upload-process').disabled, true);
  assert.match(elements.get('#workflow-upload-status-text').textContent, /12 MB limit/);
  await controller.processSelection();
  assert.equal(requests, 0);
});

test('backend processing failures remain visible and do not enable speech', async () => {
  const { controller, elements } = createController({
    fetchFn: async () => ({
      ok: false,
      json: async () => ({ detail: 'This video could not be decoded. Check its codec.' }),
    }),
  });
  controller.selectFile({ name: 'broken.webm', size: 20 });
  await controller.processSelection();

  assert.equal(elements.get('#workflow-upload-results').hidden, true);
  assert.equal(elements.get('#workflow-upload-speak').disabled, true);
  assert.match(elements.get('#workflow-upload-status-text').textContent, /could not be decoded/);
});

test('results remain visible when browser speech synthesis is unavailable', async () => {
  const elements = new Map(SELECTORS.map((selector) => [selector, new FakeElement()]));
  const controller = new BYAKUWorkflowUploadController({
    documentRef: {
      querySelector: (selector) => elements.get(selector),
      createElement: () => new FakeElement(),
    },
    fetchFn: async () => ({
      ok: true,
      json: async () => ({
        media_type: 'image',
        status: 'success',
        scene_description: 'A chair is in the center.',
        objects: [{ class_name: 'chair', horizontal_position: 'center', depth_category: 'UNKNOWN' }],
      }),
    }),
    URLRef: { createObjectURL: () => 'blob:image', revokeObjectURL() {} },
    SpeechController: null,
    FormDataClass: FakeFormData,
    windowRef: { addEventListener() {} },
  });
  controller.selectFile({ name: 'chair.jpg', size: 12 });
  await controller.processSelection();
  elements.get('#workflow-upload-speak').listeners.get('click')();

  assert.equal(elements.get('#workflow-upload-results').hidden, false);
  assert.equal(elements.get('#workflow-upload-description').textContent, 'A chair is in the center.');
});

test('uploaded video displays annotated frame gallery and per-frame detection review', async () => {
  const feedbackSubmissions = [];
  const { controller, elements } = createController({
    fetchFn: async (url, options) => {
      if (url === '/api/upload/video') {
        return {
          ok: true,
          json: async () => ({
            media_type: 'video',
            status: 'success',
            session_id: 'upload-video-test-session',
            frames_processed: 2,
            frames_failed: 0,
            detection_count: 2,
            scene_description: 'A person is visible.',
            objects: [{ class_name: 'person', track_id: 1, horizontal_position: 'center', depth_category: 'MID-RANGE' }],
            samples: [
              {
                frame_number: 1,
                timestamp_seconds: 0.0,
                status: 'success',
                annotated_image: 'ZnJhbWUx',
                annotated_size: [640, 480],
                session_id: 'upload-video-test-session',
                review_frame_token: 'token-f1',
                objects: [
                  {
                    class_name: 'person',
                    track_id: 1,
                    detection_id: 'upload-video-test-session:track:1',
                    confidence: 0.95,
                    horizontal_position: 'center',
                    depth_category: 'MID-RANGE',
                    bounding_box: [100, 100, 300, 400],
                  },
                ],
              },
              {
                frame_number: 31,
                timestamp_seconds: 1.0,
                status: 'success',
                annotated_image: 'ZnJhbWUy',
                annotated_size: [640, 480],
                session_id: 'upload-video-test-session',
                review_frame_token: 'token-f2',
                objects: [
                  {
                    class_name: 'cup',
                    track_id: null,
                    detection_id: 'upload-video-test-session:frame-31:det-0',
                    confidence: 0.88,
                    horizontal_position: 'right',
                    depth_category: 'NEAR',
                    bounding_box: [400, 200, 480, 320],
                  },
                ],
              },
            ],
          }),
        };
      }
      if (url === '/api/feedback') {
        const body = JSON.parse(options.body);
        feedbackSubmissions.push(body);
        return {
          ok: true,
          json: async () => ({ feedback_id: 'fb-123', ...body, timestamp: new Date().toISOString() }),
        };
      }
      throw new Error(`Unexpected url: ${url}`);
    },
  });

  controller.selectFile({ name: 'walk.mp4', size: 4096 });
  await controller.processSelection();

  // Status reports frames analyzed and detections
  assert.match(elements.get('#workflow-upload-status-text').textContent, /2 frames successfully analyzed/);
  assert.match(elements.get('#workflow-upload-status-text').textContent, /2 detections/);

  // Both sampled frames rendered in gallery
  const samplesContainer = elements.get('#workflow-upload-samples');
  assert.equal(samplesContainer.children.length, 2);

  // Frame 1 annotated image and meta
  const card1 = samplesContainer.children[0];
  const img1 = card1.querySelector('.upload-sample-annotated');
  assert.equal(img1.src, 'data:image/jpeg;base64,ZnJhbWUx');
  assert.match(card1.textContent, /Frame 1/);
  assert.match(card1.textContent, /\[ID: #1\]/);
  assert.match(card1.textContent, /MID-RANGE/);

  // Review detection on Frame 1 (Confirm)
  const btnConfirm = card1.querySelector('.btn-confirm');
  assert.ok(btnConfirm);
  await btnConfirm.listeners.get('click')({ stopPropagation() {} });

  assert.equal(feedbackSubmissions.length, 1);
  assert.equal(feedbackSubmissions[0].source, 'upload_video');
  assert.equal(feedbackSubmissions[0].session_id, 'upload-video-test-session');
  assert.equal(feedbackSubmissions[0].frame_number, 1);
  assert.equal(feedbackSubmissions[0].detection_id, 'upload-video-test-session:track:1');
  assert.equal(feedbackSubmissions[0].review_status, 'confirmed');
  assert.equal(feedbackSubmissions[0].track_id, 1);

  // Frame 2 has untracked detection with stable identifier
  const card2 = samplesContainer.children[1];
  assert.match(card2.textContent, /Frame 31/);
  assert.match(card2.textContent, /\[Untracked\]/);
  assert.match(card2.textContent, /NEAR/);

  // Review detection on Frame 2 (Correct/Rename)
  const formCorrect = card2.querySelector('.upload-correction-form');
  const inputCorrect = card2.querySelector('.upload-correction-input');
  inputCorrect.value = 'coffee mug';
  await formCorrect.listeners.get('submit')({
    preventDefault() {},
    stopPropagation() {},
  });

  assert.equal(feedbackSubmissions.length, 2);
  assert.equal(feedbackSubmissions[1].source, 'upload_video');
  assert.equal(feedbackSubmissions[1].session_id, 'upload-video-test-session');
  assert.equal(feedbackSubmissions[1].frame_number, 31);
  assert.equal(feedbackSubmissions[1].detection_id, 'upload-video-test-session:frame-31:det-0');
  assert.equal(feedbackSubmissions[1].track_id, null);
  assert.equal(feedbackSubmissions[1].correction_type, 'class_correction');
  assert.equal(feedbackSubmissions[1].review_status, 'corrected');
  assert.equal(feedbackSubmissions[1].human_corrected_name, 'coffee mug');
});

test('failed video frames are reported without discarding successful frames', async () => {
  const { controller, elements } = createController({
    fetchFn: async () => ({
      ok: true,
      json: async () => ({
        media_type: 'video',
        status: 'partial',
        session_id: 'upload-video-failed-frame-session',
        frames_processed: 1,
        frames_failed: 1,
        detection_count: 1,
        warnings: ['Source frame 31 failed to decode.'],
        scene_description: 'Across 1 sampled video frames, A dog is visible.',
        objects: [{ class_name: 'dog', track_id: 1, horizontal_position: 'left', depth_category: 'FAR' }],
        samples: [
          {
            frame_number: 1,
            timestamp_seconds: 0.0,
            status: 'success',
            annotated_image: 'ZG9n',
            objects: [{ class_name: 'dog', track_id: 1, detection_id: 'upload-video-failed-frame-session:track:1' }],
          },
          {
            frame_number: 31,
            timestamp_seconds: null,
            status: 'decode_error',
            annotated_image: null,
            objects: [],
          },
        ],
      }),
    }),
  });

  controller.selectFile({ name: 'partial.mp4', size: 3000 });
  await controller.processSelection();

  // Status clearly identifies 1 analyzed, 1 failed, and the specific warning
  const status = elements.get('#workflow-upload-status-text').textContent;
  assert.match(status, /1 frame successfully analyzed/);
  assert.match(status, /1 failed/);
  assert.match(status, /Source frame 31 failed to decode/);

  // Both frames are visible in the samples gallery
  const samples = elements.get('#workflow-upload-samples');
  assert.equal(samples.children.length, 2);

  // First frame is successful
  assert.equal(samples.children[0].classList.contains('upload-sample-error'), false);
  assert.match(samples.children[0].textContent, /Frame 1/);

  // Second frame is marked with decode error
  assert.equal(samples.children[1].classList.contains('upload-sample-error'), true);
  assert.match(samples.children[1].textContent, /Frame 31: Decode error/);
  assert.match(samples.children[1].textContent, /failed to decode/);
});
