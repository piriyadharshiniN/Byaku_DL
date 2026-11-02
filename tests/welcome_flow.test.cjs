const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { installWelcomeRedirect } = require('../public/js/welcome_flow.js');

const welcomePage = fs.readFileSync(path.join(__dirname, '../public/welcome/index.html'), 'utf8');
const assistantPage = fs.readFileSync(path.join(__dirname, '../public/assistant.html'), 'utf8');

test('welcome Continue goes directly to the voice-start page', () => {
  assert.match(welcomePage, /class="skip-button" href="\/assistant\.html">Continue<\/a>/);
  assert.doesNotMatch(welcomePage, /Open the assistant|Explore project workflow/);
  assert.doesNotMatch(welcomePage, /index-DGY3_KB3\.js/);
  assert.match(assistantPage, /id="start-listening"/);
});

test('welcome automatically navigates to the existing assistant route after three seconds', () => {
  const callbacks = new Map();
  const navigations = [];
  const linkListeners = new Map();
  let timerId = 0;
  const windowRef = {
    location: { assign: (url) => navigations.push(url) },
    setTimeout(callback, delay) {
      assert.equal(delay, 3000);
      callbacks.set(++timerId, callback);
      return timerId;
    },
    clearTimeout(id) { callbacks.delete(id); },
    addEventListener() {},
  };
  const continueLink = {
    href: 'http://localhost:3000/assistant.html',
    addEventListener: (type, callback) => linkListeners.set(type, callback),
  };
  const documentRef = { querySelector: () => continueLink };
  const cancel = installWelcomeRedirect(documentRef, windowRef);
  assert.equal(installWelcomeRedirect(documentRef, windowRef), cancel);
  assert.equal(callbacks.size, 1);

  assert.equal(navigations.length, 0);
  callbacks.get(1)();
  assert.deepEqual(navigations, ['http://localhost:3000/assistant.html']);
  linkListeners.get('click')({ preventDefault() {} });
  assert.equal(navigations.length, 1);
});

test('Continue navigates immediately and cancels automatic navigation', () => {
  const callbacks = new Map();
  const navigations = [];
  const linkListeners = new Map();
  const cleared = [];
  const windowRef = {
    location: { assign: (url) => navigations.push(url) },
    setTimeout(callback, delay) {
      assert.equal(delay, 3000);
      callbacks.set(1, callback);
      return 1;
    },
    clearTimeout: (id) => { cleared.push(id); callbacks.delete(id); },
    addEventListener() {},
  };
  const continueLink = {
    href: '/assistant.html',
    addEventListener: (type, callback) => linkListeners.set(type, callback),
  };
  installWelcomeRedirect({ querySelector: () => continueLink }, windowRef);
  let prevented = false;
  linkListeners.get('click')({ preventDefault: () => { prevented = true; } });
  callbacks.get(1)?.();

  assert.equal(prevented, true);
  assert.deepEqual(cleared, [1]);
  assert.deepEqual(navigations, ['/assistant.html']);
});

test('welcome cancels its pending navigation when the page is unloaded', () => {
  const callbacks = new Map();
  let pageHide;
  let cleared = false;
  const windowRef = {
    location: { assign() { assert.fail('unloaded page must not navigate'); } },
    setTimeout: (callback) => { callbacks.set(1, callback); return 1; },
    clearTimeout: (id) => { cleared = id === 1; callbacks.delete(id); },
    addEventListener: (_type, callback) => { pageHide = callback; },
  };
  const continueLink = { href: '/assistant.html', addEventListener() {} };
  installWelcomeRedirect({ querySelector: () => continueLink }, windowRef);
  pageHide();
  callbacks.get(1)?.();
  assert.equal(cleared, true);
});

test('assistant camera preview preserves camera aspect ratio for video and overlays', () => {
  const assistantStyles = fs.readFileSync(path.join(__dirname, '../public/css/assistant-runtime.css'), 'utf8');
  assert.match(assistantStyles, /\.assistant-camera-view\.has-stream\s*\{[^}]*min-height:\s*0/);
  assert.match(assistantStyles, /\.assistant-camera-view\.has-stream\s*\{[^}]*width:\s*min\(100%,\s*640px\)/);
  assert.match(assistantStyles, /\.assistant-camera-view \.camera-feed\s*\{[^}]*aspect-ratio:\s*auto/);
  assert.doesNotMatch(assistantStyles, /\.assistant-camera-view \.camera-feed\s*\{[^}]*aspect-ratio:\s*16\s*\/\s*9/);
});

test('assistant page greets once through the speech controller without starting camera or microphone', () => {
  assert.match(assistantPage, /id="assistant-greeting-fallback"/);
  const source = fs.readFileSync(path.join(__dirname, '../public/js/assistant_interface.js'), 'utf8');
  assert.match(source, /speech\.speak\('Your assistant is ready\.'\)/);
  assert.match(source, /if \(!greetingAttempted\)/);
  assert.doesNotMatch(source.slice(source.indexOf('if (!greetingAttempted)'), source.indexOf('if (!greetingAttempted)') + 120), /startCameraSession|startListening/);
});
