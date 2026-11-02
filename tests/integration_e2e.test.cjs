const test = require('node:test');
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const path = require('node:path');
const { once } = require('node:events');

async function waitForServer(url, timeoutMs = 60000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url);
      if (res.ok) return true;
    } catch {}
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error(`Timeout waiting for ${url}`);
}

test('End-to-End Integration: Image upload -> Review submit -> History fetch -> Undo -> Restart persistence', { timeout: 120000 }, async (t) => {
  const projectRoot = path.resolve(__dirname, '..');
  const backendPort = 8765;
  const relayPort = 3000;
  const python = process.env.PYTHON || (process.platform === 'win32' ? 'python' : 'python3');

  // 1. Start Python FastAPI backend
  let backend = spawn(python, ['-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', String(backendPort)], {
    cwd: projectRoot,
    stdio: 'ignore',
  });

  // 2. Start Node.js relay
  const relay = spawn(process.execPath, ['server/http_websocket_relay.js'], {
    cwd: projectRoot,
    env: {
      ...process.env,
      PORT: String(relayPort),
      PYTHON_BACKEND_URL: `http://127.0.0.1:${backendPort}`,
    },
    stdio: 'ignore',
  });

  t.after(async () => {
    if (backend && backend.exitCode === null) {
      backend.kill();
      await Promise.race([once(backend, 'exit'), new Promise((r) => setTimeout(r, 1000))]);
    }
    if (relay && relay.exitCode === null) {
      relay.kill();
      await Promise.race([once(relay, 'exit'), new Promise((r) => setTimeout(r, 1000))]);
    }
  });

  // Wait for both to be online
  await waitForServer(`http://127.0.0.1:${backendPort}/health`);
  await waitForServer(`http://127.0.0.1:${relayPort}/health`);

  // 3. Health check via relay
  const healthRes = await fetch(`http://127.0.0.1:${relayPort}/health`);
  assert.equal(healthRes.status, 200);
  const healthBody = await healthRes.json();
  assert.equal(healthBody.ok, true);
  assert.equal(healthBody.data.database_ok, true);

  // 4. Image Upload through relay (using valid 100x100 JPEG)
  const validJpegB64 = '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCABkAGQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD5UooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA//2Q==';
  const jpegBytes = Buffer.from(validJpegB64, 'base64');
  
  const boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW';
  let body = '';
  body += `--${boundary}\r\n`;
  body += 'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n';
  body += 'Content-Type: image/jpeg\r\n\r\n';
  
  const payloadBuffer = Buffer.concat([
    Buffer.from(body, 'utf8'),
    jpegBytes,
    Buffer.from(`\r\n--${boundary}--\r\n`, 'utf8')
  ]);

  const uploadRes = await fetch(`http://127.0.0.1:${relayPort}/api/image`, {
    method: 'POST',
    headers: {
      'Content-Type': `multipart/form-data; boundary=${boundary}`,
    },
    body: payloadBuffer,
  });

  assert.equal(uploadRes.status, 200);
  const uploadData = await uploadRes.json();
  assert.equal(uploadData.ok, true);
  assert.equal(uploadData.data.media_type, 'image');
  const sessionId = uploadData.data.session_id;
  assert.ok(sessionId.startsWith('img_'));

  // 5. Submit a review through the relay
  const reviewPayload = {
    session_id: sessionId,
    submission_id: `sub_${Date.now()}`,
    original_label: 'chair',
    original_confidence: 0.88,
    bbox: { x1: 10, y1: 10, x2: 100, y2: 100 },
    action: 'corrected',
    corrected_label: 'armchair',
    input_source: 'uploaded_image'
  };

  const reviewRes = await fetch(`http://127.0.0.1:${relayPort}/api/reviews`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(reviewPayload),
  });

  assert.equal(reviewRes.status, 201);
  const reviewBody = await reviewRes.json();
  assert.equal(reviewBody.ok, true);
  const reviewId = reviewBody.data.review_id;
  assert.equal(reviewBody.data.original_label, 'chair');
  assert.equal(reviewBody.data.corrected_label, 'armchair');

  // 6. Fetch review history through the relay
  const historyRes = await fetch(`http://127.0.0.1:${relayPort}/api/reviews?session_id=${sessionId}`);
  assert.equal(historyRes.status, 200);
  const historyBody = await historyRes.json();
  assert.equal(historyBody.ok, true);
  assert.equal(historyBody.data.length, 1);
  assert.equal(historyBody.data[0].review_id, reviewId);
  assert.equal(historyBody.data[0].action, 'corrected');
  assert.equal(historyBody.data[0].corrected_label, 'armchair');

  // 7. Undo the review through the relay
  const undoRes = await fetch(`http://127.0.0.1:${relayPort}/api/reviews/${reviewId}/undo`, {
    method: 'POST'
  });
  assert.equal(undoRes.status, 200);
  const undoBody = await undoRes.json();
  assert.equal(undoBody.ok, true);
  assert.equal(undoBody.data.status, 'undone');

  // Verify history shows status='undone'
  const historyAfterUndoRes = await fetch(`http://127.0.0.1:${relayPort}/api/reviews?session_id=${sessionId}`);
  const historyAfterUndo = await historyAfterUndoRes.json();
  assert.equal(historyAfterUndo.data[0].status, 'undone');

  // 8. Restart backend and test persistence
  backend.kill();
  await once(backend, 'exit');

  // Respawn backend
  backend = spawn('python3', ['-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', String(backendPort)], {
    cwd: projectRoot,
    stdio: 'ignore',
  });
  await waitForServer(`http://127.0.0.1:${backendPort}/health`);

  // Query review history through relay after backend restart
  const restartHistoryRes = await fetch(`http://127.0.0.1:${relayPort}/api/reviews?session_id=${sessionId}`);
  assert.equal(restartHistoryRes.status, 200);
  const restartHistory = await restartHistoryRes.json();
  assert.equal(restartHistory.ok, true);
  assert.equal(restartHistory.data.length, 1);
  assert.equal(restartHistory.data[0].review_id, reviewId);
  assert.equal(restartHistory.data[0].original_label, 'chair');
  assert.equal(restartHistory.data[0].status, 'undone');
});
