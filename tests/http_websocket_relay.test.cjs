const test = require('node:test');
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const http = require('node:http');
const path = require('node:path');
const { once } = require('node:events');

async function listen(server) {
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  return server.address().port;
}

async function waitForRelay(url, process) {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (process.exitCode !== null) throw new Error(`Relay exited with code ${process.exitCode}.`);
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`Relay did not respond at ${url}.`);
}

test('relay forwards the documented /health endpoint to the backend', { timeout: 10000 }, async (t) => {
  const backend = http.createServer((request, response) => {
    if (request.url === '/health') {
      response.writeHead(200, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify({ service: 'test-backend', status: 'running' }));
      return;
    }
    response.writeHead(404).end();
  });
  const backendPort = await listen(backend);
  const temporaryListener = http.createServer();
  const relayPort = await listen(temporaryListener);
  await new Promise((resolve) => temporaryListener.close(resolve));
  const relay = spawn(process.execPath, ['server/http_websocket_relay.js'], {
    cwd: path.resolve(__dirname, '..'),
    env: {
      ...process.env,
      PORT: String(relayPort),
      PYTHON_BACKEND_URL: `http://127.0.0.1:${backendPort}`,
    },
    stdio: 'ignore',
  });

  t.after(async () => {
    if (relay.exitCode === null) {
      relay.kill();
      await Promise.race([once(relay, 'exit'), new Promise((resolve) => setTimeout(resolve, 2000))]);
    }
    await new Promise((resolve) => backend.close(resolve));
  });

  const relayUrl = `http://127.0.0.1:${relayPort}`;
  await waitForRelay(`${relayUrl}/assistant`, relay);
  const response = await fetch(`${relayUrl}/health`);
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { service: 'test-backend', status: 'running' });
});

test('relay forwards upload request bodies larger than the standard API limit', { timeout: 10000 }, async (t) => {
  let forwardedBytes = 0;
  const backend = http.createServer(async (request, response) => {
    for await (const chunk of request) forwardedBytes += chunk.length;
    response.writeHead(200, { 'Content-Type': 'application/json' });
    response.end(JSON.stringify({ received: forwardedBytes }));
  });
  const backendPort = await listen(backend);
  const temporaryListener = http.createServer();
  const relayPort = await listen(temporaryListener);
  await new Promise((resolve) => temporaryListener.close(resolve));
  const relay = spawn(process.execPath, ['server/http_websocket_relay.js'], {
    cwd: path.resolve(__dirname, '..'),
    env: {
      ...process.env,
      PORT: String(relayPort),
      PYTHON_BACKEND_URL: `http://127.0.0.1:${backendPort}`,
    },
    stdio: 'ignore',
  });

  t.after(async () => {
    if (relay.exitCode === null) {
      relay.kill();
      await Promise.race([once(relay, 'exit'), new Promise((resolve) => setTimeout(resolve, 2000))]);
    }
    await new Promise((resolve) => backend.close(resolve));
  });

  const relayUrl = `http://127.0.0.1:${relayPort}`;
  await waitForRelay(`${relayUrl}/assistant`, relay);
  const requestBody = Buffer.alloc(2 * 1024 * 1024, 7);
  const response = await fetch(`${relayUrl}/api/upload/video`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream' },
    body: requestBody,
  });

  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { received: requestBody.length });
  assert.equal(forwardedBytes, requestBody.length);
});
