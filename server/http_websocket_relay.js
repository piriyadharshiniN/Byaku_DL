const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { WebSocket, WebSocketServer } = require('ws');

const root = path.resolve(__dirname, '..', 'public');
const port = Number(process.env.RELAY_PORT || process.env.PORT || 3000);
const pythonBackend = (
  process.env.PYTHON_BACKEND_URL ||
  process.env.BYAKU_PYTHON_BACKEND_URL ||
  'http://127.0.0.1:8765'
).replace(/\/$/, '');
const mimeTypes = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.mp4': 'video/mp4',
  '.webp': 'image/webp',
  '.js': 'text/javascript; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.wav': 'audio/wav',
  '.json': 'application/json; charset=utf-8',
};

const server = http.createServer(async (request, response) => {
  const pathname = decodeURIComponent(new URL(request.url, `http://${request.headers.host}`).pathname);

  if (pathname.startsWith('/api/') || pathname === '/health') {
    if (!['GET', 'POST'].includes(request.method)) {
      response.writeHead(405, { 'Content-Type': 'application/json' }).end(JSON.stringify({ error: 'Method not allowed' }));
      return;
    }
    try {
      const chunks = [];
      let size = 0;
      const maxBodyBytes = pathname === '/api/upload/image'
        ? 13 * 1024 * 1024
        : pathname === '/api/upload/video'
          ? 101 * 1024 * 1024
          : 1024 * 1024;
      const contentLength = Number(request.headers['content-length']);
      if (Number.isFinite(contentLength) && contentLength > maxBodyBytes) {
        response.writeHead(413, { 'Content-Type': 'application/json' });
        response.end(JSON.stringify({ error: 'Upload request exceeds the allowed size.' }));
        return;
      }
      if (request.method !== 'GET') {
        for await (const chunk of request) {
          size += chunk.length;
          if (size > maxBodyBytes) {
            const error = new Error('Upload request exceeds the allowed size.');
            error.statusCode = 413;
            throw error;
          }
          chunks.push(chunk);
        }
      }
      const upstream = await fetch(`${pythonBackend}${request.url}`, {
        method: request.method,
        headers: request.headers['content-type'] ? { 'Content-Type': request.headers['content-type'] } : {},
        body: request.method === 'GET' ? undefined : Buffer.concat(chunks),
      });
      const body = Buffer.from(await upstream.arrayBuffer());
      response.writeHead(upstream.status, {
        'Content-Type': upstream.headers.get('content-type') || 'application/json',
        'Cache-Control': 'no-store',
        'X-Content-Type-Options': 'nosniff',
      });
      response.end(body);
    } catch (error) {
      response.writeHead(error.statusCode === 413 ? 413 : 503, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
      response.end(JSON.stringify({ error: error.message || 'Python AI backend is unavailable' }));
    }
    return;
  }

  const requestedPath = pathname === '/'
    || pathname === '/index.html'
    || pathname === '/welcome'
    || pathname === '/welcome/'
    ? '/welcome/index.html'
    : pathname === '/assistant' || pathname === '/assistant/'
      ? '/assistant.html'
      : pathname;
  const filePath = path.resolve(root, `.${requestedPath}`);

  if (!filePath.startsWith(`${root}${path.sep}`) && filePath !== path.join(root, 'index.html')) {
    response.writeHead(403).end('Forbidden');
    return;
  }

  fs.readFile(filePath, (error, content) => {
    if (error) {
      response.writeHead(error.code === 'ENOENT' ? 404 : 500, { 'Content-Type': 'text/plain; charset=utf-8' });
      response.end(error.code === 'ENOENT' ? 'Not found' : 'Unable to read file');
      return;
    }
    response.writeHead(200, {
      'Content-Type': mimeTypes[path.extname(filePath)] || 'application/octet-stream',
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff',
    });
    response.end(content);
  });
});

const webSockets = new WebSocketServer({ noServer: true, maxPayload: 6 * 1024 * 1024 });

server.on('upgrade', (request, socket, head) => {
  const pathname = new URL(request.url, `http://${request.headers.host}`).pathname;
  if (pathname !== '/ws') {
    socket.destroy();
    return;
  }
  webSockets.handleUpgrade(request, socket, head, (client) => {
    webSockets.emit('connection', client);
  });
});

webSockets.on('connection', (client) => {
  const backendUrl = `${pythonBackend.replace(/^http/, 'ws')}/ws`;
  const backend = new WebSocket(backendUrl, { maxPayload: 6 * 1024 * 1024 });
  const pending = [];
  const connectTimeout = setTimeout(() => {
    if (backend.readyState !== WebSocket.OPEN) {
      client.send(JSON.stringify({ type: 'error', stage: 'python_backend', message: 'Python backend did not connect in time' }));
      client.close(1013, 'Python backend unavailable');
    }
  }, 10000);

  backend.on('open', () => {
    clearTimeout(connectTimeout);
    while (pending.length && backend.readyState === WebSocket.OPEN) {
      backend.send(pending.shift().toString(), { binary: false });
    }
  });
  backend.on('message', (message) => {
    if (client.readyState === WebSocket.OPEN) client.send(message, { binary: false });
  });
  backend.on('error', (error) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify({ type: 'error', stage: 'python_backend', message: error.message || 'Python backend unavailable' }));
    }
  });
  backend.on('close', () => {
    clearTimeout(connectTimeout);
    if (client.readyState === WebSocket.OPEN) client.close(1011, 'Python backend disconnected');
  });
  client.on('message', (message) => {
    if (backend.readyState === WebSocket.OPEN) {
      backend.send(message.toString());
    } else if (backend.readyState === WebSocket.CONNECTING && pending.length < 2) {
      pending.push(message);
    } else if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify({ type: 'error', stage: 'python_backend', message: 'Python backend is not connected' }));
    }
  });
  client.on('close', () => {
    clearTimeout(connectTimeout);
    backend.close();
  });
});

server.listen(port, '127.0.0.1', () => {
  console.log(`BYAKU is available at http://127.0.0.1:${port}`);
  console.log(`Python backend relay: ${pythonBackend}`);
});
