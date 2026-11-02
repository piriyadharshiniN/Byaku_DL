const { spawn } = require('node:child_process');
const path = require('node:path');

const projectRoot = path.resolve(__dirname, '..');
const host = process.env.HOST || '127.0.0.1';
const backendPort = process.env.BACKEND_PORT || process.env.PORT || '8765';
const relayPort = process.env.RELAY_PORT || '3000';
const backendUrl = process.env.PYTHON_BACKEND_URL || `http://${host}:${backendPort}`;
const python = process.env.PYTHON || (process.platform === 'win32' ? 'python' : 'python3');

const children = [];
let shuttingDown = false;

function startService(name, command, args, env = process.env) {
  const child = spawn(command, args, {
    cwd: projectRoot,
    env,
    stdio: 'inherit',
    windowsHide: true,
  });

  children.push({ name, process: child });
  child.once('error', (error) => {
    console.error(`[${name}] Could not start: ${error.message}`);
    void shutdown(1);
  });
  child.once('exit', (code, signal) => {
    if (!shuttingDown) {
      console.error(`[${name}] exited unexpectedly (${signal || `code ${code}`}).`);
      void shutdown(code || 1);
    }
  });

  return child;
}

async function shutdown(exitCode) {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log('\nStopping BYAKU services...');

  const runningChildren = children.filter(({ process: child }) => child.exitCode === null);
  for (const { process: child } of runningChildren) {
    child.kill('SIGINT');
  }

  const forceTimer = setTimeout(() => {
    for (const { process: child } of runningChildren) {
      if (child.exitCode === null) child.kill('SIGKILL');
    }
  }, 8000);
  forceTimer.unref();

  await Promise.all(runningChildren.map(({ process: child }) => {
    if (child.exitCode !== null) return Promise.resolve();
    return new Promise((resolve) => child.once('exit', resolve));
  }));
  clearTimeout(forceTimer);
  process.exitCode = exitCode;
}

process.once('SIGINT', () => void shutdown(0));
process.once('SIGTERM', () => void shutdown(0));

console.log(`Starting BYAKU backend on ${host}:${backendPort}...`);
startService('backend', python, [
  '-m',
  'uvicorn',
  'backend.main:app',
  '--host',
  host,
  '--port',
  String(backendPort),
]);

console.log(`Starting BYAKU website and relay on 127.0.0.1:${relayPort}...`);
startService('relay', process.execPath, [
  path.join(projectRoot, 'server', 'http_websocket_relay.js'),
], {
  ...process.env,
  RELAY_PORT: String(relayPort),
  PYTHON_BACKEND_URL: backendUrl,
});

console.log(`Open http://127.0.0.1:${relayPort} when the services are ready. Press Ctrl+C to stop both.`);
