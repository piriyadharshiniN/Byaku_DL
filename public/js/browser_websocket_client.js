(() => {
  const STATE_SEQUENCE = ['idle', 'connecting', 'connected', 'streaming', 'paused', 'error'];
  const SCAN_INTERVAL_MS = 5000;

  class BYAKUConnection extends EventTarget {
    constructor(url = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`) {
      super();
      this.url = url;
      this.socket = null;
      this.state = 'idle';
      this.reconnectTimer = null;
      this.reconnectMs = 250;
      this.maxReconnectMs = 10000;
      this.heartbeatTimer = null;
      this.heartbeatMs = 25000;
      this.heartbeatAck = true;
      this.shouldReconnect = true;
      this.connectionAttempt = 0;
      this.lastMessageAt = 0;
    }

    setState(nextState) {
      if (this.state === nextState) return;
      this.state = nextState;
      this.dispatchEvent(new CustomEvent('statechange', { detail: { state: nextState } }));
    }

    static getBackoffDelay(attempt) {
      const base = 250 + Math.random() * 250;
      return Math.min(base * Math.pow(2, attempt), 10000);
    }

    scheduleReconnect() {
      if (!this.shouldReconnect) return;
      const delay = BYAKUConnection.getBackoffDelay(this.connectionAttempt);
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = window.setTimeout(() => {
        this.connectionAttempt += 1;
        this.connect();
      }, delay);
    }

    clearHeartbeat() {
      if (this.heartbeatTimer) {
        window.clearInterval(this.heartbeatTimer);
        this.heartbeatTimer = null;
      }
    }

    startHeartbeat() {
      this.clearHeartbeat();
      this.heartbeatTimer = window.setInterval(() => {
        if (this.socket?.readyState !== WebSocket.OPEN) return;
        if (!this.heartbeatAck) {
          this.socket.close();
          return;
        }
        this.heartbeatAck = false;
        try {
          this.socket.send(JSON.stringify({ type: 'ping', ts: Date.now() }));
        } catch {
          this.socket.close();
        }
      }, this.heartbeatMs);
    }

    connect() {
      if (!this.url || !('WebSocket' in window)) {
        this.setState('error');
        return Promise.reject(new Error('WebSocket is not supported in this browser.'));
      }
      if (this.socket && this.socket.readyState === WebSocket.OPEN) {
        this.setState('connected');
        return Promise.resolve(true);
      }

      this.setState('connecting');
      this.socket = new WebSocket(this.url);
      return new Promise((resolve, reject) => {
        const timeout = window.setTimeout(() => {
          if (this.socket && this.socket.readyState !== WebSocket.OPEN) {
            this.socket.close();
          }
          reject(new Error('Python backend connection timed out.'));
        }, 10000);

        this.socket.addEventListener('open', () => {
          window.clearTimeout(timeout);
          this.connectionAttempt = 0;
          this.heartbeatAck = true;
          this.setState('connected');
          this.startHeartbeat();
          this.dispatchEvent(new Event('connected'));
          resolve(true);
        }, { once: true });

        this.socket.addEventListener('error', (event) => {
          window.clearTimeout(timeout);
          this.setState('error');
          this.dispatchEvent(new CustomEvent('error', { detail: { message: 'Could not connect to the BYAKU backend.', raw: event } }));
          reject(event);
        }, { once: true });
      }).then(() => {
        this.socket.addEventListener('close', () => {
          this.clearHeartbeat();
          this.setState('idle');
          this.dispatchEvent(new Event('disconnected'));
          this.scheduleReconnect();
        });

        this.socket.addEventListener('message', (event) => {
          try {
            const payload = JSON.parse(event.data);
            this.lastMessageAt = Date.now();
            if (payload?.type === 'pong') {
              this.heartbeatAck = true;
              return;
            }
            if (payload?.type === 'error') {
              this.dispatchEvent(new CustomEvent('error', { detail: payload }));
            }
            this.dispatchEvent(new CustomEvent('update', { detail: payload }));
          } catch (error) {
            this.dispatchEvent(new CustomEvent('error', { detail: { type: 'error', stage: 'protocol', code: 'invalid_json', message: 'Invalid backend message', recoverable: true, raw: error } }));
          }
        });
        return true;
      });
    }

    send(payload) {
      if (!payload || this.socket?.readyState !== WebSocket.OPEN) return false;
      try {
        this.socket.send(JSON.stringify(payload));
        return true;
      } catch (error) {
        this.dispatchEvent(new CustomEvent('error', { detail: { type: 'error', stage: 'socket', code: 'send_failed', message: 'Could not send data to the backend.', recoverable: true, raw: error } }));
        return false;
      }
    }

    close() {
      this.shouldReconnect = false;
      this.clearHeartbeat();
      window.clearTimeout(this.reconnectTimer);
      if (this.socket) {
        try {
          this.socket.close();
        } catch {
          // ignore double-close
        }
      }
      this.socket = null;
      this.setState('idle');
    }

    resume() {
      this.shouldReconnect = true;
      this.connectionAttempt = 0;
      return this.connect();
    }
  }

  class BYAKUFrameProcessor extends EventTarget {
    constructor(connection, video, maxWidth = 640) {
      super();
      this.connection = connection;
      this.video = video;
      this.maxWidth = maxWidth;
      this.canvas = document.createElement('canvas');
      this.context = this.canvas.getContext('2d', { alpha: false });
      this.timer = 0;
      this.frameId = 0;
      this.busy = false;
      this.capturing = false;
      this.running = false;
      this.pendingFrameId = null;
      this.pendingFrameStartedAt = 0;
      this.pendingFrameSentAt = 0;
      this.pendingFrameDataUrl = null;
      this.pendingQuery = null;
      this.pendingQueryRequestedAt = 0;
      this.queryFrameId = null;
      this.frameTimeoutMs = 90000;
      this.keepAliveTimer = null;
      this.onReconnect = () => {
        if (this.running) this.connection.send({ type: 'start_camera' });
      };
      this.onDisconnect = () => {
        window.clearTimeout(this.keepAliveTimer);
        this.keepAliveTimer = null;
        this.busy = false;
        this.pendingFrameId = null;
        this.pendingFrameDataUrl = null;
        this.queryFrameId = null;
      };
      this.onUpdate = (event) => {
        const message = event.detail;
        if (message?.type === 'frame_result' && this.pendingFrameId && Number(message.frame_id) === this.pendingFrameId) {
          const receivedAt = performance.now();
          message.client_frame_data_url = this.pendingFrameDataUrl;
          message.client_frame_age_ms = Math.round(receivedAt - this.pendingFrameStartedAt);
          message.client_round_trip_ms = Math.round(receivedAt - this.pendingFrameSentAt);
          window.clearTimeout(this.keepAliveTimer);
          this.keepAliveTimer = null;
          this.busy = false;
          this.pendingFrameId = null;
          this.pendingFrameDataUrl = null;
          if (this.queryFrameId === Number(message.frame_id) && this.pendingQuery) {
            const query = this.pendingQuery;
            this.pendingQuery = null;
            this.pendingQueryRequestedAt = 0;
            this.queryFrameId = null;
            this.connection.send({ type: 'query', text: query });
          }
        } else if (message?.type === 'error') {
          if (!message.frame_id || Number(message.frame_id) === this.pendingFrameId) {
            window.clearTimeout(this.keepAliveTimer);
            this.keepAliveTimer = null;
            this.busy = false;
            this.pendingFrameId = null;
            this.pendingFrameDataUrl = null;
            if (this.queryFrameId === Number(message.frame_id)) this.queryFrameId = null;
          }
        }
        this.dispatchEvent(new CustomEvent('result', { detail: message }));
      };
    }

    start() {
      if (this.running) return;
      this.running = true;
      this.connection.addEventListener('update', this.onUpdate);
      this.connection.addEventListener('connected', this.onReconnect);
      this.connection.addEventListener('disconnected', this.onDisconnect);
      this.connection.send({ type: 'start_camera' });
      this.connection.setState('streaming');
      this.timer = window.setInterval(() => this.capture(), SCAN_INTERVAL_MS);
      void this.capture();
    }

    pause() {
      if (!this.running) return;
      this.running = false;
      window.clearInterval(this.timer);
      this.timer = 0;
      this.connection.setState('paused');
    }

    resume() {
      if (this.running) return;
      this.running = true;
      this.connection.send({ type: 'start_camera' });
      this.connection.setState('streaming');
      this.timer = window.setInterval(() => this.capture(), SCAN_INTERVAL_MS);
      void this.capture();
    }

    stop() {
      this.running = false;
      this.busy = false;
      this.pendingFrameId = null;
      this.pendingFrameDataUrl = null;
      this.pendingQuery = null;
      this.queryFrameId = null;
      window.clearInterval(this.timer);
      this.timer = 0;
      if (this.keepAliveTimer) {
        window.clearTimeout(this.keepAliveTimer);
        this.keepAliveTimer = null;
      }
      this.connection.removeEventListener('update', this.onUpdate);
      this.connection.removeEventListener('connected', this.onReconnect);
      this.connection.removeEventListener('disconnected', this.onDisconnect);
      this.connection.send({ type: 'stop_camera' });
      this.connection.setState('paused');
    }

    sendQuery(text) {
      if (!this.running || typeof text !== 'string' || !text.trim() || this.pendingQuery) return false;
      this.pendingQuery = text.trim();
      this.pendingQueryRequestedAt = performance.now();
      if (!this.busy && !this.capturing) void this.capture();
      return true;
    }

    async capture() {
      if (!this.running || this.busy || this.capturing || this.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
      const sourceWidth = this.video.videoWidth;
      const sourceHeight = this.video.videoHeight;
      if (!sourceWidth || !sourceHeight) return;

      this.capturing = true;
      const frameStartedAt = performance.now();
      const scale = Math.min(1, this.maxWidth / sourceWidth);
      const width = Math.max(1, Math.round(sourceWidth * scale));
      const height = Math.max(1, Math.round(sourceHeight * scale));
      this.canvas.width = width;
      this.canvas.height = height;
      this.context.drawImage(this.video, 0, 0, width, height);

      try {
        const blob = await new Promise((resolve, reject) => {
          this.canvas.toBlob((value) => value ? resolve(value) : reject(new Error('Could not encode camera frame.')), 'image/jpeg', 0.7);
        });
        const encodedFrame = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(String(reader.result));
          reader.onerror = () => reject(new Error('Could not read encoded camera frame.'));
          reader.readAsDataURL(blob);
        });

        if (!this.running) {
          this.capturing = false;
          return;
        }
        const frameId = ++this.frameId;
        this.pendingFrameId = frameId;
        this.pendingFrameDataUrl = encodedFrame;
        this.pendingFrameStartedAt = frameStartedAt;
        this.pendingFrameSentAt = performance.now();
        this.busy = this.connection.send({
          type: 'frame',
          frame_id: frameId,
          width,
          height,
          data: encodedFrame.split(',')[1],
        });
        this.capturing = false;
        if (!this.busy) {
          this.pendingFrameId = null;
          this.pendingFrameDataUrl = null;
          return;
        }

        if (this.pendingQuery && frameStartedAt >= this.pendingQueryRequestedAt) {
          this.queryFrameId = frameId;
        }
        this.keepAliveTimer = window.setTimeout(() => {
          if (this.busy && this.pendingFrameId === frameId) {
            this.dispatchEvent(new CustomEvent('result', { detail: { type: 'error', stage: 'camera', frame_id: frameId, code: 'frame_slow', message: 'Live inference is still processing this frame. Waiting for its result; newer frames are paused to prevent a stale-frame backlog.', recoverable: true } }));
          }
        }, this.frameTimeoutMs);
      } catch (error) {
        this.capturing = false;
        this.busy = false;
        this.pendingFrameId = null;
        this.pendingFrameDataUrl = null;
        this.dispatchEvent(new CustomEvent('result', { detail: { type: 'error', stage: 'camera', code: 'capture_failed', message: error.message || 'Camera frame capture failed.', recoverable: true } }));
      }
    }

    static drawOverlay(canvas, frameSize, objects) {
      const context = canvas?.getContext('2d');
      const width = canvas?.clientWidth || 0;
      const height = canvas?.clientHeight || 0;
      const sourceWidth = Number(frameSize?.width) || 0;
      const sourceHeight = Number(frameSize?.height) || 0;
      if (!context || !width || !height || !sourceWidth || !sourceHeight) return;

      const ratio = window.devicePixelRatio || 1;
      if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
      }
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);

      const scale = Math.max(width / sourceWidth, height / sourceHeight);
      const offsetX = (width - sourceWidth * scale) / 2;
      const offsetY = (height - sourceHeight * scale) / 2;
      context.lineWidth = 2;
      context.font = '600 12px "DM Sans", sans-serif';
      for (const object of objects || []) {
        const box = object.bounding_box || object.bbox;
        if (!Array.isArray(box) || box.length !== 4) continue;
        const [x1, y1, x2, y2] = box.map(Number);
        if (![x1, y1, x2, y2].every(Number.isFinite)) continue;
        const left = offsetX + x1 * scale;
        const top = offsetY + y1 * scale;
        const boxWidth = (x2 - x1) * scale;
        const boxHeight = (y2 - y1) * scale;
        if (boxWidth <= 0 || boxHeight <= 0) continue;

        const rawDepth = object.depth_category;
        const relativeDepth = typeof object.relative_depth_value === 'number'
          && Number.isFinite(object.relative_depth_value)
          && object.relative_depth_value >= 0
          && object.relative_depth_value <= 1
          ? object.relative_depth_value
          : null;
        const depthCategory = ['NEAR', 'MID-RANGE', 'FAR'].includes(rawDepth)
          ? rawDepth
          : relativeDepth === null
            ? 'UNKNOWN'
            : relativeDepth >= 2 / 3
              ? 'NEAR'
              : relativeDepth >= 1 / 3
                ? 'MID-RANGE'
                : 'FAR';
        const colors = {
          NEAR: '#ef4444',
          'MID-RANGE': '#facc15',
          FAR: '#3b82f6',
          UNKNOWN: '#d68af0',
        };
        context.strokeStyle = colors[depthCategory];
        context.strokeRect(left, top, boxWidth, boxHeight);
        const trackId = object.track_id ?? (
          typeof object.object_id === 'string' && object.object_id.startsWith('track:')
            ? object.object_id.slice('track:'.length)
            : null
        );
        const position = object.horizontal_position || object.position;
        const label = [
          object.class_name || object.class || 'Object',
          trackId != null ? `ID ${trackId}` : 'ID unavailable',
          typeof position === 'string' && position ? position.toUpperCase() : 'POSITION UNKNOWN',
          depthCategory,
        ].join(' | ');
        if (label) {
          const labelWidth = context.measureText(label).width + 12;
          const labelTop = Math.max(0, top - 22);
          context.fillStyle = 'rgba(50, 22, 63, .88)';
          context.fillRect(left, labelTop, labelWidth, 20);
          context.fillStyle = '#fff';
          context.fillText(label, left + 6, labelTop + 14);
        }
      }
    }
  }

  window.BYAKUConnection = BYAKUConnection;
  window.BYAKUFrameProcessor = BYAKUFrameProcessor;
})();
