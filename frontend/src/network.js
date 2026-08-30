/**
 * WebSocket 客户端
 * =================
 * 管理与后端服务器的 WebSocket 连接。
 */

export class NetworkClient {
  constructor(url) {
    this.url = url;
    this.ws = null;
    this.handlers = new Map();  // msgType → [callback, ...]
    this.reconnectDelay = 2000;
    this.maxReconnectDelay = 30000;
    this.connected = false;
    this.reconnects = 0;
    this.messagesReceived = 0;
    this.messagesSent = 0;
    this.bytesReceived = 0;
    this.bytesSent = 0;
    this.sendRejected = 0;
    this.bufferedAmountHighWater = 0;
    this.rttMs = null;
    this.lastMessageAt = null;
    this._pingTimer = null;
    this._reconnectTimer = null;
    this._manualDisconnect = false;
  }

  connect() {
    if (this.ws && (
      this.ws.readyState === WebSocket.OPEN
      || this.ws.readyState === WebSocket.CONNECTING
    )) return;
    this._manualDisconnect = false;
    try {
      this.ws = new WebSocket(this.url);
      this.ws.binaryType = 'arraybuffer';

      this.ws.onopen = () => {
        console.log('[WS] Connected');
        this.connected = true;
        this.reconnectDelay = 2000;
        this._startHeartbeat();
        this._emit('connection', { connected: true });
      };

      this.ws.onmessage = (event) => {
        const size = typeof event.data === 'string'
          ? new TextEncoder().encode(event.data).byteLength
          : Number(event.data?.byteLength || event.data?.size || 0);
        this.messagesReceived += 1;
        this.bytesReceived += size;
        this.lastMessageAt = performance.now();
        if (typeof event.data !== 'string') {
          this._emit('binary', event.data);
          return;
        }
        try {
          const data = JSON.parse(event.data);
          const type = data.type;
          const payload = data.payload;

          if (type === 'pong' && Number.isFinite(Number(payload?.client_time_ms))) {
            this.rttMs = Math.max(0, performance.now() - Number(payload.client_time_ms));
          }

          // 调度到注册的处理程序
          if (this.handlers.has(type)) {
            for (const cb of this.handlers.get(type)) {
              cb(payload);
            }
          }

          // 通用消息处理
          if (this.handlers.has('*')) {
            for (const cb of this.handlers.get('*')) {
              cb(type, payload);
            }
          }
        } catch (e) {
          console.error('[WS] Message parse error:', e);
        }
      };

      this.ws.onclose = () => {
        console.log('[WS] Disconnected');
        this.connected = false;
        this._stopHeartbeat();
        this._emit('connection', { connected: false });
        if (!this._manualDisconnect) this._scheduleReconnect();
      };

      this.ws.onerror = (err) => {
        // onclose owns reconnect state; a transient transport error during a
        // backend restart is expected and should not masquerade as an app bug.
        console.warn('[WS] Transport interrupted; reconnect pending', err);
      };
    } catch (e) {
      console.error('[WS] Connection failed:', e);
      this._scheduleReconnect();
    }
  }

  send(type, payload = {}) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const message = JSON.stringify({ type, payload });
      this._recordBufferedAmount();
      this.ws.send(message);
      this.messagesSent += 1;
      this.bytesSent += new TextEncoder().encode(message).byteLength;
      return true;
    }
    this.sendRejected += 1;
    return false;
  }

  sendBinary(payload) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this._recordBufferedAmount();
      this.ws.send(payload);
      this.messagesSent += 1;
      this.bytesSent += Number(payload?.byteLength || payload?.size || 0);
      this._recordBufferedAmount();
      return true;
    }
    this.sendRejected += 1;
    return false;
  }

  on(type, callback) {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, []);
    }
    this.handlers.get(type).push(callback);
  }

  _emit(type, payload) {
    if (!this.handlers.has(type)) return;
    for (const callback of this.handlers.get(type)) {
      callback(payload);
    }
  }

  disconnect() {
    this._manualDisconnect = true;
    this._stopHeartbeat();
    clearTimeout(this._reconnectTimer);
    this._reconnectTimer = null;
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  _scheduleReconnect() {
    if (this._reconnectTimer) return;
    console.log(`[WS] Reconnecting in ${this.reconnectDelay}ms...`);
    this.reconnects += 1;
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      this.connect();
      this.reconnectDelay = Math.min(
        this.reconnectDelay * 1.5,
        this.maxReconnectDelay
      );
    }, this.reconnectDelay);
  }

  _recordBufferedAmount() {
    const buffered = Number(this.ws?.bufferedAmount || 0);
    this.bufferedAmountHighWater = Math.max(this.bufferedAmountHighWater, buffered);
    return buffered;
  }

  _startHeartbeat() {
    this._stopHeartbeat();
    const ping = () => {
      if (this.connected) this.send('ping', { client_time_ms: performance.now() });
    };
    ping();
    this._pingTimer = setInterval(ping, 5000);
  }

  _stopHeartbeat() {
    clearInterval(this._pingTimer);
    this._pingTimer = null;
  }

  statistics() {
    return {
      connected: this.connected,
      ready_state: this.ws?.readyState ?? WebSocket.CLOSED,
      rtt_ms: this.rttMs,
      buffered_amount: this._recordBufferedAmount(),
      buffered_amount_high_water: this.bufferedAmountHighWater,
      messages_received: this.messagesReceived,
      messages_sent: this.messagesSent,
      bytes_received: this.bytesReceived,
      bytes_sent: this.bytesSent,
      send_rejected: this.sendRejected,
      reconnects: this.reconnects,
      last_message_age_ms: this.lastMessageAt === null
        ? null
        : Math.max(0, performance.now() - this.lastMessageAt),
    };
  }
}
