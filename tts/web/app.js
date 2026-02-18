/**
 * VoxCPM TTS Web Client
 * Streaming audio playback support with audio caching
 */

// IndexedDB for audio storage
class AudioStorage {
    constructor() {
        this.dbName = 'VoxCPMAudioCache';
        this.storeName = 'audio';
        this.db = null;
    }

    async init() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.dbName, 1);

            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                this.db = request.result;
                resolve();
            };

            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains(this.storeName)) {
                    const store = db.createObjectStore(this.storeName, { keyPath: 'id' });
                    store.createIndex('timestamp', 'timestamp', { unique: false });
                }
            };
        });
    }

    async saveAudio(id, audioData, metadata) {
        if (!this.db) await this.init();

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);
            const request = store.put({
                id,
                audioData, // Int16Array
                metadata,
                timestamp: Date.now()
            });

            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    async getAudio(id) {
        if (!this.db) await this.init();

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readonly');
            const store = transaction.objectStore(this.storeName);
            const request = store.get(id);

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async deleteAudio(id) {
        if (!this.db) await this.init();

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);
            const request = store.delete(id);

            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    async getAll() {
        if (!this.db) await this.init();

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readonly');
            const store = transaction.objectStore(this.storeName);
            const request = store.getAll();

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async clear() {
        if (!this.db) await this.init();

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.storeName], 'readwrite');
            const store = transaction.objectStore(this.storeName);
            const request = store.clear();

            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    async getStorageInfo() {
        if (!this.db) await this.init();

        return new Promise((resolve) => {
            const transaction = this.db.transaction([this.storeName], 'readonly');
            const store = transaction.objectStore(this.storeName);
            const countRequest = store.count();

            countRequest.onsuccess = () => {
                // Estimate storage size (rough estimate)
                const estimate = countRequest.result * 2; // Rough estimate per entry
                resolve({ count: countRequest.result, sizeEstimate: estimate });
            };
        });
    }
}

class VoxCPMClient {
    constructor() {
        this.ws = null;
        this.requestId = null;
        this.audioContext = null;
        this.audioQueue = [];
        this.isPlaying = false;
        this.currentSource = null;
        this.audioBuffer = [];  // Current audio being generated
        this.completeAudio = null;  // Complete audio for replay
        this.completeAudioSampleRate = 24000;
        this.sampleRate = 24000;
        this.history = this.loadHistory();
        this.isProcessing = false;
        this.audioStorage = new AudioStorage();
        this.currentRequestId = null;
        this.historyMetadata = this.loadHistoryMetadata();
        this.voices = {};  // Available voices

        this.initAudioContext();
        this.initEventListeners();
        this.connect();
        this.audioStorage.init().catch(console.error);
        this.loadVoices();  // Load voices
    }

    async initAudioContext() {
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
    }

    connect() {
        const wsUrl = `ws://${window.location.hostname}:9300/tts`;
        this.updateStatus('connecting', '正在连接...');
        this.setReconnectButtonState('connecting');

        this.ws = new WebSocket(wsUrl);
        this.ws.binaryType = 'arraybuffer';

        this.ws.onopen = () => {
            console.log('[WebSocket] Connected to', wsUrl);
            this.updateStatus('connected', '已连接');
            this.setReconnectButtonState('connected');
        };

        this.ws.onclose = (event) => {
            console.log('[WebSocket] Connection closed:', {
                code: event.code,
                reason: event.reason,
                wasClean: event.wasClean
            });
            this.updateStatus('disconnected', `连接断开 (${event.code})`);
            this.stopPlayback();
            this.setReconnectButtonState('disconnected');

            // If we were processing, the request was interrupted
            if (this.isProcessing) {
                this.isProcessing = false;
                this.setGenerateButtonState(false);
                this.showVisualizer(false);
                this.showError('连接中断，请点击重新连接后重试');
            }
        };

        this.ws.onerror = (error) => {
            console.error('[WebSocket] Error:', error);
            this.showError('连接错误，请检查服务器是否运行');
        };

        this.ws.onmessage = (event) => this.handleMessage(event.data);
    }

    reconnect() {
        // Prevent multiple reconnect attempts
        if (this.isReconnecting) {
            console.log('[WebSocket] Already reconnecting...');
            return;
        }

        this.isReconnecting = true;
        console.log('[WebSocket] Manual reconnect initiated...');

        // Close existing connection if any
        if (this.ws) {
            this.ws.onclose = null; // Remove the onclose handler to avoid triggering disconnected state
            this.ws.close();
        }

        // Reset state
        this.stopPlayback();
        this.audioBuffer = [];
        if (this.isProcessing) {
            this.isProcessing = false;
            this.setGenerateButtonState(false);
            this.showVisualizer(false);
        }

        // Reconnect
        this.connect();

        // Clear reconnecting flag after a short delay
        setTimeout(() => {
            this.isReconnecting = false;
        }, 2000);
    }

    setReconnectButtonState(state) {
        const btn = document.getElementById('reconnectBtn');
        if (!btn) return;

        btn.className = 'reconnect-btn'; // Reset classes
        btn.disabled = false;

        switch (state) {
            case 'connecting':
                btn.disabled = true;
                btn.classList.add('connecting');
                btn.textContent = '⏳ 连接中...';
                break;
            case 'connected':
                btn.textContent = '🔄 重新连接';
                break;
            case 'disconnected':
                btn.style.background = 'var(--warning-color)';
                btn.textContent = '⚠️ 重新连接';
                break;
            default:
                btn.textContent = '🔄 重新连接';
        }
    }

    async handleMessage(data) {
        if (data instanceof ArrayBuffer) {
            // Binary audio data
            await this.handleAudioData(data);
        } else {
            // JSON control message
            const message = JSON.parse(data);
            this.handleControlMessage(message);
        }
    }

    async handleAudioData(data) {
        const { metadata, audio } = this.parseBinaryFrame(data);

        // Store audio chunk with its metadata (including sample rate)
        this.audioBuffer.push({
            data: audio,
            sampleRate: metadata.sample_rate || 24000,
            sequence: metadata.sequence
        });

        // Update global sample rate for display purposes
        if (metadata.sample_rate) {
            this.sampleRate = metadata.sample_rate;
        }

        // Start playback if not already playing (streaming mode)
        if (this.audioContext && !this.isPlaying && this.audioBuffer.length > 0) {
            await this.startStreamingPlayback();
        }
    }

    parseBinaryFrame(buffer) {
        const view = new DataView(buffer);

        // Check magic number
        const magic = view.getUint16(0, false);  // big-endian
        if (magic !== 0xAA55) {
            throw new Error('Invalid frame format');
        }

        const msgType = view.getUint8(2);
        const metadataLength = view.getUint32(4, false);  // big-endian

        // Parse metadata JSON
        const metadataBytes = new Uint8Array(buffer, 8, metadataLength);
        const metadata = JSON.parse(new TextDecoder().decode(metadataBytes));

        // Calculate offsets (no padding)
        const payloadLengthOffset = 8 + metadataLength;
        const payloadLength = view.getUint32(payloadLengthOffset, false);  // big-endian
        const audioDataOffset = payloadLengthOffset + 4;

        // Parse audio payload - extract bytes first
        const audioDataBytes = new Uint8Array(buffer, audioDataOffset, payloadLength);

        // Create Int16Array from the audio bytes (little-endian PCM 16-bit)
        const audioData = new Int16Array(audioDataBytes.length / 2);
        for (let i = 0; i < audioData.length; i++) {
            // PCM 16-bit is stored as little-endian
            const lowByte = audioDataBytes[i * 2];
            const highByte = audioDataBytes[i * 2 + 1];
            audioData[i] = lowByte | (highByte << 8);
        }

        return { metadata, audio: audioData };
    }

    async startStreamingPlayback() {
        if (this.audioBuffer.length === 0) return;

        this.isPlaying = true;
        this.updateStatus('processing', '正在播放...');

        // Use a single AudioBufferSource with scheduled playback for seamless streaming
        // We'll queue chunks as they arrive and play them continuously
        this._streamingQueue = [];
        // Start scheduling from the current AudioContext time
        this._streamingNextPlayTime = this.audioContext.currentTime;

        const processNextChunk = async () => {
            if (!this.isPlaying) {
                this._streamingQueue = [];
                return;
            }

            if (this.audioBuffer.length === 0) {
                // Wait for more audio
                setTimeout(processNextChunk, 10);
                return;
            }

            const audioChunk = this.audioBuffer.shift();
            const audioData = audioChunk.data;
            const sampleRate = audioChunk.sampleRate;

            // Convert Int16 to Float32
            const float32Buffer = new Float32Array(audioData.length);
            for (let i = 0; i < audioData.length; i++) {
                float32Buffer[i] = audioData[i] / 32768.0;
            }

            // Create audio buffer with the correct sample rate
            const audioBuffer = this.audioContext.createBuffer(
                1,
                float32Buffer.length,
                sampleRate
            );
            audioBuffer.getChannelData(0).set(float32Buffer);

            // Calculate when this chunk should start playing
            // Use the next available time in the audio context
            let startTime = this._streamingNextPlayTime;

            // If the scheduled time is in the past, start immediately
            const currentTime = this.audioContext.currentTime;
            if (startTime < currentTime) {
                startTime = currentTime;
            }

            // Create source and schedule playback
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);

            // Update the next play time BEFORE starting the source
            this._streamingNextPlayTime = startTime + audioBuffer.duration;

            // Schedule this chunk to play
            source.start(startTime);

            // Clean up the source after playing
            source.onended = () => {
                // Source will be garbage collected automatically
            };

            // Keep track of the current source for stopping
            this.currentSource = source;

            // Process next chunk immediately to queue it up
            setTimeout(processNextChunk, 0);
        };

        processNextChunk();
    }

    stopPlayback() {
        this.isPlaying = false;
        if (this.currentSource) {
            try {
                this.currentSource.stop();
            } catch (e) {
                // Ignore if already stopped
            }
            this.currentSource = null;
        }
    }

    async playCachedAudio(audioData, sampleRate) {
        // Validate audio data
        if (!audioData || audioData.length === 0) {
            this.showError('音频数据为空');
            return;
        }

        // Stop any current playback
        this.stopPlayback();

        // Convert Int16 to Float32
        const float32Buffer = new Float32Array(audioData.length);
        for (let i = 0; i < audioData.length; i++) {
            float32Buffer[i] = audioData[i] / 32768.0;
        }

        // Create audio buffer
        const audioBuffer = this.audioContext.createBuffer(1, float32Buffer.length, sampleRate);
        audioBuffer.getChannelData(0).set(float32Buffer);

        // Create source and play
        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audioContext.destination);

        source.onended = () => {
            this.isPlaying = false;
        };

        this.currentSource = source;
        this.isPlaying = true;
        source.start();
    }

    handleControlMessage(message) {
        switch (message.type) {
            case 'progress':
                this.handleProgress(message);
                break;
            case 'complete':
                this.handleComplete(message);
                break;
            case 'error':
                this.handleError(message);
                break;
            case 'pong':
                // Handle pong if needed
                break;
        }
    }

    handleProgress(message) {
        const progress = message.progress * 100;
        this.updateProgress(progress);
        this.updateStatus('processing', message.message || '处理中...');

        if (message.state === 'generating') {
            this.showVisualizer(true);
        }
    }

    async handleComplete(message) {
        this.isProcessing = false;
        this.updateProgress(100);

        const result = message.result || {};

        if (!result.cancelled) {
            // Save audio to IndexedDB
            await this.saveAudioToStorage(result);

            // Update stats
            this.updateStats(result);

            // Add to history
            this.addToHistory(result);

            // Show download button
            document.getElementById('downloadBtn').disabled = false;
            document.getElementById('playCurrentBtn').disabled = false;

            this.updateStatus('connected', '生成完成');
        } else {
            this.updateStatus('connected', '已取消');
        }

        this.showVisualizer(false);
        this.setGenerateButtonState(false);
    }

    handleError(message) {
        this.isProcessing = false;
        this.showError(message.error?.message || '生成失败');
        this.updateStatus('error', '错误');
        this.showVisualizer(false);
        this.setGenerateButtonState(false);
    }

    async saveAudioToStorage(result) {
        // Combine all audio chunks (each chunk is now an object with data property)
        const totalLength = this.audioBuffer.reduce((sum, chunk) => sum + chunk.data.length, 0);
        const combinedAudio = new Int16Array(totalLength);
        let offset = 0;

        for (const chunk of this.audioBuffer) {
            combinedAudio.set(chunk.data, offset);
            offset += chunk.data.length;
        }

        // Store complete audio for replay
        this.completeAudio = combinedAudio;
        this.completeAudioSampleRate = result.sample_rate || this.sampleRate;

        // Save to IndexedDB
        await this.audioStorage.saveAudio(
            this.currentRequestId,
            combinedAudio,
            {
                text: document.getElementById('textInput').value.trim(),
                duration: result.duration,
                sampleRate: result.sample_rate || this.sampleRate,
                timestamp: Date.now()
            }
        );
    }

    async generateSpeech() {
        const text = document.getElementById('textInput').value.trim();
        if (!text) {
            this.showError('请输入要合成的文本');
            return;
        }

        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            this.showError('WebSocket 未连接');
            return;
        }

        await this.initAudioContext();

        // Reset state
        this.audioBuffer = [];
        this.isProcessing = true;
        this.currentRequestId = crypto.randomUUID();
        this.setGenerateButtonState(true);
        document.getElementById('downloadBtn').disabled = true;
        document.getElementById('playCurrentBtn').disabled = true;
        this.hideError();

        // Gather parameters
        const params = {
            text: text,
            mode: document.getElementById('modeSelect').value,
            cfg_value: parseFloat(document.getElementById('cfgSlider').value),
            inference_timesteps: parseInt(document.getElementById('timestepsSlider').value),
            normalize: document.getElementById('normalizeToggle').checked,
            denoise: document.getElementById('denoiseToggle').checked,
            retry_badcase: document.getElementById('retryToggle').checked
        };

        // Add voice_id if selected
        const voiceId = document.getElementById('voiceSelect').value;
        if (voiceId) {
            params.voice_id = voiceId;
        }

        const promptWavUrl = document.getElementById('promptWavUrl').value.trim();
        if (promptWavUrl) {
            params.prompt_wav_url = promptWavUrl;
        }

        // Send request
        const request = {
            type: 'tts_request',
            request_id: this.currentRequestId,
            params: params
        };

        this.ws.send(JSON.stringify(request));
        this.updateStatus('processing', '请求已发送...');
    }

    async loadVoices() {
        try {
            const response = await fetch('/api/voices');
            if (!response.ok) {
                console.error('Failed to load voices:', response.statusText);
                return;
            }

            const data = await response.json();
            this.voices = data.voices || {};
            this.populateVoiceSelect();
            console.log(`Loaded ${Object.keys(this.voices).length} voice categories`);
        } catch (error) {
            console.error('Error loading voices:', error);
        }
    }

    populateVoiceSelect() {
        const select = document.getElementById('voiceSelect');
        select.innerHTML = '<option value="">默认声音</option>';

        for (const [category, voices] of Object.entries(this.voices)) {
            const optgroup = document.createElement('optgroup');
            optgroup.label = `${category} (${voices.length})`;

            for (const voice of voices) {
                const option = document.createElement('option');
                option.value = voice.id;
                option.textContent = voice.name;
                option.dataset.category = category;
                optgroup.appendChild(option);
            }

            select.appendChild(optgroup);
        }
    }

    async refreshVoices() {
        const btn = document.getElementById('refreshVoicesBtn');
        btn.disabled = true;
        btn.textContent = '⏳ 加载中...';

        await this.loadVoices();

        btn.disabled = false;
        btn.textContent = '🔄 刷新声音列表';
    }

    stopGeneration() {
        if (this.ws && this.currentRequestId) {
            const cancelRequest = {
                type: 'cancel',
                request_id: this.currentRequestId
            };
            this.ws.send(JSON.stringify(cancelRequest));
        }

        this.stopPlayback();
        this.isProcessing = false;
        this.setGenerateButtonState(false);
        this.showVisualizer(false);
    }

    async playCurrentAudio() {
        let audioToPlay = null;
        let sampleRate = this.sampleRate;

        // Use complete audio if available (saved from previous generation)
        if (this.completeAudio && this.completeAudio.length > 0) {
            audioToPlay = this.completeAudio;
            sampleRate = this.completeAudioSampleRate;
        } else if (this.audioBuffer.length > 0) {
            // Fallback to current buffer (combine chunks)
            const totalLength = this.audioBuffer.reduce((sum, chunk) => sum + chunk.data.length, 0);
            audioToPlay = new Int16Array(totalLength);
            let offset = 0;

            for (const chunk of this.audioBuffer) {
                audioToPlay.set(chunk.data, offset);
                offset += chunk.data.length;
            }
        }

        if (!audioToPlay || audioToPlay.length === 0) {
            this.showError('没有可播放的音频，请先生成语音');
            return;
        }

        await this.playCachedAudio(audioToPlay, sampleRate);
    }

    async playHistoryAudio(id) {
        const record = await this.audioStorage.getAudio(id);
        if (!record) {
            this.showError('音频缓存不存在');
            return;
        }

        const { audioData, metadata } = record;

        // Update text input with history text
        document.getElementById('textInput').value = metadata.text;

        // Update stats display
        this.updateStats({
            duration: metadata.duration,
            samples: audioData.length,
            sample_rate: metadata.sampleRate
        });

        // Show audio player (remove empty state)
        const player = document.getElementById('audioPlayer');
        player.style.display = 'block';
        player.classList.remove('empty');

        // Play the cached audio
        await this.playCachedAudio(audioData, metadata.sampleRate);

        // Enable download button
        document.getElementById('downloadBtn').disabled = false;
    }

    downloadAudio() {
        let audioToDownload = null;
        let sampleRate = this.sampleRate;

        // Use complete audio if available
        if (this.completeAudio && this.completeAudio.length > 0) {
            audioToDownload = this.completeAudio;
            sampleRate = this.completeAudioSampleRate;
        } else if (this.audioBuffer.length > 0) {
            // Fallback to current buffer (combine chunks)
            const totalLength = this.audioBuffer.reduce((sum, chunk) => sum + chunk.data.length, 0);
            audioToDownload = new Int16Array(totalLength);
            let offset = 0;

            for (const chunk of this.audioBuffer) {
                audioToDownload.set(chunk.data, offset);
                offset += chunk.data.length;
            }
        }

        if (!audioToDownload || audioToDownload.length === 0) {
            this.showError('没有可下载的音频，请先生成语音');
            return;
        }

        // Create WAV file
        const wav = this.createWavFile(audioToDownload, sampleRate);
        const blob = new Blob([wav], { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `voxcpm-tts-${Date.now()}.wav`;
        a.click();
        URL.revokeObjectURL(url);
    }

    downloadHistoryAudio(id) {
        this.audioStorage.getAudio(id).then(record => {
            if (!record) return;

            const { audioData, metadata } = record;

            // Create WAV file
            const wav = this.createWavFile(audioData, metadata.sampleRate);
            const blob = new Blob([wav], { type: 'audio/wav' });
            const url = URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.href = url;
            a.download = `voxcpm-tts-${id}.wav`;
            a.click();
            URL.revokeObjectURL(url);
        });
    }

    deleteHistoryItem(id) {
        this.audioStorage.deleteAudio(id);

        // Remove from history metadata
        this.historyMetadata = this.historyMetadata.filter(item => item.id !== id);
        this.saveHistoryMetadata();

        // Re-render history
        this.renderHistory();
    }

    async clearHistory() {
        await this.audioStorage.clear();
        this.historyMetadata = [];
        this.saveHistory();
        this.renderHistory();
    }

    createWavFile(audioData, sampleRate) {
        const numSamples = audioData.length;
        const buffer = new ArrayBuffer(44 + numSamples * 2);
        const view = new DataView(buffer);

        // WAV header
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + numSamples * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(36, 'data');
        view.setUint32(40, numSamples * 2, true);

        // Write audio data
        for (let i = 0; i < numSamples; i++) {
            view.setInt16(44 + i * 2, audioData[i], true);
        }

        return buffer;
    }

    updateStatus(state, message) {
        const indicator = document.getElementById('statusIndicator');
        const text = document.getElementById('statusText');

        indicator.className = 'status-indicator';
        if (state === 'connected') {
            indicator.classList.add('connected');
        } else if (state === 'processing') {
            indicator.classList.add('processing');
        } else if (state === 'disconnected') {
            indicator.classList.add('disconnected');
        }

        text.textContent = message;
    }

    updateProgress(percent) {
        document.getElementById('progressFill').style.width = `${percent}%`;
    }

    updateStats(result) {
        document.getElementById('statDuration').textContent =
            `${(result.duration || 0).toFixed(2)}s`;
        document.getElementById('statSamples').textContent =
            (result.samples || 0).toLocaleString();
        document.getElementById('statSampleRate').textContent =
            `${(result.sample_rate || 24000) / 1000}kHz`;

        document.getElementById('audioStats').style.display = 'grid';

        // Replace empty player with visualizer
        const player = document.getElementById('audioPlayer');
        if (player.classList.contains('empty')) {
            player.style.display = 'none';
            player.classList.remove('empty');
        }
    }

    showVisualizer(show) {
        document.getElementById('audioVisualizer').style.display = show ? 'flex' : 'none';
    }

    setGenerateButtonState(generating) {
        document.getElementById('generateBtn').disabled = generating;
        document.getElementById('stopBtn').disabled = !generating;
    }

    showError(message) {
        const errorEl = document.getElementById('errorMessage');
        errorEl.textContent = `❌ ${message}`;
        errorEl.classList.add('show');
        setTimeout(() => errorEl.classList.remove('show'), 5000);
    }

    hideError() {
        document.getElementById('errorMessage').classList.remove('show');
    }

    async addToHistory(result) {
        const text = document.getElementById('textInput').value.trim();
        const item = {
            id: this.currentRequestId,
            text: text,
            duration: result.duration,
            sampleRate: result.sample_rate || this.sampleRate,
            samples: result.samples || 0,
            timestamp: Date.now(),
            hasAudio: true
        };

        this.historyMetadata.unshift(item);
        if (this.historyMetadata.length > 20) {
            // Remove old items from IndexedDB
            const removedItem = this.historyMetadata.pop();
            if (removedItem.hasAudio) {
                await this.audioStorage.deleteAudio(removedItem.id);
            }
        }

        this.saveHistoryMetadata();
        this.renderHistory();
    }

    loadHistory() {
        try {
            return JSON.parse(localStorage.getItem('voxcpm-history') || '[]');
        } catch {
            return [];
        }
    }

    loadHistoryMetadata() {
        try {
            return JSON.parse(localStorage.getItem('voxcpm-history-metadata') || '[]');
        } catch {
            return [];
        }
    }

    saveHistory() {
        localStorage.setItem('voxcpm-history', JSON.stringify(this.history));
    }

    saveHistoryMetadata() {
        localStorage.setItem('voxcpm-history-metadata', JSON.stringify(this.historyMetadata));
    }

    async renderHistory() {
        const container = document.getElementById('historyList');
        const clearBtn = document.getElementById('clearHistoryBtn');
        const storageInfo = document.getElementById('storageInfo');

        // Update storage info
        const storageData = await this.audioStorage.getStorageInfo();
        storageInfo.textContent = `已缓存 ${storageData.count} 个音频`;

        // Show/hide clear button
        clearBtn.style.display = this.historyMetadata.length > 0 ? 'block' : 'none';

        if (this.historyMetadata.length === 0) {
            container.innerHTML = '<div class="empty-history">暂无历史记录</div>';
            return;
        }

        container.innerHTML = this.historyMetadata.map(item => `
            <div class="history-item">
                <div class="history-item-header">
                    <div class="history-item-text">${this.escapeHtml(item.text.substring(0, 80))}${item.text.length > 80 ? '...' : ''}</div>
                    <div class="history-item-actions">
                        ${item.hasAudio ? `<button class="history-btn play-btn" onclick="client.playHistoryAudio('${item.id}')">▶️ 播放</button>` : ''}
                        ${item.hasAudio ? `<button class="history-btn" onclick="client.downloadHistoryAudio('${item.id}')">💾</button>` : ''}
                        <button class="history-btn" onclick="client.loadTextFromHistory('${item.id}')">📝</button>
                        <button class="history-btn" onclick="client.deleteHistoryItem('${item.id}')">🗑️</button>
                    </div>
                </div>
                <div class="history-item-meta">
                    ${(item.duration || 0).toFixed(2)}s • ${new Date(item.timestamp).toLocaleString()}
                </div>
            </div>
        `).join('');
    }

    loadTextFromHistory(id) {
        const item = this.historyMetadata.find(h => h.id === id);
        if (item) {
            document.getElementById('textInput').value = item.text;
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    initEventListeners() {
        document.getElementById('generateBtn').addEventListener('click', () => this.generateSpeech());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopGeneration());
        document.getElementById('downloadBtn').addEventListener('click', () => this.downloadAudio());
        document.getElementById('playCurrentBtn').addEventListener('click', () => this.playCurrentAudio());
        document.getElementById('reconnectBtn').addEventListener('click', () => this.reconnect());
        document.getElementById('clearHistoryBtn').addEventListener('click', () => this.clearHistory());
        document.getElementById('refreshVoicesBtn').addEventListener('click', () => this.refreshVoices());

        // Sliders
        document.getElementById('cfgSlider').addEventListener('input', (e) => {
            document.getElementById('cfgValue').textContent = parseFloat(e.target.value).toFixed(1);
        });

        document.getElementById('timestepsSlider').addEventListener('input', (e) => {
            document.getElementById('timestepsValue').textContent = e.target.value;
        });

        // Enter key to generate (Ctrl+Enter in textarea)
        document.getElementById('textInput').addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                this.generateSpeech();
            }
        });
    }
}

// Initialize client
let client;
document.addEventListener('DOMContentLoaded', () => {
    client = new VoxCPMClient();
    client.renderHistory();
});
