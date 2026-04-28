<template>
  <div class="app">
    <!-- Header -->
    <header class="header">
      <div class="header-inner">
        <div class="logo">
          <div class="logo-icon">🧠</div>
          <div class="logo-text-wrap">
            <span class="logo-title">MNIST</span>
            <span class="logo-sub">Neural Network</span>
          </div>
        </div>
        <div class="status-pill" :class="{ active: modelReady, training: isTraining }">
          <span class="status-dot"></span>
          <span>{{ isTraining ? 'TRAINING...' : modelReady ? 'MODEL READY' : 'AWAITING TRAINING' }}</span>
        </div>
      </div>
    </header>

    <main class="main">
      <!-- Panel 01: Training -->
      <section class="panel panel-train">
        <div class="panel-badge">01</div>
        <h2 class="panel-title">TRAINING</h2>

        <div class="train-config">
          <div class="config-row">
            <div class="config-label-row">
              <span class="config-label">EPOCHS</span>
              <span class="config-val">{{ trainConfig.epochs }}</span>
            </div>
            <input class="slider" type="range" v-model.number="trainConfig.epochs" min="1" max="20" :disabled="isTraining" />
          </div>
          <div class="config-row">
            <div class="config-label-row">
              <span class="config-label">BATCH SIZE</span>
              <span class="config-val">{{ trainConfig.batchSize }}</span>
            </div>
            <input class="slider" type="range" v-model.number="trainConfig.batchSize" min="64" max="1024" step="64" :disabled="isTraining" />
          </div>
        </div>

        <button class="btn-train" @click="startTraining" :disabled="isTraining">
          <span v-if="!isTraining">▶ START TRAINING</span>
          <span v-else class="pulse">● TRAINING IN PROGRESS...</span>
        </button>

        <button class="btn-load" @click="loadModel" :disabled="isTraining || !hasSavedModel">
          📂 LOAD SAVED MODEL
          <span class="badge-saved" v-if="hasSavedModel">✓</span>
        </button>

        <!-- Progress -->
        <div v-if="isTraining || trainingProgress.status === 'done'" class="progress-area">
          <div class="progress-label-row">
            <span>EPOCH {{ trainingProgress.epoch }} / {{ trainingProgress.totalEpochs }}</span>
            <span v-if="trainingProgress.status === 'done'" class="badge-done">✓ DONE</span>
          </div>
          <div class="progress-track">
            <div class="progress-fill" :style="{ width: progressPercent + '%' }"></div>
          </div>
          <div v-if="trainingProgress.epoch > 0" class="metrics-grid">
            <div class="metric-card metric-loss">
              <div class="metric-label">LOSS</div>
              <div class="metric-number">{{ trainingProgress.loss?.toFixed(4) }}</div>
            </div>
            <div class="metric-card metric-acc">
              <div class="metric-label">ACCURACY</div>
              <div class="metric-number">{{ (trainingProgress.acc * 100)?.toFixed(1) }}%</div>
            </div>
            <div class="metric-card metric-loss">
              <div class="metric-label">VAL LOSS</div>
              <div class="metric-number">{{ trainingProgress.valLoss?.toFixed(4) }}</div>
            </div>
            <div class="metric-card metric-acc">
              <div class="metric-label">VAL ACC</div>
              <div class="metric-number">{{ (trainingProgress.valAcc * 100)?.toFixed(1) }}%</div>
            </div>
          </div>
          <div v-if="historyData.length > 0" class="chart-box">
            <div class="chart-title">TRAINING CURVE</div>
            <svg viewBox="0 0 220 80" preserveAspectRatio="none" class="chart-svg">
              <defs>
                <linearGradient id="accGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stop-color="#00E676" stop-opacity="0.4"/>
                  <stop offset="100%" stop-color="#00E676" stop-opacity="0"/>
                </linearGradient>
                <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stop-color="#FF1744" stop-opacity="0.3"/>
                  <stop offset="100%" stop-color="#FF1744" stop-opacity="0"/>
                </linearGradient>
              </defs>
              <polyline :points="accPoints" fill="none" stroke="#00E676" stroke-width="2.5"/>
              <polyline :points="lossPoints" fill="none" stroke="#FF1744" stroke-width="2" stroke-dasharray="4,2"/>
            </svg>
            <div class="chart-legend">
              <span class="legend-acc">● ACCURACY</span>
              <span class="legend-loss">● LOSS</span>
            </div>
          </div>
        </div>

        <!-- Architecture -->
        <div class="arch-box">
          <div class="arch-heading">MODEL ARCHITECTURE</div>
          <div class="arch-list">
            <div class="arch-item" v-for="(layer, i) in architecture" :key="i">
              <span class="arch-name">{{ layer.name }}</span>
              <span class="arch-info">{{ layer.info }}</span>
            </div>
          </div>
        </div>
      </section>

      <!-- Panel 02: Draw -->
      <section class="panel panel-draw">
        <div class="panel-badge">02</div>
        <h2 class="panel-title">DRAW & PREDICT</h2>

        <div class="canvas-wrap">
          <canvas
            ref="drawCanvas"
            width="280" height="280"
            class="draw-canvas"
            @mousedown="startDraw"
            @mousemove="draw"
            @mouseup="stopDraw"
            @mouseleave="stopDraw"
            @touchstart.prevent="startDrawTouch"
            @touchmove.prevent="drawTouch"
            @touchend="stopDraw"
          ></canvas>
          <div class="canvas-hint">✏️ Draw a digit (0–9) here</div>
        </div>

        <div class="draw-actions">
          <button class="btn-clear" @click="clearCanvas">⌫ CLEAR</button>
          <button class="btn-predict" @click="predict" :disabled="!modelReady">
            ⚡ PREDICT
          </button>
        </div>

        <div class="preview-wrap">
          <div class="preview-label">28 × 28 INPUT PREVIEW</div>
          <canvas ref="previewCanvas" width="84" height="84" class="preview-canvas"></canvas>
        </div>
      </section>

      <!-- Panel 03: Results -->
      <section class="panel panel-result">
        <div class="panel-badge">03</div>
        <h2 class="panel-title">RESULTS</h2>

        <div v-if="!prediction" class="empty-state">
          <div class="empty-icon">?</div>
          <p>Train model then draw a digit to see predictions</p>
        </div>

        <template v-else>
          <div class="prediction-hero">
            <div class="prediction-digit">{{ prediction.prediction }}</div>
            <div class="prediction-conf">{{ (prediction.confidence * 100).toFixed(1) }}% confidence</div>
          </div>

          <div class="prob-list">
            <div
              class="prob-row"
              v-for="(prob, i) in prediction.probabilities"
              :key="i"
              :class="{ winner: i === prediction.prediction }"
            >
              <span class="prob-label">{{ i }}</span>
              <div class="prob-track">
                <div class="prob-fill" :style="{ width: (prob * 100) + '%' }"></div>
              </div>
              <span class="prob-pct">{{ (prob * 100).toFixed(1) }}%</span>
            </div>
          </div>
        </template>

        <div v-if="predictionHistory.length > 0" class="history-box">
          <div class="history-heading">HISTORY</div>
          <div class="history-list">
            <div class="history-chip" v-for="(item, i) in predictionHistory.slice().reverse()" :key="i">
              <span class="history-digit">{{ item.prediction }}</span>
              <span class="history-conf">{{ (item.confidence * 100).toFixed(0) }}%</span>
            </div>
          </div>
        </div>
      </section>
    </main>
  </div>
</template>

<script>
import axios from 'axios';
const API = 'http://localhost:3001/api';

export default {
  name: 'App',
  data() {
    return {
      modelReady: false,
      isTraining: false,
      hasSavedModel: false,
      trainConfig: { epochs: 5, batchSize: 512 },
      trainingProgress: { epoch: 0, totalEpochs: 5, loss: 0, acc: 0, valLoss: 0, valAcc: 0, status: 'idle' },
      historyData: [],
      prediction: null,
      predictionHistory: [],
      drawing: false,
      lastX: 0, lastY: 0,
      pollInterval: null,
      architecture: [
        { name: 'Conv2D', info: '32 filters · 3×3 · ReLU' },
        { name: 'MaxPool2D', info: '2×2' },
        { name: 'Conv2D', info: '64 filters · 3×3 · ReLU' },
        { name: 'MaxPool2D', info: '2×2' },
        { name: 'Flatten', info: '→ 3136 neurons' },
        { name: 'Dense', info: '128 units · ReLU' },
        { name: 'Dropout', info: '25%' },
        { name: 'Dense', info: '10 units · Softmax' },
      ]
    };
  },
  computed: {
    progressPercent() {
      const { epoch, totalEpochs } = this.trainingProgress;
      return totalEpochs > 0 ? (epoch / totalEpochs) * 100 : 0;
    },
    accPoints() {
      if (!this.historyData.length) return '';
      return this.historyData.map((d, i) => {
        const x = (i / Math.max(this.historyData.length - 1, 1)) * 220;
        const y = 80 - d.acc * 80;
        return `${x},${y}`;
      }).join(' ');
    },
    lossPoints() {
      if (!this.historyData.length) return '';
      const maxLoss = Math.max(...this.historyData.map(d => d.loss));
      return this.historyData.map((d, i) => {
        const x = (i / Math.max(this.historyData.length - 1, 1)) * 220;
        const y = 80 - (1 - d.loss / maxLoss) * 80;
        return `${x},${y}`;
      }).join(' ');
    }
  },
  mounted() {
    this.initCanvas();
    this.checkModelStatus();
  },
  beforeUnmount() {
    if (this.pollInterval) clearInterval(this.pollInterval);
  },
  methods: {
    initCanvas() {
      const canvas = this.$refs.drawCanvas;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, 280, 280);
      this.updatePreview();
    },
    startDraw(e) {
      this.drawing = true;
      const rect = this.$refs.drawCanvas.getBoundingClientRect();
      this.lastX = e.clientX - rect.left;
      this.lastY = e.clientY - rect.top;
    },
    draw(e) {
      if (!this.drawing) return;
      const canvas = this.$refs.drawCanvas;
      const ctx = canvas.getContext('2d');
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 20;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      ctx.moveTo(this.lastX, this.lastY);
      ctx.lineTo(x, y);
      ctx.stroke();
      this.lastX = x; this.lastY = y;
      this.updatePreview();
    },
    startDrawTouch(e) {
      const touch = e.touches[0];
      const rect = this.$refs.drawCanvas.getBoundingClientRect();
      this.drawing = true;
      this.lastX = touch.clientX - rect.left;
      this.lastY = touch.clientY - rect.top;
    },
    drawTouch(e) {
      if (!this.drawing) return;
      const touch = e.touches[0];
      this.draw({ clientX: touch.clientX, clientY: touch.clientY });
    },
    stopDraw() { this.drawing = false; },
    clearCanvas() {
      const canvas = this.$refs.drawCanvas;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, 280, 280);
      this.updatePreview();
      this.prediction = null;
    },
    updatePreview() {
      const src = this.$refs.drawCanvas;
      const preview = this.$refs.previewCanvas;
      if (!preview) return;
      const ctx = preview.getContext('2d');
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(src, 0, 0, 28, 28);
      const imgData = ctx.getImageData(0, 0, 28, 28);
      ctx.clearRect(0, 0, 84, 84);
      for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
          const idx = (y * 28 + x) * 4;
          const v = imgData.data[idx];
          ctx.fillStyle = `rgb(${v},${v},${v})`;
          ctx.fillRect(x * 3, y * 3, 3, 3);
        }
      }
    },
    getPixelData() {
      const canvas = this.$refs.drawCanvas;
      const off = document.createElement('canvas');
      off.width = 28; off.height = 28;
      const ctx = off.getContext('2d');
      ctx.drawImage(canvas, 0, 0, 28, 28);
      const imgData = ctx.getImageData(0, 0, 28, 28);
      const pixels = [];
      for (let i = 0; i < imgData.data.length; i += 4) pixels.push(imgData.data[i] / 255);
      return pixels;
    },
    async startTraining() {
      try {
        this.isTraining = true;
        this.historyData = [];
        this.trainingProgress = { epoch: 0, totalEpochs: this.trainConfig.epochs, status: 'preparing', loss: 0, acc: 0 };
        await axios.post(`${API}/train`, this.trainConfig);
        this.pollInterval = setInterval(this.pollProgress, 800);
      } catch (err) {
        this.isTraining = false;
        alert('Failed to start training: ' + (err.response?.data?.error || err.message));
      }
    },
    async pollProgress() {
      try {
        const { data } = await axios.get(`${API}/train/progress`);
        this.trainingProgress = data;
        if (data.epoch > 0 && this.historyData.length < data.epoch)
          this.historyData.push({ acc: data.acc, loss: data.loss });
        if (data.status === 'done' || data.status === 'error') {
          clearInterval(this.pollInterval);
          this.isTraining = false;
          this.modelReady = data.status === 'done';
        }
      } catch (err) { console.error('Poll error:', err); }
    },
    async checkModelStatus() {
      try {
        const { data } = await axios.get(`${API}/health`);
        this.modelReady = data.modelReady;
        this.isTraining = data.isTraining;
        this.hasSavedModel = data.hasSavedModel;
      } catch { console.log('Backend not reachable'); }
    },
    async loadModel() {
      try {
        const { data } = await axios.post(`${API}/model/load`);
        this.modelReady = true;
        this.trainingProgress.status = 'done';
        alert('✅ ' + data.message);
      } catch (err) { alert('❌ ' + (err.response?.data?.error || err.message)); }
    },
    async predict() {
      if (!this.modelReady) return;
      try {
        const imageData = this.getPixelData();
        const { data } = await axios.post(`${API}/predict`, { imageData });
        this.prediction = data;
        this.predictionHistory.push(data);
        if (this.predictionHistory.length > 20) this.predictionHistory.shift();
      } catch (err) { alert('Prediction error: ' + (err.response?.data?.error || err.message)); }
    }
  }
};
</script>

<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:        #0D0D14;
  --surface:   #16161F;
  --surface2:  #1E1E2E;
  --border:    #2A2A40;
  --green:     #00E676;
  --green-dim: #00B050;
  --purple:    #A855F7;
  --red:       #FF1744;
  --yellow:    #FFD600;
  --blue:      #2979FF;
  --text:      #F0F0FF;
  --muted:     #7070A0;
  --mono:      'JetBrains Mono', monospace;
  --display:   'Outfit', sans-serif;
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--mono);
  min-height: 100vh;
}

/* ─── HEADER ─────────────────────────────────────── */
.header {
  background: var(--surface);
  border-bottom: 2px solid var(--green);
  padding: 0 32px;
  height: 64px;
  display: flex;
  align-items: center;
}
.header-inner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  max-width: 1400px;
  margin: 0 auto;
}
.logo { display: flex; align-items: center; gap: 12px; }
.logo-icon { font-size: 28px; }
.logo-title {
  font-family: var(--display);
  font-size: 26px;
  font-weight: 900;
  color: var(--green);
  letter-spacing: -1px;
}
.logo-sub {
  display: block;
  font-size: 10px;
  color: var(--muted);
  letter-spacing: 3px;
  text-transform: uppercase;
}

.status-pill {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 16px;
  border-radius: 100px;
  background: var(--surface2);
  border: 1px solid var(--border);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 2px;
  color: var(--muted);
}
.status-pill .status-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--muted);
}
.status-pill.active { border-color: var(--green); color: var(--green); }
.status-pill.active .status-dot { background: var(--green); box-shadow: 0 0 10px var(--green); }
.status-pill.training { border-color: var(--yellow); color: var(--yellow); }
.status-pill.training .status-dot {
  background: var(--yellow);
  box-shadow: 0 0 10px var(--yellow);
  animation: blink 0.8s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

/* ─── LAYOUT ─────────────────────────────────────── */
.main {
  display: grid;
  grid-template-columns: 340px 1fr 340px;
  gap: 20px;
  padding: 24px;
  max-width: 1400px;
  margin: 0 auto;
  min-height: calc(100vh - 64px);
}

/* ─── PANELS ─────────────────────────────────────── */
.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}
.panel-train  { border-top: 3px solid var(--purple); }
.panel-draw   { border-top: 3px solid var(--green); }
.panel-result { border-top: 3px solid var(--blue); }

.panel-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 32px; height: 32px;
  border-radius: 8px;
  background: var(--surface2);
  border: 1px solid var(--border);
  font-size: 12px;
  font-weight: 700;
  color: var(--muted);
}
.panel-train  .panel-badge { color: var(--purple); border-color: var(--purple); }
.panel-draw   .panel-badge { color: var(--green); border-color: var(--green); }
.panel-result .panel-badge { color: var(--blue); border-color: var(--blue); }

.panel-title {
  font-family: var(--display);
  font-size: 18px;
  font-weight: 800;
  letter-spacing: 3px;
  color: var(--text);
  margin-top: -8px;
}

/* ─── TRAINING CONFIG ────────────────────────────── */
.train-config { display: flex; flex-direction: column; gap: 14px; }
.config-row { display: flex; flex-direction: column; gap: 8px; }
.config-label-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.config-label { font-size: 11px; font-weight: 600; letter-spacing: 2px; color: var(--muted); }
.config-val {
  font-size: 18px;
  font-weight: 700;
  color: var(--purple);
  font-family: var(--display);
}

.slider {
  width: 100%;
  -webkit-appearance: none;
  appearance: none;
  height: 4px;
  background: var(--border);
  border-radius: 4px;
  outline: none;
  cursor: pointer;
}
.slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 18px; height: 18px;
  border-radius: 50%;
  background: var(--purple);
  box-shadow: 0 0 10px var(--purple);
  cursor: pointer;
}
.slider:disabled { opacity: 0.4; cursor: not-allowed; }

/* ─── BUTTONS ────────────────────────────────────── */
.btn-train {
  width: 100%;
  padding: 14px;
  background: var(--purple);
  border: none;
  border-radius: 8px;
  color: #fff;
  font-family: var(--mono);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 2px;
  cursor: pointer;
  transition: all 0.2s;
  box-shadow: 0 4px 20px rgba(168,85,247,0.4);
}
.btn-train:hover:not(:disabled) {
  background: #C084FC;
  box-shadow: 0 6px 28px rgba(168,85,247,0.6);
  transform: translateY(-1px);
}
.btn-train:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

.btn-load {
  width: 100%;
  padding: 10px 14px;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text);
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 1px;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}
.btn-load:hover:not(:disabled) { border-color: var(--green); color: var(--green); }
.btn-load:disabled { opacity: 0.35; cursor: not-allowed; }
.badge-saved {
  background: var(--green);
  color: #000;
  font-weight: 800;
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 4px;
}

/* ─── PROGRESS ───────────────────────────────────── */
.progress-area {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.progress-label-row {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 2px;
  color: var(--muted);
}
.badge-done {
  background: var(--green);
  color: #000;
  font-size: 10px;
  font-weight: 800;
  padding: 2px 8px;
  border-radius: 100px;
}
.progress-track {
  height: 6px;
  background: var(--border);
  border-radius: 100px;
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--purple), var(--green));
  border-radius: 100px;
  transition: width 0.5s ease;
  box-shadow: 0 0 12px rgba(168,85,247,0.6);
}
.metrics-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.metric-card {
  border-radius: 8px;
  padding: 10px 12px;
  border: 1px solid;
}
.metric-card.metric-loss { background: rgba(255,23,68,0.1); border-color: rgba(255,23,68,0.3); }
.metric-card.metric-acc  { background: rgba(0,230,118,0.1); border-color: rgba(0,230,118,0.3); }
.metric-label { font-size: 9px; font-weight: 600; letter-spacing: 2px; color: var(--muted); margin-bottom: 4px; }
.metric-number { font-size: 18px; font-weight: 700; font-family: var(--display); }
.metric-loss .metric-number { color: var(--red); }
.metric-acc  .metric-number { color: var(--green); }

/* ─── CHART ──────────────────────────────────────── */
.chart-box {
  background: #0A0A12;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
}
.chart-title { font-size: 9px; font-weight: 600; letter-spacing: 3px; color: var(--muted); margin-bottom: 8px; }
.chart-svg { width: 100%; height: 60px; display: block; }
.chart-legend { display: flex; gap: 16px; margin-top: 6px; font-size: 10px; font-weight: 600; }
.legend-acc  { color: var(--green); }
.legend-loss { color: var(--red); }

/* ─── ARCHITECTURE ───────────────────────────────── */
.arch-box { margin-top: auto; }
.arch-heading { font-size: 10px; font-weight: 700; letter-spacing: 3px; color: var(--muted); margin-bottom: 8px; }
.arch-list { display: flex; flex-direction: column; gap: 3px; }
.arch-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 10px;
  background: var(--surface2);
  border-radius: 6px;
  border-left: 3px solid var(--purple);
  font-size: 11px;
}
.arch-name { color: var(--text); font-weight: 600; }
.arch-info { color: var(--muted); }

/* ─── DRAW PANEL ─────────────────────────────────── */
.panel-draw { align-items: center; }
.canvas-wrap { display: flex; flex-direction: column; align-items: center; gap: 8px; }
.draw-canvas {
  display: block;
  cursor: crosshair;
  border-radius: 12px;
  border: 2px solid var(--green);
  box-shadow: 0 0 30px rgba(0,230,118,0.2), 0 0 0 4px rgba(0,230,118,0.05);
}
.canvas-hint { font-size: 11px; font-weight: 600; letter-spacing: 1px; color: var(--muted); }

.draw-actions {
  display: flex;
  gap: 12px;
  width: 100%;
  max-width: 280px;
}
.btn-clear {
  flex: 1;
  padding: 12px;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text);
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}
.btn-clear:hover { border-color: var(--red); color: var(--red); }

.btn-predict {
  flex: 1;
  padding: 12px;
  background: var(--green);
  border: none;
  border-radius: 8px;
  color: #000;
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 1px;
  cursor: pointer;
  transition: all 0.2s;
  box-shadow: 0 4px 20px rgba(0,230,118,0.4);
}
.btn-predict:hover:not(:disabled) {
  background: #00FF88;
  box-shadow: 0 6px 28px rgba(0,230,118,0.6);
  transform: translateY(-1px);
}
.btn-predict:disabled { opacity: 0.3; cursor: not-allowed; transform: none; }

.preview-wrap { display: flex; flex-direction: column; align-items: center; gap: 6px; }
.preview-label { font-size: 10px; font-weight: 700; letter-spacing: 2px; color: var(--muted); }
.preview-canvas {
  width: 84px; height: 84px;
  image-rendering: pixelated;
  border: 2px solid var(--border);
  border-radius: 6px;
}

/* ─── RESULTS PANEL ──────────────────────────────── */
.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  opacity: 0.3;
}
.empty-icon {
  font-size: 72px;
  font-family: var(--display);
  font-weight: 900;
  color: var(--muted);
  line-height: 1;
}
.empty-state p { font-size: 11px; text-align: center; line-height: 1.6; color: var(--muted); }

.prediction-hero {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 24px;
  background: rgba(41,121,255,0.1);
  border: 2px solid var(--blue);
  border-radius: 12px;
}
.prediction-digit {
  font-family: var(--display);
  font-size: 96px;
  font-weight: 900;
  color: var(--blue);
  line-height: 1;
  text-shadow: 0 0 40px rgba(41,121,255,0.6);
}
.prediction-conf {
  font-size: 13px;
  font-weight: 700;
  color: var(--blue);
  letter-spacing: 1px;
  opacity: 0.8;
}

.prob-list { display: flex; flex-direction: column; gap: 5px; }
.prob-row {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 12px;
}
.prob-label { width: 16px; font-weight: 700; color: var(--muted); }
.prob-row.winner .prob-label { color: var(--green); }
.prob-track {
  flex: 1;
  height: 8px;
  background: var(--surface2);
  border-radius: 100px;
  overflow: hidden;
}
.prob-fill {
  height: 100%;
  background: var(--border);
  border-radius: 100px;
  transition: width 0.4s ease;
}
.prob-row.winner .prob-fill {
  background: linear-gradient(90deg, var(--green-dim), var(--green));
  box-shadow: 0 0 8px rgba(0,230,118,0.5);
}
.prob-pct { width: 48px; text-align: right; font-size: 11px; color: var(--muted); }
.prob-row.winner .prob-pct { color: var(--green); font-weight: 700; }

/* ─── HISTORY ────────────────────────────────────── */
.history-box { margin-top: auto; }
.history-heading { font-size: 10px; font-weight: 700; letter-spacing: 3px; color: var(--muted); margin-bottom: 8px; }
.history-list { display: flex; flex-wrap: wrap; gap: 6px; }
.history-chip {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 8px 12px;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  gap: 2px;
  transition: border-color 0.2s;
}
.history-chip:hover { border-color: var(--blue); }
.history-digit { font-family: var(--display); font-size: 22px; font-weight: 900; color: var(--text); }
.history-conf  { font-size: 9px; font-weight: 600; color: var(--muted); }

/* ─── PULSE ANIMATION ────────────────────────────── */
.pulse { animation: pulse 1s ease-in-out infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ─── RESPONSIVE ─────────────────────────────────── */
@media (max-width: 1000px) {
  .main { grid-template-columns: 1fr; }
}
</style>