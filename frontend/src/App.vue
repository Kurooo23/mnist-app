<template>
  <div class="app">
    <div class="bg-grid"></div>
    <div class="noise"></div>

    <!-- Header -->
    <header class="header">
      <div class="header-inner">
        <div class="logo">
          <span class="logo-bracket">[</span>
          <span class="logo-text">MNIST</span>
          <span class="logo-bracket">]</span>
          <span class="logo-sub">Neural Network</span>
        </div>
        <div class="status-bar">
          <div class="status-dot" :class="{ active: modelReady, training: isTraining }"></div>
          <span class="status-text">
            {{ isTraining ? 'TRAINING...' : modelReady ? 'MODEL READY' : 'AWAITING TRAINING' }}
          </span>
        </div>
      </div>
    </header>

    <main class="main">
      <!-- Left Panel: Training -->
      <section class="panel panel-train">
        <div class="panel-header">
          <span class="panel-num">01</span>
          <h2>TRAINING</h2>
        </div>

        <div class="train-config">
          <div class="config-row">
            <label>EPOCHS</label>
            <div class="slider-wrap">
              <input type="range" v-model.number="trainConfig.epochs" min="1" max="20" :disabled="isTraining" />
              <span class="slider-val">{{ trainConfig.epochs }}</span>
            </div>
          </div>
          <div class="config-row">
            <label>BATCH SIZE</label>
            <div class="slider-wrap">
              <input type="range" v-model.number="trainConfig.batchSize" min="64" max="1024" step="64" :disabled="isTraining" />
              <span class="slider-val">{{ trainConfig.batchSize }}</span>
            </div>
          </div>
        </div>

        <button class="btn-train" @click="startTraining" :disabled="isTraining">
          <span v-if="!isTraining">▶ START TRAINING</span>
          <span v-else class="pulse">● TRAINING...</span>
        </button>

        <button class="btn-load" @click="loadModel" :disabled="isTraining || !hasSavedModel">
          📂 LOAD SAVED MODEL
          <span v-if="!hasSavedModel" class="btn-sub">(belum ada)</span>
          <span v-else class="btn-sub saved">(tersimpan ✓)</span>
        </button>

        <!-- Progress -->
        <div v-if="isTraining || trainingProgress.status === 'done'" class="progress-area">
          <div class="progress-header">
            <span>EPOCH {{ trainingProgress.epoch }} / {{ trainingProgress.totalEpochs }}</span>
            <span v-if="trainingProgress.status === 'done'" class="done-badge">✓ DONE</span>
          </div>
          <div class="progress-bar-wrap">
            <div class="progress-bar" :style="{ width: progressPercent + '%' }"></div>
          </div>
          <div v-if="trainingProgress.epoch > 0" class="metrics">
            <div class="metric">
              <span class="metric-label">LOSS</span>
              <span class="metric-val loss">{{ trainingProgress.loss?.toFixed(4) }}</span>
            </div>
            <div class="metric">
              <span class="metric-label">ACC</span>
              <span class="metric-val acc">{{ (trainingProgress.acc * 100)?.toFixed(1) }}%</span>
            </div>
            <div class="metric">
              <span class="metric-label">VAL LOSS</span>
              <span class="metric-val loss">{{ trainingProgress.valLoss?.toFixed(4) }}</span>
            </div>
            <div class="metric">
              <span class="metric-label">VAL ACC</span>
              <span class="metric-val acc">{{ (trainingProgress.valAcc * 100)?.toFixed(1) }}%</span>
            </div>
          </div>
          <!-- Chart -->
          <div v-if="historyData.length > 0" class="mini-chart">
            <svg :viewBox="`0 0 200 80`" preserveAspectRatio="none">
              <polyline :points="accPoints" fill="none" stroke="#00ff88" stroke-width="2"/>
              <polyline :points="lossPoints" fill="none" stroke="#ff4444" stroke-width="1.5" stroke-dasharray="3,2"/>
            </svg>
            <div class="chart-legend">
              <span style="color:#00ff88">— ACC</span>
              <span style="color:#ff4444">-- LOSS</span>
            </div>
          </div>
        </div>

        <!-- Architecture -->
        <div class="arch-section">
          <div class="arch-title">MODEL ARCHITECTURE</div>
          <div class="arch-layers">
            <div class="arch-layer" v-for="(layer, i) in architecture" :key="i">
              <div class="arch-layer-name">{{ layer.name }}</div>
              <div class="arch-layer-info">{{ layer.info }}</div>
            </div>
          </div>
        </div>
      </section>

      <!-- Middle Panel: Draw & Predict -->
      <section class="panel panel-draw">
        <div class="panel-header">
          <span class="panel-num">02</span>
          <h2>DRAW & PREDICT</h2>
        </div>

        <div class="canvas-container">
          <canvas
            ref="drawCanvas"
            width="280"
            height="280"
            class="draw-canvas"
            @mousedown="startDraw"
            @mousemove="draw"
            @mouseup="stopDraw"
            @mouseleave="stopDraw"
            @touchstart.prevent="startDrawTouch"
            @touchmove.prevent="drawTouch"
            @touchend="stopDraw"
          ></canvas>
          <div class="canvas-label">DRAW A DIGIT (0-9)</div>
        </div>

        <div class="canvas-actions">
          <button class="btn-secondary" @click="clearCanvas">⌫ CLEAR</button>
          <button class="btn-primary" @click="predict" :disabled="!modelReady">
            ⚡ PREDICT
          </button>
        </div>

        <!-- 28x28 preview -->
        <div class="preview-section">
          <div class="preview-label">28×28 INPUT</div>
          <canvas ref="previewCanvas" width="84" height="84" class="preview-canvas"></canvas>
        </div>
      </section>

      <!-- Right Panel: Results -->
      <section class="panel panel-result">
        <div class="panel-header">
          <span class="panel-num">03</span>
          <h2>RESULTS</h2>
        </div>

        <div v-if="!prediction" class="no-result">
          <div class="no-result-icon">?</div>
          <p>Train model & draw a digit to see predictions</p>
        </div>

        <div v-else class="result-content">
          <div class="big-prediction">
            <div class="prediction-num">{{ prediction.prediction }}</div>
            <div class="prediction-conf">{{ (prediction.confidence * 100).toFixed(1) }}% confidence</div>
          </div>

          <div class="prob-bars">
            <div
              class="prob-row"
              v-for="(prob, i) in prediction.probabilities"
              :key="i"
              :class="{ winner: i === prediction.prediction }"
            >
              <span class="prob-digit">{{ i }}</span>
              <div class="prob-bar-bg">
                <div class="prob-bar-fill" :style="{ width: (prob * 100) + '%' }"></div>
              </div>
              <span class="prob-pct">{{ (prob * 100).toFixed(1) }}%</span>
            </div>
          </div>
        </div>

        <!-- History -->
        <div v-if="predictionHistory.length > 0" class="history-section">
          <div class="history-title">HISTORY</div>
          <div class="history-list">
            <div class="history-item" v-for="(item, i) in predictionHistory.slice().reverse()" :key="i">
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
      lastX: 0,
      lastY: 0,
      pollInterval: null,
      architecture: [
        { name: 'Conv2D', info: '32 filters, 3×3, ReLU' },
        { name: 'MaxPool2D', info: '2×2' },
        { name: 'Conv2D', info: '64 filters, 3×3, ReLU' },
        { name: 'MaxPool2D', info: '2×2' },
        { name: 'Flatten', info: '→ 1D vector' },
        { name: 'Dense', info: '128 units, ReLU' },
        { name: 'Dropout', info: '25%' },
        { name: 'Dense', info: '10 units, Softmax' },
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
      const max = 1, min = 0;
      return this.historyData.map((d, i) => {
        const x = (i / Math.max(this.historyData.length - 1, 1)) * 200;
        const y = 80 - ((d.acc - min) / (max - min)) * 80;
        return `${x},${y}`;
      }).join(' ');
    },
    lossPoints() {
      if (!this.historyData.length) return '';
      const maxLoss = Math.max(...this.historyData.map(d => d.loss));
      return this.historyData.map((d, i) => {
        const x = (i / Math.max(this.historyData.length - 1, 1)) * 200;
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
      this.lastX = x;
      this.lastY = y;
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
      const fakeEvent = { clientX: touch.clientX, clientY: touch.clientY };
      this.draw(fakeEvent);
    },

    stopDraw() {
      this.drawing = false;
    },

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
      // Scale up for display
      const imgData = ctx.getImageData(0, 0, 28, 28);
      ctx.clearRect(0, 0, 84, 84);
      const scale = 3;
      for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
          const idx = (y * 28 + x) * 4;
          const v = imgData.data[idx];
          ctx.fillStyle = `rgb(${v},${v},${v})`;
          ctx.fillRect(x * scale, y * scale, scale, scale);
        }
      }
    },

    getPixelData() {
      const canvas = this.$refs.drawCanvas;
      const offscreen = document.createElement('canvas');
      offscreen.width = 28;
      offscreen.height = 28;
      const ctx = offscreen.getContext('2d');
      ctx.drawImage(canvas, 0, 0, 28, 28);
      const imgData = ctx.getImageData(0, 0, 28, 28);
      const pixels = [];
      for (let i = 0; i < imgData.data.length; i += 4) {
        pixels.push(imgData.data[i] / 255);
      }
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
        if (data.epoch > 0 && this.historyData.length < data.epoch) {
          this.historyData.push({ acc: data.acc, loss: data.loss });
        }
        if (data.status === 'done' || data.status === 'error') {
          clearInterval(this.pollInterval);
          this.isTraining = false;
          this.modelReady = data.status === 'done';
        }
      } catch (err) {
        console.error('Poll error:', err);
      }
    },

    async checkModelStatus() {
      try {
        const { data } = await axios.get(`${API}/health`);
        this.modelReady = data.modelReady;
        this.isTraining = data.isTraining;
        this.hasSavedModel = data.hasSavedModel;
      } catch (err) {
        console.log('Backend not reachable');
      }
    },

    async loadModel() {
      try {
        const { data } = await axios.post(`${API}/model/load`);
        this.modelReady = true;
        this.trainingProgress.status = 'done';
        alert('✅ ' + data.message);
      } catch (err) {
        alert('❌ ' + (err.response?.data?.error || err.message));
      }
    },

    async predict() {
      if (!this.modelReady) return;
      try {
        const imageData = this.getPixelData();
        const { data } = await axios.post(`${API}/predict`, { imageData });
        this.prediction = data;
        this.predictionHistory.push(data);
        if (this.predictionHistory.length > 20) this.predictionHistory.shift();
      } catch (err) {
        alert('Prediction error: ' + (err.response?.data?.error || err.message));
      }
    }
  }
};
</script>

<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #0a0a0f;
  --panel: #111118;
  --border: #222233;
  --accent: #00ff88;
  --accent2: #7c3aff;
  --red: #ff4444;
  --text: #e0e0ff;
  --muted: #555577;
  --font-mono: 'Space Mono', monospace;
  --font-head: 'Syne', sans-serif;
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-mono);
  min-height: 100vh;
  overflow-x: hidden;
}

.app {
  position: relative;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* Background grid */
.bg-grid {
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(0,255,136,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,255,136,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
  z-index: 0;
}

.noise {
  position: fixed;
  inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
  opacity: 0.4;
  pointer-events: none;
  z-index: 0;
}

/* Header */
.header {
  position: relative;
  z-index: 10;
  border-bottom: 1px solid var(--border);
  padding: 16px 32px;
  background: rgba(10,10,15,0.9);
  backdrop-filter: blur(10px);
}

.header-inner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  max-width: 1400px;
  margin: 0 auto;
}

.logo {
  display: flex;
  align-items: baseline;
  gap: 6px;
  font-family: var(--font-head);
  font-size: 22px;
  font-weight: 800;
  letter-spacing: -0.5px;
}

.logo-bracket { color: var(--accent); }
.logo-text { color: #fff; }
.logo-sub { font-size: 11px; color: var(--muted); font-family: var(--font-mono); letter-spacing: 2px; margin-left: 4px; }

.status-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 11px;
  letter-spacing: 2px;
  color: var(--muted);
}

.status-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--muted);
  transition: all 0.3s;
}
.status-dot.active { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
.status-dot.training { background: #ffaa00; box-shadow: 0 0 8px #ffaa00; animation: blink 1s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* Main layout */
.main {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: 320px 1fr 320px;
  gap: 0;
  flex: 1;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
  padding: 24px;
  gap: 16px;
}

/* Panels */
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 2px;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  position: relative;
  overflow: hidden;
}

.panel::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--accent), transparent);
}

.panel-header {
  display: flex;
  align-items: center;
  gap: 12px;
}

.panel-num {
  font-size: 10px;
  color: var(--accent);
  letter-spacing: 2px;
  font-family: var(--font-mono);
}

.panel-header h2 {
  font-family: var(--font-head);
  font-size: 14px;
  font-weight: 700;
  letter-spacing: 4px;
  color: var(--text);
}

/* Training config */
.train-config { display: flex; flex-direction: column; gap: 16px; }

.config-row {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.config-row label {
  font-size: 10px;
  letter-spacing: 2px;
  color: var(--muted);
}

.slider-wrap {
  display: flex;
  align-items: center;
  gap: 12px;
}

.slider-wrap input[type=range] {
  flex: 1;
  -webkit-appearance: none;
  height: 2px;
  background: var(--border);
  outline: none;
  cursor: pointer;
}

.slider-wrap input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 14px; height: 14px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 8px var(--accent);
  cursor: pointer;
}

.slider-val {
  font-size: 14px;
  color: var(--accent);
  min-width: 36px;
  text-align: right;
}

/* Buttons */
.btn-train {
  width: 100%;
  padding: 14px;
  background: transparent;
  border: 1px solid var(--accent);
  color: var(--accent);
  font-family: var(--font-mono);
  font-size: 12px;
  letter-spacing: 3px;
  cursor: pointer;
  transition: all 0.2s;
  position: relative;
  overflow: hidden;
}

.btn-train::before {
  content: '';
  position: absolute;
  inset: 0;
  background: var(--accent);
  transform: scaleX(0);
  transform-origin: left;
  transition: transform 0.2s;
  z-index: -1;
}

.btn-train:hover:not(:disabled)::before { transform: scaleX(1); }
.btn-train:hover:not(:disabled) { color: #000; }
.btn-train:disabled { opacity: 0.4; cursor: not-allowed; }

.btn-load {
  width: 100%;
  padding: 10px;
  background: transparent;
  border: 1px solid var(--accent2);
  color: var(--accent2);
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 2px;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}
.btn-load:hover:not(:disabled) { background: rgba(124,58,255,0.1); }
.btn-load:disabled { opacity: 0.35; cursor: not-allowed; }
.btn-sub { font-size: 9px; letter-spacing: 1px; color: var(--muted); }
.btn-sub.saved { color: var(--accent); }

.btn-primary, .btn-secondary {
  padding: 10px 20px;
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 2px;
  cursor: pointer;
  transition: all 0.2s;
  border: 1px solid;
}

.btn-primary {
  background: var(--accent);
  border-color: var(--accent);
  color: #000;
  font-weight: 700;
}
.btn-primary:hover:not(:disabled) { box-shadow: 0 0 16px var(--accent); }
.btn-primary:disabled { opacity: 0.3; cursor: not-allowed; }

.btn-secondary {
  background: transparent;
  border-color: var(--border);
  color: var(--muted);
}
.btn-secondary:hover { border-color: var(--text); color: var(--text); }

/* Progress */
.progress-area { display: flex; flex-direction: column; gap: 12px; }

.progress-header {
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  letter-spacing: 2px;
  color: var(--muted);
}

.done-badge { color: var(--accent); }

.progress-bar-wrap {
  height: 3px;
  background: var(--border);
  border-radius: 2px;
  overflow: hidden;
}

.progress-bar {
  height: 100%;
  background: var(--accent);
  transition: width 0.5s ease;
  box-shadow: 0 0 8px var(--accent);
}

.metrics {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.metric {
  display: flex;
  flex-direction: column;
  gap: 2px;
  background: rgba(0,0,0,0.3);
  padding: 8px;
  border: 1px solid var(--border);
}

.metric-label { font-size: 9px; color: var(--muted); letter-spacing: 2px; }
.metric-val { font-size: 14px; }
.metric-val.acc { color: var(--accent); }
.metric-val.loss { color: var(--red); }

.mini-chart {
  background: rgba(0,0,0,0.4);
  border: 1px solid var(--border);
  padding: 8px;
}

.mini-chart svg { width: 100%; height: 60px; }
.chart-legend { display: flex; gap: 12px; font-size: 9px; margin-top: 4px; }

/* Architecture */
.arch-section { margin-top: auto; }
.arch-title { font-size: 9px; letter-spacing: 3px; color: var(--muted); margin-bottom: 8px; }
.arch-layers { display: flex; flex-direction: column; gap: 2px; }

.arch-layer {
  display: flex;
  justify-content: space-between;
  padding: 4px 8px;
  background: rgba(0,0,0,0.3);
  border-left: 2px solid var(--accent2);
  font-size: 10px;
}

.arch-layer-name { color: var(--text); }
.arch-layer-info { color: var(--muted); }

/* Draw Panel */
.panel-draw { align-items: center; }

.canvas-container {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.draw-canvas {
  display: block;
  cursor: crosshair;
  border: 1px solid var(--border);
  box-shadow: 0 0 30px rgba(0,255,136,0.05), inset 0 0 20px rgba(0,0,0,0.5);
}

.canvas-label {
  font-size: 9px;
  letter-spacing: 2px;
  color: var(--muted);
}

.canvas-actions {
  display: flex;
  gap: 12px;
  width: 100%;
  max-width: 280px;
}

.canvas-actions .btn-secondary,
.canvas-actions .btn-primary { flex: 1; text-align: center; }

.preview-section { display: flex; flex-direction: column; align-items: center; gap: 6px; }
.preview-label { font-size: 9px; letter-spacing: 2px; color: var(--muted); }
.preview-canvas {
  image-rendering: pixelated;
  border: 1px solid var(--border);
  width: 84px; height: 84px;
}

/* Results panel */
.no-result {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  gap: 12px;
  opacity: 0.4;
}

.no-result-icon {
  font-size: 60px;
  font-family: var(--font-head);
  font-weight: 800;
  color: var(--muted);
}

.no-result p { font-size: 10px; letter-spacing: 1px; text-align: center; }

.result-content { display: flex; flex-direction: column; gap: 20px; }

.big-prediction {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 20px;
  background: rgba(0,255,136,0.05);
  border: 1px solid rgba(0,255,136,0.2);
}

.prediction-num {
  font-family: var(--font-head);
  font-size: 80px;
  font-weight: 800;
  color: var(--accent);
  line-height: 1;
  text-shadow: 0 0 40px var(--accent);
}

.prediction-conf { font-size: 11px; color: var(--muted); letter-spacing: 2px; }

.prob-bars { display: flex; flex-direction: column; gap: 4px; }

.prob-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 11px;
}

.prob-digit {
  width: 14px;
  color: var(--muted);
  font-weight: 700;
}

.prob-row.winner .prob-digit { color: var(--accent); }

.prob-bar-bg {
  flex: 1;
  height: 6px;
  background: var(--border);
  border-radius: 0;
  overflow: hidden;
}

.prob-bar-fill {
  height: 100%;
  background: var(--muted);
  transition: width 0.4s ease;
}

.prob-row.winner .prob-bar-fill {
  background: var(--accent);
  box-shadow: 0 0 6px var(--accent);
}

.prob-pct {
  width: 44px;
  text-align: right;
  font-size: 10px;
  color: var(--muted);
}

.prob-row.winner .prob-pct { color: var(--accent); }

/* History */
.history-section { margin-top: auto; }
.history-title { font-size: 9px; letter-spacing: 3px; color: var(--muted); margin-bottom: 8px; }

.history-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.history-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 6px 10px;
  background: rgba(0,0,0,0.4);
  border: 1px solid var(--border);
  gap: 2px;
}

.history-digit { font-size: 18px; font-family: var(--font-head); font-weight: 700; color: var(--text); }
.history-conf { font-size: 8px; color: var(--muted); }

.pulse { animation: pulse 1s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

@media (max-width: 900px) {
  .main { grid-template-columns: 1fr; }
}
</style>
