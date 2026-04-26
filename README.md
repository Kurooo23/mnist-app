# 🧠 MNIST Neural Network App

Website interaktif untuk melatih dan menggunakan Convolutional Neural Network (CNN) pada dataset MNIST.

## Stack
- **Frontend**: Vue 3 + Vite
- **Backend**: Node.js + Express + TensorFlow.js

## Cara Menjalankan

### Prasyarat
- Node.js >= 18

### Install & Build Frontend
```bash
cd frontend
npm install
npm run build
```

### Jalankan Server
```bash
cd backend
npm install
node server.js
```

Buka browser di: **http://localhost:3001**

### Development Mode (hot reload frontend)
```bash
# Terminal 1 - Backend
cd backend && node server.js

# Terminal 2 - Frontend dev server
cd frontend && npm run dev
# Frontend di http://localhost:3000 (proxy ke backend port 3001)
```

## Fitur
- **Panel 01 - Training**: Konfigurasi epochs & batch size, lalu train model CNN
- **Panel 02 - Draw**: Gambar angka 0-9 di kanvas 280×280, lihat preview 28×28
- **Panel 03 - Results**: Prediksi digit beserta confidence score per kelas

## Arsitektur Model
```
Input (28×28×1)
  → Conv2D (32 filters, 3×3, ReLU)
  → MaxPooling2D (2×2)
  → Conv2D (64 filters, 3×3, ReLU)
  → MaxPooling2D (2×2)
  → Flatten
  → Dense (128, ReLU)
  → Dropout (25%)
  → Dense (10, Softmax)
```

## API Endpoints
| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| POST | `/api/train` | Mulai training (body: `{epochs, batchSize}`) |
| GET | `/api/train/progress` | Status training |
| POST | `/api/predict` | Prediksi (body: `{imageData: float[784]}`) |
| GET | `/api/health` | Status server & model |

## Catatan
- Model menggunakan **synthetic data** untuk demo cepat (training ~30 detik)
- Untuk akurasi lebih baik, hubungkan ke dataset MNIST asli (65.000 gambar)
- Model di-reset setiap restart server
