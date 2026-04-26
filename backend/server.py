from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import threading
import os

app = Flask(__name__)
CORS(app)

MODEL_SAVE_PATH = "./saved_model.pth"

model = None
is_training = False
training_progress = {
    "epoch": 0,
    "totalEpochs": 0,
    "loss": 0,
    "acc": 0,
    "valLoss": 0,
    "valAcc": 0,
    "status": "idle"
}

# ==============================
# GPU CHECK
# ==============================
def check_gpu():
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU ditemukan: {gpu_name} ({vram:.1f} GB VRAM)")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Training akan menggunakan GPU!")
        return "cuda"
    else:
        print("⚠️  GPU tidak ditemukan, menggunakan CPU.")
        return "cpu"

# ==============================
# BUILD MODEL (CNN PyTorch)
# ==============================
def build_model(device):
    import torch.nn as nn

    class MnistCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),  # -> 32x28x28
                nn.ReLU(),
                nn.MaxPool2d(2),                              # -> 32x14x14
                nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> 64x14x14
                nn.ReLU(),
                nn.MaxPool2d(2),                              # -> 64x7x7
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(128, 10)
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return MnistCNN()

# ==============================
# LOAD MNIST DATASET
# ==============================
def load_mnist():
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    print("📦 Loading dataset MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test_dataset  = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False, num_workers=0)

    print(f"✅ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    return train_loader, test_loader

# ==============================
# AUTO LOAD MODEL SAAT START
# ==============================
def try_load_model():
    global model
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            import torch
            device = check_gpu()
            m = build_model(device)
            m.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
            m.to(device)
            m.eval()
            model = {"net": m, "device": device}
            training_progress["status"] = "done"
            print("✅ Model berhasil di-load! Siap dipakai tanpa training ulang.")
        except Exception as e:
            print(f"⚠️  Gagal load model: {e}")
    else:
        print("ℹ️  Belum ada model tersimpan. Silakan train dulu.")

# ==============================
# TRAINING THREAD
# ==============================
def train_thread(epochs, batch_size):
    global model, is_training, training_progress
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        training_progress["status"] = "loading_data"

        # Load data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True,  download=True, transform=transform)
        test_dataset  = datasets.MNIST('./data', train=False, download=True, transform=transform)
        train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
        test_loader   = DataLoader(test_dataset,  batch_size=256,        shuffle=False, num_workers=0)

        training_progress["status"] = "training"

        # Build model
        net = build_model(device).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        for epoch in range(epochs):
            # --- TRAIN ---
            net.train()
            total_loss, correct, total = 0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            train_loss = total_loss / len(train_loader)
            train_acc  = correct / total

            # --- VALIDATION ---
            net.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total += labels.size(0)

            val_loss = val_loss / len(test_loader)
            val_acc  = val_correct / val_total

            training_progress.update({
                "epoch": epoch + 1,
                "totalEpochs": epochs,
                "loss": round(train_loss, 4),
                "acc": round(train_acc, 4),
                "valLoss": round(val_loss, 4),
                "valAcc": round(val_acc, 4),
                "status": "training"
            })
            print(f"Epoch {epoch+1}/{epochs} — loss: {train_loss:.4f}, acc: {train_acc*100:.1f}%, val_acc: {val_acc*100:.1f}%")

        # Simpan model
        print("💾 Menyimpan model...")
        torch.save(net.state_dict(), MODEL_SAVE_PATH)
        print(f"✅ Model tersimpan di: {MODEL_SAVE_PATH}")

        net.eval()
        model = {"net": net, "device": device}
        training_progress["status"] = "done"
        is_training = False

    except Exception as e:
        import traceback
        traceback.print_exc()
        training_progress["status"] = "error"
        training_progress["error"] = str(e)
        is_training = False

# ==============================
# ROUTES
# ==============================

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "modelReady": model is not None,
        "isTraining": is_training,
        "hasSavedModel": os.path.exists(MODEL_SAVE_PATH)
    })

@app.route("/api/train", methods=["POST"])
def train():
    global is_training, training_progress
    if is_training:
        return jsonify({"error": "Training already in progress"}), 400

    data = request.get_json() or {}
    epochs     = int(data.get("epochs", 5))
    batch_size = int(data.get("batchSize", 128))

    is_training = True
    training_progress = {
        "epoch": 0, "totalEpochs": epochs,
        "loss": 0, "acc": 0,
        "valLoss": 0, "valAcc": 0,
        "status": "preparing"
    }

    t = threading.Thread(target=train_thread, args=(epochs, batch_size))
    t.daemon = True
    t.start()

    return jsonify({"message": "Training started", "epochs": epochs, "batchSize": batch_size})

@app.route("/api/train/progress")
def progress():
    return jsonify(training_progress)

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model belum siap. Train atau load model dulu."}), 400
    try:
        import torch
        data       = request.get_json()
        image_data = data.get("imageData", [])
        if len(image_data) != 784:
            return jsonify({"error": "Expected 784 pixels"}), 400

        net    = model["net"]
        device = model["device"]

        # Normalize sama seperti saat training
        img = np.array(image_data, dtype=np.float32).reshape(1, 1, 28, 28)
        img = (img - 0.1307) / 0.3081
        tensor = torch.tensor(img).to(device)

        net.eval()
        with torch.no_grad():
            outputs = net(tensor)
            probs   = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        pred = int(np.argmax(probs))
        return jsonify({
            "prediction":    pred,
            "probabilities": probs.tolist(),
            "confidence":    float(probs[pred])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/model/save", methods=["POST"])
def save_model():
    if model is None:
        return jsonify({"error": "Tidak ada model untuk disimpan."}), 400
    try:
        import torch
        torch.save(model["net"].state_dict(), MODEL_SAVE_PATH)
        return jsonify({"message": "Model berhasil disimpan!", "path": MODEL_SAVE_PATH})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/model/load", methods=["POST"])
def load_model_route():
    global model, training_progress
    if not os.path.exists(MODEL_SAVE_PATH):
        return jsonify({"error": "Tidak ada model tersimpan. Train dulu!"}), 404
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net    = build_model(device)
        net.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        net.to(device)
        net.eval()
        model = {"net": net, "device": device}
        training_progress["status"] = "done"
        return jsonify({"message": f"Model berhasil di-load! (device: {device})"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    import torch
    print("\n🧠 MNIST Backend (Python + PyTorch)")
    print("=" * 40)
    device = check_gpu()
    print(f"   PyTorch version: {torch.__version__}")
    print("=" * 40)
    try_load_model()
    print(f"\n🚀 Server running at http://localhost:3001\n")
    app.run(host="0.0.0.0", port=3001, debug=False, threaded=True)