#!/bin/bash
echo "🧠 Starting MNIST Neural Network App..."
echo ""
cd "$(dirname "$0")/backend"
node server.js
