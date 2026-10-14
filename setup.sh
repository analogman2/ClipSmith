#!/usr/bin/env bash
# One-time setup: Python deps (uv), caption font, face-detection model.
set -euo pipefail
cd "$(dirname "$0")"
uv sync
mkdir -p assets
[ -s assets/Montserrat-ExtraBold.ttf ] || curl -fsSL -o assets/Montserrat-ExtraBold.ttf \
  https://cdn.jsdelivr.net/fontsource/fonts/montserrat@latest/latin-800-normal.ttf
[ -s assets/face_detection_yunet_2023mar.onnx ] || curl -fsSL -o assets/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
echo "Ready. Try: ./clipsmith.py my_video.mp4"
