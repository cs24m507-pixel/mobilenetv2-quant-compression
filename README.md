
# MobileNetV2 CIFAR-10 — Quantization & Compression (PTQ-style, calibrated)

This repo trains a MobileNetV2 baseline on CIFAR-10 and applies a calibrated post-training quantization:
- **Weights**: per-channel symmetric INT (Conv/Linear)
- **Activations**: per-tensor affine INT with **EMA min/max** calibration
- **Evaluation**: accuracy & storage metrics (model/weights/activations)
- **W&B**: grid sweep + parallel coordinates chart

## 1) Environment

- Python ≥ 3.10
- PyTorch ≥ 2.8 (tested with torch 2.8/2.9), torchvision ≥ 0.23
- Weights & Biases ≥ 0.22

Install:
```bash
python -m venv .venv && source .venv/bin/activate  # or conda create -n mv2-quant python=3.10
pip install -r requirements.txt
