
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

#####################################################################
#Execution commands################################################
# 1) login
import wandb
wandb.login(key="cbc7f79e1989a692a9e611f430cc14e032004826")

# 2) Clone repo (if not already)
!git clone https://github.com/cs24m507-pixel/mobilenetv2-quant-compression.git
%cd /kaggle/working/mobilenetv2-quant-compression
%env PYTHONPATH=/kaggle/working/mobilenetv2-quant-compression

# 3) Install minimal dependencies (optional; Kaggle has torch/torchvision)
!pip install -q -r requirement.txt

# 4) Train FP32 baseline
!python -m src.train \
  --project mobilenetv2-cifar10 \
  --run_name kaggle_fp32_run \
  --epochs 200 \
  --base_lr 0.05 \
  --momentum 0.9 \
  --wd 5e-4 \
  --label_smoothing 0.1 \
  --seed 42
# 5) Evaluate
run_path = "cs24m507-qualcomm/mobilenetv2-cifar10/opseb9nk"
ckpt_name = "mobilenetv2_cifar10_best.pth"
restored = wandb.restore(ckpt_name, run_path=run_path, replace=True)
!python -m src.evaluate --checkpoint mobilenetv2_cifar10_best.pth --seed 42

# 6) Compress (if you want, and have a W&B run path)
!python -m src.compress \
  --run_path cs24m507-qualcomm/mobilenetv2-cifar10/opseb9nk \
  --filename mobilenetv2_cifar10_best.pth \
  --weight_bits 4 \
  --activation_bits 8 \
  --calib_samples 2000 \
  --calib_batches 50 \
  --seed 42
# 7) swipe call
%cd /kaggle/working/mobilenetv2-quant-compression
%env PYTHONPATH=/kaggle/working/mobilenetv2-quant-compression
!python src/sweep.py
