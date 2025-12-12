
# compress.py
import argparse, wandb, torch
from torchvision import models
from torch.utils.data import DataLoader, Subset
from src.utils import set_seed, make_device, make_dataloaders, evaluate
from src.quant import (compress_model_calibrated, report_sizes)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_path", type=str, required=True,
                    help="W&B run path e.g. cs24m507-qualcomm/mobilenetv2-cifar10/opseb9nk")
    ap.add_argument("--filename", type=str, default="mobilenetv2_cifar10_best.pth")
    ap.add_argument("--weight_bits", type=int, default=4)
    ap.add_argument("--activation_bits", type=int, default=8)
    ap.add_argument("--calib_samples", type=int, default=1000)
    ap.add_argument("--calib_batches", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed); device = make_device()
    trainset, _, _, testloader = make_dataloaders()

    # Download checkpoint from W&B
    wandb.login()
    local_ckpt = wandb.restore(args.filename, run_path=args.run_path, replace=True).name
    print(f"[W&B] Downloaded checkpoint: {local_ckpt}")

    # Build calibration loader from train subset
    idxs = torch.randperm(len(trainset))[:args.calib_samples]
    calib_loader = DataLoader(Subset(trainset, idxs), batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # Load model FP32
    model = models.mobilenet_v2(num_classes=10).to(device)
    model.load_state_dict(torch.load(local_ckpt, map_location=device))

    # Compress with calibration
    model_q = compress_model_calibrated(model, calib_loader, device,
                                        weight_bits=args.weight_bits, activation_bits=args.activation_bits,
                                        act_symmetric=False, ema_momentum=0.9, calib_batches=args.calib_batches)

    # Accuracy
    acc = evaluate(model_q, testloader, device)
    print(f"[Quantized W{args.weight_bits}/A{args.activation_bits}] Val Acc: {acc:.2f}%")

    # Q4 metrics
    sizes = report_sizes(model_q, weight_bits=args.weight_bits, activation_bits=args.activation_bits,
                         testloader=testloader, device=device, act_profile_samples=20)
    print(f"Model CR: {sizes['compression_ratio_model']:.2f}x | Final approx model size (MB): {sizes['model_mb_approx_total']:.3f}")
    print(f"Weights CR: {sizes['compression_ratio_weights']:.2f}x | Weights total (MB): {sizes['weights_mb_total']:.3f} "
          f"(payload {sizes['weights_mb_payload']:.3f} + scales {sizes['weights_mb_overhead']:.3f})")
    print(f"Activations CR: {sizes['compression_ratio_activations']:.2f}x "
          f"| Activations quant (MB): {sizes['activations_mb_quant']:.3f} vs FP32 (MB): {sizes['activations_mb_fp32']:.3f}")

if __name__ == "__main__":
    main()
