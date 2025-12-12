
# sweep.py
import wandb, torch
from torchvision import models
from torch.utils.data import DataLoader, Subset
from src.utils import set_seed, make_device, make_dataloaders, evaluate
from src.quant import compress_model_calibrated, report_sizes

def train_compressed():
    with wandb.init(project="mobilenetv2-compression") as run:
        cfg = wandb.config
        run.name = f"W{cfg.weight_bits}_A{cfg.activation_bits}"

        set_seed(cfg.seed); device = make_device()
        trainset, _, _, testloader = make_dataloaders()

        # restore checkpoint once (best FP32)
        local_ckpt = wandb.restore(cfg.filename, run_path=cfg.run_path, replace=True).name

        # calibration subset
        idxs = torch.randperm(len(trainset))[:cfg.calib_samples]
        calib_loader = DataLoader(Subset(trainset, idxs), batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

        # load FP32
        base = models.mobilenet_v2(num_classes=10).to(device)
        base.load_state_dict(torch.load(local_ckpt, map_location=device))

        # compress
        model_q = compress_model_calibrated(base, calib_loader, device,
                                            weight_bits=cfg.weight_bits, activation_bits=cfg.activation_bits,
                                            act_symmetric=False, ema_momentum=0.9, calib_batches=cfg.calib_batches)

        # accuracy
        acc = evaluate(model_q, testloader, device)

        # sizes
        sizes = report_sizes(model_q, weight_bits=cfg.weight_bits, activation_bits=cfg.activation_bits,
                             testloader=testloader, device=device, act_profile_samples=20)

        # log for parallel coordinates
        wandb.log({
            "weight_bits": cfg.weight_bits,
            "activation_bits": cfg.activation_bits,
            "quant_acc": acc,
            # weights
            "weights_mb_payload": sizes["weights_mb_payload"],
            "weights_mb_overhead": sizes["weights_mb_overhead"],
            "weights_mb_total": sizes["weights_mb_total"],
            "cr_weights": sizes["compression_ratio_weights"],
            # activations (runtime)
            "activations_mb_fp32": sizes["activations_mb_fp32"],
            "activations_mb_quant": sizes["activations_mb_quant"],
            "cr_activations": sizes["compression_ratio_activations"],
            # model (persistent)
            "model_mb_fp32_total": sizes["model_mb_fp32_total"],
            "model_mb_approx_total": sizes["model_mb_approx_total"],
            "cr_model": sizes["compression_ratio_model"],
            # calibration metadata
            "calib_samples": cfg.calib_samples,
            "calib_batches": cfg.calib_batches,
        })

if __name__ == "__main__":
    # create sweep programmatically; then agent will run all combinations
    sweep_config = {
        "method": "grid",
        "metric": {"name": "quant_acc", "goal": "maximize"},
        "parameters": {
            "weight_bits": {"values": [8, 6, 4]},
            "activation_bits": {"values": [8, 6, 4]},
            "seed": {"values": [42]},
            "run_path": {"values": ["cs24m507-qualcomm/mobilenetv2-cifar10/opseb9nk"]},
            "filename": {"values": ["mobilenetv2_cifar10_best.pth"]},
            "calib_samples": {"values": [1000]},
            "calib_batches": {"values": [50]}
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="mobilenetv2-compression")
    wandb.agent(sweep_id, function=train_compressed)
