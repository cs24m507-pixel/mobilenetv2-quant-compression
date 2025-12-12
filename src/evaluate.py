
# evaluate.py
import argparse, torch
from torchvision import models
from src.utils import set_seed, make_device, make_dataloaders, evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed); device = make_device()
    _, _, _, testloader = make_dataloaders()

    model = models.mobilenet_v2(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    acc = evaluate(model, testloader, device)
    print(f"[Eval] Val Acc: {acc:.2f}%")

if __name__ == "__main__":
    main()
