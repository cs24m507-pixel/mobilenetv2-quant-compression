
# train.py
import argparse, wandb, torch
import torch.nn as nn, torch.optim as optim
from torchvision import models
from src.utils import set_seed, make_device, make_dataloaders, evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=str, default="mobilenetv2-cifar10")
    ap.add_argument("--run_name", type=str, default="baseline-training_Final")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--base_lr", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = make_device()
    _, _, trainloader, testloader = make_dataloaders()

    # Login reads WANDB_API_KEY from env
    wandb.login()
    config = dict(arch="MobileNetV2", epochs=args.epochs, base_lr=args.base_lr,
                  momentum=args.momentum, weight_decay=args.wd,
                  label_smoothing=args.label_smoothing, seed=args.seed)
    run = wandb.init(project=args.project, name=args.run_name, config=config)

    model = models.mobilenet_v2(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr,
                          momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train(); running_loss = 0.0; correct = 0; total = 0
        for x, y in trainloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            running_loss += loss.item()
            _, pred = logits.max(1); total += y.size(0); correct += pred.eq(y).sum().item()

        train_acc = 100.0 * correct / total
        avg_train_loss = running_loss / len(trainloader)

        # validation
        val_acc = evaluate(model, testloader, device)
        avg_val_loss = 0.0  # (optional: compute if you want)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "mobilenetv2_cifar10_best.pth")
            wandb.save("mobilenetv2_cifar10_best.pth")

        scheduler.step()
        wandb.log({"Train Loss": avg_train_loss, "Train Acc": train_acc,
                   "Val Loss": avg_val_loss, "Val Acc": val_acc,
                   "lr": optimizer.param_groups[0]["lr"], "epoch": epoch})
        print(f"Epoch [{epoch+1}/{args.epochs}] Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}% "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    torch.save(model.state_dict(), "mobilenetv2_cifar10_last.pth"); wandb.save("mobilenetv2_cifar10_last.pth")
    wandb.finish()

if __name__ == "__main__":
    main()
