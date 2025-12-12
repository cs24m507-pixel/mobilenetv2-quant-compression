
# utils.py
import os, random, torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def set_seed(seed: int = 42):
    import numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # True => slower, more deterministic
    torch.backends.cudnn.benchmark = True

def make_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dataloaders(batch_train=128, batch_test=256, num_workers=2, pin_memory=True):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    testset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
    trainloader = DataLoader(trainset, batch_size=batch_train, shuffle=True,
                             num_workers=num_workers, pin_memory=pin_memory)
    testloader  = DataLoader(testset,  batch_size=batch_test, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return trainset, testset, trainloader, testloader

def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x); _, pred = logits.max(1)
            total += y.size(0); correct += pred.eq(y).sum().item()
    return 100.0 * correct / total
