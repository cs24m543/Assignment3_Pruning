# === train.py ===
import argparse
import math
import os
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import get_mobilenet_v2


def mixup_data(x, y, alpha=0.8):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class EMA:
    def __init__(self, model, decay=0.9999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.copy_(v * self.decay + msd[k].detach() * (1 - self.decay))


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, accum_steps=1, use_mixup=False, mixup_alpha=0.8, ema=None):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    optimizer.zero_grad()
    for step, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        if use_mixup:
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=mixup_alpha)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                if use_mixup:
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    loss = criterion(outputs, labels)
                loss = loss / accum_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(images)
            if use_mixup:
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, labels)
            loss = loss / accum_steps
            loss.backward()

        if (step + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * images.size(0) * accum_steps
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

        if ema is not None:
            ema.update(model)

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def make_warmup_cosine_lr(optimizer, warmup_epochs, total_epochs, base_lr, final_lr=0.0):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def auto_scale_lr(base_lr, base_batch=128, batch_size=64):
    # linear scaling rule (simple). Adjust further if using OneCycle or if fine-tuning.
    return base_lr * (batch_size / base_batch)


def main():
    parser = argparse.ArgumentParser(description="Train MobileNet-V2 on CIFAR-10 (patched)")
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=None, help='base lr; if not set, will scale from 0.01 with batch_size')
    parser.add_argument('--no-pretrained', action='store_true', help='Set to disable ImageNet pretrained weights (default: pretrained)')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-path', type=str, default='best_mobilenetv2.pth')
    parser.add_argument('--width-mult', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup-alpha', type=float, default=0.8)
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--ema-decay', type=float, default=0.9999)
    parser.add_argument('--accum-steps', type=int, default=1)
    args = parser.parse_args()

    pretrained = not args.no_pretrained
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # LR default
    if args.lr is None:
        # base lr for batch 64 when fine-tuning
        default_base = 0.01
        args.lr = auto_scale_lr(default_base, base_batch=64, batch_size=args.batch_size)

    # ---- Transforms (ImageNet-style fine-tuning) ----
    # choose normalization based on whether we're fine-tuning ImageNet weights
    if pretrained:
    	mean = (0.485, 0.456, 0.406)
    	std  = (0.229, 0.224, 0.225)
    else:
    # CIFAR-10 normalization (used when training from scratch)
    	mean = (0.4914, 0.4822, 0.4465)
    	std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
    	transforms.RandomResizedCrop(224),
    	transforms.RandomHorizontalFlip(),
    	transforms.RandAugment(num_ops=2, magnitude=9),
    	transforms.ToTensor(),
    	transforms.RandomErasing(p=0.2),
    	transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
    	transforms.Resize(256),
    	transforms.CenterCrop(224),
    	transforms.ToTensor(),
    	transforms.Normalize(mean, std),
    ])


    train_dataset = datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Model: if pretrained requested and width !=1.0, warn user and use width 1.0 for pretrained loading
    load_width = args.width_mult
    if pretrained and args.width_mult != 1.0:
        print(f"Warning: pretrained weights are for width_mult=1.0. Using width_mult=1.0 for weight loading but building model with width_mult={args.width_mult}.")
        load_width = 1.0

    model = get_mobilenet_v2(num_classes=10, pretrained=pretrained, device=device, width_mult=args.width_mult, dropout_prob=args.dropout)

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # LR schedule: warmup + cosine annealing
    scheduler = make_warmup_cosine_lr(optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs, base_lr=args.lr)

    scaler = torch.cuda.amp.GradScaler() if (args.use_amp and device.type == 'cuda') else None

    ema = EMA(model, decay=args.ema_decay) if args.ema else None

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler,
                                                accum_steps=args.accum_steps, use_mixup=args.mixup,
                                                mixup_alpha=args.mixup_alpha, ema=ema)

        # Evaluate EMA model if available, otherwise model
        if ema is not None:
            val_loss, val_acc = evaluate(ema.ema, test_loader, criterion, device)
        else:
            val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step()
        t1 = time.time()

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc}, args.save_path)
            # also save EMA if exists
            if ema is not None:
                torch.save({'epoch': epoch,
                            'model_state_dict': ema.ema.state_dict(),
                            'val_acc': val_acc}, args.save_path.replace('.pth', '_ema.pth'))

        print(f"Epoch {epoch}/{args.epochs} | Time: {t1-t0:.1f}s | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% {'[BEST]' if is_best else ''}")

    print(f"Training complete. Best val acc: {best_acc:.2f}% | Saved to: {args.save_path}")


if __name__ == '__main__':
    main()
