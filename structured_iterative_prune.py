#!/usr/bin/env python3
# structured_iterative_prune.py
# Structured iterative channel pruning + BN recalibration (Windows-ready)
import os
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import your model and training utilities
# Assumes model.py exposes get_mobilenet_v2(...) and train.py exposes train_one_epoch, evaluate, EMA, auto_scale_lr
from model import get_mobilenet_v2
from train import train_one_epoch, evaluate, EMA, auto_scale_lr

# Windows-safe multiprocessing
if os.name == 'nt':
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

# ---------------- Dataset / transforms ----------------
def build_dataloaders(data_dir, batch, num_workers, pretrained=True):
    """
    Build CIFAR10 loaders that match ImageNet pretrained transforms (224x224 + ImageNet mean/std)
    This matches your reported baseline (pretrained MobileNet-V2).
    """
    if pretrained:
        mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225)
        train_t = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_t = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        mean = (0.4914, 0.4822, 0.4465); std = (0.2470, 0.2435, 0.2616)
        train_t = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_t)
    testset  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_t)
    pin = True if torch.cuda.is_available() else False
    train_loader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(testset, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader

def export_pruned_cfg(model):
    """
    Extract per-block output channels so the pruned architecture
    can be reconstructed exactly later.
    """
    cfg = []

    for blk in model.features:
        if hasattr(blk, "conv"):
            # find last Conv2d in the block (projection layer)
            out_ch = None
            for m in reversed(blk.conv):
                if isinstance(m, torch.nn.Conv2d):
                    out_ch = m.out_channels
                    break
            cfg.append(out_ch)
        else:
            cfg.append(None)

    return cfg


# ---------------- helper to find pointwise conv in MobileNetV2 blocks ----------------
def find_pw_conv(block):
    """
    Return (sequential_container, idx) for the 1x1 conv (pointwise conv) in a block,
    or (None, None) if not found.
    """
    seq = block.conv if hasattr(block, 'conv') else None
    if seq is None:
        return None, None
    idx = None
    for i, m in enumerate(seq):
        if isinstance(m, torch.nn.Conv2d) and m.kernel_size == (1, 1) and m.groups == 1:
            idx = i
    if idx is None:
        return None, None
    return seq, idx

# ---------------- robust module replacement helper ----------------
def _replace_child_module(root_module, target_mod, new_mod):
    """
    Find and replace target_mod which is a direct child of some parent module within root_module's tree.
    Returns True on success.
    """
    for parent in root_module.modules():
        for name, child in list(parent.named_children()):
            if child is target_mod:
                setattr(parent, name, new_mod)
                return True
    return False

# ---------------- structured pruning function ----------------
def prune_pointwise_l1(model, prune_fraction, skip_first=True, skip_last=True):
    """
    Simple structured pruning:
    - For each block, find pointwise 1x1 conv, compute L1 score per output channel,
      keep top (1-prune_fraction) channels, rebuild that 1x1 conv & following BN,
      then update downstream convs that consume the pruned outputs (best-effort).
    Note: This performs *structural* reduction in channel counts (not just zeroing weights).
    """
    features = model.features
    for i, blk in enumerate(features):
        if i == 0 and skip_first:  # skip initial stem
            continue
        if i == len(features) - 1 and skip_last:
            continue
        seq, idx = find_pw_conv(blk)
        if seq is None:
            continue
        pw = seq[idx]
        # channel scores (L1)
        scores = pw.weight.data.abs().mean(dim=(1, 2, 3))
        n_out = scores.numel()
        k = int(math.floor((1.0 - prune_fraction) * n_out))
        if k < 1:
            continue
        _, order = torch.sort(scores, descending=True)
        keep = order[:k].sort().values  # sorted indices of kept channels (ascending)
        # make new pointwise conv with fewer out channels
        new_pw = torch.nn.Conv2d(in_channels=pw.in_channels,
                                  out_channels=len(keep),
                                  kernel_size=pw.kernel_size,
                                  stride=pw.stride,
                                  padding=pw.padding,
                                  bias=(pw.bias is not None))
        with torch.no_grad():
            new_pw.weight[:] = pw.weight[keep.tolist()].clone()
            if pw.bias is not None:
                new_pw.bias[:] = pw.bias[keep].clone()
        seq[idx] = new_pw

        # adjust BN if present right after conv
        if idx + 1 < len(seq) and isinstance(seq[idx + 1], torch.nn.BatchNorm2d):
            old_bn = seq[idx + 1]
            new_bn = torch.nn.BatchNorm2d(len(keep), eps=old_bn.eps, momentum=old_bn.momentum,
                                          affine=old_bn.affine, track_running_stats=old_bn.track_running_stats)
            with torch.no_grad():
                if old_bn.affine:
                    new_bn.weight[:] = old_bn.weight[keep].clone()
                    new_bn.bias[:] = old_bn.bias[keep].clone()
                if old_bn.track_running_stats:
                    new_bn.running_mean[:] = old_bn.running_mean[keep].clone()
                    new_bn.running_var[:] = old_bn.running_var[keep].clone()
            seq[idx + 1] = new_bn

        # update downstream convs that expect the original n_out input channels (best-effort)
        for j in range(i + 1, len(features)):
            sub = features[j].conv if hasattr(features[j], 'conv') else features[j]
            for mod in list(sub.modules()):
                if isinstance(mod, torch.nn.Conv2d) and mod.in_channels == n_out:
                    new_in = len(keep)
                    if mod.groups == n_out:
                        # depthwise conv case (in_channels == groups == out_channels)
                        new_conv = torch.nn.Conv2d(new_in, new_in, kernel_size=mod.kernel_size,
                                                   stride=mod.stride, padding=mod.padding, groups=new_in,
                                                   bias=(mod.bias is not None))
                        with torch.no_grad():
                            new_conv.weight[:] = mod.weight[keep.tolist()].clone()
                    else:
                        new_conv = torch.nn.Conv2d(new_in, mod.out_channels, kernel_size=mod.kernel_size,
                                                   stride=mod.stride, padding=mod.padding,
                                                   bias=(mod.bias is not None))
                        with torch.no_grad():
                            new_conv.weight[:] = mod.weight[:, keep.tolist(), :, :].clone()
                            if mod.bias is not None:
                                new_conv.bias[:] = mod.bias.clone()
                    # replace in parent
                    replaced = _replace_child_module(model, mod, new_conv)
                    if not replaced:
                        _replace_child_module(sub, mod, new_conv)
    return model

# ---------------- BatchNorm recalibration ----------------
@torch.no_grad()
def recalibrate_bn(model, loader, device, n_batches=200):
    """
    Run forward pass on a subset (n_batches) of train loader in train() mode to update running_mean/var.
    Model must be in training mode for BN updates; we don't update weights (no grad).
    """
    model.train()
    i = 0
    for xb, _ in loader:
        xb = xb.to(device)
        model(xb)
        i += 1
        if i >= n_batches:
            break
    model.eval()

# ---------------- robust checkpoint loader ----------------
def _extract_state_dict(ck):
    """
    Return a cleaned state_dict from common checkpoint formats.
    Handles:
      - raw state_dict (OrderedDict)
      - dict with 'model_state_dict' or 'state_dict'
      - strips 'module.' prefixes
    """
    if isinstance(ck, dict):
        if 'model_state_dict' in ck:
            sd = ck['model_state_dict']
        elif 'state_dict' in ck:
            sd = ck['state_dict']
        else:
            sd = ck
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        return sd
    return ck

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(description="Structured iterative pruning + BN recalibration (Windows-ready)")
    parser.add_argument('--checkpoint', default='best_mobilenetv2.pth', help='path to baseline checkpoint')
    parser.add_argument('--data', default='./data', help='data directory for CIFAR10')
    parser.add_argument('--iterations', type=int, default=8, help='num pruning iterations')
    parser.add_argument('--prune-per-iter', type=float, default=0.04, help='fraction pruned per iteration (e.g. 0.04)')
    parser.add_argument('--epochs-per-iter', type=int, default=6, help='finetune epochs after each prune')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--kd', action='store_true', help='use knowledge distillation from teacher checkpoint (optional)')
    parser.add_argument('--teacher', default=None, help='teacher checkpoint path for KD')
    parser.add_argument('--save-dir', default='./prune_runs_structured', help='where to save intermediate & final models')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # instantiate model then load ckpt
    # get_mobilenet_v2 should return nn.Module (your model.py)
    model = get_mobilenet_v2(num_classes=10, pretrained=True, device=device)
    ck = torch.load(args.checkpoint, map_location='cpu')
    sd = _extract_state_dict(ck)
    load_res = model.load_state_dict(sd, strict=False)
    if load_res.missing_keys:
        print("Warning: missing keys when loading checkpoint:", load_res.missing_keys)
    if load_res.unexpected_keys:
        print("Warning: unexpected keys when loading checkpoint:", load_res.unexpected_keys)
    model.to(device)

    # optional teacher for KD
    teacher = None
    if args.kd:
        teacher = get_mobilenet_v2(num_classes=10, pretrained=True, device=device)
        tck = torch.load(args.teacher if args.teacher is not None else args.checkpoint, map_location='cpu')
        tsd = _extract_state_dict(tck)
        tload_res = teacher.load_state_dict(tsd, strict=False)
        if tload_res.missing_keys:
            print("Teacher load missing keys:", tload_res.missing_keys)
        teacher.to(device).eval()

    # loaders
    train_loader, test_loader = build_dataloaders(args.data, args.batch_size, args.num_workers, pretrained=True)

    # default lr scaling helper from your train.py; fall back to 0.001 if not present
    if args.lr is None:
        try:
            args.lr = auto_scale_lr(0.01, base_batch=64, batch_size=args.batch_size)
        except Exception:
            args.lr = 0.001

    best_val = -1.0
    for it in range(args.iterations):
        print(f"\n=== Iter {it+1}/{args.iterations} | prune-per-iter {args.prune_per_iter:.4f} ===")
        t0 = time.time()

        # structured prune
        model = prune_pointwise_l1(model, args.prune_per_iter, skip_first=True, skip_last=True)
        model.to(device).float()

        # BN recalibration (important step)
        print("Running BN recalibration (forward-only) on train set...")
        recalibrate_bn(model, train_loader, device, n_batches=200)
        print("BN recalibration done. Time:", time.time() - t0)

        # finetune after pruning
        finetune_lr = args.lr * 0.5
        optimz = optim.SGD(model.parameters(), lr=finetune_lr, momentum=0.9, weight_decay=5e-4)
        ema = EMA(model, decay=0.9999)

        for ep in range(1, args.epochs_per_iter + 1):
            if teacher is not None:
                # simple KD training loop (explicit) so we can combine CE+KD
                model.train()
                total_n = 0
                total_correct = 0
                for xb, yb in train_loader:
                    xb = xb.to(device); yb = yb.to(device)
                    optimz.zero_grad()
                    s = model(xb)
                    with torch.no_grad():
                        t_logits = teacher(xb)
                    ce = nn.CrossEntropyLoss()(s, yb)
                    T = 4.0; alpha = 0.7
                    kd_loss = nn.KLDivLoss(reduction='batchmean')(torch.log_softmax(s / T, dim=1), torch.softmax(t_logits / T, dim=1)) * (T * T)
                    loss = alpha * ce + (1.0 - alpha) * kd_loss
                    loss.backward()
                    optimz.step()
                    ema.update(model)
                    preds = s.argmax(dim=1)
                    total_correct += (preds == yb).sum().item()
                    total_n += yb.size(0)
                tr_acc = 100.0 * total_correct / total_n
            else:
                # call your train_one_epoch helper (it should support the same args used in baseline)
                tr_loss, tr_acc = train_one_epoch(model, train_loader, nn.CrossEntropyLoss(), optimz, device,
                                                 scaler=None, accum_steps=1, use_mixup=False, mixup_alpha=0.8, ema=ema)

            # evaluate with EMA weights if present
            eval_model = ema.ema if hasattr(ema, 'ema') else model
            val_loss, val_acc = evaluate(eval_model, test_loader, nn.CrossEntropyLoss(), device)
            print(f"[Iter {it+1}] Ep {ep}/{args.epochs_per_iter} | Train Acc: {tr_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            # save best per-iteration
            if val_acc > best_val:
                best_val = val_acc
                model_cfg = export_pruned_cfg(model)

                torch.save(
                    {
                        "model_cfg": model_cfg,
                        "model_state_dict": model.state_dict()
                    },
                    os.path.join(args.save_dir, f"best_iter{it+1}_ep{ep}_{val_acc:.2f}.pth")
                )


    # final save
    # ===== FINAL SAVE (ARCH + WEIGHTS) =====
    model_cfg = export_pruned_cfg(model)

    torch.save(
        {
            "model_cfg": model_cfg,          # 🔑 architecture
            "model_state_dict": model.state_dict()  # 🔑 weights
        },
        os.path.join(args.save_dir, "final_pruned.pth")
    )

    print("Saved final pruned model with architecture + weights")
    

if __name__ == "__main__":
    main()
