# test.py (fixed)
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import get_mobilenet_v2

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

def main():
    parser = argparse.ArgumentParser(description='Test MobileNet-V2 on CIFAR-10')
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--checkpoint', type=str, default='best_mobilenetv2.pth')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Set to disable ImageNet pretrained weights (default: pretrained)')
    parser.add_argument('--width-mult', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    pretrained = not args.no_pretrained
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # use ImageNet normalization if we're evaluating a model that was fine-tuned from ImageNet
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_dataset = datasets.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # build model (matches how train.py built it)
    model = get_mobilenet_v2(num_classes=10, pretrained=pretrained, device=device,
                             width_mult=args.width_mult, dropout_prob=args.dropout)

    # Load checkpoint: supports both {'model_state_dict':...} and raw state_dict
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif isinstance(ckpt, dict) and all(k.startswith('module.') or k in model.state_dict() for k in ckpt.keys()):
        # handles raw state_dict saved directly or from DataParallel
        try:
            model.load_state_dict(ckpt)
        except RuntimeError:
            # try stripping 'module.' if present
            new_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            model.load_state_dict(new_ckpt)
    else:
        # if ckpt contains outer dict, try to get nested state dicts
        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
            model.load_state_dict(sd)
        else:
            raise RuntimeError("Unrecognized checkpoint format. Please pass a valid checkpoint.")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss, acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {loss:.4f} | Test Acc: {acc:.2f}%")

if __name__ == "__main__":
    main()
