"""EcoSort Edge-AI training and ONNX export script.

Run from the project folder, for example:
    python train_template.py --data_dir data/public --epochs 30
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGE_SIZE = 96
MAX_PARAMETERS = 350_000
MAX_MODEL_BYTES = int(1.50 * 1024 * 1024)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


class ConvBNReLU(nn.Sequential):
    """A convolution followed by batch normalization and ReLU6."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding,
                groups=groups, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    """Expand channels, apply a cheap depthwise filter, then project back."""

    def __init__(self, in_channels, out_channels, stride, expansion=4):
        super().__init__()
        hidden_channels = in_channels * expansion
        layers = []

        # Expand the feature channels to give the block more learning capacity.
        layers.append(ConvBNReLU(in_channels, hidden_channels, kernel_size=1))
        # A depthwise convolution filters each channel separately to save size.
        layers.append(
            ConvBNReLU(hidden_channels, hidden_channels, kernel_size=3,
                       stride=stride, groups=hidden_channels)
        )
        # Project to the requested output width without adding another activation.
        layers.extend([
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x):
        result = self.block(x)
        # Skip connections help information and gradients pass through the network.
        return x + result if self.use_residual else result


class EcoSortClassifier(nn.Module):
    """Compact six-class CNN designed for 96x96 images and the model budget."""

    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(3, 32, kernel_size=3, stride=2),       # 96x96 -> 48x48
            InvertedResidual(32, 40, stride=1),
            InvertedResidual(40, 80, stride=2),               # -> 24x24
            InvertedResidual(80, 80, stride=1),
            InvertedResidual(80, 128, stride=2),              # -> 12x12
            InvertedResidual(128, 160, stride=2),             # -> 6x6
            nn.Conv2d(160, 160, kernel_size=1, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),                           # -> 1x1 per channel
        )
        self.classifier = nn.Linear(160, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)  # Return six class scores (logits).


def count_parameters(model):
    """Count trainable model parameters before training or export."""
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


def get_dataloaders(data_dir, batch_size):
    """Load class-folder images and apply matching train/validation scaling."""
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Expected train and val folders inside '{data_dir}'."
        )

    # Small random changes help the model generalize to new waste photos.
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])
    # Keep validation deterministic so epoch results can be compared fairly.
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])

    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    val_set = datasets.ImageFolder(val_dir, transform=val_transform)
    if train_set.classes != val_set.classes:
        raise ValueError("Train and validation folders must have the same class names.")
    if train_set.classes != CLASSES:
        raise ValueError(
            f"Expected class folders in this order: {CLASSES}; found {train_set.classes}"
        )

    # zero workers is reliable on Windows and easy to run from VS Code.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def evaluate(model, loader, criterion, device):
    """Return average loss and accuracy for one complete data split."""
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / max(total, 1), 100.0 * correct / max(total, 1)


def export_to_onnx(model, output_path):
    """Export dynamic-batch ONNX and stop if the file-size budget is exceeded."""
    model.eval().cpu()
    sample = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
    export_options = dict(
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    try:
        torch.onnx.export(
            model, sample, output_path, dynamo=False, **export_options
        )
    except TypeError:
        # Some earlier PyTorch 2.x versions do not accept the dynamo argument.
        torch.onnx.export(model, sample, output_path, **export_options)
    size_bytes = os.path.getsize(output_path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"ONNX file size: {size_mb:.2f} MB / 1.50 MB")
    if size_bytes > MAX_MODEL_BYTES:
        raise ValueError("Exported model exceeds the 1.50 MB file-size limit.")


def main():
    parser = argparse.ArgumentParser(description="Train and export the EcoSort image classifier")
    parser.add_argument("--data_dir", default="data/public", help="Folder containing train/ and val/")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=32, help="Images per training batch")
    parser.add_argument("--lr", type=float, default=0.001, help="Starting learning rate")
    parser.add_argument("--output", default="model.onnx", help="Output ONNX file path")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs and batch_size must be positive, and lr must be greater than zero")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size)
    model = EcoSortClassifier(num_classes=len(CLASSES)).to(device)

    parameter_count = count_parameters(model)
    print(f"Trainable parameters: {parameter_count:,} / {MAX_PARAMETERS:,}")
    if parameter_count > MAX_PARAMETERS:
        raise ValueError("Model exceeds the 350,000-parameter limit.")
    print(f"Training images: {len(train_loader.dataset)} | Validation images: {len(val_loader.dataset)}")

    # Label smoothing and weight decay provide mild regularization on small datasets.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_accuracy = -1.0
    best_weights = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        train_loss = train_loss_sum / max(train_total, 1)
        train_accuracy = 100.0 * train_correct / max(train_total, 1)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.3f}, acc {train_accuracy:.1f}% | "
            f"val loss {val_loss:.3f}, acc {val_accuracy:.1f}%"
        )

        # Keep the weights from the best validation epoch, not just the last one.
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_weights = {name: value.detach().cpu().clone()
                            for name, value in model.state_dict().items()}
        scheduler.step()

    if best_weights is None:
        raise RuntimeError("Training ended without a validation checkpoint.")
    model.load_state_dict(best_weights)
    print(f"Best validation accuracy: {best_accuracy:.2f}% (target: at least 75%)")
    export_to_onnx(model, args.output)
    print("Next, check the exported model with verify_submission.py.")
    print(f"python verify_submission.py --model {args.output} --val_dir {os.path.join(args.data_dir, 'val')}")


if __name__ == "__main__":
    main()
