"""
Training loop for ResNet-50 vision classifier.
Usage: python src/vision/train.py --config configs/vision_train.yaml
"""
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from src.vision.model import build_model
from src.vision.dataset import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vision_train.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # Model
    model = build_model(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"] == "imagenet",
    ).to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    save_dir = Path("models/vision")
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_resnet50.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
