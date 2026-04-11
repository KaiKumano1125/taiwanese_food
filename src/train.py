import os
import argparse
import logging
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from dataset_loader import get_train_dataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a food classification model.")
    parser.add_argument(
        "--config",
        type=str,
        default="../config/config.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_dataloaders(cfg: dict, transform: transforms.Compose):
    dataset = get_train_dataset(cfg["dataset"]["train_dir"], transform)

    train_ratio, val_ratio = 0.8, 0.1
    val_size = int(len(dataset) * val_ratio)
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size - val_size

    train_subset, val_subset, test_subset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    logger.info("Dataset split — Train: %d, Val: %d, Test: %d", train_size, val_size, test_size)

    loader_kwargs = dict(
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
    )
    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_subset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, dataset.classes


def build_model(cfg: dict, num_classes: int, device: torch.device) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if cfg["model"]["pretrained"] else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss, running_corrects = 0.0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * imgs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects / len(loader.dataset)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, total_corrects = 0.0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            total_loss += loss.item() * imgs.size(0)
            total_corrects += torch.sum(preds == labels.data).item()

    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc = total_corrects / len(loader.dataset)
    return epoch_loss, epoch_acc


def run_training(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    transform = build_transforms()
    train_loader, val_loader, test_loader, classes = build_dataloaders(cfg, transform)
    model = build_model(cfg, num_classes=len(classes), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]))

    num_epochs = cfg["training"]["epochs"]
    os.makedirs(cfg["output"]["save_dir"], exist_ok=True)
    save_path = os.path.join(cfg["output"]["save_dir"], cfg["output"]["save_name"])

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        logger.info(
            "Epoch [%d/%d] | Train Loss: %.4f, Acc: %.4f | Val Loss: %.4f, Acc: %.4f",
            epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            logger.info("New best model saved (val_acc=%.4f) -> %s", best_val_acc, save_path)

    logger.info("Training complete. Best val accuracy: %.4f", best_val_acc)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    logger.info("Test Accuracy: %.4f | Test Loss: %.4f", test_acc, test_loss)


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    run_training(cfg)
