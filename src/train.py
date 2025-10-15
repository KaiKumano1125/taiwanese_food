import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset_loader import get_train_dataset


with open("../config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = get_train_dataset(cfg["dataset"]["train_dir"], train_tf)
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg["training"]["batch_size"],
    shuffle=True,
    num_workers=cfg["training"]["num_workers"]
)


from torchvision.models import resnet18, ResNet18_Weights
weights = ResNet18_Weights.DEFAULT if cfg["model"]["pretrained"] else None
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]))

if __name__ == "__main__":
    num_epochs = cfg["training"]["epochs"]
    for epoch in range(num_epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * imgs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

    os.makedirs(cfg["output"]["save_dir"], exist_ok=True)
    save_path = os.path.join(cfg["output"]["save_dir"], cfg["output"]["save_name"])
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")
