import os
import yaml
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
from dataset_loader import get_train_dataset


with open("../config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -----------------------
# 3. Load model
# -----------------------
from torchvision.models import resnet18, ResNet18_Weights

# Load class names dynamically from train dataset
train_dataset = get_train_dataset(cfg["dataset"]["train_dir"], test_tf)
class_names = train_dataset.classes

weights = ResNet18_Weights.DEFAULT if cfg["model"]["pretrained"] else None
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(
    os.path.join(cfg["output"]["save_dir"], cfg["output"]["save_name"]),
    map_location=device
))
model = model.to(device)
model.eval()

# print(f"[INFO] Model loaded with {len(class_names)} classes.")
print(f"[INFO] Model name: {cfg['output']['save_name']}")

# -----------------------
# 4. Inference function
# -----------------------
def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        return None

    image = Image.open(img_path).convert("RGB")
    image = test_tf(image).unsqueeze(0).to(device)  

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
        predicted_class = class_names[pred.item()]

    print(f"The food is: {predicted_class}")
    return predicted_class

# -----------------------
# 5. Example usage
# -----------------------

img_path = "../dataset/kai3.jpg"
predict_image(img_path)
