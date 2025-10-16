## Taiwanese Food-101 Classification (PyTorch)

This repository contains a deep learning project for classifying Taiwanese foods using a CNN (ResNet-18) trained on the **TW Food-101** dataset.  
The dataset includes 101 traditional Taiwanese food categories such as beef noodles, bubble tea, bawan, and more.

---

## ⚙️ Setup

### 1️⃣ Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux / macOS
```
## Training model
```bash
python train.py
```

## Defualt Model
Architecture: ResNet-18 (pretrained on ImageNet)

Framework: PyTorch

Loss: Cross-Entropy

Optimizer: Adam

Image size: 224 × 224

Classes: 101 Taiwanese foods

## Configuration(config.yaml)
```
dataset:
  train_dir: "../dataset/tw_food_101/train"
  test_dir: "../dataset/tw_food_101/test"

model:
  name: "resnet18"
  num_classes: 101
  pretrained: true

training:
  batch_size: 32
  epochs: 15
  learning_rate: 0.0001
  num_workers: 0
  device: "cuda"

output:
  save_dir: "./output"
  save_name: "tw_food101_best.pth"
```












