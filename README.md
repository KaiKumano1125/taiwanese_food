# Taiwanese Food-101 Classification

This repository presents a deep learning project for recognizing **traditional Taiwanese dishes** using **Convolutional Neural Networks (CNNs)** built with **PyTorch**.  
A **ResNet-18** model (pretrained on ImageNet) is fine-tuned on the **TW Food-101 dataset**, which contains **over 20,000 images** across **101 iconic Taiwanese food categories** such as *beef noodles*, *bubble tea*, *bawan*, and more.

The goal of this project is to develop an efficient and accurate food image classifier that can serve as a foundation for applications in **smart restaurant menus**, **AI-assisted food recognition**, or **cultural food datasets**.


---

## Setup

### Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux / macOS
```

Before running the project, install all dependencies listed in `requirements.txt`.

```bash
pip install -r requirements.txt

```
## Training model
```bash
python train.py
```

## Default Model

| Component | Description |
|------------|-------------|
| **Architecture** | ResNet-18 (pretrained on ImageNet) |
| **Framework** | PyTorch |
| **Loss Function** | Cross-Entropy |
| **Optimizer** | Adam |
| **Input Size** | 224 × 224 pixels |
| **Number of Classes** | 101 Taiwanese food categories |


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

## Author
Kai Kumano -Graduate School of Engineering, Chiba University










