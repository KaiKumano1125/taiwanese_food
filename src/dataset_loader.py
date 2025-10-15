import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets


def get_train_dataset(train_dir, transform):
    """Load training dataset from folder structure."""
    dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    print(f"[INFO] Found {len(dataset)} training images across {len(dataset.classes)} classes.")
    return dataset


class TWFoodDataset(Dataset):
    """Load test images from CSV where column 1 has 'test/xxx.jpg'."""
    def __init__(self, csv_file, root_dir, transform=None):
        df = pd.read_csv(csv_file, header=None)
        df = df.dropna().reset_index(drop=True)

        # ✅ Use column 1 as the filename column
        self.data = df.iloc[:, 1].astype(str)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx].strip()
        img_name = os.path.basename(img_name)  # e.g., "0.jpg"
        img_path = os.path.join(self.root_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, -1  
