import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class TWFoodDataset(Dataset):
    def __init__(self, csvfile, root_dir, transform=None):
        