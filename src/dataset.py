import os
import json
import cv2
import torch
from torch.utils.data import Dataset


class GCPDataset(Dataset):

    def __init__(self, image_dir, label_path, img_size=256):
        self.image_dir = image_dir
        self.img_size = img_size

        with open(label_path) as f:
            self.labels = json.load(f)

        self.keys = [
            k for k in self.labels
            if "verified_shape" in self.labels[k]
        ]

        # shape encoding
        self.shape_map = {
            "Cross": 0,
            "Square": 1,
            "L-Shaped": 2
        }

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):

        key = self.keys[idx]

        img_path = os.path.join(self.image_dir, key)

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # resize image
        img = cv2.resize(img, (256, 256))

        # convert to tensor format
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        label = self.labels[key]

        x = label["mark"]["x"]
        y = label["mark"]["y"]

        # normalize coordinates
        x = x / w
        y = y / h

        coords = torch.tensor([x, y], dtype=torch.float32)

        shape = label.get("verified_shape")

        if shape is None:
            shape = -1
        else:
            shape = self.shape_map[shape]

        shape = torch.tensor(shape, dtype=torch.long)

        return img, coords, shape