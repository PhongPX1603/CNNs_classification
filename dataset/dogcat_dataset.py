import cv2
import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class DogCatDataset(Dataset):
    def __init__(self, image_dir, image_pattern, classes,
                 image_size=(224, 224),
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 transforms=None):
        super(DogCatDataset, self).__init__()
        self.classes = classes
        self.image_size = image_size
        self.mean = torch.tensor(mean, dtype=torch.float).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float).view(3, 1, 1)
        self.transforms = transforms if transforms is not None else []
        self.image_paths = list(Path(image_dir).glob(image_pattern))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # agumentation
        sample = torch.from_numpy(image)  # torch.uint8, HWC
        sample = sample.permute(2, 0, 1).contiguous()  # torch.uint8, CHW
        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            sample = transform(sample)

        sample = transforms.Resize(size=self.image_size)(sample)

        # normalization
        sample = (sample.float().div(255.) - self.mean) / self.std

        # target
        class_name = image_path.stem[:3]
        label = self.classes[class_name]
        label = torch.tensor(label, dtype=torch.int64)

        return sample, label