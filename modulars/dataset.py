
# Dataset.py
import albumentations as A
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import torchvision.transforms as T

def define_transformer(basic=False, augment=False):
    if basic:
        train_transform = A.Compose([
            A.Resize(width=128, height=128)
        ])

        test_transform = A.Compose([
            A.Resize(width=128, height=128)
        ])
    elif augment:
        train_transform = A.Compose([
            A.Resize(width=128, height=128),  # Resize images to 128x128 pixels
            A.HorizontalFlip(p=0.5),  # Apply horizontal flip with 50% probability
            A.VerticalFlip(p=0.5),  # Apply vertical flip with 50% probability
            A.RandomRotate90(p=0.5),  # Rotate randomly by 90 degrees with 50% probability
            A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),  # Randomly shift, scale, and rotate
        ])

        test_transform = A.Compose([
            A.Resize(width=128, height=128)
        ])
    
    return train_transform, test_transform


# Create dataset
class BrainDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path, mask_path = self.df.iloc[idx, 0], self.df.iloc[idx, 1]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']

        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = image / 255.
        image = torch.from_numpy(image)
        image = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)

        mask = np.expand_dims(mask, 0).astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = (mask > 0).float()
        
        return image, mask
