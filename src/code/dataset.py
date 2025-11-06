import os
import cv2
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class PairedImageDataset(Dataset):
    """
    Loads paired images (original, noisy) from two folders for residual denoising.
    Returns: original image, noisy image (both normalized to [0,1])
    """
    def __init__(self, original_folder, noisy_folder, img_size=(256,256)):
        self.original_paths = sorted([
            os.path.join(original_folder, f)
            for f in os.listdir(original_folder)
            if f.lower().endswith(('.png','.jpg','.jpeg'))
        ])
        self.noisy_paths = sorted([
            os.path.join(noisy_folder, f)
            for f in os.listdir(noisy_folder)
            if f.lower().endswith(('.png','.jpg','.jpeg'))
        ])
        assert len(self.original_paths) == len(self.noisy_paths), "Original and noisy image counts do not match!"
        self.img_size = img_size

    def __len__(self):
        return len(self.original_paths)

    def __getitem__(self, idx):
        orig = cv2.imread(self.original_paths[idx])
        noisy = cv2.imread(self.noisy_paths[idx])
        orig = cv2.resize(orig, self.img_size)
        noisy = cv2.resize(noisy, self.img_size)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
        # Morphological mask creation (foreground extraction)
        gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = (mask > 0).astype(np.float32)  # shape (H, W)
        # Set background to pure black in both orig and noisy
        orig = orig * mask[..., None]
        noisy = noisy * mask[..., None]
        orig = torch.tensor(orig.transpose(2,0,1), dtype=torch.float32) / 255.0
        noisy = torch.tensor(noisy.transpose(2,0,1), dtype=torch.float32) / 255.0
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        # Ensure mask is 1 for foreground, 0 for background, and images are 256x256
        assert orig.shape[1] == 256 and orig.shape[2] == 256, f"Image shape is {orig.shape}"
        assert noisy.shape[1] == 256 and noisy.shape[2] == 256, f"Image shape is {noisy.shape}"
        assert mask_tensor.shape[0] == 256 and mask_tensor.shape[1] == 256, f"Mask shape is {mask_tensor.shape}"
        return noisy, orig, mask_tensor
