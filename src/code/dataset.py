import os
import cv2
import torch
from torch.utils.data import Dataset

class PairedImageDataset(Dataset):
    """
    Loads paired images (original, noisy) from two folders for residual denoising.
    Returns: original image, noisy image (both normalized to [0,1])
    """
    def __init__(self, original_folder, noisy_folder, img_size=(180,180)):
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
        orig = torch.tensor(orig.transpose(2,0,1), dtype=torch.float32) / 255.0
        noisy = torch.tensor(noisy.transpose(2,0,1), dtype=torch.float32) / 255.0
        return orig, noisy
