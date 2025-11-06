from PIL import Image
import numpy as np
import cv2
import random
import os
import cv2
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import random


def augment_foreground_only(src_folder, save_folder, augmentations_per_image=10):
    """
    For each image, separate foreground/background using thresholding, augment only the foreground, and save with pure black background.
    """

    os.makedirs(save_folder, exist_ok=True)
    aug = advanced_augmentations()
    image_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', 'tif'))]
    for img_name in tqdm(image_files, desc='Augmenting foreground only'):
        img_path = os.path.join(src_folder, img_name)
        image = Image.open(img_path).convert('RGB').resize((256, 256), Image.BILINEAR)
        img_np = np.array(image).astype(np.float32) / 255.0
        # Morphological mask creation (foreground extraction)
        gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = (mask > 0).astype(np.float32)[..., None]  # shape (H, W, 1)
        for i in range(augmentations_per_image):
            seed = random.randint(0, 10000)
            random.seed(seed)
            np.random.seed(seed)
            try:
                # Augment only the foreground
                fg = img_np * mask
                fg_img = Image.fromarray((fg * 255).astype(np.uint8))
                aug_fg = aug(fg_img)
                aug_fg = aug_fg.resize((256, 256), Image.BILINEAR)
                aug_fg_np = np.array(aug_fg).astype(np.float32) / 255.0
                # Compose: augmented foreground + pure black background
                out_img = aug_fg_np * mask
                out_img = np.clip(out_img, 0, 1)
                out_img_uint8 = (out_img * 255).astype(np.uint8)
                aug_img_name = f"{os.path.splitext(img_name)[0]}_fgaug{i+1}.jpg"
                Image.fromarray(out_img_uint8).save(os.path.join(save_folder, aug_img_name), format='JPEG', quality=95)
            except Exception as e:
                print(f"Augmentation failed for {img_name}: {e}")
                continue
    print(f"Foreground-only augmented images saved to {save_folder}")

def add_noise_foreground_only(src_folder, save_folder, noise_sigma=0.05):
    """
    Reads images from src_folder, adds random Gaussian noise only to the foreground (non-black), and saves to save_folder.
    """
    os.makedirs(save_folder, exist_ok=True)
    image_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_name in image_files:
        img_path = os.path.join(src_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        if img.shape[0] == 256 and img.shape[1] == 256:
            pass
        else:
            img = cv2.resize(img, (256, 256))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        # Create mask: 1 where any channel is nonzero, 0 where all channels are zero (black)
        mask = (img_norm.sum(axis=2) > 0).astype(np.float32)[..., None]  # shape (H, W, 1)
        # Add noise only to foreground
        noise = noise_sigma * np.random.randn(*img_norm.shape)
        noisy_img = img_norm + noise * mask
        noisy_img = np.clip(noisy_img, 0, 1)
        # Keep background unchanged
        out_img = img_norm * (1 - mask) + noisy_img * mask
        out_img = (out_img * 255).astype(np.uint8)
        out_img_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(save_folder, f"{os.path.splitext(img_name)[0]}_noisy_fg.png")
        cv2.imwrite(save_path, out_img_bgr)
    print(f"Foreground-only noisy images saved to {save_folder}")


def advanced_augmentations():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomResizedCrop(size=(180, 180), scale=(0.8, 1.0)),
    ])


if __name__ == "__main__":
    src_folder = r"D:\Hello\images\testing\original"
    augmented_save_folder = r"D:\Hello\images\testing\augmented"
    noise_save_folder = r"D:\Hello\images\testing\noisy"

    # augment_foreground_only(src_folder, augmented_save_folder, augmentations_per_image=10)
    add_noise_foreground_only(augmented_save_folder, noise_save_folder, noise_sigma=0.005)
# 