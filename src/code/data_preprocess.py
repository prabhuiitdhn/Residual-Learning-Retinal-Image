import os
import cv2
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import random

def advanced_augmentations():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomResizedCrop(size=(180, 180), scale=(0.8, 1.0)),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    ])

def augment_and_save_images(src_folder, save_folder, augmentations_per_image=10):
    os.makedirs(save_folder, exist_ok=True)
    aug = advanced_augmentations()
    image_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', 'tif'))]
    for img_name in tqdm(image_files, desc='Augmenting images'):
        img_path = os.path.join(src_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        for i in range(augmentations_per_image):
            seed = random.randint(0, 10000)
            random.seed(seed)
            np.random.seed(seed)
            torch_seed = random.randint(0, 10000)
            try:
                aug_img = aug(image)
            except Exception as e:
                print(f"Augmentation failed for {img_name}: {e}")
                continue
            aug_img_name = f"{os.path.splitext(img_name)[0]}_aug{i+1}.png"
            aug_img.save(os.path.join(save_folder, aug_img_name))
    print(f"Augmented images saved to {save_folder}")

def add_random_noise_to_folder(src_folder, save_folder, noise_sigma=0.05, img_size=(180, 180)):
    """
    Reads images from src_folder, adds random Gaussian noise, and saves to save_folder.
    """
    import cv2
    import numpy as np
    import os
    os.makedirs(save_folder, exist_ok=True)
    image_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_name in image_files:
        img_path = os.path.join(src_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        noisy_img = np.clip(img + noise_sigma * np.random.randn(*img.shape), 0, 1)
        noisy_img = (noisy_img * 255).astype(np.uint8)
        noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(save_folder, f"{os.path.splitext(img_name)[0]}_noisy.png")
        cv2.imwrite(save_path, noisy_img)
    print(f"Noisy images saved to {save_folder}")

if __name__ == "__main__":
    src_folder = r"D:\Hello\image\dataset\testing\original"
    augmented_save_folder = r"D:\Hello\image\dataset\testing\augmented"
    noise_save_folder = r"D:\Hello\image\dataset\testing\noisy"

    augment_and_save_images(src_folder, augmented_save_folder, augmentations_per_image=10)
    add_random_noise_to_folder(augmented_save_folder, noise_save_folder, noise_sigma=0.05, img_size=(180, 180))
