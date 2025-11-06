import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Paths
noisy_folder = r"D:\Hello\Residual_learning_Retina_Image\src\dataset\testing\noisy"
original_folder = r"D:\Hello\Residual_learning_Retina_Image\src\dataset\testing\original"
den_output_folder = r"D:\Hello\Residual_learning_Retina_Image\src\results\test_outputs"
img_size = (180, 180)

# Helper to load and normalize image
def load_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

# List files
noisy_files = sorted([f for f in os.listdir(noisy_folder) if f.lower().endswith(('.png','.jpg','.jpeg'))])
original_files = sorted([f for f in os.listdir(original_folder) if f.lower().endswith(('.png','.jpg','.jpeg'))])
den_files = sorted([f for f in os.listdir(den_output_folder) if f.startswith('denoised_') and f.endswith('.png')])

# Visualize a few samples
for idx in range(min(5, len(noisy_files), len(original_files), len(den_files))):
    noisy_img = load_img(os.path.join(noisy_folder, noisy_files[idx]))
    original_img = load_img(os.path.join(original_folder, original_files[idx]))
    denoised_img = load_img(os.path.join(den_output_folder, den_files[idx]))

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(noisy_img)
    axs[0].set_title('Noisy')
    axs[0].axis('off')
    axs[1].imshow(original_img)
    axs[1].set_title('Original')
    axs[1].axis('off')
    axs[2].imshow(denoised_img)
    axs[2].set_title('Denoised')
    axs[2].axis('off')
    plt.tight_layout()
    plt.show()
