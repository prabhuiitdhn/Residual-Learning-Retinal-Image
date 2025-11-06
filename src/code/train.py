import os
import torch
from model import ResidualDenoiser
from dataset import PairedImageDataset
from trainer import Trainer

# Define explicit paths for each split
train_original_folder = r"D:\Hello\Residual_learning_Retina_Image\src\dataset\training\original"
train_noisy_folder    = r"D:\Hello\Residual_learning_Retina_Image\src\dataset\training\noisy"
val_original_folder   = r"D:\Hello\Residual_learning_Retina_Image\src\dataset\validation\original"
val_noisy_folder      = r"D:\Hello\Residual_learning_Retina_Image\src\dataset\validation\noisy"

checkpoint_path = r"D:\Hello\Residual_learning_Retina_Image\src\checkpoints"

# Dataset setup
img_size = (256, 256)
train_dataset = PairedImageDataset(train_original_folder, train_noisy_folder, img_size=img_size)
val_dataset   = PairedImageDataset(val_original_folder, val_noisy_folder, img_size=img_size)

# Model setup
model = ResidualDenoiser()

# Trainer setup
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=1,
    lr=1e-4,
    epochs=500,
    save_dir=checkpoint_path
)

if __name__ == "__main__":
    trainer.train()