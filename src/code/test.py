import os
import torch
from model import ResidualDenoiser
from dataset import PairedImageDataset
import numpy as np
from PIL import Image

# Define paths for testing data
test_original_folder = r"D:\Hello\image\testing_original"
test_noisy_folder    = r"D:\Hello\image\testing_noisy"
img_size = (180, 180)

# Load test dataset
test_dataset = PairedImageDataset(test_original_folder, test_noisy_folder, img_size=img_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResidualDenoiser()
model.load_state_dict(torch.load(os.path.join("results", "residual_denoiser_final.pt"), map_location=device))
model.to(device)
model.eval()

save_dir = os.path.join("results", "test_outputs")
os.makedirs(save_dir, exist_ok=True)

with torch.no_grad():
    for idx, (orig, noisy) in enumerate(test_loader):
        orig = orig.to(device)
        noisy = noisy.to(device)
        residual_pred = model(orig)
        denoised_pred = orig + residual_pred
        # Convert to numpy and save as image
        denoised_img = denoised_pred.squeeze().cpu().numpy()
        denoised_img = np.clip(denoised_img, 0, 1)
        denoised_img = (denoised_img * 255).astype(np.uint8)
        denoised_img = np.transpose(denoised_img, (1,2,0))
        img_pil = Image.fromarray(denoised_img)
        img_pil.save(os.path.join(save_dir, f"denoised_{idx+1}.png"))
        # Optionally save residual as image
        residual_img = residual_pred.squeeze().cpu().numpy()
        residual_img = (residual_img - residual_img.min()) / (residual_img.max() - residual_img.min() + 1e-8)
        residual_img = (residual_img * 255).astype(np.uint8)
        residual_img = np.transpose(residual_img, (1,2,0))
        residual_pil = Image.fromarray(residual_img)
        residual_pil.save(os.path.join(save_dir, f"residual_{idx+1}.png"))
print(f"✅ Test outputs saved to {save_dir}")
