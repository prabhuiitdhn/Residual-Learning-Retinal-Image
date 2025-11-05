import os
import torch
from model import ResidualDenoiser
from dataset import PairedImageDataset
import numpy as np
from PIL import Image
import cv2

# Define paths for testing data
test_noisy_folder    = r"D:\Hello\Residual_learning_Retina_Image\src\dataset\testing\noisy"
checkpoint_path = r"D:\Hello\Residual_learning_Retina_Image\src\checkpoints"
save_results_path = r"D:\Hello\Residual_learning_Retina_Image\src\results"
img_size = (180, 180)

# Load test dataset (only noisy images)
class NoisyImageDataset(torch.utils.data.Dataset):
    def __init__(self, noisy_folder, img_size=(180,180)):
        self.noisy_paths = sorted([
            os.path.join(noisy_folder, f)
            for f in os.listdir(noisy_folder)
            if f.lower().endswith(('.png','.jpg','.jpeg'))
        ])
        self.img_size = img_size
    def __len__(self):
        return len(self.noisy_paths)
    def __getitem__(self, idx):
        noisy = cv2.imread(self.noisy_paths[idx])
        noisy = cv2.resize(noisy, self.img_size)
        noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
        noisy = torch.tensor(noisy.transpose(2,0,1), dtype=torch.float32) / 255.0
        return noisy

test_dataset = NoisyImageDataset(test_noisy_folder, img_size=img_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResidualDenoiser()
model.load_state_dict(torch.load(os.path.join(checkpoint_path, "residual_denoiser_final.pt"), map_location=device))
model.to(device)
model.eval()

save_dir = os.path.join(save_results_path, "test_outputs")
os.makedirs(save_dir, exist_ok=True)

with torch.no_grad():
    for idx, noisy in enumerate(test_loader):
        noisy = noisy.to(device)
        residual_pred = model(noisy)
        denoised_pred = noisy - residual_pred
        # Convert to numpy and save as image
        denoised_img = denoised_pred.squeeze().cpu().numpy()
        denoised_img = np.clip(denoised_img, 0, 1)
        denoised_img = (denoised_img * 255).astype(np.uint8)
        denoised_img = np.transpose(denoised_img, (1,2,0))
        img_pil = Image.fromarray(denoised_img)
        img_pil.save(os.path.join(save_dir, f"denoised_{idx+1}.png"))
       
print(f"✅ Denoised test outputs saved to {save_dir}")
