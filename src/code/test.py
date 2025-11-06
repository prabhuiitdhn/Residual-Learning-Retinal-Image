import os
import torch
from model import ResidualDenoiser
import numpy as np
from PIL import Image
import cv2

# Define paths for testing data
test_noisy_folder    = r"D:\Hello\Residual_learning_Retina_Image\src\dataset\testing\noisy"
checkpoint_path = r"D:\Hello\Residual_learning_Retina_Image\src\checkpoints"
save_results_path = r"D:\Hello\Residual_learning_Retina_Image\src\results"

img_size = (256, 256)

# Load test dataset (only noisy images)
class NoisyImageDataset(torch.utils.data.Dataset):
    def __init__(self, noisy_folder, img_size=(256,256)):
        self.noisy_paths = sorted([
            os.path.join(noisy_folder, f)
            for f in os.listdir(noisy_folder)
            if f.lower().endswith(('.png','.jpg','.jpeg'))
        ])
        self.img_size = img_size
    def __len__(self):
        return len(self.noisy_paths)
    def __getitem__(self, idx):
        noisy_path = self.noisy_paths[idx]
        noisy = cv2.imread(noisy_path)
        noisy = cv2.resize(noisy, self.img_size)
        noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
        # Morphological mask creation (foreground extraction)
        gray = cv2.cvtColor(noisy, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = (mask > 0).astype(np.float32)  # shape (H, W)
        # Set background to pure black
        noisy = noisy * mask[..., None]
        noisy_tensor = torch.tensor(noisy.transpose(2,0,1), dtype=torch.float32) / 255.0
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        base_name = os.path.splitext(os.path.basename(noisy_path))[0]
        # Ensure correct shape
        assert noisy_tensor.shape[1] == 256 and noisy_tensor.shape[2] == 256, f"Image shape is {noisy_tensor.shape}"
        assert mask_tensor.shape[0] == 256 and mask_tensor.shape[1] == 256, f"Mask shape is {mask_tensor.shape}"
        return noisy_tensor, mask_tensor, base_name

test_dataset = NoisyImageDataset(test_noisy_folder, img_size=img_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResidualDenoiser()
model.load_state_dict(torch.load(os.path.join(checkpoint_path, "best_model.pt"), map_location=device))
model.to(device)
model.eval()

save_dir = os.path.join(save_results_path, "test_outputs")
os.makedirs(save_dir, exist_ok=True)


# Save bottleneck features for foreground only
def save_bottleneck_features(features, mask, base_name, save_dir):
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[2:] != features.shape[2:]:
        mask = torch.nn.functional.interpolate(mask.float(), size=features.shape[2:], mode='nearest')
    fg_features = (features * mask).cpu().numpy()
    np.save(os.path.join(save_dir, f"{base_name}_fg_features.npy"), fg_features)

with torch.no_grad():
    for idx, (noisy, mask, base_name) in enumerate(test_loader):
        noisy = noisy.to(device)
        mask = mask.to(device)
        # If mask is 2D, expand to 3D for broadcasting
        if mask.ndim == 3:
            mask3d = mask
        else:
            mask3d = mask.unsqueeze(0)
        # Zero out background in noisy image
        noisy_fg = noisy * mask3d
        denoised_pred = model(noisy_fg)
        # Save bottleneck feature space for foreground only
        bottleneck_features = model.get_last_feature_map(mask)
        feature_save_dir = os.path.join(save_results_path, "test_features")
        os.makedirs(feature_save_dir, exist_ok=True)
        save_bottleneck_features(bottleneck_features, mask, base_name, feature_save_dir)
        # Convert to numpy
        denoised_np = denoised_pred.squeeze().cpu().numpy()
        noisy_np = noisy.squeeze().cpu().numpy()
        mask_np = mask.cpu().numpy()  # shape (H, W)
        if mask_np.ndim == 2:
            mask_np = np.expand_dims(mask_np, axis=0)
        # Compose output: set background to black, foreground from denoised
        out_np = denoised_np * mask_np  # background will be 0
        out_np = np.clip(out_np, 0, 1)
        out_img = (out_np * 255).astype(np.uint8)
        out_img = np.transpose(out_img, (1,2,0))
        # Convert RGB to BGR for cv2
        out_img_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, f"{base_name}_denoised.png"), out_img_bgr)
print(f"✅ Denoised test outputs saved to {save_dir}")




