import os
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, device=None, batch_size=32, lr=1e-3, epochs=50, save_dir="results"):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)
        self.criterion = torch.nn.MSELoss()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def accuracy_fn(self, pred, target):
        mse = ((pred - target) ** 2).mean().item()
        return 1 - mse

    def save_curves(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'))
        plt.close()

        plt.figure(figsize=(10,5))
        plt.plot(self.train_accs, label='Train Accuracy')
        plt.plot(self.val_accs, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'accuracy_curve.png'))
        plt.close()

    def extract_and_save_all_feature_maps(self, x_batch, epoch):
        """
        Extracts and saves intermediate feature maps from all layers for a given batch.
        Saves each layer's output as .npy and as RGB images in a folder inside save_dir.
        """
        import inspect
        from PIL import Image
        epoch_dir = os.path.join(self.save_dir, f'feature_maps_epoch_{epoch+1}')
        os.makedirs(epoch_dir, exist_ok=True)
        # Get all modules (layers) in the model
        for name, module in self.model.named_modules():
            if name == "" or isinstance(module, torch.nn.Sequential):
                continue  # skip top-level and Sequential containers
            try:
                with torch.no_grad():
                    output = module(x_batch)
                    # Save as RGB images (for first 3 channels only)
                    fmap = output.cpu().detach().numpy()
                    # fmap shape: (batch, channels, H, W)
                    batch_size, channels, H, W = fmap.shape
                    for b in range(min(1, batch_size)):
                        for c in range(min(3, channels)):
                            img = fmap[b, c]
                            img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # normalize to [0,1]
                            img = (img * 255).astype(np.uint8)
                            img_rgb = np.stack([img]*3, axis=-1)  # grayscale to RGB
                            img_pil = Image.fromarray(img_rgb)
                            img_pil.save(os.path.join(epoch_dir, f'{name}_b{b}_c{c}.png'))
            except Exception:
                continue  # skip layers that can't be called directly

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss, epoch_acc = 0, 0
            for batch_idx, (Xb, yb) in enumerate(self.train_loader):
                Xb = Xb.to(self.device)
                yb = yb.to(self.device)
                self.optimizer.zero_grad()
                if batch_idx == 0:
                    self.extract_and_save_all_feature_maps(Xb, epoch)
                out = self.model(Xb)
                loss = self.criterion(out, yb - Xb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * Xb.size(0)
                epoch_acc += self.accuracy_fn(out, yb - Xb) * Xb.size(0)
            epoch_loss /= len(self.train_loader.dataset)
            epoch_acc /= len(self.train_loader.dataset)
            self.train_losses.append(epoch_loss)
            self.train_accs.append(epoch_acc)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_loss, val_acc = 0, 0
                for Xv, yv in self.val_loader:
                    Xv = Xv.to(self.device)
                    yv = yv.to(self.device)
                    outv = self.model(Xv)
                    val_loss += self.criterion(outv, yv - Xv).item() * Xv.size(0)
                    val_acc += self.accuracy_fn(outv, yv - Xv) * Xv.size(0)
                val_loss /= len(self.val_loader.dataset)
                val_acc /= len(self.val_loader.dataset)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
            self.scheduler.step()
            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Train Acc: {epoch_acc:.4f} - Val Acc: {val_acc:.4f}")


        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "residual_denoiser_final.pt"))
        self.save_curves()
        print(f"✅ Model and results saved to {self.save_dir}")
