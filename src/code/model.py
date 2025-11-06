import torch
import torch.nn as nn

class ResidualDenoiser(nn.Module):
    """
    Encoder-Decoder architecture for image denoising.
    Input: (batch_size, 3, H, W) - noisy image
    Output: (batch_size, 3, H, W) - denoised image (should match original)
    """
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256]):
        super(ResidualDenoiser, self).__init__()

        # Encoder
        self.encoder1 = self._block(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features[0], features[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self._block(features[1], features[2])

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder1 = self._block(features[1]*2, features[1])
        self.upconv2 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder2 = self._block(features[0]*2, features[0])

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.output_activation = nn.Sigmoid()

        # For feature extraction (foreground only)
        self._last_feature_map = None

    def get_last_feature_map(self, mask=None):
        """
        Returns the last bottleneck feature map (optionally masked for foreground only).
        mask: (batch, 1, 256, 256) or (batch, 256, 256) foreground mask, 1=fg, 0=bg
        """
        if self._last_feature_map is None:
            return None
        if mask is not None:
            # Resize mask to match feature map spatial size if needed
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            if mask.shape[2:] != self._last_feature_map.shape[2:]:
                mask = torch.nn.functional.interpolate(mask.float(), size=self._last_feature_map.shape[2:], mode='nearest')
            return self._last_feature_map * mask
        return self._last_feature_map

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        # x: noisy image, shape (batch, 3, 256, 256)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        self._last_feature_map = bottleneck.detach()  # Save for feature extraction
        dec1 = self.upconv1(bottleneck)
        dec1 = torch.cat((dec1, enc2), dim=1)
        dec1 = self.decoder1(dec1)
        dec2 = self.upconv2(dec1)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.decoder2(dec2)
        out = self.final_conv(dec2)
        out = self.output_activation(out)
        return out  # denoised image in [0,1]