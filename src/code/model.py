import torch
import torch.nn as nn

class ResidualDenoiser(nn.Module):
    """
    Encoder-Decoder architecture for residual image prediction.
    Input: (batch_size, 3, H, W) - original image
    Output: (batch_size, 3, H, W) - residual image (noisy - original)
    """
    def __init__(self, in_channels=3, out_channels=3, features=[32, 64, 128]):
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

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool2(enc2))

        # Decoder
        dec1 = self.upconv1(bottleneck)
        dec1 = torch.cat((dec1, enc2), dim=1)
        dec1 = self.decoder1(dec1)
        dec2 = self.upconv2(dec1)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.decoder2(dec2)

        # Output
        out = self.final_conv(dec2)
        return out
