"""
=============================================================================
Basic 1D ResNet Baseline for HAR
=============================================================================

A standard ResNet-18-style architecture adapted for 1D time-series (IMU data).
This serves as a clean baseline CNN to compare against:
  - ssl-wearables (same ResNet backbone but with SSL pretraining on UK Biobank)
  - HART (Lightweight Transformer)
  - LIMU-BERT (BERT-style Transformer)

Architecture (ResNet-18 style, 1D):
  Input (N, C, T)
    -> Conv1d(C, 64, 7, stride=2) -> BN -> ReLU -> MaxPool
    -> Layer1: 2 x BasicBlock(64,  64)
    -> Layer2: 2 x BasicBlock(64,  128, stride=2)
    -> Layer3: 2 x BasicBlock(128, 256, stride=2)
    -> Layer4: 2 x BasicBlock(256, 512, stride=2)
    -> AdaptiveAvgPool1d(1)
    -> Linear(512, num_classes)

No circular padding, no anti-aliased downsampling, no SSL pretraining.
Just a clean, standard ResNet for fair baseline comparison.
"""

import torch
import torch.nn as nn


class BasicBlock1D(nn.Module):
    """Standard ResNet basic block for 1D signals.

    Structure:  x -> Conv -> BN -> ReLU -> Conv -> BN -> (+shortcut) -> ReLU
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection (identity or projection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity

        return self.relu(out)


class ResNet1DBaseline(nn.Module):
    """Basic 1D ResNet for HAR -- ResNet-18 style, adapted for time series.

    Args:
        n_channels: Number of input channels (3=acc, 6=acc+gyro).
        num_classes: Number of output classes.
        kernel_size: Kernel size for residual blocks (default 5).
    """

    def __init__(self, n_channels=6, num_classes=5, kernel_size=5):
        super().__init__()

        # Stem: initial convolution + pooling
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, stride=2, padding=3,
                      bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual layers
        self.layer1 = self._make_layer(64,  64,  num_blocks=2,
                                       kernel_size=kernel_size, stride=1)
        self.layer2 = self._make_layer(64,  128, num_blocks=2,
                                       kernel_size=kernel_size, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2,
                                       kernel_size=kernel_size, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2,
                                       kernel_size=kernel_size, stride=2)

        # Global average pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

        # Kaiming initialization (standard for ResNets)
        self._init_weights()

    @staticmethod
    def _make_layer(in_channels, out_channels, num_blocks, kernel_size,
                    stride=1):
        layers = [BasicBlock1D(in_channels, out_channels,
                               kernel_size=kernel_size, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1D(out_channels, out_channels,
                                       kernel_size=kernel_size, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (N, C, T) tensor -- e.g. (batch, 6, 300)
        Returns:
            logits: (N, num_classes)
        """
        x = self.stem(x)       # (N, 64,  75)
        x = self.layer1(x)     # (N, 64,  75)
        x = self.layer2(x)     # (N, 128, 38)
        x = self.layer3(x)     # (N, 256, 19)
        x = self.layer4(x)     # (N, 512, 10)
        x = self.avgpool(x)    # (N, 512,  1)
        x = x.flatten(1)       # (N, 512)
        x = self.fc(x)         # (N, num_classes)
        return x


# ---- Quick parameter count ----
if __name__ == '__main__':
    model = ResNet1DBaseline(n_channels=6, num_classes=5)
    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResNet1DBaseline(6ch, 5cls): {n_params:,} params ({n_train:,} trainable)")

    # Quick forward pass test
    x = torch.randn(4, 6, 300)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
