import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNImageEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=128, out_dim=64):
        super(CNNImageEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # [B, 32, H, W]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, H/2, W/2]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 64, H/2, W/2]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 64, H/4, W/4]
        )

        self.fc = nn.Linear(64 * 7 * 7, out_dim)  # 假设输入是28×28

    def forward(self, x):
        x = self.conv(x)  # B×64×7×7
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # [B, out_dim]
