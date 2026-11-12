
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True)
        )

        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.conv = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        return self.conv(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.conv = ConvBlock(
            in_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class HybridISPNet(nn.Module):
    def __init__(self, base_channels=32, num_tasks=2):
        super().__init__()

        c = base_channels

        self.enc1 = ConvBlock(3, c)
        self.enc2 = DownBlock(c, c * 2)
        self.enc3 = DownBlock(c * 2, c * 4)

        self.bottleneck = DownBlock(c * 4, c * 8)

        self.task_embedding = nn.Embedding(num_tasks, 64)

        self.film = nn.Sequential(
            nn.Linear(64, c * 16),
            nn.SiLU(inplace=True),
            nn.Linear(c * 16, c * 16)
        )

        self.up3 = UpBlock(c * 8, c * 4, c * 4)
        self.up2 = UpBlock(c * 4, c * 2, c * 2)
        self.up1 = UpBlock(c * 2, c, c)

        self.output = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, x, task_id):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        embedding = self.task_embedding(task_id)
        scale, bias = self.film(embedding).chunk(2, dim=1)

        scale = scale.unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)

        b = b * (1 + scale) + bias

        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        residual = self.output(d1)

        return torch.clamp(x + residual, 0, 1)
