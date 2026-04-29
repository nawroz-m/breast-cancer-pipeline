import torch
from torch import nn
import albumentations as A


# ----------------Model Classes-----------------------

# Double Convolution Block (with GroupNorm)
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# U-Net Architecture (Optimized) 
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder (lighter)
        self.down1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            DoubleConv(256, 512),
            nn.Dropout(0.3)
        )

        # Decoder (bilinear upsampling)
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv4 = DoubleConv(512 + 256, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv3 = DoubleConv(256 + 128, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv2 = DoubleConv(128 + 64, 64)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = DoubleConv(64 + 32, 32)

        # Output layer (NO sigmoid here)
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.down1(x)
        p1 = self.pool1(c1)

        c2 = self.down2(p1)
        p2 = self.pool2(c2)

        c3 = self.down3(p2)
        p3 = self.pool3(c3)

        c4 = self.down4(p3)
        p4 = self.pool4(c4)

        # Bottleneck
        bn = self.bottleneck(p4)

        # Decoder
        u4 = self.up4(bn)
        u4 = torch.cat([u4, c4], dim=1)
        c5 = self.conv4(u4)

        u3 = self.up3(c5)
        u3 = torch.cat([u3, c3], dim=1)
        c6 = self.conv3(u3)

        u2 = self.up2(c6)
        u2 = torch.cat([u2, c2], dim=1)
        c7 = self.conv2(u2)

        u1 = self.up1(c7)
        u1 = torch.cat([u1, c1], dim=1)
        c8 = self.conv1(u1)

        # Return raw logits (use BCEWithLogitsLoss)
        return self.out(c8)



class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c)
        )

        if in_c != out_c or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


# Builing the network
class Cnn_network(nn.Module):
  def __init__(self):
    super(Cnn_network, self).__init__()

    # Encoder
    self.block1 = nn.Sequential(
        ResidualBlock(3, 32, stride=2),
        ResidualBlock(32, 32, stride=1)
    )

    self.block2 = nn.Sequential(
        ResidualBlock(32, 64, stride=2),
        ResidualBlock(64, 64, stride=1)
    )

    self.block3 = nn.Sequential(
        ResidualBlock(64, 128, stride=2),
        ResidualBlock(128, 128, stride=1)
    )

    self.block4 = nn.Sequential(
        ResidualBlock(128, 256, stride=2),
        ResidualBlock(256, 256, stride=1)
    )

    # self.dropout = nn.Dropout2d(0.1)
    self.fuse_head = nn.Sequential(
        nn.Conv2d(256 + 128, 256, 1),
        nn.ReLU()
    )

    self.head = nn.Sequential(
        nn.Conv2d(256, 256, 3, padding=1),
        nn.ReLU(),

        nn.Conv2d(256, 128, 3, padding=1),
        nn.ReLU(),

        nn.Conv2d(128, 64, 3, padding=1),
        nn.ReLU(),

        nn.Conv2d(64, 5, 1), # objectness = probability that a box actually contains an object

    )
    self.attention_head = nn.Conv2d(256, 5, 1)

  def forward(self, x):

    # Encoder
    x1 = self.block1(x)
    x2 = self.block2(x1)
    x3 = self.block3(x2)
    x4 = self.block4(x3) # <-- Bottleneck
    # x4 = self.dropout(x4) # <-- reduce noise at bottleneck

    # -------------------------
    # 🔥 FEATURE FUSION
    # -------------------------

    x4_up = nn.functional.interpolate(x4, size=x3.shape[2:])

    # combine deep + mid features
    fused = torch.cat([x3, x4_up], dim=1)

    fused = self.fuse_head(fused)

    # -------------------------
    # HEAD (bbox prediction)
    # -------------------------

    # output objects
    features = self.head(fused)
    attention_logits = self.attention_head(fused)  # [B, 5, H, W]

    B, C, H, W = attention_logits.shape

    # flatten spatial dims
    attention = attention_logits.view(B, C, -1)
    # apply softmax over spatial locations
    attention = torch.softmax(attention, dim=2)
    # reshape back
    attention = attention.view(B, C, H, W)
      
    weighted = features * attention # Let network learn where to look
    out = weighted.sum(dim=(2,3))   
    return out

class Segment_ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c)
        )

        # match dimensions for skip connection
        if in_c != out_c or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.skip(x)      # original path

        out = self.conv(x)           # transformed path

        out = out + residual
        out = self.relu(out)  #  RESIDUAL ADDED

        return out

# Builing the network
class Segment_Cnn_network(nn.Module):
  def __init__(self, num_class):
    super(Segment_Cnn_network, self).__init__()

    # Encoder
    self.block1 = nn.Sequential(
        Segment_ResidualBlock(3, 32, stride=2),
        Segment_ResidualBlock(32, 32, stride=1)
    )

    self.block2 = nn.Sequential(
        Segment_ResidualBlock(32, 64, stride=2),
        Segment_ResidualBlock(64, 64, stride=1)
    )

    self.block3 = nn.Sequential(
        Segment_ResidualBlock(64, 128, stride=2),
        Segment_ResidualBlock(128, 128, stride=1)
    )

    self.block4 = nn.Sequential(
        Segment_ResidualBlock(128, 256, stride=2),
        Segment_ResidualBlock(256, 256, stride=1)
    )

    self.dropout = nn.Dropout2d(0.3)

    # Decoder
    self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
    self.up4 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)

    self.conv1 = Segment_ResidualBlock(256, 128, stride=1)
    self.conv2 = Segment_ResidualBlock(128, 64, stride=1)
    self.conv3 = Segment_ResidualBlock(64, 32, stride=1)
    self.conv4 = Segment_ResidualBlock(32, 32, stride=1)


    self.final = nn.Conv2d(32, 1, kernel_size=1)


  def forward(self, x):

    # Encoder
    x1 = self.block1(x)
    x2 = self.block2(x1)
    x3 = self.block3(x2)
    x4 = self.block4(x3) # <-- Bottleneck
    x4 = self.dropout(x4) # <-- reduce noise at bottleneck

    # Decoder
    x = self.up1(x4)
    x = nn.functional.interpolate(x, size=x3.shape[2:])
    x = torch.cat([x, x3], dim=1)
    x = self.conv1(x)

    x = self.up2(x)
    x = nn.functional.interpolate(x, size=x2.shape[2:])
    x = torch.cat([x, x2], dim=1)
    x = self.conv2(x)

    x = self.up3(x)
    x = nn.functional.interpolate(x, size=x1.shape[2:])
    x = torch.cat([x, x1], dim=1)
    x = self.conv3(x)

    x = self.up4(x)
    x = self.conv4(x)

    x = self.final(x)
    return x

# ══════════════════════════════════════════════════════════════════════
#  STEP 2: MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════

class TumorClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)

