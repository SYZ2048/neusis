import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet34


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.resnet = resnet34(True)
        # 修改第一层卷积层以适应单通道输入
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.bn1 = nn.BatchNorm2d(64)
        self.resnet.relu = nn.ReLU(inplace=True)
        self.resnet.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 降维层，将通道数降到例如128
        self.reduce_dim = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, x):   # x shape: (B,C,H,W) --> C=3 or 1, latents(B,512,H/2,W/2)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        feats1 = self.resnet.relu(x)    # (B, 64, H/2, W/2)

        feats2 = self.resnet.layer1(self.resnet.maxpool(feats1))    # (B, 64, H/4, W/4)
        feats3 = self.resnet.layer2(feats2)     # (B, 128, H/8, W/8)
        feats4 = self.resnet.layer3(feats3)     # (B, 256, H/16, W/16)

        latents = [feats1, feats2, feats3, feats4]
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i], latent_sz, mode="bilinear", align_corners=True
            )   # 插值

        latents = torch.cat(latents, dim=1) # (B, 512, H/2, W/2)

        latents = self.reduce_dim(latents)  # (B, 128, H/2, W/2)

        return latents