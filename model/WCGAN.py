import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class Generator_Bi(nn.Module):
    def __init__(self, z_dim, c_dim, dim=128):
        super(Generator_Bi, self).__init__()
        self.label_embedding = nn.Embedding(c_dim, c_dim)

        self.initial_layer = nn.Sequential(
            nn.Conv2d(z_dim + c_dim, dim, 3, 1, 1),  # (N, dim, 1, 1)
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # (N, dim, 2, 2)
            nn.Conv2d(dim, dim, 3, 1, 1),  # (N, dim, 2, 2)
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # (N, dim, 4, 4)
            nn.Conv2d(dim, dim, 3, 1, 1),  # (N, dim, 4, 4)
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # (N, dim, 8, 8)
            nn.Conv2d(dim, dim, 3, 1, 1),  # (N, dim, 8, 8)
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # (N, dim, 16, 16)
            nn.Conv2d(dim, dim, 3, 1, 1),  # (N, dim, 16, 16)
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # (N, dim, 32, 32)
            nn.Conv2d(dim, dim, 3, 1, 1),  # (N, dim, 32, 32)
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # (N, dim, 64, 64)
            nn.Conv2d(dim, dim, 3, 1, 1),  # (N, dim, 64, 64)
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # (N, dim, 128, 128)
            nn.Conv2d(dim, 3, 3, 1, 1),  # (N, 3, 128, 128)
            nn.Tanh()
        )

    def forward(self, z, c):
        # z: (N, z_dim), c: (N, c_dim)
        c = self.label_embedding(c.long())  # (N, c_dim) -> (N, c_dim, 1, 1)
        x = torch.cat([z, c], 1)  # 将潜在向量z和标签c拼接 -> (N, z_dim + c_dim, 1, 1)
        x = x.view(x.size(0), x.size(1), 1, 1)  # 重新调整张量形状
        x = self.initial_layer(x)  # (N, dim, 1, 1)
        x = self.upsample(x)  # 进行双线性插值和卷积操作 -> (N, 3, 128, 128)
        return x

class Generator_TConv(nn.Module):
    def __init__(self, z_dim, c_dim, dim=128):
        super(Generator_TConv, self).__init__()
        self.label_embedding = nn.Embedding(dim, dim)
        def dconv_bn_relu(in_dim, out_dim, kernel_size=4, stride=2, padding=1, output_padding=0):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, output_padding),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )
        self.ls = nn.Sequential(
            dconv_bn_relu(z_dim + c_dim, dim * 16, 4, 1, 0, 0),  # (N, dim * 4, 4, 4)
            dconv_bn_relu(dim * 16, dim * 8),  # (N, dim * 2, 8, 8)
            dconv_bn_relu(dim * 8, dim*4),   # (N, dim, 16, 16)
            dconv_bn_relu(dim * 4, dim*2),   # (N, dim, 32, 32)
            dconv_bn_relu(dim * 2, dim),   # (N, dim, 64, 64)
            nn.ConvTranspose2d(dim, 3, 4, 2, padding=1), # (N, 3, 128, 128)
            nn.Tanh()  
        )

    def forward(self, z, c):
        # z: (N, z_dim), c: (N, c_dim)
        c=self.label_embedding(c.long())
        x = torch.cat([z, c], 1)
        x = self.ls(x.view(x.size(0), x.size(1), 1, 1))
        return x
  
class Discriminator(nn.Module):
    def __init__(self, label_dim, img_dim):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(label_dim, label_dim)
        img_flat_dim = int(np.prod(img_dim))
        self.model = nn.Sequential(
            nn.Linear(img_flat_dim + label_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.validity_layer = nn.Linear(64, 1)
        self.class_layer = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, label_dim),
        )

    def forward(self, img, labels):
        c = self.label_embedding(labels.long())
        x = torch.cat([img.view(img.size(0), -1), c], 1)
        x = x + 0.05 * torch.randn_like(x)  # 添加噪声
        features = self.model(x)
        validity = self.validity_layer(features)
        class_pred = self.class_layer(features)
        return validity, class_pred
    
