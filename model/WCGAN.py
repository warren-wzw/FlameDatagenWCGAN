import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


class Generator_Conv_linear(nn.Module):
    def __init__(self, generator_layer_size, z_dim,img_channels, img_size, class_num):
        super().__init__()

        self.z_size = z_dim
        self.img_size = img_size
        self.img_channels = img_channels  # 生成 3 通道图像

        self.label_emb = nn.Embedding(class_num, class_num)

        self.model = nn.Sequential(
            nn.Linear(self.z_size + class_num, generator_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[0], generator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[1], generator_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[2], 256 * 16 * 16),  # 生成 256 通道的 16x16 特征图
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, self.img_channels, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z, labels):
        z = z.view(-1, self.z_size)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        out = out.view(-1, 256, 16, 16)  # Reshape to feature map
        out = self.upsample(out)
        out = self.conv1(out)
        out = self.upsample(out)
        out = self.conv2(out)
        out = self.upsample(out)
        out = self.conv3(out)

        return self.tanh(out)

class Generator_Bi(nn.Module):
    def __init__(self, generator_layer_size, z_dim, img_channels, img_size, class_num):
        super(Generator_Bi, self).__init__()
        self.z_size = z_dim
        self.img_channels = img_channels
        self.img_size = img_size
        
        self.label_emb = nn.Embedding(class_num, class_num)
        
        self.model = nn.Sequential(
            nn.Linear(self.z_size + class_num, generator_layer_size[0]),
            nn.BatchNorm1d(generator_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[0], generator_layer_size[1]),
            nn.BatchNorm1d(generator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[1], generator_layer_size[2]),
            nn.BatchNorm1d(generator_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[2], generator_layer_size[3]),
            nn.BatchNorm1d(generator_layer_size[3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[3], self.img_channels * self.img_size * self.img_size),
            nn.Tanh()
        )
    
    def forward(self, z, labels):     
        z = z.view(-1, self.z_size)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(-1, self.img_channels, self.img_size, self.img_size)  # 返回形状为 (batch_size, 3, 128, 128) 的张量

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
    
