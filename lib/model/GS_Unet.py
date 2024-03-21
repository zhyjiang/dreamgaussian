import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTModel, ViTConfig




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,padding=1):
        super(ResidualBlock, self).__init__()

        # Main block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        
        # import ipdb;ipdb.set_trace()
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out
    
    
    
# class ResUnet(nn.Module):
#     def __init__(self, channel, filters=[64, 128, 256, 512]):
#         super(ResUnet, self).__init__()

#         self.input_layer = nn.Sequential(
#             nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
#             nn.BatchNorm2d(filters[0]),
#             nn.ReLU(),
#             nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
#         )
#         self.input_skip = nn.Sequential(
#             nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
#         )

#         self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
#         self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

#         self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

#         self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
#         self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

#         self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
#         self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

#         self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
#         self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

#         self.output_layer = nn.Sequential(
#             nn.Conv2d(filters[0], 1, 1, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         # Encode
#         x1 = self.input_layer(x) + self.input_skip(x)
#         x2 = self.residual_conv_1(x1)
#         x3 = self.residual_conv_2(x2)
#         # Bridge
#         x4 = self.bridge(x3)
#         # Decode
#         x4 = self.upsample_1(x4)
#         x5 = torch.cat([x4, x3], dim=1)

#         x6 = self.up_residual_conv1(x5)

#         x6 = self.upsample_2(x6)
#         x7 = torch.cat([x6, x2], dim=1)

#         x8 = self.up_residual_conv2(x7)

#         x8 = self.upsample_3(x8)
#         x9 = torch.cat([x8, x1], dim=1)

#         x10 = self.up_residual_conv3(x9)

#         output = self.output_layer(x10)

#         return output

class GS_Unet(nn.Module):
    def __init__(self, in_channels, out_channels, dim=128,img_dim=384,device='cuda'):
        super(GS_Unet, self).__init__()

        # Encoder (contracting path)
        def __init__(self, in_channels, out_channels):
            super(UNetResidual, self).__init__()
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = dim
        hidden_dim = self.hidden_dim
        self.img_dim = img_dim
        self.device = device
        
        
        
        self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
        config=ViTConfig.from_pretrained('facebook/dino-vits16')
        self.dino_encoder = ViTModel.from_pretrained('facebook/dino-vits16',config=config,ignore_mismatched_sizes=True).to(self.device)



        # self.input_layer = nn.Sequential(
        #     nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64,64, kernel_size=3, padding=1),
        # )
        # self.input_skip = nn.Sequential(
        #     nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        # )
        # Encoder
        self.enc0 = ResidualBlock(in_channels, 32,1,1)
        self.enc1 = ResidualBlock(32, 64,2,1)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ResidualBlock(64, 128,2,1)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Middle
        self.middle = ResidualBlock(128, 256,2,1)

        # Decoder
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(128, 64)
        
        # self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # self.dec2 = ResidualBlock(64, 32)
        self.upconv0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = ResidualBlock( 64,32)

        # Output
        self.outconv = nn.Conv2d(32, out_channels,kernel_size=1)
        
        self.mean3D_head = self._make_head(hidden_dim, 3)
        self.opacity_head = self._make_head(hidden_dim, 1)
        self.shs_head = self._make_head(hidden_dim, 3)
        self.rotations_head = self._make_head(hidden_dim, 4)
        self.scales_head = self._make_head(hidden_dim, 3)
        
        # self.gs_norm = nn.LayerNorm(self.img_dim)
        self.gs_encoder = nn.Linear(3, self.hidden_dim)



    def _make_head(self, hidden_dim, out_dim):
        layers = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                               nn.GELU(),
                               nn.Dropout(0.1),
                               nn.Linear(hidden_dim, out_dim))
        return layers    

    def forward(self, x):
        
        
        # x = x.permute(0, 2, 3, 1)
        
        # with torch.no_grad():
        #     self.dino_encoder.eval()
        #     inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        #     img_emb = self.dino_encoder(**inputs)
        #     img_emb = img_emb['last_hidden_state'][...,1:,:]
        # img_emb = self.gs_norm(img_emb)    
        # x = self.gs_encoder(x)
        
        
        # x = x.permute(0, 3, 1, 2)
        
        # x1self.input_layer(x) + self.input_skip(x)
        enc0 = self.enc0(x)
        # Encoder
        enc1 = self.enc1(enc0)
        # pool1 = self.pool1(enc1)
        enc2 = self.enc2(enc1)
        # pool2 = self.pool2(enc2)

        # Middle
        middle = self.middle(enc2)

        # Decoder
        upconv2 = self.upconv2(middle)
        # import ipdb;ipdb.set_trace()
        cat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = self.dec2(cat2)
        upconv1 = self.upconv1(dec2)
        cat1 = torch.cat([upconv1, enc1], dim=1)
        dec1 = self.dec1(cat1)
        upconv0 = self.upconv0(dec1)
        cat0 = torch.cat([upconv0, enc0], dim=1)
        # import ipdb;ipdb.set_trace()
        dec0 = self.dec0(cat0)

        # Output
        output = self.outconv(dec0)
        x = output
        
        x = x.permute(0, 2,3,1)
        
        x = self.gs_encoder(x)
        
        # import ipdb;ipdb.set_trace()
# 
        
        means3D = self.mean3D_head(x)
        opacity = self.opacity_head(x).sigmoid()
        shs = self.shs_head(x).sigmoid()
        # rotations = torch.nn.functional.normalize(self.rotations_head(x))
        rotations = F.normalize(self.rotations_head(x))
        scales = torch.exp(self.scales_head(x))
        
        
        
        
        

        return output, means3D, opacity, scales,shs, rotations