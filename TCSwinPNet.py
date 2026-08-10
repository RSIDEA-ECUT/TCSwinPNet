import torch
import torch.nn as nn
import torch.nn.functional as F
from KANLinear import KANLinear
from FastKANConv2D import FastKANConvLayer
import torch.nn.init as init
from Modify_Swin_Transformer import BasicLayer
from torchstat import stat
from thop import profile, clever_format


class DepthWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthWiseConv, self).__init__()

        self.depth_conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=in_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=padding,
                                    groups=in_channels)

        self.point_conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


# Spatial Attention Module
class Spa_Att(nn.Module):
    def __init__(self):
        super(Spa_Att, self).__init__()
        self.conv = FastKANConvLayer(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # (B, C, H, W)

        ave_x = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_x, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)

        x_ = torch.cat([ave_x, max_x], dim=1)  # (B, 2, H, W)
        x_ = self.conv(x_)  # (B, 1, H, W)

        weights = self.sigmoid(x_)  # (B, 1, H, W): [0,1]
        x = x * weights  # (B, C, H, W)
        return x


# Channel Attention Module
class Cha_Att(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(Cha_Att, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.linear1 = KANLinear(in_features=in_channels, out_features=in_channels // ratio)
        self.linear2 = KANLinear(in_features=in_channels // ratio, out_features=in_channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        x_ave = self.avg_pool(x)  # (B, C, 1, 1)
        x_max = self.max_pool(x)  # (B, C, 1, 1)

        x_ave = self.linear1(x_ave.view(B, -1))  # (B, C)-->(B, C//ratio)
        x_ave = self.linear2(x_ave.view(B, -1))  # (B, C//ratio)-->(B, C)

        x_max = self.linear1(x_max.view(B, -1))  # (B, C)-->(B, C//ratio)
        x_max = self.linear2(x_max.view(B, -1))  # (B, C//ratio)-->(B, C)

        x_ = x_ave + x_max  # (B, C)
        weights = self.sigmoid(x_)  # (B, C): [0,1]
        weights = weights.view(B, C, 1, 1)  # (B, C, 1, 1)
        x = x * weights  # (B, C, H, W)
        return x


class PAN_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PAN_block, self).__init__()
        self.conv = DepthWiseConv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.block = nn.Sequential(
            Spa_Att(),
            DepthWiseConv(in_channels=out_channels, out_channels=out_channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),

            DepthWiseConv(in_channels=out_channels, out_channels=out_channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),

            DepthWiseConv(in_channels=out_channels, out_channels=out_channels, kernel_size=7, padding=3),
        )

    def forward(self, x):
        x_ = F.relu(self.conv(x))
        return F.relu(self.block(x_) + x_)


class MS_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MS_block, self).__init__()
        self.conv = DepthWiseConv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.block = nn.Sequential(
            Cha_Att(out_channels),
            DepthWiseConv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            DepthWiseConv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            DepthWiseConv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),

        )

    def forward(self, x):
        x_ = F.relu(self.conv(x))
        return F.relu(self.block(x_) + x_)


# general swin-transformer
class PAN_MS_block(nn.Module):
    def __init__(self):
        super(PAN_MS_block, self).__init__()
        self.basic_layer = BasicLayer()

    def forward(self, x):
        x = self.basic_layer(x)
        return x


class Fused_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fused_block, self).__init__()
        self.block = nn.Sequential(
            DepthWiseConv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),

            DepthWiseConv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            DepthWiseConv(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TCSwinPNet(nn.Module):
    def __init__(self, spectral_nums, in_channels=32, out_channels=32):
        super(TCSwinPNet, self).__init__()
        self.panconv = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=3, padding=1)
        self.msconv = nn.Conv2d(in_channels=spectral_nums, out_channels=out_channels, kernel_size=3, padding=1)
        self.pmconv = nn.Conv2d(in_channels=spectral_nums + 1, out_channels=out_channels, kernel_size=3, padding=1)
        self.fusedconv = nn.Conv2d(in_channels=out_channels, out_channels=spectral_nums, kernel_size=3, padding=1)

        self.PB1 = PAN_block(in_channels, out_channels)
        self.PB2 = PAN_block(2 * in_channels, out_channels)
        self.PB3 = PAN_block(2 * in_channels, out_channels)
        self.PB4 = PAN_block(2 * in_channels, out_channels)

        self.MB1 = MS_block(in_channels, out_channels)
        self.MB2 = MS_block(2 * in_channels, out_channels)
        self.MB3 = MS_block(2 * in_channels, out_channels)
        self.MB4 = MS_block(2 * in_channels, out_channels)

        self.PMB1 = PAN_MS_block()
        self.PMB2 = PAN_MS_block()
        self.PMB3 = PAN_MS_block()
        self.PMB4 = PAN_MS_block()

        self.FB1 = Fused_block(3 * out_channels, out_channels)
        self.FB2 = Fused_block(in_channels, out_channels)
        self.FB3 = Fused_block(in_channels, out_channels)
        self.FB4 = Fused_block(in_channels, out_channels)

    def forward(self, ms, pan):
        ms_ = F.relu(self.msconv(ms))  # (B, 4 or 8, 64, 64)--> (B, 32, 64, 64)
        pan_ = F.relu(self.panconv(pan))  # (B, 1, 64, 64)--> (B, 32, 64, 64)
        pm = torch.cat([ms, pan], dim=1)  # (B, 5, 64, 64)
        pm_ = F.relu(self.pmconv(pm))  # (B, 5, 64, 64)--> (B, 32, 64, 64)

        pm1 = self.PMB1(pm_)
        ms1 = self.MB1(ms_)
        pan1 = self.PB1(pan_)

        pm2 = self.PMB2(pm1)
        ms2 = self.MB2(torch.cat([ms1, pm1], dim=1))
        pan2 = self.PB2(torch.cat([pan1, pm1], dim=1))

        pm3 = self.PMB3(pm2)
        ms3 = self.MB3(torch.cat([ms2, pm2], dim=1))
        pan3 = self.PB3(torch.cat([pan2, pm2], dim=1))

        pm4 = self.PMB4(pm3)
        ms4 = self.MB4(torch.cat([ms3, pm3], dim=1))
        pan4 = self.PB4(torch.cat([pan3, pm3], dim=1))

        cat_features = torch.cat([pan4, pm4, ms4], dim=1)
        x1 = self.FB1(cat_features)
        x2 = self.FB2(x1)
        x3 = self.FB3(x2)
        x4 = self.FB4(x3 + x2)

        fused_img = self.fusedconv(x4 + x1)  # (B, 4 or 8, 64, 64)
        return fused_img + ms


if __name__ == '__main__':
    model = TCSwinPNet(spectral_nums=4)
    print("===> Parameter numbers : %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    ms = torch.rand(1, 4, 256, 256)
    pan = torch.rand(1, 1, 256, 256)
    out = model(ms, pan)
    print(out.shape)
    flops, params = profile(model, inputs=(ms, pan))
    flops, params = clever_format([flops, params], "%.2f")
    print(flops, params)
