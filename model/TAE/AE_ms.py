from black import diff
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F


class AEModule(nn.Module):
    """ Motion exciation module
    
    :param reduction=16
    :param n_segment=8/16
    """
    def __init__(self, channel, reduction=8, n_segment=8):
        super(AEModule, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.n_segment = n_segment
        self.conv1 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel//self.reduction,
            kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel//self.reduction)

        self.conv2 = nn.Conv2d(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel//self.reduction,
            kernel_size=3,
            padding=1,
            groups=channel//self.reduction,
            bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_forward = nn.AvgPool2d(kernel_size=2, stride=2)

        self.sigmoid = nn.Sigmoid()

        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)

        self.conv3 = nn.Conv2d(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.channel)

        self.conv4 = nn.Conv2d(in_channels=self.channel//self.reduction,
                      out_channels=self.channel//self.reduction,padding=1,kernel_size=3, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=self.channel//self.reduction)

        self.conv5 = nn.Conv2d(in_channels=self.channel//self.reduction,
                      out_channels=self.channel//self.reduction,padding=1,kernel_size=3, bias=False)
        self.bn5 = nn.BatchNorm2d(num_features=self.channel//self.reduction)


    def forward(self, x):
        nt, c, h, w = x.size()
        bottleneck = self.conv1(x) # nt, c//r, h, w
        bottleneck = self.bn1(bottleneck) # nt, c//r, h, w

        # t feature
        reshape_bottleneck = bottleneck.view((-1, self.n_segment) + bottleneck.size()[1:])  # n, t, c//r, h, w
        t_fea, __ = reshape_bottleneck.split([self.n_segment-1, 1], dim=1) # n, t-1, c//r, h, w

        # apply transformation conv to t+1 feature
        conv_bottleneck = self.conv2(bottleneck)  # nt, c//r, h, w
        # reshape fea: n, t, c//r, h, w
        reshape_conv_bottleneck = conv_bottleneck.view((-1, self.n_segment) + conv_bottleneck.size()[1:])
        __, tPlusone_fea = reshape_conv_bottleneck.split([1, self.n_segment-1], dim=1)  # n, t-1, c//r, h, w
        
        # motion fea = t+1_fea - t_fea
        # pad the last timestamp
        diff_fea = tPlusone_fea - t_fea # n, t-1, c//r, h, w --> 相邻两帧所对应的特征图在特征通道上对应相减，这样就可以找出每一帧的运动敏感通道(或者说是伪影敏感通道)，并对这些通道进行加强
        # pad = (0,0,0,0,0,0,0,1)
        diff_fea_pluszero = F.pad(diff_fea, self.pad, mode="constant", value=0)  # n, t, c//r, h, w --> 由于差异增强特征图是由后一帧减去前一帧进行计算的， 那么最后一帧也就没有可减的帧，所以对应的差异特征图全部置0
        diff_fea_pluszero = diff_fea_pluszero.view((-1,) + diff_fea_pluszero.size()[2:])  #nt, c//r, h, w  将获取的差异信息拼接到一起,获取到最终的差异信息


        #=================下采样+卷积4+上采样分支==========
        diff1 = self.avg_pool_forward(diff_fea_pluszero)
        diff1 = self.bn4(self.conv4(diff1))
        diff1 = F.interpolate(diff1, diff_fea_pluszero.size()[2:])
        #================卷积分支=========================
        diff2 = self.bn5(self.conv5(diff_fea_pluszero))
        #================原分支===========================
        diff3 = diff_fea_pluszero
        #================进行多尺度融合===================
        diff_fusion = 1.0/3.0*diff1 + 1.0/3.0*diff2 + 1.0/3.0*diff3
        atten_map = self.avg_pool(diff_fusion) # [nt, c, 1, 1]
        atten_map = self.bn3(self.conv3(atten_map)) # [nt, c, 1, 1]
        #================获取增强通道注意力图=============
        atten_map = self.sigmoid(atten_map) # [nt, c, 1, 1]
        atten_map = atten_map - 0.5 # [nt, c, 1, 1]
    
        output = x + x * atten_map.expand_as(x)
        return output


if __name__ == '__main__':
    x = torch.rand(48, 728, 14, 14)
    channel = x.shape[1]
    AE = AEModule(728)
    
    out = AE(x)
    print(out.shape)
