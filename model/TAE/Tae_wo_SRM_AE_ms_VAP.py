from re import T
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.TAE.xception_AE_ms import Xception
from model.TAE.attention import ChannelAttention, SpatialAttention, DualCrossModalAttention

from model.TAE.VAP import VAP


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#SRM attention map
class SRMPixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(SRMPixelAttention, self).__init__()
        # self.srm = SRMConv2d_simple()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.pa = SpatialAttention()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_srm):
        # x_srm = self.srm(x)
        fea = self.conv(x_srm)
        att_map = self.pa(fea)

        return att_map

# rgb features and srm features fusion -- 直接拼接后加入了 channel attention
class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan=2048 * 2, out_chan=2048, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )
        self.ca = ChannelAttention(out_chan, ratio=16)
        self.init_weight()

    def forward(self, x, y):
        fuse_fea = self.convblk(torch.cat((x, y), dim=1))
        fuse_fea = fuse_fea + fuse_fea * self.ca(fuse_fea)
        return fuse_fea

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

# 定义的双流网络，以xception为backbone ,在entry flow, middle flow, exit flow 处加入不同的模块
class Two_Stream_Net(nn.Module):
    def __init__(self, segments=8, num_class=2):
        super().__init__()
        self.num_class = num_class
        self.segments = segments
        self.xception_rgb = self.init_xcep()
        self.xception_srm = self.init_xcep()
        

        self.att_map = None
        self.srm_sa = SRMPixelAttention(3)  # 利用SRM得到的残差图送入空间注意力中获得注意力图
        self.srm_sa_post = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.sm = nn.Sigmoid()
        self.ln = nn.LayerNorm(512)
        self.ln1 = nn.LayerNorm(1024)
        self.lr = nn.LeakyReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fusion = FeatureFusionModule()
        self.linear1 = nn.Linear(512, 1024)
        self.drop = nn.Dropout(0.2)
        #=================================在这里增加多尺度时间特征融合策略====================================
        self.VAP = VAP(self.segments)
        self.fc = nn.Linear(1024, self.num_class, bias=False)
        self.att_dic = {}
    def init_xcep(self):
        self.xcep = Xception(self.num_class, self.segments).to(device) # 这里去掉to(device)

        # To get a good performance, using ImageNet-pretrained Xception model is recommended
        state_dict = get_xcep_state_dict()
        self.xcep.load_state_dict(state_dict, False)
        return self.xcep
    def features(self, x):
        # 与generalizing face forgery detection with high-frequency前期处理相同，利用SRM进行滤波与做空间注意力
        #从dataloader 中读出来的数据[bs,3*segments,h,w],送进模型之前将其维度改变成[bs*segments,3,h,w]
        x = x.view((-1, 3) + x.size()[-2:]) 
    

        x = self.xception_rgb.fea_part1_0(x)


        x = self.xception_rgb.fea_part1_1(x)


        x = self.xception_rgb.features(x) # [bs, 512, 7, 7]

        return x

   
    def forward(self, x):
        '''
        x: original rgb

        Return:
        out: (B, 2) the output for loss computing
        fea: (B, 1024) the flattened features before the last FC
        att_map: srm spatial attention map
        '''
        x = self.features(x)
        x = self.avgpool(x)# [bs*segments, 512, 1, 1]
       
        x = x.view(x.size(0), -1) # [bs*segments, 512]
        
 
        # 这里重新reshape一下输出的维度，转换成[bs, segments, 512]
        #x = x.view((-1, self.segments)+ x.size()[-1:])
    # ==========================================================================================================
        # 在这里经过段融合函数，将片段级别的特征向量融合成视频级的向量，这里采用最为简单的mean(),后续的修改在这里产生
        x = self.VAP(x) # [bs, 512]
    # ==============================================================================================================    
        # 计算出自适应的权重矩阵
        
        # 最后的融合形式 [bs, 1024]
        fusion_feature = self.linear1(x)

        # 做完双流融合之后，记得在这里加入一个dropout层和linear层进行最后的分类 classification
        feature = self.fc(self.lr(self.ln1(fusion_feature)))
        feature = self.drop(feature)
        return feature

def get_xcep_state_dict(pretrained_path='/root/autodl-tmp/cipucnet/pretrained/xception-b5690688.pth'):
    # load Xception
    state_dict = torch.load(pretrained_path)
    for name, weights in state_dict.items():
        if 'pointwise' in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
    return state_dict

if __name__ == '__main__':
    model = Two_Stream_Net(segments=8)
    model=model.to(device)
    dummy = torch.rand((6, 24, 320, 320))
    dummy = dummy.to(device)
    output = model(dummy)

    print(output.shape)
    print(output)