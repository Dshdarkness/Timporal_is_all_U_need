from re import X
import torch
import torch.nn as nn
import math

# 一种多帧向量的融合策略
class VAP(nn.Module):
    def __init__(self, n_segment):
        super(VAP, self).__init__()
        VAP_level = int(math.log(n_segment, 2))
        self.n_segment = n_segment
        self.VAP_level = VAP_level
        total_timescale = 0
        for i in range(VAP_level):
           timescale = 2**i
           total_timescale += timescale
           setattr(self, "VAP_{}".format(timescale), nn.MaxPool3d((n_segment//timescale,1,1),1,0,(timescale,1,1)))
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.TES = nn.Sequential(
            nn.Linear(total_timescale, total_timescale*4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(total_timescale*4, total_timescale, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

        
        # fc init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.001)
                if hasattr(m.bias, 'data'):
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        _, d = x.size()
        x = x.view(-1, self.n_segment, d, 1, 1).permute(0,2,1,3,4)
        x = torch.cat(tuple([getattr(self, "VAP_{}".format(2**i))(x) for i in range(self.VAP_level)]), 2).squeeze(3).squeeze(3).permute(0,2,1)
        w = self.GAP(x).squeeze(2)
        w = self.softmax(self.TES(w))
        x = x * w.unsqueeze(2)
        x = x.sum(dim=1) #加权求和
        return x

if __name__ == '__main__':
    input = torch.rand(24, 512)
    print('输入的维度', input.shape)
    vap = VAP(n_segment=8)
    out = vap(input)
    print(out.shape)