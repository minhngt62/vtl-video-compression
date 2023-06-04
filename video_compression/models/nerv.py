import os
import torch
from torch import nn, Tensor
import torch.nn.functional as F

def mlp(fc_dims, bias=True):
    fcs = []
    for i in range(len(fc_dims)-1):
        fcs += [nn.Linear(fc_dims[i], fc_dims[i+1], bias=bias), 
                nn.GELU()]
    return nn.Sequential(*fcs)

class NervBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        stride,
        bias: bool = True,
    ):
        super().__init__()
        
        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels * stride**2, 3, 1, 1, bias=bias),
            nn.PixelShuffle(stride)
        ])
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.upsample(x))

class Nerv(nn.Module):
    def __init__(
        self,
        stem_dim_num,
        fc_hw_dim,
        embed_length,
        stride_list,
        expansion,
        reduction,
        lower_width,
        num_blocks,
        bias,
        sin_res,
        sigmoid
    ):
        super().__init__()
        
        stem_dim, stem_num = [int(x) for x in stem_dim_num.split('_')]
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in fc_hw_dim.split('_')]
        mlp_dim_list = [embed_length] + [stem_dim] * stem_num + [self.fc_h *self.fc_w *self.fc_dim]
        self.stem = mlp(fc_dims=mlp_dim_list)

        self.layers, self.head_layers = [nn.ModuleList() for _ in range(2)]
        ngf = self.fc_dim
        for i, stride in enumerate(stride_list):
            if i == 0:
                new_ngf = int(ngf * expansion)
            else:
                new_ngf = max(ngf // (1 if stride == 1 else reduction), lower_width)

            for j in range(num_blocks):
                self.layers.append(NervBlock(
                    in_channels=ngf, out_channels=new_ngf, stride=1 if j else stride, bias=bias))

            head_layer = [None]
            if sin_res:
                if i == len(stride_list) - 1:
                    head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=bias)
                else:
                    head_layer = None
            else:
                head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=bias)
            self.head_layers.append(head_layer)
        
        self.sigmoid = sigmoid
    
    def forward(self, x):
        x = self.stem(x)
        x = x.view(x.size(0), self.fc_dim, self.fc_h, self.fc_w)

        frames = []
        for layer, head_layer in zip(self.layers, self.head_layers):
            x = layer(x)
            if head_layer is not None:
                frame = head_layer(x)
                frame = torch.sigmoid(frame) if self.sigmoid else (torch.tanh(frame) + 1) * 0.5
                frames.append(frame)
        
        return frames