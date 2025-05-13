import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, ModuleList,auto_fp16
from mmcv.cnn import build_conv_layer
from mmcv.cnn.bricks.context_block import ContextBlock
class DCB(BaseModule):
    """ASPP (Atrous Spatial Pyramid Pooling)

    This is an implementation of the ASPP module used in DetectoRS
    (https://arxiv.org/pdf/2006.02334.pdf)

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels produced by this module
        dilations (tuple[int]): Dilations of the four branches.
            Default: (1, 3, 6, 1)
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 #(1, 3, 6, 1),
                 init_cfg=dict(type='Kaiming', layer='Conv2d')):
        super().__init__(init_cfg)

        self.dc = nn.ModuleList()
        kernel_size = 3
        padding = 2
        self.dc = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=2,
            padding=padding,
            bias=True)

        self.gelu = nn.ReLU()
        self.bn = nn.BatchNorm2d(in_channels)
        self.gc = ContextBlock(in_channels, ratio=1. / 1)
    @auto_fp16()
    def forward(self, x,H,W):
        _, _, num_features = x.shape
        x = x.view(-1, H, W, num_features).permute(0, 3, 1, 2).contiguous()
        out = self.gc(x)
        #print(out.shape)
        out = out.flatten(2).transpose(1, 2)
        return out