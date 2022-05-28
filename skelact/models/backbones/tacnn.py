# Copyright (c) Hikvision Research Institute. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import SqueezeExcitation

from mmaction.models.builder import BACKBONES
from .hcn import HCN, ConvBN, Permute


class SELayer(SqueezeExcitation):

    def __init__(self, input_channels, squeeze_factor=1, bias=True):
        squeeze_channels = input_channels // squeeze_factor
        super().__init__(input_channels, squeeze_channels)

        if not bias:
            self.fc1.register_parameter('bias', None)
            self.fc2.register_parameter('bias', None)


class DualGroupConv(nn.Module):
    """Dual grouped convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 groups, bias):
        super().__init__()
        assert out_channels % groups == 0
        assert groups % 2 == 0
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias)
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            groups=groups // 2,
            bias=bias)

    def forward(self, input):
        out1 = F.relu(self.conv1(input), inplace=True)
        out2 = F.relu(self.conv2(input), inplace=True)
        out = out1 + out2
        return out


class CrossChannelFeatureAugment(nn.Module):
    """Cross-channel feature augmentation.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=10,
                 squeeze_factor=1):
        super().__init__()
        inter_channels = (out_channels // groups) * groups
        self.map1 = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.attend = SELayer(inter_channels, squeeze_factor, bias=False)
        self.group = DualGroupConv(
            inter_channels,
            inter_channels,
            kernel_size,
            stride,
            padding,
            groups,
            bias=False)
        self.map2 = nn.Conv2d(inter_channels, out_channels, 1, bias=False)

    def forward(self, x):
        out = F.relu(self.map1(x), inplace=True)
        out = self.attend(out)
        out = self.group(out)
        out = F.relu(self.map2(out), inplace=True)
        return out


class TaCNNBlock(nn.Sequential):
    """Building block for Ta-CNN.

    Args:
        in_channels (int): Number of channels in the input sequence data.
        out_channels (int): Number of channels produced by the convolution.
        num_joints (int): Number of joints in each skeleton.
        groups (tuple): Number of groups for conv2 (CAG) and conv3 (VAG).
        squeeze_factor (int): Squeeze factor in the SE layer.

    Shape:
        - Input: Input skeleton sequence in :math:`(N, in_channels, T_{in}, V)`
            format
        - Output: Output feature map in :math:`(N, out_channels, T_{out},
            C_{out})` format

        where
            :math:`N` is a batch size,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of joints,
            :math:`C_{out}` is the output size of the coordinate dimension.
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=64,
                 num_joints=25,
                 groups=(10, 6),
                 squeeze_factor=1):
        inter_channels = out_channels // 2
        super().__init__(
            # conv1
            ConvBN(in_channels, out_channels, 1),
            nn.ReLU(),
            # conv2
            CrossChannelFeatureAugment(
                out_channels,
                inter_channels, (3, 1),
                stride=1,
                padding=(1, 0),
                groups=groups[0],
                squeeze_factor=squeeze_factor),
            Permute((0, 3, 2, 1)),

            # conv3
            CrossChannelFeatureAugment(
                num_joints,
                inter_channels,
                3,
                stride=1,
                padding=1,
                groups=groups[1],
                squeeze_factor=squeeze_factor),
            nn.MaxPool2d(2, stride=2),

            # conv4
            nn.Conv2d(inter_channels, out_channels, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.5),
        )


@BACKBONES.register_module()
class TaCNN(HCN):
    """Backbone of Topology-aware Convolutional Neural Network for Efficient
    Skeleton-based Action Recognition.

    Args:
        in_channels (int): Number of channels in the input data.
        num_joints (int): Number of joints in each skeleton.
        groups (tuple): Number of groups for conv2 (CAG) and conv3 (VAG).
        squeeze_factor (int): Squeeze factor in the SE layer.
        pretrained (str | None): Name of pretrained model.

    Shape:
        - Input: :math:`(N, in_channels, T, V, M)`
        - Output: :math:`(N, D)` where
            :math:`N` is a batch size,
            :math:`T` is a length of input sequence,
            :math:`V` is the number of joints,
            :math:`M` is the number of instances in a frame.
    """

    def __init__(self,
                 in_channels=3,
                 num_joints=25,
                 groups=(10, 6),
                 squeeze_factor=1,
                 pretrained=None):
        clip_len = 64
        with_bn = True
        reduce = 'mean'
        super().__init__(in_channels, num_joints, clip_len, with_bn, reduce,
                         pretrained)

        self.net_l = TaCNNBlock(in_channels, 64, num_joints, groups,
                                squeeze_factor)
        self.net_m = TaCNNBlock(in_channels, 64, num_joints, groups,
                                squeeze_factor)
