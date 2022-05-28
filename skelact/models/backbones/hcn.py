# Copyright (c) Hikvision Research Institute. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import constant_init, normal_init, xavier_init
from mmcv.runner import load_checkpoint

from mmaction.models.builder import BACKBONES
from mmaction.utils import get_root_logger


class Permute(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, input):
        return input.permute(self.dims).contiguous()


class ConvBN(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False),
            nn.BatchNorm2d(out_channels),
        )


class HCNBlock(nn.Sequential):
    """Extracts hierarchical co-occurrence feature from an input skeleton
    sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data.
        out_channels (int): Number of channels produced by the convolution.
        num_joints (int): Number of joints in each skeleton.
        with_bn (bool): Whether to append a BN layer after conv1.

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
                 with_bn=False):
        inter_channels = out_channels // 2
        conv1 = ConvBN if with_bn else nn.Conv2d
        super().__init__(
            # conv1
            conv1(in_channels, out_channels, 1),
            nn.ReLU(),
            # conv2
            nn.Conv2d(out_channels, inter_channels, (3, 1), padding=(1, 0)),
            Permute((0, 3, 2, 1)),

            # conv3
            nn.Conv2d(num_joints, inter_channels, 3, padding=1),
            nn.MaxPool2d(2, stride=2),

            # conv4
            nn.Conv2d(inter_channels, out_channels, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.5),
        )


@BACKBONES.register_module()
class HCN(nn.Module):
    """Backbone of Co-occurrence Feature Learning from Skeleton Data for Action
    Recognition and Detection with Hierarchical Aggregation.

    Args:
        in_channels (int): Number of channels in the input data.
        num_joints (int): Number of joints in each skeleton.
        clip_len (int): Skeleton sequence length.
        with_bn (bool): Whether to append a BN layer after conv1.
        reduce (str): Reduction mode along the temporal dimension,'flatten' or
            'mean'.
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
                 clip_len=64,
                 with_bn=False,
                 reduce='flatten',
                 pretrained=None):
        super().__init__()
        assert reduce in ('flatten', 'mean')
        self.reduce = reduce

        self.net_l = HCNBlock(in_channels, 64, num_joints, with_bn)
        self.net_m = HCNBlock(in_channels, 64, num_joints, with_bn)

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
        )
        self.drop6 = nn.Dropout(p=0.5)

        self.fc7 = nn.Sequential(
            nn.Linear(256 * 2 * clip_len // 16, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        ) if self.reduce == 'flatten' else None

        self.pretrained = pretrained

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform')
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        n, c, t, v, m = x.size()
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # N M C T V
        x = x.view(-1, c, t, v)  # N*M C T V

        vel1 = x[:, :, :1] * 0
        vel2 = x[:, :, 1:] - x[:, :, :-1]
        vel = torch.cat((vel1, vel2), dim=2)

        out_l = self.net_l(x)
        out_m = self.net_m(vel)

        out = torch.cat((out_l, out_m), dim=1)

        out = self.conv5(out)
        out = self.conv6(out)

        if self.reduce == 'mean':
            out = out.mean(dim=2)

        out = out.view(n, m, -1)  # N M D
        out = out.max(dim=1)[0]  # N D
        out = self.drop6(out)

        if self.fc7 is not None:
            out = self.fc7(out)

        return out
