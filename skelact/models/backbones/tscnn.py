# Copyright (c) Hikvision Research Institute. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import normal_init, xavier_init
from mmcv.runner import load_checkpoint

from mmaction.models.builder import BACKBONES
from mmaction.utils import get_root_logger


class SkeletonTransformer(nn.Module):

    def __init__(self, in_joints=25, out_joints=30):
        super().__init__()
        self.in_joints = in_joints
        self.out_joints = out_joints

        self.trans_mat = nn.Parameter(torch.empty(in_joints, out_joints))
        nn.init.orthogonal_(self.trans_mat)

    def forward(self, x):
        n, c, t, v = x.size()
        x = x.view(-1, v)
        y = torch.matmul(x, self.trans_mat)
        y = y.view(n, c, t, -1)
        return y

    def extra_repr(self):
        return 'in_joints={}, out_joints={}'.format(self.in_joints,
                                                    self.out_joints)


class TSBlock(nn.Sequential):
    """Extracts two-stream feature from an input skeleton sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data.
        out_channels (int): Number of channels produced by the convolution.
        in_joints (int): Number of input joints.
        out_joints (int): Number of output joints.

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
                 in_joints=25,
                 out_joints=30):
        super().__init__(
            # fc1
            SkeletonTransformer(in_joints, out_joints),

            # conv2
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=(1, 0)),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.5),
        )


@BACKBONES.register_module()
class TSCNN(nn.Module):
    """Backbone of Skeleton-based Action Recognition with Convolutional Neural
    Networks.

    Args:
        in_channels (int): Number of channels in the input data.
        num_joints (int): Number of joints in each skeleton.
        clip_len (int): Skeleton sequence length.
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
                 clip_len=32,
                 pretrained=None):
        super().__init__()
        self.net_l = TSBlock(in_channels, 64, num_joints, 30)
        self.net_m = TSBlock(in_channels, 64, num_joints, 30)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), padding=(1, 0)),
            nn.MaxPool2d(2, stride=2),
            nn.PReLU(init=0.1),
            nn.Dropout(p=0.5),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), padding=(1, 0)),
            nn.MaxPool2d(2, stride=2),
            nn.PReLU(init=0.1),
        )
        self.drop4 = nn.Dropout(p=0.5)

        self.fc5 = nn.Sequential(
            nn.Linear(256 * 2 * clip_len // 8, 256),
            nn.PReLU(init=0.1),
            nn.Dropout(p=0.5),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(256, 128),
            nn.PReLU(init=0.1),
        )

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

        out = self.conv3(out)
        out = self.conv4(out)

        out = out.view(n, m, -1)  # N M D
        out = out.max(dim=1)[0]  # N D
        out = self.drop4(out)

        out = self.fc5(out)
        out = self.fc6(out)

        return out
