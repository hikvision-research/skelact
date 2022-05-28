# Copyright (c) Hikvision Research Institute. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from mmaction.models.builder import BACKBONES
from mmaction.models.skeleton_gcn.utils import Graph
from mmaction.utils import get_root_logger


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(
        weight, mean=0, std=math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def zero(x):
    """return zero."""
    return 0


def identity(x):
    """return input itself."""
    return x


class CeN(nn.Module):

    def __init__(self, in_channels, num_joints=25, clip_len=64):
        super().__init__()
        self.num_joints = num_joints
        self.conv_c = nn.Conv2d(
            in_channels=in_channels, out_channels=1, kernel_size=1)
        self.conv_t = nn.Conv2d(
            in_channels=clip_len, out_channels=1, kernel_size=1)
        self.conv_v = nn.Conv2d(
            in_channels=num_joints,
            out_channels=num_joints * num_joints,
            kernel_size=1)
        self.bn = nn.BatchNorm2d(num_joints)

    def forward(self, x):
        x = self.conv_c(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # N T V C
        x = self.conv_t(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # N V C T
        x = self.bn(x)
        x = self.conv_v(x)

        n = x.size(0)
        A = x.view(n, self.num_joints, self.num_joints)
        d = torch.sum(torch.pow(A, 2), dim=1, keepdim=True)
        A = torch.div(A, torch.sqrt(d))
        return A


class STCAttention(nn.Module):

    def __init__(self, out_channels, num_joints):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # temporal attention
        self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
        nn.init.constant_(self.conv_ta.weight, 0)
        nn.init.constant_(self.conv_ta.bias, 0)

        # s attention
        ker_jpt = num_joints - 1 if not num_joints % 2 else num_joints
        pad = (ker_jpt - 1) // 2
        self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
        nn.init.xavier_normal_(self.conv_sa.weight)
        nn.init.constant_(self.conv_sa.bias, 0)

        # channel attention
        rr = 2
        self.fc1c = nn.Linear(out_channels, out_channels // rr)
        self.fc2c = nn.Linear(out_channels // rr, out_channels)
        nn.init.kaiming_normal_(self.fc1c.weight)
        nn.init.constant_(self.fc1c.bias, 0)
        nn.init.constant_(self.fc2c.weight, 0)
        nn.init.constant_(self.fc2c.bias, 0)

    def forward(self, x):
        y = x
        # spatial attention
        se = y.mean(-2)  # N C V
        se1 = self.sigmoid(self.conv_sa(se))
        y = y * se1.unsqueeze(-2) + y

        # temporal attention
        se = y.mean(-1)
        se1 = self.sigmoid(self.conv_ta(se))
        y = y * se1.unsqueeze(-1) + y

        # channel attention
        se = y.mean(-1).mean(-1)
        se1 = self.relu(self.fc1c(se))
        se2 = self.sigmoid(self.fc2c(se1))
        y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
        return y


class JointProject(nn.Module):

    def __init__(self, in_channels, in_joints=25, out_joints=15):
        super().__init__()
        self.in_joints = in_joints
        self.out_joints = out_joints

        self.proj_mat = nn.Parameter(torch.empty(in_joints, out_joints))
        self.bn = nn.BatchNorm2d(in_channels)

        nn.init.kaiming_normal_(self.proj_mat)
        constant_init(self.bn, 1)

    def forward(self, x):
        n, c, t, v = x.size()
        x = x.view(n, c * t, v)
        y = torch.matmul(x, self.proj_mat)
        y = y.view(n, c, t, -1)
        y = self.bn(y)
        return y

    def extra_repr(self):
        return 'in_joints={}, out_joints={}'.format(self.in_joints,
                                                    self.out_joints)


class DGCNBlock(nn.Module):
    """Applies spatial graph convolution and  temporal convolution over an
    input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and
            graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        adj_len (int, optional): The length of the adjacency matrix.
            Default: 17
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)`
            format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out},
            V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V,
            V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]
                `,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 lamb=1.0,
                 A=None,
                 adj_len=25,
                 clip_len=64,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1], lamb, A, adj_len,
                                         clip_len)
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1),
                      (stride, 1), padding), nn.BatchNorm2d(out_channels))

        # tcn init
        for m in self.tcn.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

        if not residual:
            self.residual = zero

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = identity

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)), nn.BatchNorm2d(out_channels))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.gcn(x)

        x = self.tcn(x) + res

        return self.relu(x)


class ConvTemporalGraphical(nn.Module):
    """The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        lamb (float):  The lambda parameter for fusion of static and dynamic
            branches in Eq. (4)
        A (torch.Tensor | None): The adjacency matrix
        adj_len (int, optional): The length of the adjacency matrix
            Default: 17
        clip_len (int): Input clip length

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)`
            format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}
            , V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)
            ` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]
                `,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 lamb=1.0,
                 A=None,
                 adj_len=25,
                 clip_len=64):
        super().__init__()

        self.kernel_size = kernel_size
        self.lamb = nn.Parameter(torch.empty(1))
        nn.init.constant_(self.lamb, lamb)

        if A.size(1) == adj_len:
            assert A is not None
            self.PA = nn.Parameter(A.clone())
        else:
            self.PA = nn.Parameter(torch.empty(3, adj_len, adj_len))
            nn.init.constant_(self.PA, 1e-6)

        self.cen = CeN(in_channels, num_joints=adj_len, clip_len=clip_len)
        self.conv_cen = nn.Conv2d(in_channels, out_channels, 1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels))
        else:
            self.down = lambda x: x

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

        self.num_subset = 3
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            conv = nn.Conv2d(in_channels, out_channels, 1)
            conv_branch_init(conv, self.num_subset)
            self.conv_d.append(conv)

        self.bn = nn.BatchNorm2d(out_channels)
        constant_init(self.bn, 1e-6)

        self.relu = nn.ReLU()
        self.attention = STCAttention(out_channels, adj_len)

    def forward(self, x):
        """Defines the computation performed at every call."""
        n, c, t, v = x.size()
        x1 = x.view(n, c * t, v)

        y = None
        for i in range(self.num_subset):
            A1 = self.PA[i]
            z = self.conv_d[i](torch.matmul(x1, A1).view(n, c, t, v))
            y = z + y if y is not None else z

        A2 = self.cen(x)
        z2 = torch.matmul(x1, A2).view(n, c, t, v)
        z2 = self.conv_cen(z2)
        y += self.lamb * z2

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        y = self.attention(y)
        return y


@BACKBONES.register_module()
class DGCN(nn.Module):
    """Backbone of Two-Stream Adaptive Graph Convolutional Networks for
    Skeleton-Based Action Recognition.

    Args:
        in_channels (int): Number of channels in the input data.
        graph_cfg (dict): The arguments for building the graph.
        data_bn (bool): If 'True', adds data normalization to the inputs.
            Default: True.
        alpha (float): The alpha parameter for joint reduction.
        lamb (float):  The lambda parameter for fusion of static and dynamic
            branches in Eq. (4).
        clip_len (int): Input clip length.
        pretrained (str | None): Name of pretrained model.
        **kwargs (optional): Other parameters for graph convolution units.

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self,
                 in_channels,
                 graph_cfg,
                 data_bn=True,
                 alpha=0.6,
                 lamb=1.0,
                 clip_len=64,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(
            self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels *
                                      A.size(1)) if data_bn else identity

        v1 = A.size(1)
        v2 = int(v1 * alpha)
        v3 = int(v2 * alpha)
        self.agcn_networks = nn.ModuleList((
            DGCNBlock(
                in_channels,
                64,
                kernel_size,
                1,
                lamb,
                A,
                v1,
                clip_len,
                residual=False),
            DGCNBlock(64, 64, kernel_size, 1, lamb, A, v1, clip_len, **kwargs),
            DGCNBlock(64, 64, kernel_size, 1, lamb, A, v1, clip_len, **kwargs),
            DGCNBlock(64, 64, kernel_size, 1, lamb, A, v1, clip_len, **kwargs),
            DGCNBlock(64, 128, kernel_size, 2, lamb, A, v1, clip_len,
                      **kwargs),
            JointProject(128, v1, v2),
            DGCNBlock(128, 128, kernel_size, 1, lamb, A, v2, clip_len // 2,
                      **kwargs),
            DGCNBlock(128, 128, kernel_size, 1, lamb, A, v2, clip_len // 2,
                      **kwargs),
            DGCNBlock(128, 256, kernel_size, 2, lamb, A, v2, clip_len // 2,
                      **kwargs),
            JointProject(256, v2, v3),
            DGCNBlock(256, 256, kernel_size, 1, lamb, A, v3, clip_len // 4,
                      **kwargs),
            DGCNBlock(256, 256, kernel_size, 1, lamb, A, v3, clip_len // 4,
                      **kwargs),
        ))

        self.pretrained = pretrained

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        # data normalization
        x = x.float()
        n, c, t, v, m = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # N M V C T
        x = x.view(n * m, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(n * m, c, t, v)

        for gcn in self.agcn_networks:
            x = gcn(x)

        return x
