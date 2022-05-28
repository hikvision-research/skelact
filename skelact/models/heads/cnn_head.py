# Copyright (c) Hikvision Research Institute. All rights reserved.
import torch.nn as nn
from mmcv.cnn import normal_init

from mmaction.models.builder import HEADS
from mmaction.models.heads.base import BaseHead


@HEADS.register_module()
class CNNHead(BaseHead):
    """The CNN classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.init_std = init_std

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        normal_init(self.fc, std=self.init_std)

    def forward(self, x):
        x = self.fc(x)
        return x
