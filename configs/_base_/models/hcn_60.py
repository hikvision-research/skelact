# model settings
model = dict(
    type='SkeletonGCN',
    backbone=dict(
        type='HCN',
        in_channels=3,
        num_joints=25,
        clip_len=64,
        with_bn=False,
        reduce='flatten'),
    cls_head=dict(
        type='CNNHead',
        num_classes=60,
        in_channels=256,
        loss_cls=dict(type='CrossEntropyLoss')),
    train_cfg=None,
    test_cfg=None)
