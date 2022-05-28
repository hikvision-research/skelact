# model settings
model = dict(
    type='SkeletonGCN',
    backbone=dict(
        type='TaCNN',
        in_channels=3,
        num_joints=25,
        groups=(10, 6),
        squeeze_factor=1),
    cls_head=dict(
        type='CNNHead',
        num_classes=60,
        in_channels=512,
        loss_cls=dict(type='CrossEntropyLoss')),
    train_cfg=None,
    test_cfg=None)
