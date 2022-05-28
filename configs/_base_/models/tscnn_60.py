# model settings
model = dict(
    type='SkeletonGCN',
    backbone=dict(type='TSCNN', in_channels=3, num_joints=25, clip_len=32),
    cls_head=dict(
        type='CNNHead',
        num_classes=60,
        in_channels=128,
        loss_cls=dict(type='CrossEntropyLoss')),
    train_cfg=None,
    test_cfg=None)
