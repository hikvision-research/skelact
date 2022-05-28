# model settings
model = dict(
    type='SkeletonGCN',
    backbone=dict(
        type='DGCN',
        in_channels=3,
        alpha=0.6,
        lamb=1.0,
        clip_len=64,
        graph_cfg=dict(layout='ntu-rgb+d', strategy='agcn')),
    cls_head=dict(
        type='STGCNHead',
        num_classes=60,
        in_channels=256,
        loss_cls=dict(type='CrossEntropyLoss')),
    train_cfg=None,
    test_cfg=None)
