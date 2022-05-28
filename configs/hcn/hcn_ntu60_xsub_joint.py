_base_ = [
    '../_base_/models/hcn_60.py', '../_base_/datasets/ntu60_xsub_hcn.py',
    '../_base_/schedules/adam_500e.py', '../_base_/default_runtime.py'
]

# optimizer
optimizer = dict(
    type='Adam',
    lr=0.001,
    weight_decay=0.0001,
    paramwise_cfg=dict(custom_keys={'backbone.fc7': dict(decay_mult=10)}))

# runtime settings
work_dir = './work_dirs/hcn_ntu60_xsub_joint/'
