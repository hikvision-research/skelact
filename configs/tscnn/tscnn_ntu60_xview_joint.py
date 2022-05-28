_base_ = [
    '../_base_/models/tscnn_60.py', '../_base_/datasets/ntu60_xview_tscnn.py',
    '../_base_/schedules/adam_500e.py', '../_base_/default_runtime.py'
]

# optimizer
optimizer = dict(
    type='Adam',
    lr=0.001,
    weight_decay=0.0001,
    paramwise_cfg=dict(custom_keys={'backbone.fc5': dict(decay_mult=10)}))

# runtime settings
work_dir = './work_dirs/tscnn_ntu60_xview_joint/'
