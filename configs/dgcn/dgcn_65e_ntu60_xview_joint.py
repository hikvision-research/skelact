_base_ = [
    '../_base_/models/dgcn_60.py', '../_base_/datasets/ntu60_xview_rot.py',
    '../_base_/schedules/sgd_65e.py', '../_base_/default_runtime.py'
]

checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])

# runtime settings
work_dir = './work_dirs/dgcn_ntu60_xview_joint/'
