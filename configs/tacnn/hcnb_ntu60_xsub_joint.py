_base_ = [
    '../_base_/models/hcnb_60.py', '../_base_/datasets/ntu60_xsub_rot.py',
    '../_base_/schedules/adam_800e.py', '../_base_/default_runtime.py'
]

# runtime settings
work_dir = './work_dirs/hcnb_ntu60_xsub_joint/'
