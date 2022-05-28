# optimizer
optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[650, 730, 770])
total_epochs = 800
