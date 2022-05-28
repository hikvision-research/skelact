# optimizer
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0004, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[35, 45, 55])
total_epochs = 65
