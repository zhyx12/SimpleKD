backbone_optimizer = dict(
    type='AdamW',
    lr=0.0005,
    weight_decay=0.0001,
)

backbone = dict(
    type='CLIPLoRADistill',
    name='ViT-B/16',
    model_device='cpu',
    template="a photo of a {}.",
    jit=False,
    lora_r=1,
    lora_alpha=1.0,
    optimizer=backbone_optimizer,
)

scheduler = dict(
    type='ConstantLR',
    total_iters=1,
    factor=1.0,
)

models = dict(
    base_model=backbone,
    lr_scheduler=scheduler,
    find_unused_parameters=True,
)
