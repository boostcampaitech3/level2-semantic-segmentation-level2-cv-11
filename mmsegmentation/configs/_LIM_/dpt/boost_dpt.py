_base_ = [
    './dpt_vit-b16.py', './dataset_trash.py',
    './default_runtime.py', './schedule_adamw.py'
]

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    # _delete_=True,
    policy="CosineRestart",
    periods=[30000, 50000],
    restart_weights=[1.0, 0.3],
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.1,
    min_lr=0,
    by_epoch=False,
)


# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=8, workers_per_gpu=2)
