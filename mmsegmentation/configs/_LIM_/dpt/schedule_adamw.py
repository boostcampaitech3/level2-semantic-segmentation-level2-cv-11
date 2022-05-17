# optimizer
# optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer = dict(
    # _delete_=True,
    type="AdamW",
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
            "head": dict(lr_mult=2.0),
        }
    ),
)
optimizer_config = dict()
# learning policy
# lr_config = dict(policy="poly", power=0.9, min_lr=1e-4, by_epoch=True)
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

# runtime settings
runner = dict(type="EpochBasedRunner", max_epochs=90)
checkpoint_config = dict(max_keep_ckpts=2, by_epoch=True, interval=5)
evaluation = dict(interval=1, metric="mIoU", pre_eval=True)