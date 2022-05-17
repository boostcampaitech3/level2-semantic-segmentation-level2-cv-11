# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="mmseg",
                name="LIM-DPT-baseline", #FIXME
                entity="canvas11",
            ),
        ),
    ],
)
# yapf:enable
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
cudnn_benchmark = True