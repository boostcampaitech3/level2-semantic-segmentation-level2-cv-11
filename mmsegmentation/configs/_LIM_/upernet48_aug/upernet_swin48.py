# model settings
# from /mmsegmentation/configs/_base_/models/upernet_swin.py
_base_ = [
    "dataset_trash.py",
    "default_runtime.py",
    "schedule_adamw.py",
]

norm_cfg = dict(type="BN", requires_grad=True)
backbone_norm_cfg = dict(type="LN", requires_grad=True)
model = dict(
    type="EncoderDecoder",
    pretrained="./pretrained/upernetswin48.out",
    backbone=dict(
        type="SwinTransformer",
        pretrain_img_size=384,
        embed_dims=192,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type="GELU"),
        norm_cfg=backbone_norm_cfg,
    ),
    decode_head=dict(
        type="UPerHead",
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type="CrossEntropyLoss", loss_name="loss_ce", loss_weight=0.75),
            dict(type="DiceLoss", loss_name="loss_dice", loss_weight=0.25),
        ],
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type="CrossEntropyLoss", loss_name="loss_ce", loss_weight=0.75 * 0.4),
            dict(type="DiceLoss", loss_name="loss_dice", loss_weight=0.25 * 0.4),
        ],
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
seed = 42