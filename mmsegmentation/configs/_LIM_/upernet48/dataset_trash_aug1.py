# dataset settings
dataset_type = "CustomDataset"
data_root = "/opt/ml/input/data/mmseg/"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

classes = [
    "Background",
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
]
palette = [
    [0, 0, 0], [192, 0, 128], [0, 128, 192],
    [0, 128, 64],[128, 0, 0],[64, 0, 128],
    [64, 0, 192],[192, 128, 64],[192, 192, 128],[64, 64, 128],[128, 0, 192],
]
alb_transform = [
    # dict(type='VerticalFlip', p=0.3),
    dict(type='HorizontalFlip', p=0.3),
    dict(
        type='OneOf',
        transforms=[
            dict(type='ShiftScaleRotate', p=1.0),
            dict(type='RandomRotate90', p=1.0),
            dict(type='PiecewiseAffine', p=1.0),
            dict(type='CoarseDropout',max_height=8, max_width=8, mask_fill_value=0, p=1.0),
            dict(type='ElasticTransform', border_mode=0, p=1.0),
            dict(type='GridDistortion', border_mode=0, p=1.0),
            dict(type='OpticalDistortion', distort_limit=0.5, p=1.0),
            dict(type='RandomGridShuffle', p=1.0)
        ],
        p=0.3),
    dict(type='RandomCrop', height=512, width=512, p=1.0),
    dict(
        type='OneOf',
        transforms=[
            dict(type='GaussianBlur', p=1.0),
            dict(type='Blur', p=1.0)
        ],
        p=0.3),
    dict(
        type='OneOf',
        transforms=[
            dict(type='RandomGamma', p=1.0),
            dict(type='ChannelDropout', p=1.0),
            dict(type='ChannelShuffle', p=1.0),
            dict(type='RGBShift', p=1.0)
        ],
        p=0.3)
]
imscale = [(x,x) for x in range(512, 1024+1, 128)]
crop_size = (512, 512)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=imscale, multiscale_mode='value', keep_ratio=True),
    dict(type="Albu", transforms=alb_transform),
    # dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.3),
    # dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    # dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            # dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        classes=classes,
        palette=palette,
        type=dataset_type,
        reduce_zero_label=False,
        img_dir=data_root + "images/training",
        ann_dir=data_root + "annotations/training",
        pipeline=train_pipeline,
    ),
    val=dict(
        classes=classes,
        palette=palette,
        type=dataset_type,
        reduce_zero_label=False,
        img_dir=data_root + "images/validation",
        ann_dir=data_root + "annotations/validation",
        pipeline=test_pipeline,
    ),
    test=dict(
        classes=classes,
        palette=palette,
        type=dataset_type,
        reduce_zero_label=False,
        img_dir=data_root + "/test",
        pipeline=test_pipeline,
    ),
)