_base_ = ['./dataset_recycle.py', './schedule_hrnet.py', './runner_hrnet.py']

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='CascadeEncoderDecoder',
    num_stages=2,
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        type='HRNet',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384)))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[48, 96, 192, 384],
            channels=720,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            kernel_size=1,
            num_convs=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            concat_input=False,
            dropout_ratio=-1,
            num_classes=11, # CLASSES
            align_corners=False,
            loss_decode=dict(
                type='FocalLoss',
                alpha=[6.4, 6.8, 6.5, 7.7, 7.5, 7.6, 6.7, 7.0, 6.4, 29.1, 8.3],
                loss_name='loss_focal',
                loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[48, 96, 192, 384],
            channels=512,
            ocr_channels=256,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            dropout_ratio=-1,
            num_classes=11, # CLASSES
            align_corners=False,
            loss_decode=dict(
                type='FocalLoss',
                alpha=[6.4, 6.8, 6.5, 7.7, 7.5, 7.6, 6.7, 7.0, 6.4, 29.1, 8.3],
                loss_name='loss_focal',
                loss_weight=1.0))
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
work_dir = './work_dirs/ocrnet_hr48'
gpu_ids = range(0, 1)