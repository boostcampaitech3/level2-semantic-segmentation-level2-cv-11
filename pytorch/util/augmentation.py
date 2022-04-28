import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(args):
    return A.Compose([
                    # A.Resize(width=args["RESIZE"][0], height=args["RESIZE"][1]),
                # A.RandomScale(scale_limit=0.3, p=0.5),
                # A.PadIfNeeded(512, 512, p=1),
                # A.RandomCrop(512, 512, p=1.),
                # A.Downscale(scale_min=0.5, scale_max=0.75, p=0.05),

                # color transforms
                # A.OneOf(
                #     [
                #         A.RandomBrightnessContrast(p=1),
                #         A.RandomGamma(p=1),
                #         A.ChannelShuffle(p=0.2),
                #         A.HueSaturationValue(p=1),
                #         A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1),
                #     ],
                #     p=0.5,
                # ),
                # # distortion
                # A.OneOf(
                #     [
                #         A.ElasticTransform(p=1),
                #         A.OpticalDistortion(p=1),
                #         A.GridDistortion(p=1),
                #         A.Perspective(p=1),
                #     ],
                #     p=0.1,
                # ),
                # noise transforms
                A.OneOf(
                    [
                        # A.ColorJitter(0.5, 0.5, 0.5, 0.25, p=1),
                        A.GaussNoise(p=1),
                        # A.MultiplicativeNoise(p=1),
                        A.Sharpen(p=1),
                        # A.CLAHE(p=2,clip_limit=5),
                        A.MedianBlur(blur_limit=3, p=1),
                    ],
                    p=0.4,
                ),
                    # A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ToTensorV2()
                    ])

def get_valid_transform(args):
    return A.Compose([
                    # A.Resize(width=args["RESIZE"][0], height=args["RESIZE"][1]),
                    ToTensorV2()
                    ])

def get_test_transform(args):
    return A.Compose([
                    # A.Resize(width=args["RESIZE"][0], height=args["RESIZE"][1]),
                    ToTensorV2()
                    ])