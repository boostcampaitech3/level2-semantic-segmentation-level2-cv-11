import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(args):
    return A.Compose([
                    # A.Resize(width=args["RESIZE"][0], height=args["RESIZE"][1]),
                    ToTensorV2()
                    ])

def get_valid_transform(args):
    return A.Compose([
                    # A.Resize(width=args["RESIZE"][0], height=args["RESIZE"][1]),
                    ToTensorV2()
                    ])

def get_test_transform(args):
    return A.Compose([
                    A.Resize(width=256, height=256),
                    ToTensorV2()
                    ])