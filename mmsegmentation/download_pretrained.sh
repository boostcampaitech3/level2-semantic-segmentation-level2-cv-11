wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth

mkdir pretrained
touch pretrained/upernetswin48.out

python tools/model_converters/swin2mmseg.py swin_large_patch4_window12_384_22k.pth pretrained/upernetswin48.out