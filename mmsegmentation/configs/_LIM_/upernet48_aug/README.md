## UperNet(Swin-Transformer)

### Annotation Directory 
MMSegmentation 라이브러리는 각 파일마다 annotation이 저장된 mask.png를 참고합니다. 
이는 convert_seg.py를 /opt/ml/input 위치에 놓고 `python convert_seg.py`를 돌려서 만들 수 있습니다. (위치 중요!!)
현재 level2~ git에 있는 convert_seg.py를 /opt/ml/input 에 놓고 실행시켜주세요

### Pretrained
```
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth

mkdir pretrained
touch pretrained/upernetswin48.out

python tools/model_converters/swin2mmseg.py swin_large_patch4_window12_384_22k.pth pretrained/upernetswin48.out
```
복사해서 실행합니다.(또는 `sh download_pretrained.sh` 으로도 실행 가능합니다.)
swin2mmseg.py는 pth file을 mmseg가 읽을 수 있는 형태로 바꿔줍니다. 

### Train 
```
python tools/train.py configs/_LIM_/upernet48/upernet_swin48.py
```
복사해서 실행합니다. (또는 `sh run.sh` )
실행하기 전에 wandb 프로젝트 이름 바꿔주세요. 

### Inference 
```
python tools/inference.py
```
파라미터를 고정시켜놨기 때문에 파일에 들어가서 config file 위치, checkpoint 위치를 바꿔야 합니다. 
submission 파일은 실행 ./ 경로에 {모델 이름}_sub.csv 라는 이름으로 저장됩니다. 

