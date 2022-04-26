## Project Structure
```
pytorch
├── trainer
│   ├── train.py
│   └── validate.py
├── util
│   ├── augmentation.py
│   ├── dataset.py
│   ├── model.py
│   ├── metrics.py
│   └── utils.py
├── args.py
├── inference.py
├── main.py
├── README.md
├── class_dict.csv
├── requirements.txt
└── visualization.ipynb
```

---

## 환경 설정

### Installation

- Clone the repository:
```
git clone https://github.com/boostcampaitech3/level2-semantic-segmentation-level2-cv-11.git
```

- Install required libraries.

```
pip install -r requirements.txt
```

- WandB 설정 바꿔주기
```
if args["WANDB_PLOT"]:
    wandb.init(project="semantic-segmentation", entity="canvas11", name = f"NAME-{args['MODEL']}") # name 수정 필요

```

---

## 학습 및 추론

### Train

- `fcn_resnet50` 모델 학습 예시
- 학습 시 `saved` 폴더에 Best valid mIoU의 모델이 `EPOCH_{에폭}_{모델 이름}_pretrained.pt`의 이름으로 저장된다.
```
python main.py --mode train --wandb_plot True --model fcn_resnet50

```

### Inference
- `fcn_resnet50` 모델 추론 예시
- 추론 후 `submission/sub_{모델 이름}.csv` 제출 파일이 생성된다. 
```
# python main.py --mode test --model fcn_resnet50 --pretrained_path /opt/ml/input/code/pytorch_baseline/saved/EPOCH_20_fcn_resnet50_pretrained.pt;
```


### Arguments
- `--model`: torchvision.models의 모델(['fcn_resnet50', 'deeplabv3_resnet50']). `model.py`에서 커스텀하여 추가 가능
- `--resize`: 입력 resolution(default=(512, 512))
- `--learning_rate`: 학습률(default=1e-4)
- `--weight_decay`: L2 penalty(모델 weight의 제곱합을 loss항에 추가하여 weight가 지나치게 커지는 것을 방지한다. 오버피팅 방지 효과. deafult=1e-6)
- `--num_epochs`: 학습할 Epoch 수(default=30)
- `--num_workers`: Num workers(default=4)
- `--batch_size`: 배치 크기(default=16)
- `--device`: cuda
- `--save_dir`: 학습한 모델을 저장할 디렉토리(default='./saved')
- `--pretrained_path`: 학습 시에는 fine-tuning할 모델.pt 경로 / 추론 시에는 추론할 모델.pt 경로
- `--num_classes`: class 수(default=11)
- `--val_every`: 몇 에폭마다 validation할 것인지(default=1)
- `--random_seed`: 랜덤 시드(default=42)
- `--train_json_path`: train.json 경로 
- `--valid_json_path`: val.json 경로 
- `--test_json_path`: test.json 경로 
- `--mode`: 학습/추론 모드 설정(default='train')
- `--wandb_plot`: WandB 시각화 여부(default=False)
