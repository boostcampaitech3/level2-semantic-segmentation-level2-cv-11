# smp_baseline

### Train
```
python train.py --config-dir [CONFIG YAML FILE] --name [MODEL_NAME] [--metric]
```

### File 내부 설명
* train.py : optimizer, scheduler 설정 가능
* dataset.py : Dataset, transform 설정 가능
* config.yaml : Model 및 hyperparameter 설정가능

### 구현 기능
* wandb 연동 : train/val loss 및 각종 metric logging, output visualize 기능
  * 구현 Metric : mean IoU, f1 score, precision, recall, Pixel Accuracy
* model save : best loss 나 metric 둘 중 하나로 선택 가능 (default : loss) --metric 입력시 mean IoU로 저장
* loss 함수 : DiceLoss, CrossEntropy, FocalLoss, SoftCrossEntropy
