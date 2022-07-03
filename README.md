# 🎨 [Pstage] CV 11조 CanVas 

<p align="center">
<img width="1071" alt="Screen Shot 2022-07-03 at 6 56 45 PM" src="https://user-images.githubusercontent.com/68208055/177034575-b63d52b4-4115-4d67-aa65-1ddb37a3e263.png">

</p>

- 대회 기간 : 2022.04.25 ~ 2022.05.12
- 목적 : 재활용 품목 분류를 위한 Semantic Segmentation

---

## 🔎 Overview

 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제가 나오고 있다. 분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나이다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문이다. 따라서 우리는 사진에서 쓰레기를 segmentation 하는 모델을 만들어 이러한 문제점을 해결하고자 한다.

---

## 💾  데이터셋

<p align="center">
<img width="600" alt="스크린샷 2022-04-27 오전 12 02 43" src="https://user-images.githubusercontent.com/68208055/164621090-2ac83869-d6b6-4b6a-bde4-fe5275252d83.png">

</p>

- 학습 이미지 개수 : 3272장
- 테스트 이미지 개수 : 624장 
- 11개 클래스 : Background, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- 이미지 크기 : 512x512
---

## 🧑 멤버

| [김영운](https://github.com/Cronople) | [이승현](https://github.com/sseunghyuns) | [임서현](https://github.com/seohl16) | [전성휴](https://github.com/shhommychon) | [허석용](https://github.com/HeoSeokYong) |  
| :-: | :-: | :-: | :-: | :-: |  
|<img src="https://avatars.githubusercontent.com/u/57852025?v=4" width=100>|<img src="https://avatars.githubusercontent.com/u/63924704?v=4" width=100> |<img src="https://avatars.githubusercontent.com/u/68208055?v=4" width=100> | <img src="https://avatars.githubusercontent.com/u/38153357?v=4" width=100> |<img src="https://avatars.githubusercontent.com/u/67945696?v=4" width=100>


---

## 🛰️ 프로젝트 수행 결과 

<p align="center">
<img width="629" alt="Screen Shot 2022-07-03 at 7 02 54 PM" src="https://user-images.githubusercontent.com/68208055/177034823-5f74b14a-7f28-4485-8070-e357c64c1a5e.png">

</p>

> 최종 mIoU : 0.7287

## 모델 개발 과정  
### 1. CV Strategy

기본으로 train.json, valid.json이 제공되었지만 더 일반화 성능을 높이기 위해 따로 Stratified Group K-Fold 함수를 사용하여 5 fold로 나누었다. 5개의 fold로 나누어진 train-valid 셋 중, local mAP와 leader board mAP간 차이가 가장 적은 CV셋을 베이스라인 실험 데이터셋으로 선택하였다. 

### 2. AMP 적용

AMP(Automatic Mixed Precision)는 학습시 처리 속도를 높이기 위한 FP16(16 bit floating point)연산과 정확도 유지를 위한 FP32 연산을 섞어 학습하는 방법이다. Swin transformer base+PAN 모델을 기준으로 실험했을 때, AMP를 사용할 경우 1 epoch 학습 소요 시간을 1분28초 → 59초까지 단축시킬 수 있었다.

### 3. Backbone 변경
<img width="757" alt="Screen Shot 2022-07-03 at 7 05 34 PM" src="https://user-images.githubusercontent.com/68208055/177034902-e1867877-8c04-4379-b4fd-7804c568d6a1.png">


위 Figure는 좌측부터 input image, predicted mask, ground truth mask이다. 모델이 이미지 내 객체의 위치는 어느정도 탐지하고 있으나, class 분류가 잘 이루어지지 않고 있음을 확인할 수 있다. 이러한 결과를 통해 backbone을 보다 무거운 모델로 교체한다면 객체의 semantic한 정보를 더욱 잘 학습할 수 있을 것이라는 가정을 했고, 실제로 backbone을 swin transformer Base → swin transformer Large로 교체하였을 때 리더보드 성능이 0.6950 → 0.7231로 향상되었다. 

### 4. Focal Loss

Focal loss는 cross entropy loss을 기반으로 상대적으로 예측하기 쉬운 class에 대해서는 down-weight하고 상대적으로 맞추기 어려운 class에 큰 weight를 주는 방식으로 작동하는 손실함수이다.
대회에서 제공된 데이터의 class 별 객체의 분포는 위 figure과 같다. 특히 Battery, Clothing의 데이터가 다른 class에 비해 많이 부족했고, 이를 해결하기 위해 imbalance dataset에 효과적인 Focal loss를 적용했다. 대부분의 모델에서 이를 사용했을 때 성능이 개선되었다. 

### 5. Augmentation

GridDistortion, PhotoMetricDistortion 등 강의에서 소개된 Distortion augmentation을 적용해봤다. PhotoMetricDistortion이 가장 결과가 좋아서 최종 모델에 적용했다. 
저번 Object Detection task에서는 CLAHE, RGBShift 등 augmentation이 효과가 있는 것이었지만 이번 Task에서는 유의미한 차이가 보이지 않았다. 

### 6. Ensemble

Hard-voting 방식으로 모델 결과를 Ensemble했다. 
Upernet 싱글 모델로 0.7123 정도 나왔지만 5 fold 결과를 모두 앙상블 한 결과 0.7256으로 상승했다. 




## Reference
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- [UperNet](https://paperswithcode.com/paper/unified-perceptual-parsing-for-scene)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
