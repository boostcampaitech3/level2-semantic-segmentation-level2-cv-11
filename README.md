# ๐จ [Pstage] CV 11์กฐ CanVas 

<p align="center">
<img width="1071" alt="Screen Shot 2022-07-03 at 6 56 45 PM" src="https://user-images.githubusercontent.com/68208055/177034575-b63d52b4-4115-4d67-aa65-1ddb37a3e263.png">

</p>

- ๋ํ ๊ธฐ๊ฐ : 2022.04.25 ~ 2022.05.12
- ๋ชฉ์  : ์ฌํ์ฉ ํ๋ชฉ ๋ถ๋ฅ๋ฅผ ์ํ Semantic Segmentation

---

## ๐ Overview

 '์ฐ๋ ๊ธฐ ๋๋', '๋งค๋ฆฝ์ง ๋ถ์กฑ'๊ณผ ๊ฐ์ ์ฌ๋ฌ ์ฌํ ๋ฌธ์ ๊ฐ ๋์ค๊ณ  ์๋ค. ๋ถ๋ฆฌ์๊ฑฐ๋ ์ด๋ฌํ ํ๊ฒฝ ๋ถ๋ด์ ์ค์ผ ์ ์๋ ๋ฐฉ๋ฒ ์ค ํ๋์ด๋ค. ์ ๋ถ๋ฆฌ๋ฐฐ์ถ ๋ ์ฐ๋ ๊ธฐ๋ ์์์ผ๋ก์ ๊ฐ์น๋ฅผ ์ธ์ ๋ฐ์ ์ฌํ์ฉ๋์ง๋ง, ์๋ชป ๋ถ๋ฆฌ๋ฐฐ์ถ ๋๋ฉด ๊ทธ๋๋ก ํ๊ธฐ๋ฌผ๋ก ๋ถ๋ฅ๋์ด ๋งค๋ฆฝ ๋๋ ์๊ฐ๋๊ธฐ ๋๋ฌธ์ด๋ค. ๋ฐ๋ผ์ ์ฐ๋ฆฌ๋ ์ฌ์ง์์ ์ฐ๋ ๊ธฐ๋ฅผ segmentation ํ๋ ๋ชจ๋ธ์ ๋ง๋ค์ด ์ด๋ฌํ ๋ฌธ์ ์ ์ ํด๊ฒฐํ๊ณ ์ ํ๋ค.

---

## ๐พ  ๋ฐ์ดํฐ์

<p align="center">
<img width="600" alt="์คํฌ๋ฆฐ์ท 2022-04-27 ์ค์  12 02 43" src="https://user-images.githubusercontent.com/68208055/164621090-2ac83869-d6b6-4b6a-bde4-fe5275252d83.png">

</p>

- ํ์ต ์ด๋ฏธ์ง ๊ฐ์ : 3272์ฅ
- ํ์คํธ ์ด๋ฏธ์ง ๊ฐ์ : 624์ฅ 
- 11๊ฐ ํด๋์ค : Background, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- ์ด๋ฏธ์ง ํฌ๊ธฐ : 512x512
---

## ๐ง ๋ฉค๋ฒ

| [๊น์์ด](https://github.com/Cronople) | [์ด์นํ](https://github.com/sseunghyuns) | [์์ํ](https://github.com/seohl16) | [์ ์ฑํด](https://github.com/shhommychon) | [ํ์์ฉ](https://github.com/HeoSeokYong) |  
| :-: | :-: | :-: | :-: | :-: |  
|<img src="https://avatars.githubusercontent.com/u/57852025?v=4" width=100>|<img src="https://avatars.githubusercontent.com/u/63924704?v=4" width=100> |<img src="https://avatars.githubusercontent.com/u/68208055?v=4" width=100> | <img src="https://avatars.githubusercontent.com/u/38153357?v=4" width=100> |<img src="https://avatars.githubusercontent.com/u/67945696?v=4" width=100>


---

## ๐ฐ๏ธ ํ๋ก์ ํธ ์ํ ๊ฒฐ๊ณผ 

<p align="center">
<img width="629" alt="Screen Shot 2022-07-03 at 7 02 54 PM" src="https://user-images.githubusercontent.com/68208055/177034823-5f74b14a-7f28-4485-8070-e357c64c1a5e.png">

</p>

> ์ต์ข mIoU : 0.7287

## ๋ชจ๋ธ ๊ฐ๋ฐ ๊ณผ์   
### 1. CV Strategy

๊ธฐ๋ณธ์ผ๋ก train.json, valid.json์ด ์ ๊ณต๋์์ง๋ง ๋ ์ผ๋ฐํ ์ฑ๋ฅ์ ๋์ด๊ธฐ ์ํด ๋ฐ๋ก Stratified Group K-Fold ํจ์๋ฅผ ์ฌ์ฉํ์ฌ 5 fold๋ก ๋๋์๋ค. 5๊ฐ์ fold๋ก ๋๋์ด์ง train-valid ์ ์ค, local mAP์ leader board mAP๊ฐ ์ฐจ์ด๊ฐ ๊ฐ์ฅ ์ ์ CV์์ ๋ฒ ์ด์ค๋ผ์ธ ์คํ ๋ฐ์ดํฐ์์ผ๋ก ์ ํํ์๋ค. 

### 2. AMP ์ ์ฉ

AMP(Automatic Mixed Precision)๋ ํ์ต์ ์ฒ๋ฆฌ ์๋๋ฅผ ๋์ด๊ธฐ ์ํ FP16(16 bit floating point)์ฐ์ฐ๊ณผ ์ ํ๋ ์ ์ง๋ฅผ ์ํ FP32 ์ฐ์ฐ์ ์์ด ํ์ตํ๋ ๋ฐฉ๋ฒ์ด๋ค. Swin transformer base+PAN ๋ชจ๋ธ์ ๊ธฐ์ค์ผ๋ก ์คํํ์ ๋, AMP๋ฅผ ์ฌ์ฉํ  ๊ฒฝ์ฐ 1 epoch ํ์ต ์์ ์๊ฐ์ 1๋ถ28์ด โ 59์ด๊น์ง ๋จ์ถ์ํฌ ์ ์์๋ค.

### 3. Backbone ๋ณ๊ฒฝ
<img width="757" alt="Screen Shot 2022-07-03 at 7 05 34 PM" src="https://user-images.githubusercontent.com/68208055/177034902-e1867877-8c04-4379-b4fd-7804c568d6a1.png">


์ Figure๋ ์ข์ธก๋ถํฐ input image, predicted mask, ground truth mask์ด๋ค. ๋ชจ๋ธ์ด ์ด๋ฏธ์ง ๋ด ๊ฐ์ฒด์ ์์น๋ ์ด๋์ ๋ ํ์งํ๊ณ  ์์ผ๋, class ๋ถ๋ฅ๊ฐ ์ ์ด๋ฃจ์ด์ง์ง ์๊ณ  ์์์ ํ์ธํ  ์ ์๋ค. ์ด๋ฌํ ๊ฒฐ๊ณผ๋ฅผ ํตํด backbone์ ๋ณด๋ค ๋ฌด๊ฑฐ์ด ๋ชจ๋ธ๋ก ๊ต์ฒดํ๋ค๋ฉด ๊ฐ์ฒด์ semanticํ ์ ๋ณด๋ฅผ ๋์ฑ ์ ํ์ตํ  ์ ์์ ๊ฒ์ด๋ผ๋ ๊ฐ์ ์ ํ๊ณ , ์ค์ ๋ก backbone์ swin transformer Base โ swin transformer Large๋ก ๊ต์ฒดํ์์ ๋ ๋ฆฌ๋๋ณด๋ ์ฑ๋ฅ์ด 0.6950 โ 0.7231๋ก ํฅ์๋์๋ค. 

### 4. Focal Loss

Focal loss๋ cross entropy loss์ ๊ธฐ๋ฐ์ผ๋ก ์๋์ ์ผ๋ก ์์ธกํ๊ธฐ ์ฌ์ด class์ ๋ํด์๋ down-weightํ๊ณ  ์๋์ ์ผ๋ก ๋ง์ถ๊ธฐ ์ด๋ ค์ด class์ ํฐ weight๋ฅผ ์ฃผ๋ ๋ฐฉ์์ผ๋ก ์๋ํ๋ ์์คํจ์์ด๋ค.
๋ํ์์ ์ ๊ณต๋ ๋ฐ์ดํฐ์ class ๋ณ ๊ฐ์ฒด์ ๋ถํฌ๋ ์ figure๊ณผ ๊ฐ๋ค. ํนํ Battery, Clothing์ ๋ฐ์ดํฐ๊ฐ ๋ค๋ฅธ class์ ๋นํด ๋ง์ด ๋ถ์กฑํ๊ณ , ์ด๋ฅผ ํด๊ฒฐํ๊ธฐ ์ํด imbalance dataset์ ํจ๊ณผ์ ์ธ Focal loss๋ฅผ ์ ์ฉํ๋ค. ๋๋ถ๋ถ์ ๋ชจ๋ธ์์ ์ด๋ฅผ ์ฌ์ฉํ์ ๋ ์ฑ๋ฅ์ด ๊ฐ์ ๋์๋ค. 

### 5. Augmentation

GridDistortion, PhotoMetricDistortion ๋ฑ ๊ฐ์์์ ์๊ฐ๋ Distortion augmentation์ ์ ์ฉํด๋ดค๋ค. PhotoMetricDistortion์ด ๊ฐ์ฅ ๊ฒฐ๊ณผ๊ฐ ์ข์์ ์ต์ข ๋ชจ๋ธ์ ์ ์ฉํ๋ค. 
์ ๋ฒ Object Detection task์์๋ CLAHE, RGBShift ๋ฑ augmentation์ด ํจ๊ณผ๊ฐ ์๋ ๊ฒ์ด์์ง๋ง ์ด๋ฒ Task์์๋ ์ ์๋ฏธํ ์ฐจ์ด๊ฐ ๋ณด์ด์ง ์์๋ค. 

### 6. Ensemble

Hard-voting ๋ฐฉ์์ผ๋ก ๋ชจ๋ธ ๊ฒฐ๊ณผ๋ฅผ Ensembleํ๋ค. 
Upernet ์ฑ๊ธ ๋ชจ๋ธ๋ก 0.7123 ์ ๋ ๋์์ง๋ง 5 fold ๊ฒฐ๊ณผ๋ฅผ ๋ชจ๋ ์์๋ธ ํ ๊ฒฐ๊ณผ 0.7256์ผ๋ก ์์นํ๋ค. 




## Reference
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- [UperNet](https://paperswithcode.com/paper/unified-perceptual-parsing-for-scene)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
