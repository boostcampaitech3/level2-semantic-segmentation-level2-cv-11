from random import sample
import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                        wrap_fp16_model)
from mmcv.utils import DictAction
from mmcv import Config
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from glob import glob

import matplotlib.pyplot as plt
import albumentations as A
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

from pycocotools.coco import COCO
import warnings 
warnings.filterwarnings('ignore')

# FIXME
# model_cfg = '/opt/ml/input/level2-semantic-segmentation-level2-cv-11/mmsegmentation/configs/_LIM_/upernet48/upernet_swin48.py'
# ckpt = '/opt/ml/input/level2-semantic-segmentation-level2-cv-11/mmsegmentation/work_dirs/upernet_swin48/latest.pth'
model_cfg = '/opt/ml/input/level2-semantic-segmentation-level2-cv-11/mmsegmentation/configs/_LIM_/dpt/boost_dpt.py'
ckpt = '/opt/ml/input/level2-semantic-segmentation-level2-cv-11/mmsegmentation/work_dirs/boost_dpt/latest.pth'
# ckpt = ''
# ckpt = ''


cfg = Config.fromfile(model_cfg)
cfg.data.test.test_mode = True

# imsize = 512
# cfg.data.test.pipeline[1]['img_scale'] = (imsize,imsize)
cfg.data.test.ann_dir = None

print('building dataset')
test_dataset = build_dataset(cfg.data.test)
print('finished building dataset')
test_loader = build_dataloader(
    test_dataset,
    samples_per_gpu=1,
    workers_per_gpu=1,
    dist=False,
    shuffle=False,
    drop_last=False
)

cfg.model.pretrained = None
cfg.model.train_cfg = None

model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
ckpt = load_checkpoint(model, ckpt, map_location='cpu')

model.CLASSES = ckpt['meta']['CLASSES']
model = MMDataParallel(model.cuda(), device_ids=[0])

output = single_gpu_test(model, test_loader)


# output_size = 512
output_size = 256
transform = A.Compose([A.Resize(output_size, output_size)])

test_json_path = '/opt/ml/input/data/test.json'
coco = COCO(test_json_path)

sub_dict = {}
img_infos = test_dataset.img_infos
print('resize for submit')
file_names = []

for idx,  out in enumerate(tqdm(output)):
    image_info = coco.loadImgs(coco.getImgIds(imgIds=idx))[0]
    file_names.append(image_info['file_name'])
    image = np.zeros((1,1,1))
    transformed = transform(image=image, mask=out)
    mask = transformed['mask']
    mask = mask.reshape(-1, output_size*output_size).astype(int)
    sub_dict[image_info['file_name']] = mask[0]
print('resize for submit complete')


submission = pd.DataFrame()
submission['image_id'] = file_names
preds = [sub_dict[imId].flatten() for imId in submission['image_id']]
submission['PredictionString'] = [' '.join([str(dot) for dot in mask]) for mask in preds]
submission.to_csv(f"/opt/ml/input/level2-semantic-segmentation-level2-cv-11/mmsegmentation/{model_cfg.split('/')[-1].split('.')[0]}_sub.csv", index=False)
print('final submission file at ./~_sub.csv')