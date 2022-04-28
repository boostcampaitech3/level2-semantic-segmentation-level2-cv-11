import argparse
import os
import warnings

import albumentations as A
import numpy as np
import pandas as pd
import torch
import yaml
from munch import Munch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CustomDataLoader, get_transform
from model import build_model

warnings.filterwarnings('ignore')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/opt/ml/input/data')
    parser.add_argument('--model-dir', type=str)
    args = parser.parse_args()
    return args


def load_config():
    args = get_parser()
    with open(args.model_dir + '/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    config = Munch(config)
    config.model_dir = args.model_dir
    config.data_dir = args.data_dir
    return config


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    args = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, preprocessing_fn = build_model(args)
    _, val_transform = get_transform(preprocessing_fn)
    test_dataset = CustomDataLoader(data_dir=os.path.join(args.data_dir, 'test.json'), mode='test',
                                    transform=val_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_worker,
                                 collate_fn=collate_fn, pin_memory=True)
    model_dir = os.path.join(args.model_dir, 'latest.pth')
    model.load_state_dict(torch.load(model_dir))
    model.to(device)

    model.eval()
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    file_name_list = []
    preds_array = np.empty((0, size * size), dtype=np.long)

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dataloader, total=len(test_dataloader))):
            images, image_infos = data
            output = model(torch.stack(images).float().to(device))
            oms = torch.argmax(output.squeeze(), dim=1).detach().cpu().numpy()

            temp_mask = []
            for img, mask in zip(np.stack(images), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
            oms = np.array(temp_mask)

            oms = oms.reshape([oms.shape[0], size * size]).astype(int)
            preds_array = np.vstack((preds_array, oms))

            file_name_list.append([i['file_name'] for i in image_infos])
        print("End prediction.")
        file_names = [y for x in file_name_list for y in x]

    submission = pd.read_csv('/opt/ml/input/sample_submission.csv', index_col=None)
    for file_name, string in zip(file_names, preds_array):
        submission = submission.append(
            {"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())},
            ignore_index=True)
    submission.to_csv(f"./{args.model_dir}.csv", index=False)


if __name__ == "__main__":
    main()
