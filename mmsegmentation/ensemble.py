import os
import cv2
import csv
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import webcolors
import copy
import glob
from tqdm import tqdm

if __name__ == "__main__":
    submission_path = []
    saved_models = "outputs/"
    submission_path = sorted(glob.glob(saved_models + "/*csv"))
    print(submission_path)
    root = "../input/data/"
    image_ids = []
    masks = []
    submission = pd.read_csv("/opt/ml/input/code/submission/sample_submission.csv", index_col=None)
    read_submission = None
    for i in range(len(submission_path)):
        read_submission = pd.read_csv(submission_path[i], index_col=None)
        mas2 = read_submission["PredictionString"].values
        LE = len(mas2)
        mas = []
        print(LE, "LE", i)
        for j in range(LE):
            mas.append(list(map(int, mas2[j].split())))
        masks.append(mas)
    image_ids = list(read_submission["image_id"].values)
    image_ids = np.array(image_ids)
    # print(image_ids)
    masks = np.array(masks)
    
    for j in tqdm(range(len(image_ids))):
        image_id = image_ids[j]
        ensemble = []
        mask = []
        for k in range(len(masks)):
            mask.append(masks[k][j])
        voting = np.array(mask).T
        # print(j, "/", len(image_ids))
        for i in range(len(voting)):
            vot = str(np.bincount(voting[i]).argmax())
            ensemble.append(vot)
        result = " ".join(ensemble)
        submission = submission.append({"image_id": image_id, "PredictionString": result}, ignore_index=True)
    print(
        len(read_submission["PredictionString"]) == len(submission["PredictionString"]),
        len(submission) == len(read_submission),
    )
    os.makedirs(saved_models + "ensemble", exist_ok=True)
    save_path = saved_models + "ensemble/ensemble6.csv"
    submission.to_csv(save_path, index=False)