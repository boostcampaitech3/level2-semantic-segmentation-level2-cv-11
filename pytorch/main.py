import os
import wandb
import torch
import torch.nn as nn 
import pandas as pd
from args import Args
from inference import test_fn
from trainer.train import train_fn
from util.dataset import TrashDataset
from util.utils import collate_fn, seed_all, load_checkpoint
from util.model import load_model
from util.augmentation import get_train_transform, get_valid_transform, get_test_transform

'''
**기존 노트북 코드에서 수정된 사항**
1. train/val 학습시 tqdm bar 추가
2. best min loss 모델 저장 -> best max mIoU 모델 저장
3. torchvision.models의 deeplabv3 모델 추가
4. wandb 연결
'''

def main(args):

    # Model
    model = load_model(args)

    # Loss function 정의
    criterion = nn.CrossEntropyLoss().to(args["DEVICE"])

    # Optimizer 정의
    optimizer = torch.optim.Adam(params = model.parameters(), lr = args["LEARNING_RATE"], weight_decay=args["WEIGHT_DECAY"])

    if args["PRETRAINED_PATH"] is not None:
        load_checkpoint(
            args["PRETRAINED_PATH"], model, optimizer, args["LEARNING_RATE"], args["DEVICE"]
        )

    if args["MODE"] == 'train':

        ###########
        ## Train ##
        ###########
        print("Train mode..")

        if args["WANDB_PLOT"]:
            wandb.init(project="semantic-segmentation", entity="canvas11", name = f"NAME-{args['MODEL']}")

        train_dataset = TrashDataset(data_dir=args["TRAIN_JSON_PATH"], mode='train', transform=get_train_transform(args))
        val_dataset = TrashDataset(data_dir=args["VALID_JSON_PATH"], mode='val', transform=get_valid_transform(args))

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                batch_size=args["BATCH_SIZE"],
                                                shuffle=True,
                                                num_workers=args["NUM_WORKERS"],
                                                collate_fn=collate_fn)

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                                batch_size=args["BATCH_SIZE"],
                                                shuffle=False,
                                                num_workers=args["NUM_WORKERS"],
                                                collate_fn=collate_fn)

        train_fn(args, model, train_loader, val_loader, criterion, optimizer, wandb=wandb)
    
    else:
        
        ###############
        ## Inference ##
        ###############
        print("Inference mode..")
        model.eval()
        model.to(args["DEVICE"])
        # test dataset
        test_dataset = TrashDataset(data_dir=args["TEST_JSON_PATH"], mode='test', transform=get_test_transform(args))

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args["BATCH_SIZE"],
                                            num_workers=args["NUM_WORKERS"],
                                            collate_fn=collate_fn)

        # sample_submisson.csv 열기
        submission = pd.DataFrame(columns={'image_id', 'PredictionString'})

        # test set에 대한 prediction
        file_names, preds = test_fn(model, test_loader, args["DEVICE"])

        # PredictionString 대입
        for file_name, string in zip(file_names, preds):
            submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                        ignore_index=True)

        if not os.path.exists("./submission"):
            os.makedirs("./submission")

        # submission.csv로 저장
        submission.to_csv(f"./submission/sub_{args['MODEL']}.csv", index=False)


if __name__ == "__main__":
    args = Args().params

    # 실험 비교를 위한 랜덤 시드 고정
    seed_all(args)
    main(args)

