import os
import torch
import numpy as np
from tqdm import tqdm
from trainer.validate import validation
from util.utils import save_checkpoint
from util.metrics import add_hist, label_accuracy_score

def train_fn(args, model, train_loader, val_loader, criterion, optimizer, wandb):
    print(f'Start training..')

    if not os.path.exists(args["SAVE_DIR"]):
        os.makedirs(args["SAVE_DIR"])
    
    n_class = args["NUM_CLASSES"]
    device = args["DEVICE"]
    best_mIoU = 0.
    model.to(device)

    for epoch in range(args["NUM_EPOCHS"]):
        model.train()
        hist = np.zeros((n_class, n_class))

        with tqdm(total=len(train_loader)) as pbar:
            for step, (images, masks, _) in enumerate(train_loader):
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                images = torch.stack(images)       
                masks = torch.stack(masks).long() 
                
                # gpu 연산을 위해 device 할당
                images, masks = images.to(device), masks.to(device)
                
                # inference
                outputs = model(images)['out']
                
                # loss 계산 (cross entropy loss)
                loss = criterion(outputs, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()
                
                hist = add_hist(hist, masks, outputs, n_class=n_class)
                acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
                
                pbar.update(1)

                pbar.set_postfix(
                    Train_loss = loss.item(),
                    Train_mIoU = mIoU
                )

                # step 주기에 따른 loss 출력
                if (step + 1) % 25 == 0:
                    print(f'Epoch [{epoch+1}/{args["NUM_EPOCHS"]}], Step [{step+1}/{len(train_loader)}], \
                            Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
                            
                    if args["WANDB_PLOT"]:
                        wandb.log({'Train/Loss': loss.item(),
                                    'Train/mIoU': mIoU, 
                                    })
                                
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % args["VAL_EVERY"] == 0:
            avrg_loss, val_mIoU = validation(epoch + 1, model, val_loader, criterion, n_class, device)
            if val_mIoU > best_mIoU:
                print(f"Best performance at epoch: {epoch + 1}")
                best_mIoU = val_mIoU

                # 기존 모델 제거
                try:
                    os.remove(saved_dir)
                except:
                    pass
                saved_dir = os.path.join(args["SAVE_DIR"], f'EPOCH_{epoch+1}_{args["MODEL"]}_pretrained.pt')
                print(f"Save model in {saved_dir}")
                save_checkpoint(model, optimizer, saved_dir)

            if args["WANDB_PLOT"]:
                wandb.log({'Val/Loss': avrg_loss,
                            'Val/mIoU': val_mIoU, 
                            })