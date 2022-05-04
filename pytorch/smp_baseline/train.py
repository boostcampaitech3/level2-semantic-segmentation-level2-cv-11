import os
import argparse
import warnings

import wandb
from torch.optim import *
from tqdm import tqdm

from dataset import load_dataset
from loss import get_loss
from model import build_model
from utils import *
from torch.cuda.amp import GradScaler, autocast

warnings.filterwarnings('ignore')

class_labels = {
    0: "Background",
    1: "General trash",
    2: "Paper",
    3: "Paper pack",
    4: "Metal",
    5: "Glass",
    6: "Plastic",
    7: "Styrofoam",
    8: "Plastic bag",
    9: "Battery",
    10: "Clothing",
}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/opt/ml/input/data')
    parser.add_argument('--config-dir', type=str, default='./config.yaml')
    parser.add_argument('--name', type=str, default="test")
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--viz-log', type=int, default=20)
    parser.add_argument('--metric', action='store_true')
    parser.add_argument('--loss', action='store_true')
    parser.add_argument('--save-interval', default=1)
    parser.add_argument('--wandb_plot', action='store_true')
    arg = parser.parse_args()
    return arg


def main():
    args = get_parser()
    args = load_config(args)
    set_seed(args.seed)

    model, preprocessing_fn = build_model(args)
    train_loader, val_loader = load_dataset(args, preprocessing_fn)
    model.to(args.device)
    # Loss
    criterion = get_loss(args.criterion)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4)
    scaler = GradScaler(enabled=True)

    # Wandb init
    if args.wandb_plot:
        wandb.init(project="semantic-segmentation", entity="canvas11", name=f"LEE-{args.name}")
        wandb.config = {
            "learning_rate": args.lr,
            "encoder": args.encoder,
            "epochs": args.epoch,
            "batch_size": args.batch_size
        }
        wandb.watch(model)

    device = args.device
    best_loss = 9999999.0
    best_score = 0.0
    for epoch in range(1, args.epoch + 1):
        model.train()
        train_loss, train_miou_score, train_accuracy = 0, 0, 0
        train_f1_score, train_recall, train_precision = 0, 0, 0
        hist = np.zeros((args.classes, args.classes))
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch{epoch} : Train")
        for i, data in enumerate(pbar):
            image, mask = data
            image = torch.stack(image).float().to(device)
            mask = torch.stack(mask).long().to(device)
            output = model(image)

            optimizer.zero_grad()
            with autocast(True):
                output = model(image)
                loss = criterion(output, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            hist = add_hist(hist, mask, output, n_class=args.classes)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            train_miou_score += mIoU
            train_accuracy += acc

            f1_score, recall, precision = get_metrics(output, mask)
            train_f1_score += f1_score.item()
            train_recall += recall.item()
            train_precision += precision.item()

            pbar.set_postfix(
                Train_Loss=f" {train_loss / (i + 1):.3f}",
                Train_Iou=f" {train_miou_score / (i + 1):.3f}",
                Train_Acc=f" {train_accuracy / (i + 1):.3f}",
            )
        if args.wandb_plot:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss / len(train_loader),
                'train/miou_score': train_miou_score / len(train_loader),
                'train/pixel_accuracy': train_accuracy / len(train_loader),
                'train/train_f1_score': train_f1_score / len(train_loader),
                'train/train_recall': train_recall / len(train_loader),
                'train/train_precision': train_precision / len(train_loader),
            })
        scheduler.step()

        val_loss, val_miou_score, val_accuracy = 0, 0, 0
        val_f1_score, val_recall, val_precision = 0, 0, 0
        val_pbar = tqdm(val_loader, total=len(val_loader), desc=f"Epoch{epoch} : Val")
        with torch.no_grad():
            model.eval()
            hist = np.zeros((args.classes, args.classes))
            for i, data in enumerate(val_pbar):
                image, mask = data
                image = torch.stack(image).float().to(device)
                mask = torch.stack(mask).long().to(device)
                output = model(image)

                loss = criterion(output, mask)
                val_loss += loss.item()

                hist = add_hist(hist, mask, output, n_class=args.classes)
                acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
                val_miou_score += mIoU
                val_accuracy += acc

                f1_score, recall, precision = get_metrics(output, mask)
                val_f1_score += f1_score.item()
                val_recall += recall.item()
                val_precision += precision.item()

                val_pbar.set_postfix(
                    Val_Loss=f" {val_loss / (i + 1):.3f}",
                    Val_Iou=f" {val_miou_score / (i + 1):.3f}",
                    Val_Acc=f" {val_accuracy / (i + 1):.3f}",
                )
                output = torch.argmax(output, dim=1).detach().cpu().numpy()
                if args.viz_log == i and args.wandb_plot:
                    wandb.log({
                        'visualize': wandb.Image(
                            image[0, :, :, :],
                            masks={
                                "predictions": {
                                    "mask_data": output[0, :, :],
                                    "class_labels": class_labels
                                },
                                "ground_truth": {
                                    "mask_data": mask[0, :, :].detach().cpu().numpy(),
                                    "class_labels": class_labels
                                }
                            }
                        )
                    })
            if args.wandb_plot:
                wandb.log({
                    'epoch': epoch,
                    'val/loss': val_loss / len(val_loader),
                    'val/miou_score': val_miou_score / len(val_loader),
                    'val/pixel_accuracy': val_accuracy / len(val_loader),
                    'val/f1_score': val_f1_score / len(val_loader),
                    'val/recall': val_recall / len(val_loader),
                    'val/precision': val_precision / len(val_loader),
                })
        # save_model
        if args.metric:
            if best_score < val_miou_score:
                best_score = val_miou_score
                try:
                    os.remove(ckpt_path)
                except:
                    pass
                ckpt_path = os.path.join(args.save_dir, f'epoch{epoch}_best_miou_{(best_score/len(val_loader)):.4f}.pth')
                torch.save(model.state_dict(), ckpt_path)
        if not args.metric:
            if best_loss > val_loss:
                best_loss = val_loss
                try:
                    os.remove(ckpt_path)
                except:
                    pass
                ckpt_path = os.path.join(args.save_dir, f'epoch{epoch}_best_loss_{(best_loss/len(val_loader)):.4f}.pth')
                torch.save(model.state_dict(), ckpt_path)
        if (epoch + 1) % args.save_interval == 0:
            ckpt_fpath = os.path.join(args.save_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)


if __name__ == "__main__":
    main()
