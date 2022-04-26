import torch
import numpy as np
from tqdm import tqdm
from util.metrics import add_hist, label_accuracy_score

category_names = ['Backgroud',
                  'General trash',
                  'Paper',
                  'Paper pack',
                  'Metal',
                  'Glass',
                  'Plastic',
                  'Styrofoam',
                  'Plastic bag',
                  'Battery',
                  'Clothing']

def validation(epoch, model, data_loader, criterion, n_class, device):
    print(f'Start validation #{epoch}')
    model.eval()

    with torch.no_grad():
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))
        with tqdm(total=len(data_loader)) as pbar:
            for step, (images, masks, _) in enumerate(data_loader):
                
                images = torch.stack(images)       
                masks = torch.stack(masks).long()  

                images, masks = images.to(device), masks.to(device)            
                
                # device 할당
                model = model.to(device)
                
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                total_loss += loss
                cnt += 1
                
                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()
                
                hist = add_hist(hist, masks, outputs, n_class=n_class)
                pbar.update(1)

        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        
        avrg_loss = total_loss / cnt
        
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}')
        print(f'{"CLASS_NAME":14s}| {"IoU":6s} |')
        print('ㅡ'*12 + '|')
        for classes, iou  in zip(category_names, IoU):
            print(f'{classes:14s}| {iou:0.4f} |')        
        
    return avrg_loss, mIoU