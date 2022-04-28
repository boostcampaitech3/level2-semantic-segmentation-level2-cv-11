import torch.nn as nn
from torchvision import models


def load_model(args):
    model_type = args['MODEL']
    num_classes = args['NUM_CLASSES']

    if model_type == 'fcn_resnet50':
        model = models.segmentation.fcn_resnet50(pretrained=True)
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    
    elif model_type == 'deeplabv3_resnet50':
        model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    else:
        raise Exception(f"No model named: {model_type}")
        
    return model