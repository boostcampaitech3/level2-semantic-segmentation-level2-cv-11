import torch
import argparse

class Args(object):
    parser = argparse.ArgumentParser(description='Arguments for Mask Classification train/test')
    parser.add_argument('--model', default='fcn_resnet50', help='[\'fcn_resnet50\', \'deeplabv3_resnet50\']')
    parser.add_argument('--resize', type=int, nargs="+", default=(512, 512), help='Resize input image')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.000001, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=30, help='Max epoch')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for data split')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--save_dir', default='./saved', help='Directory to save model')
    parser.add_argument('--pretrained_path', default=None, help='Pre-trained model path') # Train-> Fine tuning, Test-> Inference
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--train_json_path', default='/opt/ml/input/data/train.json')
    parser.add_argument('--valid_json_path', default='/opt/ml/input/data/val.json')
    parser.add_argument('--test_json_path', default='/opt/ml/input/data/test.json')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--wandb_plot', type=lambda s: s.lower() in ['true', '1'], default=False)


    parse = parser.parse_args()
    params = {
        "MODEL": parse.model, 
        "RESIZE": parse.resize, 
        "LEARNING_RATE": parse.learning_rate,
        "WEIGHT_DECAY": parse.weight_decay, 
        "NUM_EPOCHS": parse.num_epochs,
        "BATCH_SIZE": parse.batch_size,
        "NUM_WORKERS": parse.num_workers,
        "RANDOM_SEED": parse.random_seed,
        "DEVICE": parse.device,
        "SAVE_DIR": parse.save_dir, 
        "PRETRAINED_PATH": parse.pretrained_path,
        "NUM_CLASSES": parse.num_classes,
        "VAL_EVERY": parse.val_every,
        "TRAIN_JSON_PATH": parse.train_json_path,
        "VALID_JSON_PATH": parse.valid_json_path,
        "TEST_JSON_PATH": parse.test_json_path,
        "MODE": parse.mode,
        "WANDB_PLOT": parse.wandb_plot
        
    }