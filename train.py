import json
import argparse
import os
import datetime
from pathlib import Path
from lib import trainManager
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Attention_OOD')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'debug'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model_arch', type=str, default='deit_small_patch16_224', choices=['deit_tiny_distilled_patch16_224',
                                                                                            'deit_tiny_patch16_224', 'deit_small_patch16_224'])
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--drop_path', type=float, default=0.1)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--averagemeter', type=str, default='Meter_cls')
    parser.add_argument('--loss', type=str, default='CELoss', choices=['CELoss', 'FocalLoss', 'FGLoss'])
    parser.add_argument('--distillation_type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation_alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation_tau', default=1.0, type=float, help="")
    parser.add_argument('--teacher_path', type=str, default='/home/xuanli/OOD_compression/output/'
                                                            '04_03_17_16_41_resnet34_dermnet_train_0_0/'
                                                            'models/best_checkpoint.pth')
    parser.add_argument('--num_classes', type=int, default=23)
    parser.add_argument('--metric', type=str, default='accuracy')
    parser.add_argument('--img_h', type=int, default=224)
    parser.add_argument('--img_w', type=int, default=224)
    parser.add_argument('--train_file', type=str, default='data/dermnet/dermnet_train_0.txt')
    parser.add_argument('--val_file', type=str, default='data/dermnet/dermnet_val_0.txt')
    parser.add_argument('--file_prefix', type=str, default='/home/xuanli/Data/DermNet/DermNet')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--scheduler', type=str, default='Step')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--step_size', type=int, default=60)
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate decay multiplier')
    parser.add_argument('--lr_scheduler', type=str, choices=['StepLR'], default='StepLR')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'], default='cuda')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default='output')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--tqdm_minintervals', type=float, default=2.0)
    parser.add_argument('--resume', type=str, default='', help='path to the pth/tar checkpoints')
    parser.add_argument('--reset_epoch', action='store_true')

    args = parser.parse_args()
    time = datetime.datetime.now().strftime('%m_%d_%H_%M_%S') + \
           '_{}_{}_{}_{}'.format(args.model_arch, Path(args.train_file).stem, args.distillation_type, args.seed)

    # Make root directory for all outupts
    if args.resume != '':
        print('Resuming ...')
#        args.out_dir = os.path.join(args.out_dir, args.resume)
#        args.resume =  args.resume#os.path.join(".", args.model_dir, 'checkpoint.pth')
    elif args.mode == 'train':
        if not os.path.exists(os.path.join(args.out_dir, time)):
            os.makedirs(os.path.join(args.out_dir, time))
        args.out_dir = os.path.join(args.out_dir, time)
        if not os.path.exists(os.path.join(args.out_dir, args.model_dir)):
            os.makedirs(os.path.join(args.out_dir, args.model_dir))

        with open(os.path.join(args.out_dir, 'config.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    elif args.mode == 'debug':
        args.num_workers = 0
        args.device = args.device if torch.cuda.is_available() else 'cpu'
    elif args.mode == 'eval':
        pass

    return args


if __name__ == '__main__':
    args = parse_args()
    print(args.__dict__)
    train_manager = trainManager(args)
    train_manager.fit()

