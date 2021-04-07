import json
import argparse
import os
import datetime
from pathlib import Path
from lib import trainManager
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Attention_OOD')
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--model_arch', type=str, default='deit_small_patch16_224',
    #                     choices=['deit_tiny_distilled_patch16_224',
    #                              'deit_tiny_patch16_224', 'deit_small_patch16_224'])
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--drop_path', type=float, default=0.1)
    parser.add_argument('--averagemeter', type=str, default='Meter_ood')
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
    parser.add_argument('--val_file', type=str, default='data/dermnet/dermnet_val_0.txt')
    parser.add_argument('--file_prefix', type=str, default='/home/xuanli/Data/DermNet/DermNet')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'], default='cuda')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--out_dir', type=str, default='output')
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--tqdm_minintervals', type=float, default=2.0)
    parser.add_argument('--resume', type=str, default='output/04_01_19_38_05_deit_tiny_patch16_224_dermnet_train_0_0/models/best_checkpoint.pth', help='path to the pth/tar checkpoints')
    parser.add_argument('--reset_epoch', action='store_true')

    args = parser.parse_args()

    assert args.resume != ''
    parent_folder = Path(args.resume).parent.parent
    args.out_dir = os.path.join(parent_folder, 'ood')
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if 'isic' in args.resume:
        ind = 'isic'
    else:
        assert 'dermnet' in args.resume
        ind = 'dermnet'
    args.model_arch = parent_folder.stem[(parent_folder.stem).index('deit'): (parent_folder.stem).index(ind)-1]

    return args

# def modify_ood(args, ood, ood_root='/data/xuanli/OOD'):
#     args.val_file = f'data/{ood}/{ood}_ood.txt'
#     args.file_prefix = os.path.join(ood_root, ood)


if __name__ == '__main__':
    args = parse_args()
    print(args.__dict__)
    # For InD
    train_manager = trainManager(args)
    train_manager.fit_ood()

    # For OOD
    args.val_file = 'data/dermnet/dermnet_unseen_0.txt'
    train_manager.set_data_loader(args, 'val', shuffle=False)
    train_manager.fit_ood()


