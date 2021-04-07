import glob
from PIL import Image
import os
from pathlib import Path
import random
import numpy as np
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor


def process_folder(cur_dir):
    imgs = glob.glob(os.path.join(cur_dir, '*.JPEG'))
    for i in range(2):
        img = Image.open(imgs[i]).convert('RGB')
        img = transforms.RandomCrop(32)(img)
        img.save(f'{file_path_1.replace("train","imagenet_c")}/{Path(imgs[i]).parent.stem}_{Path(imgs[i]).name}')


def step_1_crop_image_and_save():
    folders = [os.path.join(file_path_1, p) for p in os.listdir(file_path_1) if os.path.isdir(os.path.join(file_path_1, p))]
    with ThreadPoolExecutor(20) as exe:
        res = [exe.submit(process_folder, f) for f in folders]
    for r in res:
        print(r.result())


def step_2_generate_txt():
    imgs = glob.glob(os.path.join(file_path_2, '*.JPEG'))
    with open('imagenet_c_ood.txt', 'w') as f:
        for img in imgs:
            f.write(f'{Path(img).name},0\n')


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    file_path_1 = '/data/xuanli/ImageNet/train'
    # step_1_crop_image_and_save()

    file_path_2 = '/data/xuanli/ImageNet/imagenet_c'
    step_2_generate_txt()

