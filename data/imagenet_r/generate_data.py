import glob
from PIL import Image
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def process_folder(cur_dir):
    imgs = glob.glob(os.path.join(cur_dir, '*.JPEG'))
    for i in range(2):
        img = Image.open(imgs[i]).convert('RGB')
        img = img.resize((32, 32))
        img.save(f'{file_path_1.replace("train","imagenet_r")}/{Path(imgs[i]).parent.stem}_{Path(imgs[i]).name}')


def step_1_resize_image_and_save():
    folders = [os.path.join(file_path_1, p) for p in os.listdir(file_path_1) if os.path.isdir(os.path.join(file_path_1, p))]
    with ThreadPoolExecutor(20) as exe:
        res = [exe.submit(process_folder, f) for f in folders]
    for r in res:
        print(r.result())


def step_2_generate_txt():
    imgs = glob.glob(os.path.join(file_path_2, '*.JPEG'))
    with open('imagenet_r_ood.txt', 'w') as f:
        for img in imgs:
            f.write(f'{Path(img).name},0\n')


if __name__ == '__main__':
    file_path_1 = '/data/xuanli/ImageNet/train'
    # step_1_resize_image_and_save()

    file_path_2 = '/data/xuanli/ImageNet/imagenet_r'
    step_2_generate_txt()

