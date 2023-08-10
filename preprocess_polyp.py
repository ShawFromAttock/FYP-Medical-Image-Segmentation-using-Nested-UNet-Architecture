import os
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

def main():
    img_size = 256

    image_dir = 'inputs\\images'
    mask_dir = 'inputs\\masks'
    os.makedirs('inputs\\polyp_%d\\images' % img_size, exist_ok=True)
    os.makedirs('inputs\\polyp_%d\\masks\\0' % img_size, exist_ok=True)

    image_paths = glob(os.path.join(image_dir, '*.png'))

    for image_path in tqdm(image_paths):
        img = cv2.imread(image_path)

        mask = np.zeros((img.shape[0], img.shape[1]))
        mask_paths = os.path.join(mask_dir, os.path.basename(image_path))

        mask_ = cv2.imread(mask_paths, cv2.IMREAD_GRAYSCALE) > 127
        mask[mask_] = 1

        if len(img.shape) == 2:
            img = np.tile(img[..., None], (1, 1, 3))
        if img.shape[2] == 4:
            img = img[..., :3]

        img = cv2.resize(img, (img_size, img_size))
        mask = cv2.resize(mask, (img_size, img_size))

        cv2.imwrite(os.path.join('inputs\\dsb2018_%d\\images' % img_size, os.path.basename(image_path)), img)
        cv2.imwrite(os.path.join('inputs\\dsb2018_%d\\masks\\0' % img_size, os.path.basename(image_path)), (mask * 255).astype('uint8'))


if __name__ == '__main__':
    main()
