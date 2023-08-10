import os
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

def main():
    img_size = 512

    image_dir = 'inputs\\images'
    mask_dir = 'inputs\\masks'
    print(mask_dir)
    print(image_dir)

    os.makedirs('inputs\\retina_%d\\images' % img_size, exist_ok=True)
    os.makedirs('inputs\\retina_%d\\masks\\0' % img_size, exist_ok=True)

    image_paths = glob(os.path.join(image_dir, '*.tif'))

    for image_path in tqdm(image_paths):
        img = cv2.imread(image_path)
        img_basename = os.path.splitext(os.path.basename(image_path))[0][:2]
        print(img_basename)

        mask = np.zeros((img.shape[0], img.shape[1]))
        mask_paths = os.path.join(mask_dir, f'{img_basename}_manual1.gif')
        print("Mask path:", mask_paths)

        # Convert GIF mask to PNG
        converted_mask_path = os.path.join(mask_dir, f'{img_basename}.png')
        mask_image = Image.open(mask_paths)
        mask_image.save(converted_mask_path)

        mask_ = cv2.imread(converted_mask_path, cv2.IMREAD_GRAYSCALE) > 0
        mask[mask_] = 1

        if len(img.shape) == 2:
            img = np.tile(img[..., None], (1, 1, 3))
        if img.shape[2] == 4:
            img = img[..., :3]

        img = cv2.resize(img, (img_size, img_size))
        mask = cv2.resize(mask, (img_size, img_size))

        cv2.imwrite(os.path.join('inputs\\dsb2018_%d\\images' % img_size, img_basename + '.png'), img)
        cv2.imwrite(os.path.join('inputs\\dsb2018_%d\\masks\\0' % img_size, img_basename + '.png'), (mask * 255).astype('uint8'))


if __name__ == '__main__':
    main()
