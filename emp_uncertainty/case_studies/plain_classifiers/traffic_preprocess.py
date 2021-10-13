import glob
import math
import os
import random
import re

from skimage import color, exposure, transform, io
import numpy as np

from emp_uncertainty.case_studies.image_corruptor import random_corruption
from emp_uncertainty.case_studies.plain_classifiers.traffic import ds_path


def preprocess_img(input):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(input)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)
    # rescale to standard size
    img = transform.resize(img, (48, 48))
    # roll color axis to axis 0
    img = np.rollaxis(img, -1)
    return img


def create_splits_on_fs():
    # Read all image paths with extension ppm
    paths = dict()
    training_folder_img_paths = glob.glob(ds_path('Training/*/*.ppm'))
    random.Random(4).shuffle(training_folder_img_paths)
    num_val_samples = math.floor(len(training_folder_img_paths) * 0.2)
    paths['val'] = training_folder_img_paths[:num_val_samples]
    paths['train'] = training_folder_img_paths[num_val_samples:]

    test_paths = glob.glob(ds_path('Testing/*/*.ppm'))
    random.Random(5).shuffle(test_paths)
    paths['test'] = test_paths[:1000]

    save_preprocessed_images('train', paths['train'], save_individually=True, transform=lambda x: x)
    save_preprocessed_images('val', paths['val'], save_individually=False, transform=lambda x: x)
    save_preprocessed_images('test', paths['test'], save_individually=False, transform=lambda x: x)
    for severity in range(1, 6):
        save_preprocessed_images(f"ood_{severity}", paths['test'], save_individually=False,
                                 transform=lambda x: random_corruption(np.copy(x), severity=severity))


def get_class(img_path):
    split_path = re.split('[/\\\\]', img_path)
    return int(split_path[-2]) - 1


def save_preprocessed_images(name, image_paths, save_individually: bool, transform):
    # Only used if not save_individually
    if save_individually:
        os.makedirs(ds_path(f"preprocessed/{name}"))
    all_imgs = []
    all_labels = []
    for img_id, img_path in enumerate(image_paths):
        img = preprocess_img(io.imread(img_path))
        img = np.moveaxis(img, 0, -1)
        assert np.all(img < 1.00000001) and np.all(img > -0.0000000001)
        img = img.clip(0, 1)
        img = transform(img)
        label = get_class(img_path)
        if save_individually:
            np.save(ds_path(f"preprocessed/{name}/img-{img_id}-label-{label}.npy"), img, allow_pickle=False)
        else:
            all_imgs.append(img)
            all_labels.append(label)

    if not save_individually:
        np.save(ds_path(f"preprocessed/{name}-imgs.npy"), np.array(all_imgs), allow_pickle=False)
        np.save(ds_path(f"preprocessed/{name}-labels.npy"), np.array(all_labels), allow_pickle=False)


if __name__ == '__main__':
    create_splits_on_fs()
