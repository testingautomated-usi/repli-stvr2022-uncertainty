import abc
import math
import os
from concurrent.futures.process import ProcessPoolExecutor
from typing import Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from emp_uncertainty.case_studies.case_study import ClassificationCaseStudy

IMAGENET_CORRUPTION_TYPES = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'glass_blur',
    # 'motion_blur', # TODO Requires ImageMagick
    'zoom_blur',
    # 'snow', # TODO Requires ImageMagick
    'frost',
    'fog',
    'brightness',
    'contrast',
    'elastic_transform',
    'pixelate',
    'jpeg_compression',
    'gaussian_blur',
    'saturate',
    # 'spatter', # TODO Requires cv2.cv2
    'speckle_noise'
]


class ImagenetPretrainedCaseStudy(ClassificationCaseStudy, abc.ABC):

    y_test = None

    def __init__(self) -> None:
        super().__init__()


    @classmethod
    def _get_train_and_val_data(cls) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        raise RuntimeError("For pretrained Imagenet Studies, the training data should never be requested.")

    @classmethod
    def get_test_data(cls) -> Tuple[Union[np.ndarray, tf.data.Dataset], np.ndarray]:
        # Cite
        # Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh,
        # Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein,
        # Alexander C. Berg and Li Fei-Fei. (* = equal contribution)
        # ImageNet Large Scale Visual Recognition Challenge. arXiv:1409.0575, 2014.

        # Test set not publicly available. We use val set. 2 % of 50'000 images => 1000 images
        imagenet_val = tfds.load('imagenet2012', split='validation[:2%]', shuffle_files=False, as_supervised=True)
        x_test_resized, y_test = cls._preprocess_tfds(imagenet_val)
        return x_test_resized, y_test

    @classmethod
    def _preprocess_tfds(cls, imagenet_val):
        x_test = imagenet_val.map(lambda x, _: x).prefetch(tf.data.experimental.AUTOTUNE)
        x_test_resized = x_test.map(lambda x: tf.image.resize(x, [224, 224], method='bicubic'),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        x_test_resized = x_test_resized.prefetch(tf.data.experimental.AUTOTUNE)
        y_test_iter = imagenet_val.map(lambda _, y: y).as_numpy_iterator()
        y_test = np.fromiter(y_test_iter, dtype=int)
        return x_test_resized, y_test

    @classmethod
    def get_val_data(cls) -> Tuple[Union[np.ndarray, tf.data.Dataset], np.ndarray]:
        imagenet_val = tfds.load('imagenet2012', split='validation[98%:]', shuffle_files=False, as_supervised=True)
        x_test_resized, y_test = cls._preprocess_tfds(imagenet_val)
        cls.y_test = y_test
        return x_test_resized, y_test

    @classmethod
    def get_val_labels(cls) -> np.ndarray:
        if cls.y_test is None:
            cls.get_val_data()
        return cls.y_test

    @classmethod
    def get_outlier_data(cls, severity: str) -> Tuple[Union[np.ndarray, tf.data.Dataset], np.ndarray]:
        img_per_corr = math.floor(1000 / len(IMAGENET_CORRUPTION_TYPES))
        all_images = None
        for i, corr_type in enumerate(IMAGENET_CORRUPTION_TYPES):
            ds = tfds.load(f'imagenet2012_corrupted/{corr_type}_{severity}',
                           split=tfds.core.ReadInstruction('validation',
                                                           from_=i * img_per_corr,
                                                           to=(i + 1) * img_per_corr,
                                                           unit='abs'),
                           as_supervised=True)
            if all_images is None:
                all_images = ds
            else:
                all_images = all_images.concatenate(ds)
        x, y = cls._preprocess_tfds(all_images)
        return x, y

    def load_imagenet_model(self):
        self.ensemble_model = None
        self.stochastic_model = self._create_stochastic_model()

    @classmethod
    def _create_ensemble_model(cls) -> None:
        return None

    def run_imgnet_inference(self):
        val_set = self.get_val_data()
        self.run_nn_inference(epoch=None, ood_severities=["1", "2", "3", "4", "5"], val_dataset=val_set)


def download_and_prepare(c, s):
    tfds.builder(f"imagenet2012_corrupted/{c}_{s}").download_and_prepare()


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    futures = []
    with ProcessPoolExecutor(max_workers=20) as executor:
        for severity in ["1", "2", "3", "4", "5"]:
            for c_type in IMAGENET_CORRUPTION_TYPES:
                future = executor.submit(download_and_prepare, c_type, severity)
                futures.append(future)
        [future.result() for future in futures]
    print("Done")
