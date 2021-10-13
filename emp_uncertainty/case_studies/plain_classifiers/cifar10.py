import math
import os
from typing import Union, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import uncertainty_wizard as uwiz
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD

from emp_uncertainty.case_studies.case_study import ClassificationCaseStudy

CIFAR10_CORRUPTION_TYPES = [
    'brightness',
    'contrast',
    'defocus_blur',
    'elastic',
    'fog',
    'frost',
    'frosted_glass_blur',
    'gaussian_blur',
    'gaussian_noise',
    'impulse_noise',
    'jpeg_compression',
    'motion_blur',
    'pixelate',
    'saturate',
    'shot_noise',
    'snow',
    'spatter',
    'speckle_noise',
    'zoom_blur',
]


def build_sequential(x):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
                                     input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    _compile(model)

    return model, None


def _compile(model):
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


class Cifar10(ClassificationCaseStudy):
    """
    This model was taken from
    https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
    (the one with dropout regularization)
    """
    y_val = None

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_val_labels(cls) -> np.ndarray:
        if cls.y_val is None:
            cls._get_train_and_val_data()
        return cls.y_val

    @classmethod
    def _case_study_id(cls) -> str:
        return "cifar10"

    @classmethod
    def _get_train_and_val_data(cls, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train / 255.
        y_train = y_train.flatten()
        x_t, x_val, y_t, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)
        y_t = tf.keras.backend.one_hot(y_t, 10).numpy()
        cls.y_val = y_val
        return x_t, y_t, x_val, y_val

    @classmethod
    def get_test_data(cls) -> Tuple[Union[np.ndarray, tf.data.Dataset], np.ndarray]:
        (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_test = x_test / 255.
        return x_test, y_test

    @classmethod
    def get_outlier_data(cls, severity: str) -> Tuple[Union[np.ndarray, tf.data.Dataset], np.ndarray]:
        img_per_corr = math.floor(10000 / len(CIFAR10_CORRUPTION_TYPES))
        all_images = None
        for i, corr_type in enumerate(CIFAR10_CORRUPTION_TYPES):
            ds = tfds.load(f'cifar10_corrupted/{corr_type}_{severity}',
                           split=tfds.core.ReadInstruction('test',
                                                           from_=i * img_per_corr,
                                                           to=(i + 1) * img_per_corr,
                                                           unit='abs'))
            if all_images is None:
                all_images = ds
            else:
                all_images.concatenate(ds)
        x = []
        y = []
        as_numpy = tfds.as_numpy(all_images)
        for i in as_numpy:
            x.append(i["image"])
            y.append(i["label"])
        x = np.array(x) / 255.
        y = np.array(y)
        return x, y

    @classmethod
    def _create_stochastic_model(cls) -> uwiz.models.StochasticSequential:
        keras_model, _ = build_sequential(x=None)
        model = uwiz.models.stochastic_from_keras(keras_model)
        _compile(model)
        return model

    @classmethod
    def _create_ensemble_model(cls) -> Union[None, uwiz.models.LazyEnsemble]:
        model = uwiz.models.LazyEnsemble(num_models=50, model_save_path=cls.ensemble_save_path(), delete_existing=True)
        model.create(create_function=build_sequential,
                     # No gpu load; use more processes for speedup (limiting factor: num cores)
                     num_processes=cls.num_ensemble_processes() * 3)
        return model

    @classmethod
    def _create_sample_bayesian_nn(cls) -> uwiz.models.StochasticSequential:
        # Code based on
        #  - https://github.com/tensorflow/probability/blob/main/tensorflow_probability/examples/bayesian_neural_network.py
        #  - https://keras.io/examples/keras_recipes/bayesian_neural_networks/
        #
        # Architecture based on the same as the regular stochastic model

        kl_divergence_function = (
            lambda q, p, _: tfp.distributions.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                            tf.cast(54000, dtype=tf.float32))

        model = uwiz.models.StochasticSequential()
        model.add(tfp.layers.Convolution2DFlipout(32, kernel_size=(3, 3), activation='relu', padding='same',
                                                  kernel_divergence_fn=kl_divergence_function,
                                                  input_shape=(32, 32, 3)))
        model.add(tfp.layers.Convolution2DFlipout(32, kernel_size=(3, 3), activation='relu', padding='same',
                                                  kernel_divergence_fn=kl_divergence_function))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tfp.layers.Convolution2DFlipout(64, kernel_size=(3, 3), activation='relu', padding='same',
                                                  kernel_divergence_fn=kl_divergence_function))
        model.add(tfp.layers.Convolution2DFlipout(64, kernel_size=(3, 3), activation='relu', padding='same',
                                                  kernel_divergence_fn=kl_divergence_function))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tfp.layers.Convolution2DFlipout(128, kernel_size=(3, 3), activation='relu', padding='same',
                                                  kernel_divergence_fn=kl_divergence_function))
        model.add(tfp.layers.Convolution2DFlipout(128, kernel_size=(3, 3), activation='relu', padding='same',
                                                  kernel_divergence_fn=kl_divergence_function))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tfp.layers.DenseFlipout(128, activation='relu',
                                          kernel_divergence_fn=kl_divergence_function))
        model.add(tfp.layers.DenseFlipout(10, activation='softmax',
                                          kernel_divergence_fn=kl_divergence_function))

        _compile(model)
        return model

    @classmethod
    def get_batch_size(cls) -> int:
        return 64

    @classmethod
    def num_ensemble_processes(cls):
        return 3


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    study = Cifar10()
    study.clear_nn_outputs_folders()
    study.clear_db_results()
    study.train_with_epoch_eval(num_epochs=200, ood_severities=["1", "2", "3", "4", "5"])
    for i in range(200):
        study.run_quantifiers(i, ood_severities=["1", "2", "3", "4", "5"])
