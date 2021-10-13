import math
from typing import Union, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import uncertainty_wizard as uwiz
from sklearn.model_selection import train_test_split

from emp_uncertainty.case_studies.case_study import ClassificationCaseStudy


def build_sequential(x):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                                     input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    _compile(model)
    return model, None


def _compile(model):
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adadelta(),
                  metrics=['accuracy'])


MNIST_CORRUPTION_TYPES = [
    "shot_noise",
    "impulse_noise",
    "glass_blur",
    "motion_blur",
    "shear",
    "scale",
    "rotate",
    "brightness",
    "translate",
    "stripe",
    "fog",
    "spatter",
    "dotted_line",
    "zigzag",
    "canny_edges",
]


class Mnist(ClassificationCaseStudy):
    """
    Model taken from https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    """

    y_val = None

    @classmethod
    def get_val_labels(cls) -> np.ndarray:
        if cls.y_val is None:
            cls._get_train_and_val_data()
        return cls.y_val

    @classmethod
    def get_outlier_data(cls, severity: str) -> Tuple[Union[np.ndarray, tf.data.Dataset], np.ndarray]:
        assert severity == "noseverity"
        img_per_corr = math.floor(10000 / len(MNIST_CORRUPTION_TYPES))
        all_images = None
        for i, corr_type in enumerate(MNIST_CORRUPTION_TYPES):
            ds = tfds.load(f'mnist_corrupted/{corr_type}',
                           split=tfds.core.ReadInstruction('test',
                                                           from_=i * img_per_corr,
                                                           to=(i + 1) * img_per_corr,
                                                           unit='abs'))
            if all_images is None:
                all_images = ds
            else:
                all_images = all_images.concatenate(ds)
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
    def _case_study_id(cls) -> str:
        return "mnist"

    @classmethod
    def _get_train_and_val_data(cls, seed=42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
        x_train = tf.expand_dims(x_train / 255., 3).numpy()
        x_t, x_val, y_t, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)
        cls.y_val = y_val
        y_t = tf.keras.backend.one_hot(y_t, 10).numpy()
        return x_t, y_t, x_val, y_val

    @classmethod
    def get_test_data(cls) -> Tuple[Union[np.ndarray, tf.data.Dataset], np.ndarray]:
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test = tf.expand_dims(x_test / 255., 3).numpy()
        return x_test, y_test

    @classmethod
    def _create_stochastic_model(cls) -> uwiz.models.StochasticSequential:
        keras_model, _ = build_sequential(x=None)
        model = uwiz.models.stochastic_from_keras(keras_model)
        _compile(model)
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

        model = uwiz.models.StochasticSequential([
            tfp.layers.Convolution2DFlipout(
                32, kernel_size=(3, 3), padding='SAME',
                kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.relu),
            tfp.layers.Convolution2DFlipout(
                64, kernel_size=(3, 3), padding='SAME',
                kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(
                pool_size=[2, 2], padding='SAME'),
            tf.keras.layers.Flatten(),
            tfp.layers.DenseFlipout(
                128, kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.relu),
            tfp.layers.DenseFlipout(
                10, kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.softmax)
        ])
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adadelta(),
                      metrics=['accuracy'],
                      expect_deterministic=True)
        return model

    @classmethod
    def _create_ensemble_model(cls) -> Union[None, uwiz.models.LazyEnsemble]:
        model = uwiz.models.LazyEnsemble(num_models=50, model_save_path=cls.ensemble_save_path(), delete_existing=True)
        model.create(create_function=build_sequential,
                     num_processes=15)
        return model

    @classmethod
    def get_batch_size(cls) -> int:
        return 64

    @classmethod
    def num_ensemble_processes(cls):
        return 3


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    study = Mnist()
    study.clear_nn_outputs_folders()
    study.clear_db_results()
    study.train_with_epoch_eval(num_epochs=200, ood_severities=["noseverity"])
    for e in range(200):
        study.run_quantifiers(epoch=e, ood_severities=["noseverity"])
