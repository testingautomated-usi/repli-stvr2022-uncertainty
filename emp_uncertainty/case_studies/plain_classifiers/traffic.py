import glob
import os
import random
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Union, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import uncertainty_wizard as uwiz

from emp_uncertainty.case_studies import case_study
from emp_uncertainty.case_studies.case_study import ClassificationCaseStudy

NUM_CLASSES = 164

ROOT_DIR: str = "/root/manual_datasets/traffic_signs"


# ROOT_DIR: str = "C:\\Users\\41763\\traffic_signs"


def get_training_set():
    def gen():
        paths = glob.glob(ds_path('preprocessed/train/*.npy'))
        random.shuffle(paths)
        for img_path in paths:
            label = img_path[:-4].split("-")[-1]
            label = int(label) - 1
            img = np.load(img_path, allow_pickle=False)
            yield (img, label)

    ds = tf.data.Dataset.from_generator(
        gen,
        (tf.float32, tf.int64),
        (tf.TensorShape([48, 48, 3]), tf.TensorShape([])))
    ds = ds.map(lambda x, y: (x, tf.one_hot(y, NUM_CLASSES)))
    ds = ds.batch(Traffic.get_batch_size())
    return ds.prefetch(200)


def train_epoch(model_id, model):
    dataset = get_training_set()
    print(f"Fitting model {model_id} for one epoch...")
    model.fit(dataset, epochs=1, verbose=1)
    print(f"done fitting model {model_id} for one epoch.")
    return model, "history_not_returned"


def ds_path(inner_path):
    return os.path.join(ROOT_DIR, inner_path)


def build_sequential(x):
    model = tf.keras.models.Sequential()
    """ This model uses the structure (but not the original code!) of Serna et al.
    https://github.com/citlag/European-Traffic-Sings/blob/master/models/cnn_8-layers/cnn_model.ipynb
    """
    model.add(tf.keras.layers.InputLayer([48, 48, 3]))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

    _compile(model)

    return model, None


def _compile(model):
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])


class Traffic(ClassificationCaseStudy):
    y_val = None

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def _get_train_and_val_data(cls) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        val_x = np.load(ds_path(f"preprocessed/val-imgs.npy"), allow_pickle=False)
        val_y = np.load(ds_path(f"preprocessed/val-labels.npy"), allow_pickle=False)
        val_y = val_y - 1
        cls.y_val = val_y
        return None, None, val_x, val_y

    @classmethod
    def get_val_labels(cls) -> np.ndarray:
        if cls.y_val is None:
            cls._get_train_and_val_data()
        return cls.y_val

    @classmethod
    def _case_study_id(cls) -> str:
        return "traffic"

    @classmethod
    def _create_stochastic_model(cls) -> uwiz.models.StochasticSequential:
        keras_model, _ = build_sequential(None)
        model = uwiz.models.stochastic_from_keras(keras_model)
        _compile(model)
        return model

    @classmethod
    def _create_ensemble_model(cls) -> Union[None, uwiz.models.LazyEnsemble]:
        model = uwiz.models.LazyEnsemble(num_models=50, model_save_path=cls.ensemble_save_path(), delete_existing=True)
        model.create(create_function=build_sequential,
                     num_processes=20)
        return model

    def train_with_epoch_eval(self, num_epochs: int, ood_severities):
        if case_study.TRAIN_DROPOUT:
            self.stochastic_model = self._create_stochastic_model()
        if case_study.TRAIN_ENSEMBLE:
            self.ensemble_model = self._create_ensemble_model()
        if case_study.TRAIN_FLIPOUT:
            self.sample_bayesian_nn = self._create_sample_bayesian_nn()

        _, _, val_x, val_y = self._get_train_and_val_data()

        for epoch in range(num_epochs):
            if case_study.TRAIN_DROPOUT:
                train_epoch("stochastic", self.stochastic_model)
                path = os.path.abspath(case_study.BASE_MODEL_SAVE_FOLDER + self._case_study_id() + "/stochastic/")
                self.stochastic_model.save(path, include_optimizer=True)

            if case_study.TRAIN_ENSEMBLE:
                self.ensemble_model.modify(
                    map_function=train_epoch,
                    num_processes=self.num_ensemble_processes()
                )

            if case_study.TRAIN_FLIPOUT:
                train_epoch("sampling_bnn", self.sample_bayesian_nn)
                if epoch == 199:
                    path = os.path.abspath(case_study.BASE_MODEL_SAVE_FOLDER + self._case_study_id() + "/sampling_bnn/")
                    self.sample_bayesian_nn.save(path, include_optimizer=True)

            self.run_nn_inference(epoch=epoch, ood_severities=ood_severities, val_dataset=(val_x, val_y))

    @classmethod
    def get_test_data(cls) -> Tuple[Union[np.ndarray, tf.data.Dataset], np.ndarray]:
        x = np.load(ds_path(f"preprocessed/test-imgs.npy"), allow_pickle=False)
        y = np.load(ds_path(f"preprocessed/test-labels.npy"), allow_pickle=False)
        y = y - 1
        return x, y

    @classmethod
    def get_outlier_data(cls, severity: str) -> Tuple[Union[np.ndarray, tf.data.Dataset], np.ndarray]:
        x = np.load(ds_path(f"preprocessed/ood_{severity}-imgs.npy"), allow_pickle=False)
        y = np.load(ds_path(f"preprocessed/ood_{severity}-labels.npy"), allow_pickle=False)
        y = y - 1
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

        kl_divergence_function = (lambda q, p, ignore: tfp.distributions.kl_divergence(q, p) /
                                                       tf.cast(37398, dtype=tf.float32))

        model = uwiz.models.StochasticSequential()
        model.add(tf.keras.layers.InputLayer([48, 48, 3]))
        model.add(tfp.layers.Convolution2DFlipout(32, (3, 3), padding='same', activation='relu',
                                                  kernel_divergence_fn=kl_divergence_function))
        model.add(tfp.layers.Convolution2DFlipout(32, (3, 3), activation='relu',
                                                  kernel_divergence_fn=kl_divergence_function))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tfp.layers.Convolution2DFlipout(64, (3, 3), padding='same', activation='relu',
                                                  kernel_divergence_fn=kl_divergence_function))
        model.add(tfp.layers.Convolution2DFlipout(64, (3, 3), activation='relu',
                                                  kernel_divergence_fn=kl_divergence_function))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tfp.layers.Convolution2DFlipout(128, (3, 3), padding='same', activation='relu',
                                                  kernel_divergence_fn=kl_divergence_function))
        model.add(tfp.layers.Convolution2DFlipout(128, (3, 3), activation='relu',
                                                  kernel_divergence_fn=kl_divergence_function))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tfp.layers.DenseFlipout(512, activation='relu',
                                          kernel_divergence_fn=kl_divergence_function))
        model.add(tfp.layers.DenseFlipout(NUM_CLASSES, activation='softmax',
                                          kernel_divergence_fn=kl_divergence_function))
        # Note: Changed learning rate for bnn
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.SGD(),
                      metrics=['accuracy'])
        return model

    @classmethod
    def get_batch_size(cls) -> int:
        return 64

    @classmethod
    def num_ensemble_processes(cls):
        return 8


if __name__ == '__main__':
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

    study = Traffic()
    # study.clear_nn_outputs_folders()
    study.clear_db_results()
    # study.train_with_epoch_eval(num_epochs=200, ood_severities=["1", "2", "3", "4", "5"])

    # for i in range(200):
    #     study.run_quantifiers(i, ood_severities=["1", "2", "3", "4", "5"])

    def quantify_epoch(epoch):
        try:
            Traffic().run_quantifiers(epoch, ood_severities=["1", "2", "3", "4", "5"])
        except Exception as exep:
            print(exep)
            print(exep.message)

    with ProcessPoolExecutor(max_workers=8) as executor:
        for i in range(200):
            executor.submit(quantify_epoch, i)

    executor.shutdown(wait=True)
