import datetime
import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import uncertainty_wizard as uwiz

from emp_uncertainty.case_studies.case_study import ASSETS_PATH
from emp_uncertainty.case_studies.plain_classifiers.traffic import ds_path, get_training_set, NUM_CLASSES
from emp_uncertainty.traffic_bnn_search.settings import SETTINGS

uwiz.models.ensemble_utils.DynamicGpuGrowthContextManager.enable_dynamic_gpu_growth()

RE_RUN = "RERUN_9"


def _create_sample_bayesian_nn(num_conv_flipout: int,
                               num_dense_flipout: int) -> tf.keras.Model:
    assert num_conv_flipout in [0, 2, 4, 6]
    assert num_dense_flipout in [0, 1, 2]

    # Code based on
    #  - https://github.com/tensorflow/probability/blob/main/tensorflow_probability/examples/bayesian_neural_network.py
    #  - https://keras.io/examples/keras_recipes/bayesian_neural_networks/
    #
    # Architecture based on the same as the regular stochastic model

    kl_divergence_function = (lambda q, p, ignore: tfp.distributions.kl_divergence(q, p) /
                                                   tf.cast(37398, dtype=tf.float32))

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input([48, 48, 3]))
    if num_conv_flipout == 6:
        model.add(tfp.layers.Convolution2DFlipout(32, (3, 3), padding='same', activation='relu',
                                                  kernel_divergence_fn=kl_divergence_function))
        model.add(tfp.layers.Convolution2DFlipout(32, (3, 3), activation='relu',
                                                  kernel_divergence_fn=kl_divergence_function))
    else:
        model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    if num_conv_flipout >= 4:
        model.add(tfp.layers.Convolution2DFlipout(64, (3, 3), padding='same', activation='relu',
                                                  kernel_divergence_fn=kl_divergence_function))
        model.add(tfp.layers.Convolution2DFlipout(64, (3, 3), activation='relu',
                                                  kernel_divergence_fn=kl_divergence_function))
    else:
        model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    if num_conv_flipout >= 2:
        model.add(tfp.layers.Convolution2DFlipout(128, (3, 3), padding='same', activation='relu',
                                                  kernel_divergence_fn=kl_divergence_function))
        model.add(tfp.layers.Convolution2DFlipout(128, (3, 3), activation='relu',
                                                  kernel_divergence_fn=kl_divergence_function))
    else:
        model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())

    if num_dense_flipout == 2:
        model.add(tfp.layers.DenseFlipout(512, activation='relu',
                                          kernel_divergence_fn=kl_divergence_function))
    else:
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(512, activation='relu'))

    if num_dense_flipout >= 1:
        model.add(tfp.layers.DenseFlipout(NUM_CLASSES, activation='softmax',
                                          kernel_divergence_fn=kl_divergence_function))
    else:
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

    return model


def _compile(model,
             optimizer,
             learning_rate: float,
             momentum: float):
    if optimizer == "sgd":
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, decay=1e-6, momentum=momentum)
    elif optimizer == "RMSProp":
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, decay=1e-6, momentum=momentum)
    else:
        raise ValueError("Unknown optimizer " + optimizer)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])


def train(model_id: int):
    model: tf.keras.Model = _create_sample_bayesian_nn(num_conv_flipout=SETTINGS[model_id]["cf"],
                                                       num_dense_flipout=SETTINGS[model_id]["cd"])
    _compile(model,
             optimizer=SETTINGS[model_id]["opt"],
             learning_rate=SETTINGS[model_id]["lr"],
             momentum=SETTINGS[model_id]["mom"])

    training_set = get_training_set()
    val_x = np.load(ds_path(f"preprocessed/val-imgs.npy"), allow_pickle=False)
    val_y = np.load(ds_path(f"preprocessed/val-labels.npy"), allow_pickle=False)

    val_y_oh = np.zeros((val_y.shape[0], NUM_CLASSES))
    val_y_oh[np.arange(val_y.shape[0]), val_y - 1] = 1

    log_dir = f"{ASSETS_PATH}/traffic_bnn_grid/{RE_RUN}/tb_logs/fit/{model_id}/" + datetime.datetime.now().strftime(
        "%Y%m%d-%H")
    tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = None
    try:
        history = model.fit(training_set, epochs=100, validation_data=(val_x, val_y_oh), callbacks=[tb])
    except Exception as e:
        history_message = repr(e)

    # Save training history as to file (could get them through tensorboard, but the following is more convenient)
    if not os.path.exists(f"{ASSETS_PATH}/traffic_bnn_grid/{RE_RUN}/training_histories/"):
        os.makedirs(f"{ASSETS_PATH}/traffic_bnn_grid/{RE_RUN}/training_histories/")

    with open(f"{ASSETS_PATH}/traffic_bnn_grid/{RE_RUN}/training_histories/{model_id}-history.pickle", "wb") as f:
        if history:
            pickle.dump(history.history, f)
            del history
        else:
            pickle.dump(history_message, f)

    return model, None


if __name__ == '__main__':
    ensemble = uwiz.models.LazyEnsemble(num_models=len(SETTINGS),
                                        model_save_path=f"{ASSETS_PATH}/traffic_bnn_grid/{RE_RUN}/models/",
                                        default_num_processes=0)
    ensemble.create(train)
