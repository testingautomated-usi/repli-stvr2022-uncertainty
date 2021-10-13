import os

import numpy as np
import tensorflow as tf
import uncertainty_wizard as uwiz

from emp_uncertainty.case_studies.case_study import BASE_MODEL_SAVE_FOLDER, BASE_OUTPUTS_SAVE_FOLDER
from emp_uncertainty.case_studies.plain_classifiers import mnist
from emp_uncertainty.dropout_rate.utils import rate_from_model_id, pred_identity, save_quantifications, \
    CpuOnlyContext

RUN_CONSTANT = 0
MODEL_FOLDER = f"{BASE_MODEL_SAVE_FOLDER}/dropout-experiments/mnist/{RUN_CONSTANT}"
OUTPUTS_FOLDER = f"{BASE_OUTPUTS_SAVE_FOLDER}/dropout-experiments/mnist/{RUN_CONSTANT}"

OOD_SEVERITY = "noseverity"


def create_stochastic_model(model_id: int):
    dropout_rate = rate_from_model_id(model_id)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                                     input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    x_train, y_train, x_val, y_val = mnist.Mnist._get_train_and_val_data(seed=RUN_CONSTANT * 1000 + model_id)
    y_val = tf.keras.backend.one_hot(y_val, 10).numpy()
    model.fit(x_train, y_train,
              batch_size=mnist.Mnist.get_batch_size(),
              epochs=150,
              validation_data=(x_val, y_val),
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])
    return model, None


def run_benchmark(model_id: int, model: tf.keras.Model):
    stochastic = uwiz.models.stochastic_from_keras(model, temp_weights_path=f"/tmp/model-{model_id}")
    dropout_rate = rate_from_model_id(model_id)

    x_test, _ = mnist.Mnist.get_test_data()
    r_nominal = pred_identity(stochastic, x_test)
    if not os.path.exists(f"{OUTPUTS_FOLDER}/nominal/"):
        os.makedirs(f"{OUTPUTS_FOLDER}/nominal/")
    np.save(f"{OUTPUTS_FOLDER}/nominal/{dropout_rate}-m{model_id}.npy", r_nominal)

    oods_x, _ = mnist.Mnist.oods_for_severities([OOD_SEVERITY])
    oods_x = oods_x[OOD_SEVERITY]

    r = pred_identity(stochastic, oods_x)
    if not os.path.exists(f"{OUTPUTS_FOLDER}/ood-{OOD_SEVERITY}"):
        os.makedirs(f"{OUTPUTS_FOLDER}/ood-{OOD_SEVERITY}")
    np.save(f"{OUTPUTS_FOLDER}/ood-{OOD_SEVERITY}/{dropout_rate}-m{model_id}.npy", r)


def quantify(model_id: int):
    _, truth_nom = mnist.Mnist.get_test_data()
    _, truth_ood = mnist.Mnist.oods_for_severities([OOD_SEVERITY])
    truth_ood = truth_ood[OOD_SEVERITY]
    sources = [
        (
            truth_nom,
            "nominal",
        ),
        (
            truth_ood,
            f"ood-{OOD_SEVERITY}",
        )
    ]
    save_quantifications(model_id=model_id, sources=sources, outputs_folder=OUTPUTS_FOLDER, case_study="mnist")


if __name__ == '__main__':
    ensemble = uwiz.models.LazyEnsemble(num_models=90,
                                        model_save_path=MODEL_FOLDER,
                                        delete_existing=False,
                                        expect_model=False,
                                        default_num_processes=3)
    # ensemble.create(create_stochastic_model)
    # ensemble.consume(run_benchmark)
    ensemble.run_model_free(quantify, context=CpuOnlyContext, num_processes=24)
