import os

import numpy as np
import tensorflow as tf
import uncertainty_wizard as uwiz

from emp_uncertainty.case_studies.case_study import BASE_MODEL_SAVE_FOLDER, BASE_OUTPUTS_SAVE_FOLDER
from emp_uncertainty.case_studies.plain_classifiers import cifar10
from emp_uncertainty.dropout_rate.utils import rate_from_model_id, pred_identity, save_quantifications, \
    CpuOnlyContext

MODEL_FOLDER = f"{BASE_MODEL_SAVE_FOLDER}/dropout-experiments/cifar"
OUTPUTS_FOLDER = f"{BASE_OUTPUTS_SAVE_FOLDER}/dropout-experiments/cifar"

OOD_SEVERITY = 3


def create_stochastic_model(model_id: int):
    dropout_rate = rate_from_model_id(model_id)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
                                     input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    x_train, y_train, x_val, y_val = cifar10.Cifar10._get_train_and_val_data(seed=model_id)
    y_val = tf.keras.backend.one_hot(y_val, 10).numpy()
    model.fit(x_train, y_train,
              batch_size=cifar10.Cifar10.get_batch_size(),
              epochs=200,
              validation_data=(x_val, y_val),
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])
    return model, None


def run_benchmark(model_id: int, model: tf.keras.Model):
    stochastic = uwiz.models.stochastic_from_keras(model, temp_weights_path=f"/tmp/model-{model_id}")
    dropout_rate = rate_from_model_id(model_id)

    x_test, _ = cifar10.Cifar10.get_test_data()
    r_nominal = pred_identity(stochastic, x_test)
    if not os.path.exists(f"{OUTPUTS_FOLDER}/nominal/"):
        os.makedirs(f"{OUTPUTS_FOLDER}/nominal/")
    np.save(f"{OUTPUTS_FOLDER}/nominal/{dropout_rate}-m{model_id}.npy", r_nominal)

    oods_x, _ = cifar10.Cifar10.oods_for_severities([OOD_SEVERITY])
    oods_x = oods_x[OOD_SEVERITY]

    r = pred_identity(stochastic, oods_x)
    if not os.path.exists(f"{OUTPUTS_FOLDER}/ood-{OOD_SEVERITY}"):
        os.makedirs(f"{OUTPUTS_FOLDER}/ood-{OOD_SEVERITY}")
    np.save(f"{OUTPUTS_FOLDER}/ood-{OOD_SEVERITY}/{dropout_rate}-m{model_id}.npy", r)


def quantify(model_id: int):
    _, truth_nom = cifar10.Cifar10.get_test_data()
    _, truth_ood = cifar10.Cifar10.oods_for_severities([OOD_SEVERITY])
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
    save_quantifications(model_id=model_id, sources=sources, outputs_folder=OUTPUTS_FOLDER, case_study="cifar10")


if __name__ == '__main__':
    ensemble = uwiz.models.LazyEnsemble(num_models=90,
                                        model_save_path=MODEL_FOLDER,
                                        delete_existing=False,
                                        expect_model=False,
                                        default_num_processes=3)
    ensemble.create(create_stochastic_model)
    ensemble.consume(run_benchmark)
    ensemble.run_model_free(quantify, context=CpuOnlyContext, num_processes=15)
