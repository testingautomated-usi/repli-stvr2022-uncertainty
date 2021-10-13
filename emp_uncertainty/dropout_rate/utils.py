import os
import pickle

import numpy as np
import tensorflow as tf
import uncertainty_wizard as uwiz

from emp_uncertainty.case_studies.case_study import ClassificationCaseStudy
from emp_uncertainty.utils.identity_quantifier import SamplingIdentity


def rate_from_model_id(model_id):
    return (model_id % 9 + 1) / 10


def pred_identity(model, x):
    _, samples = model.predict_quantified(x=x,
                                          quantifier=SamplingIdentity(),
                                          sample_size=200,
                                          batch_size=128,
                                          verbose=1)
    return samples


def save_quantifications(model_id, sources, outputs_folder, case_study):
    dropout_rate = rate_from_model_id(model_id)

    for y, src in sources:
        x = np.load(f"{outputs_folder}/{src}/{dropout_rate}-m{model_id}.npy", allow_pickle=True)
        for q_name in ["var_ratio", "pred_entropy", "mutu_info", "mean_softmax"]:
            thresholds = ClassificationCaseStudy.calculate_thresholds(
                y_labels=y.flatten(),
                model_type="stochastic",
                nn_outputs=x
            )
            res = ClassificationCaseStudy.eval_classification_quantifier(
                quantifier_name="mean_sm" if q_name == "mean_softmax" else q_name,
                quantifier=uwiz.quantifiers.quantifier_registry.QuantifierRegistry.find(q_name),
                src=src,
                true_labels=y,
                nn_outputs=x,
                model_type=f"stochastic-with-d={dropout_rate}",
                thresholds=thresholds,
                case_study_id=case_study,
                epochs=None,
                sample_size=None,
            )
            with open(f"{outputs_folder}/{src}/{dropout_rate}-{q_name}-m{model_id}.pickle", 'wb') as f:
                pickle.dump(res, f)


class CpuOnlyContext(uwiz.models.ensemble_utils.EnsembleContextManager):

    # docstr-coverage:inherited
    def __enter__(self) -> "DynamicGpuGrowthContextManager":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            # Disable all GPUS
            tf.config.set_visible_devices([], "GPU")
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != "GPU"
        except RuntimeError as e:
            raise ValueError(
                f"Uncertainty Wizard was unable to disable gpu use."
            ) from e
        return self
