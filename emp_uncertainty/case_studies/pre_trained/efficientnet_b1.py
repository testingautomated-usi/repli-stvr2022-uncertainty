import uncertainty_wizard as uwiz
import tensorflow as tf

from emp_uncertainty import gpu_config
from emp_uncertainty.case_studies.pre_trained.imagenet_pretrained_cs import ImagenetPretrainedCaseStudy


class EfficientNetB1(ImagenetPretrainedCaseStudy):

    @classmethod
    def get_batch_size(cls) -> int:
        return 64

    @classmethod
    def num_ensemble_processes(cls):
        return 1

    @classmethod
    def _case_study_id(cls) -> str:
        return "efficientnet_b1"

    @classmethod
    def _create_stochastic_model(cls) -> uwiz.models.StochasticSequential:
        keras_model = tf.keras.applications.EfficientNetB1(
            include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
            pooling=None, classes=1000, classifier_activation='softmax')

        return uwiz.models.stochastic_from_keras(keras_model)


if __name__ == '__main__':
    gpu_config.use_only_gpu(1)
    study = EfficientNetB1()
    # study.load_imagenet_model()
    # study.clear_nn_outputs_folders()
    study.clear_db_results()
    # study.run_imgnet_inference()
    study.run_quantifiers(epoch=None, ood_severities=["1", "2", "3", "4", "5"])
