from emp_uncertainty import gpu_config
from emp_uncertainty.case_studies.pre_trained.efficientnet_b0 import EfficientNetB0
from emp_uncertainty.case_studies.pre_trained.efficientnet_b1 import EfficientNetB1
from emp_uncertainty.case_studies.pre_trained.efficientnet_b2 import EfficientNetB2
from emp_uncertainty.case_studies.pre_trained.efficientnet_b3 import EfficientNetB3
from emp_uncertainty.case_studies.pre_trained.efficientnet_b4 import EfficientNetB4
from emp_uncertainty.case_studies.pre_trained.efficientnet_b5 import EfficientNetB5
from emp_uncertainty.case_studies.pre_trained.efficientnet_b6 import EfficientNetB6
from emp_uncertainty.case_studies.pre_trained.efficientnet_b7 import EfficientNetB7


def get_study(b):
    if b == 0:
        return EfficientNetB0()
    elif b == 1:
        return EfficientNetB1()
    elif b == 2:
        return EfficientNetB2()
    elif b == 3:
        return EfficientNetB3()
    elif b == 4:
        return EfficientNetB4()
    elif b == 5:
        return EfficientNetB5()
    elif b == 6:
        return EfficientNetB6()
    elif b == 7:
        return EfficientNetB7()


def run(b):
    study = get_study(b)
    # study.load_imagenet_model()
    # study.clear_nn_outputs_folders()
    # study.run_imgnet_inference()
    study.clear_db_results()
    study.run_quantifiers(epoch=None, ood_severities=["1", "2", "3", "4", "5"])


if __name__ == '__main__':
    gpu_config.use_only_gpu(0)
    # for b in range(4):
    # for b in range(4, 8):
    for b in range(8):
        run(b)
