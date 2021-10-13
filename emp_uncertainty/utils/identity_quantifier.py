import abc
from typing import List

import numpy as np
import uncertainty_wizard as uwiz
from uncertainty_wizard import ProblemType


class _IdentityQuantifer(uwiz.quantifiers.Quantifier, abc.ABC):
    @classmethod
    def aliases(cls) -> List[str]:
        return ["identity"]

    @classmethod
    def is_confidence(cls) -> bool:
        # Value does not matter
        return False

    @classmethod
    def calculate(cls, nn_outputs: np.ndarray):
        return None, nn_outputs

    @classmethod
    def problem_type(cls) -> ProblemType:
        # Value does not matter
        return ProblemType.CLASSIFICATION


class SamplingIdentity(_IdentityQuantifer):
    @classmethod
    def takes_samples(cls) -> bool:
        return True


class PointPredictionIdentity(_IdentityQuantifer):
    @classmethod
    def takes_samples(cls) -> bool:
        return False
