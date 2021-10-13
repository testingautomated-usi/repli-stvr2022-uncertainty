from typing import List

import numpy as np
import uncertainty_wizard as uwiz
from uncertainty_wizard import ProblemType


class DummyConstantUncertainty(uwiz.quantifiers.Quantifier):
    @classmethod
    def aliases(cls) -> List[str]:
        return "dummy_all"

    @classmethod
    def is_confidence(cls) -> bool:
        return False

    @classmethod
    def takes_samples(cls) -> bool:
        return False

    @classmethod
    def problem_type(cls) -> ProblemType:
        return uwiz.ProblemType.CLASSIFICATION

    @classmethod
    def calculate(cls, nn_outputs: np.ndarray):
        softmax_winners, _ = uwiz.quantifiers.MaxSoftmax().calculate(nn_outputs=nn_outputs)
        return softmax_winners, np.ones_like(softmax_winners, dtype=float)
