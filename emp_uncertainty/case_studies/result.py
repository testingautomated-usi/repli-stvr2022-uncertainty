from dataclasses import dataclass
from typing import Union, Optional


@dataclass
class Result:
    study_id: str
    model_type: str
    epochs: Optional[int]  # None for pre-trained models
    src: str
    num_samples: Optional[int]  # Sampling based only
    num_inputs: int
    num_misclassified: Optional[int]  # classification only
    num_correctly_classified: Optional[int]  # classification only
    metric: str
    point_biserial_r: Optional[float]
    point_biserial_p: Optional[float]
    auc_roc: float
    avg_precision_score: float
    s1_acceptantance_rate: float
    s5_acceptantance_rate: float
    s10_acceptantance_rate: float
    s1_accuracy: float
    s5_accuracy: float
    s10_accuracy: float
