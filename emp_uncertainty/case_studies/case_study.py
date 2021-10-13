import abc
import math
import os
import shutil
import sqlite3
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Tuple, Union, Iterable, Optional, Dict

import numpy as np
import tensorflow as tf
import uncertainty_wizard as uwiz
from scipy.stats import pointbiserialr
from sklearn.metrics import roc_auc_score, average_precision_score

from emp_uncertainty.case_studies.result import Result
from emp_uncertainty.utils.identity_quantifier import SamplingIdentity, PointPredictionIdentity

MAX_QUANT_WORKERS = 2

BAYES_CLASSIFICATION_QUANTIFIERS = {
    "var_ratio": uwiz.quantifiers.VariationRatio(),
    "pred_entropy": uwiz.quantifiers.PredictiveEntropy(),
    "mutu_info": uwiz.quantifiers.MutualInformation(),
    "mean_sm": uwiz.quantifiers.MeanSoftmax()
}

PP_CLASSIFICATION_QUANTIFIERS = {
    "max_softmax": uwiz.quantifiers.MaxSoftmax(),
    "softmax_entropy": uwiz.quantifiers.SoftmaxEntropy(),
    "pcs": uwiz.quantifiers.PredictionConfidenceScore()
}

TRAIN_DROPOUT = False
TRAIN_FLIPOUT = True
TRAIN_ENSEMBLE = False

# ASSETS_PATH = "E:\\stvr_extension_traffic\\assets"
ASSETS_PATH = "/root/assets/"
BASE_MODEL_SAVE_FOLDER = f"{ASSETS_PATH}/trained_models/"
BASE_OUTPUTS_SAVE_FOLDER = f"{ASSETS_PATH}/nn_outputs/"


class CaseStudy(abc.ABC):

    def __init__(self) -> None:
        self.stochastic_model = None
        self.ensemble_model: Optional[uwiz.models.LazyEnsemble] = None
        self.sample_bayesian_nn: Optional[uwiz.models.StochasticSequential] = None
        self.init_db()

    @classmethod
    def init_db(cls):
        cls.conn = sqlite3.connect(f'{ASSETS_PATH}/{cls._case_study_id()}-results.db')
        cls.conn.execute(
            """
            create table if not exists res (
                study_id TEXT NOT NULL,
                model_type TEXT NOT NULL,
                epochs INTEGER,
                src VARCHAR ,
                num_samples INTEGER,
                num_inputs INTEGER NOT NULL,
                num_misclassified INTEGER,
                num_correctly_classified INTEGER,
                metric TEXT NOT NULL,
                point_biserial_r NUMERIC,
                point_biserial_p NUMERIC,
                auc_roc NUMERIC,
                avg_precision_score NUMERIC,       
                s1_acceptantance_rate NUMERIC,
                s5_acceptantance_rate NUMERIC,
                s10_acceptantance_rate NUMERIC,
                s1_accuracy NUMERIC,
                s5_accuracy NUMERIC,
                s10_accuracy NUMERIC
            )
            """
        )
        cls.conn.commit()

    @classmethod
    @abc.abstractmethod
    def _case_study_id(cls) -> str:
        pass

    def train_with_epoch_eval(self, num_epochs: int, ood_severities):
        x_train, y_train, x_val, y_val = self._get_train_and_val_data()

        self.stochastic_model, self.ensemble_model = None, None
        if TRAIN_DROPOUT:
            self.stochastic_model = self._create_stochastic_model()
        if TRAIN_ENSEMBLE:
            self.ensemble_model = self._create_ensemble_model()
        if TRAIN_FLIPOUT:
            self.sample_bayesian_nn = self._create_sample_bayesian_nn()

        batch_size = self.get_batch_size()
        for epoch in range(num_epochs):
            if self.stochastic_model is not None and TRAIN_DROPOUT:
                self.stochastic_model.fit(x_train, y_train,
                                          batch_size=batch_size,
                                          epochs=1)
                path = os.path.abspath(BASE_MODEL_SAVE_FOLDER + self._case_study_id() + "/stochastic/")
                self.stochastic_model.save(path,
                                           include_optimizer=True)
            if self.ensemble_model is not None:
                self.ensemble_model.fit(x_train, y_train,
                                        shuffle=True,
                                        batch_size=batch_size,
                                        num_processes=self.num_ensemble_processes(),
                                        epochs=1)
            if self.sample_bayesian_nn is not None:
                self.sample_bayesian_nn.fit(x_train, y_train,
                                            batch_size=batch_size,
                                            epochs=1)
                path = os.path.abspath(BASE_MODEL_SAVE_FOLDER + self._case_study_id() + "/bnn_sampling/")
                self.sample_bayesian_nn.save(path,
                                             include_optimizer=True)
            self.run_nn_inference(epoch=epoch,
                                  ood_severities=ood_severities,
                                  val_dataset=(x_val, y_val))

    def load(self, load_sequential=True, load_ensemble=True):
        self.stochastic_model = None
        self.ensemble_model = None
        if load_sequential:
            path = os.path.abspath(BASE_MODEL_SAVE_FOLDER + self._case_study_id() + "/stochastic/")
            self.stochastic_model = \
                uwiz.models.load_model(path)
        if load_ensemble:
            self.ensemble_model = uwiz.models.load_model(self.ensemble_save_path())

    @classmethod
    @abc.abstractmethod
    def get_outlier_data(cls, severity: str) -> Tuple[Union[np.ndarray, tf.data.Dataset], np.ndarray]:
        pass

    @classmethod
    def ensemble_save_path(cls):
        path = os.path.abspath(BASE_MODEL_SAVE_FOLDER + cls._case_study_id() + "/ensemble/")
        return path

    @classmethod
    def save_eval_result(cls, results: Iterable[Result]):
        def res_as_tuple(res: Result):
            return (
                res.study_id,
                res.model_type,
                res.epochs,
                res.src,
                res.num_samples,
                res.num_inputs,
                res.num_misclassified,
                res.num_correctly_classified,
                res.metric,
                res.point_biserial_r,
                res.point_biserial_p,
                res.auc_roc,
                res.avg_precision_score,
                res.s1_acceptantance_rate,
                res.s5_acceptantance_rate,
                res.s10_acceptantance_rate,
                res.s1_accuracy,
                res.s5_accuracy,
                res.s10_accuracy,
            )

        cls.conn.executemany(
            """
            INSERT INTO res VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [res_as_tuple(res) for res in results]
        )
        cls.conn.commit()

    def run_nn_inference(self,
                         epoch: Optional[int],
                         ood_severities: Iterable[str],
                         val_dataset: Tuple[np.ndarray, np.ndarray]):
        """
        Run inference on the models and save the plain NN outputs to the file system
        :return: None
        """

        x, _ = self.get_test_data()

        oods_x, _ = self.oods_for_severities(ood_severities)

        nn_outputs_folder = os.path.dirname(self._nn_outputs_folder(epoch=epoch,
                                                                    model_type="stochastic",
                                                                    # Does not matter, we're taking parent anyways
                                                                    src="nominal"))
        if not os.path.exists(nn_outputs_folder):
            try:
                os.makedirs(nn_outputs_folder)
            except:
                pass

        if self.stochastic_model is not None:
            print("Running nn inference for stochastic & point pred models")
            r = self._stochastic_predict_quantified(x, self.stochastic_model)
            self._save_nn_outputs(epoch=epoch, values=r[0][1], src="nominal", m_type="stochastic")
            self._save_nn_outputs(epoch=epoch, values=r[1][1], src="nominal", m_type="point")

            r_val = self._stochastic_predict_quantified(val_dataset[0], self.stochastic_model)
            self._save_nn_outputs(epoch=epoch, values=r_val[0][1], src="val", m_type="stochastic")
            self._save_nn_outputs(epoch=epoch, values=r_val[1][1], src="val", m_type="point")

            for ood_severity, ood_x in oods_x.items():
                r = self._stochastic_predict_quantified(ood_x, self.stochastic_model)
                self._save_nn_outputs(epoch=epoch, values=r[0][1], src=f"ood-{ood_severity}", m_type="stochastic")
                self._save_nn_outputs(epoch=epoch, values=r[1][1], src=f"ood-{ood_severity}", m_type="point")

        if self.sample_bayesian_nn is not None:
            print("Running nn inference sampling based dnn")
            r = self._stochastic_predict_quantified(x, self.sample_bayesian_nn)
            self._save_nn_outputs(epoch=epoch, values=r[0][1], src="nominal", m_type="sampling_bnn")
            assert not np.any(np.isnan(r[0][1]))

            r_val = self._stochastic_predict_quantified(val_dataset[0], self.sample_bayesian_nn)
            self._save_nn_outputs(epoch=epoch, values=r_val[0][1], src="val", m_type="sampling_bnn")
            assert not np.any(np.isnan(r_val[0][1]))

            for ood_severity, ood_x in oods_x.items():
                r = self._stochastic_predict_quantified(ood_x, self.sample_bayesian_nn)
                self._save_nn_outputs(epoch=epoch, values=r[0][1], src=f"ood-{ood_severity}", m_type="sampling_bnn")
                assert not np.any(np.isnan(r[0][1]))

        if self.ensemble_model is not None:
            print("Running nn inference for ensemble models")
            _, nn_outputs = self._ensemble_predict_quantified(x)
            np.save(self._nn_outputs_folder(src="nominal", epoch=epoch, model_type="ensemble"), nn_outputs)

            _, nn_outputs = self._ensemble_predict_quantified(val_dataset[0])
            np.save(self._nn_outputs_folder(src="val", epoch=epoch, model_type="ensemble"), nn_outputs)

            for ood_severity, ood_x in oods_x.items():
                _, nn_outputs = self._ensemble_predict_quantified(ood_x)
                np.save(self._nn_outputs_folder(src=f"ood-{ood_severity}", epoch=epoch, model_type="ensemble"),
                        nn_outputs)

    def _save_nn_outputs(self, epoch: Optional[int], values: np.ndarray, src: str, m_type: str):
        np.save(self._nn_outputs_folder(src=src,
                                        epoch=epoch,
                                        model_type=m_type), values)

    @classmethod
    def oods_for_severities(cls, ood_severities):
        oods_x = dict()
        oods_y = dict()
        for ood_severity in ood_severities:
            ood_x, ood_y = cls.get_outlier_data(ood_severity)
            oods_x[ood_severity] = ood_x
            oods_y[ood_severity] = ood_y
        return oods_x, oods_y

    def _ensemble_predict_quantified(self, x):
        batch_size = self.get_batch_size()
        return self.ensemble_model.predict_quantified(x=x,
                                                      batch_size=batch_size,
                                                      quantifier=SamplingIdentity(),
                                                      num_processes=self.num_ensemble_processes(),
                                                      verbose=1)

    def _stochastic_predict_quantified(self, x, model):
        batch_size = self.get_batch_size()
        return model.predict_quantified(x=x,
                                        quantifier=[SamplingIdentity(), PointPredictionIdentity()],
                                        sample_size=100,
                                        # Increase batch size (as faster with same results)
                                        # There's actually only once process,
                                        # but we know that gpu has enough memory for
                                        # num-proc * batch_size many inputs at a time
                                        batch_size=batch_size * self.num_ensemble_processes(),
                                        verbose=1)

    @classmethod
    def _nn_outputs_folder(cls, src: str, epoch: int, model_type: str):
        if epoch is None:
            epoch = "epoch_indep"

        if model_type == "stochastic":
            return BASE_OUTPUTS_SAVE_FOLDER + cls._case_study_id() + f"/e-{epoch}/{src}-stoch_outputs.npy"
        elif model_type == "point":
            return BASE_OUTPUTS_SAVE_FOLDER + cls._case_study_id() + f"/e-{epoch}/{src}-pp_outputs.npy"
        elif model_type == "ensemble":
            return BASE_OUTPUTS_SAVE_FOLDER + cls._case_study_id() + f"/e-{epoch}/{src}-ensemble_outputs.npy"
        elif model_type == "sampling_bnn":
            return BASE_OUTPUTS_SAVE_FOLDER + cls._case_study_id() + f"/e-{epoch}/{src}-sampling_bnn_outputs.npy"
        else:
            assert False, f"model type {model_type} is unexpected"

    def clear_nn_outputs_folders(self):
        # Delete previously stored NN outputs
        if os.path.exists(BASE_OUTPUTS_SAVE_FOLDER + self._case_study_id()):
            shutil.rmtree(BASE_OUTPUTS_SAVE_FOLDER + self._case_study_id())
        os.makedirs(BASE_OUTPUTS_SAVE_FOLDER + self._case_study_id())

    @classmethod
    def clear_db_results(cls):
        cls.conn.execute(f"delete from res where study_id = '{cls._case_study_id()}'")
        cls.conn.commit()

    @classmethod
    @abc.abstractmethod
    def get_val_labels(cls) -> np.ndarray:
        raise NotImplementedError("Not yet implemented")
        pass

    @classmethod
    def run_quantifiers(cls, epoch, ood_severities: Iterable[str]) -> None:
        """
        Load the nn outputs from the file systems and run the quantifiers on them.

        Calls save_eval_result to store the results in the database
        :return: None
        """

        print(f"Start running quantifiers for epoch {epoch}")

        _, y = cls.get_test_data()
        cls._run_quantifiers_for_src(epoch, "nominal", y)

        y_val = cls.get_val_labels()
        cls._run_quantifiers_for_src(epoch, "val", y_val)

        _, oods_y = cls.oods_for_severities(ood_severities)
        for severity, ood_y in oods_y.items():
            cls._run_quantifiers_for_src(epoch, f"ood-{severity}", ood_y)

        if len(oods_y.values()) > 0:
            merged_oody = None
            for ood_y in oods_y.values():
                if merged_oody is not None:
                    merged_oody = np.concatenate((merged_oody, ood_y))
                else:
                    merged_oody = ood_y
            cls._run_quantifiers_for_src(epoch, f"ood-all", merged_oody, ood_severities)

    @classmethod
    def calculate_thresholds(cls, y_labels: np.ndarray,
                             model_type: str,
                             epoch: Optional[int] = None,
                             nn_outputs: np.ndarray = None):
        if nn_outputs is None:
            nn_outputs = np.load(cls._nn_outputs_folder(src="val", epoch=epoch, model_type=model_type),
                                 allow_pickle=True)
        quantifiers = BAYES_CLASSIFICATION_QUANTIFIERS
        if model_type == "point":
            quantifiers = PP_CLASSIFICATION_QUANTIFIERS

        thresholds = dict()
        for quantifier_name, quantifier in quantifiers.items():
            for target_frp in [1, 5, 10]:
                predictions, quant = quantifier.calculate(nn_outputs)
                uncertainties = quantifier.cast_conf_or_unc(as_confidence=False, superv_scores=quant)

                assert y_labels.shape == predictions.shape
                is_correctly_classified = y_labels == predictions
                neg_scores = uncertainties[np.argwhere(is_correctly_classified == True)].flatten()
                sorted_neg_scores = np.sort(neg_scores)
                threshold_index = math.floor(sorted_neg_scores.shape[0] * (1 - target_frp / 100))
                thresholds[f"{quantifier_name}_tfpr_{target_frp}"] = sorted_neg_scores[threshold_index]
        return thresholds

    @classmethod
    def _run_quantifiers_for_src(cls, epoch, src, y, ood_severities=None):
        # STOCHASTIC MODEL
        y_val = cls.get_val_labels()
        try:
            stoch_thresholds = cls.calculate_thresholds(y_labels=y_val, model_type="stochastic", epoch=epoch)
            stoch_sampled_nn_outputs = cls.load_nn_outputs(epoch, src, model_type="stochastic",
                                                           ood_severities=ood_severities)
            samples_eval: Iterable[Result] = cls._evaluate_stochastic(epoch=epoch,
                                                                      src=src,
                                                                      true_labels=y,
                                                                      nn_outputs=stoch_sampled_nn_outputs,
                                                                      thresholds=stoch_thresholds)
            cls.save_eval_result(samples_eval)
        except IOError:
            print("Was not able to load stoch_sampled_nn_outputs")
        # FLIPOUT BASED PW-BNN MODEL
        try:
            bnn_thresholds = cls.calculate_thresholds(y_labels=y_val, model_type="sampling_bnn", epoch=epoch)
            bnn_sampled_nn_outputs = cls.load_nn_outputs(epoch, src, model_type="sampling_bnn",
                                                         ood_severities=ood_severities)
            samples_eval: Iterable[Result] = cls._evaluate_stochastic(epoch=epoch,
                                                                      src=src,
                                                                      true_labels=y,
                                                                      nn_outputs=bnn_sampled_nn_outputs,
                                                                      thresholds=bnn_thresholds,
                                                                      model_type="sampling_bnn")
            cls.save_eval_result(samples_eval)
        except IOError:
            print("Was not able to load stoch_sampled_nn_outputs")

        # POINT PREDICTOR MODEL
        try:
            point_thresholds = cls.calculate_thresholds(y_labels=y_val, model_type="point", epoch=epoch)
            pp_nn_outputs = cls.load_nn_outputs(epoch, src, model_type="point",
                                                ood_severities=ood_severities)
            pp_eval: Iterable[Result] = cls._evaluate_pp(epoch=epoch,
                                                         src=src,
                                                         true_labels=y,
                                                         nn_outputs=pp_nn_outputs,
                                                         thresholds=point_thresholds)
            cls.save_eval_result(pp_eval)
        except IOError:
            print("Was not able to load pp_outputs")

        # ENSEMBLE MODEL
        try:
            ensemble_thresholds = cls.calculate_thresholds(y_labels=y_val, model_type="ensemble", epoch=epoch)
            ensemble_nn_outputs = cls.load_nn_outputs(epoch, src, model_type="ensemble",
                                                      ood_severities=ood_severities)
            ens_eval: Iterable[Result] = cls._evaluate_ensemble(epoch=epoch,
                                                                src=src,
                                                                true_labels=y,
                                                                nn_outputs=ensemble_nn_outputs,
                                                                thresholds=ensemble_thresholds)
            cls.save_eval_result(ens_eval)
        except IOError:
            print("Was not able to load ensemble_nn_outputs")

    @classmethod
    def load_nn_outputs(cls, epoch, src, model_type, ood_severities=None):
        if ood_severities is None:
            return np.load(cls._nn_outputs_folder(src=src, epoch=epoch, model_type=model_type),
                           allow_pickle=True)
        else:
            all_values = None
            for severity in ood_severities:
                for_severity = np.load(
                    cls._nn_outputs_folder(src=f"ood-{severity}", epoch=epoch, model_type=model_type),
                    allow_pickle=True)
                if all_values is None:
                    all_values = for_severity
                else:
                    all_values = np.concatenate((all_values, for_severity))
            return all_values

    @classmethod
    @abc.abstractmethod
    def _get_train_and_val_data(cls) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    @classmethod
    @abc.abstractmethod
    def get_test_data(cls) -> Tuple[Union[np.ndarray, tf.data.Dataset], np.ndarray]:
        pass

    @classmethod
    @abc.abstractmethod
    def _create_stochastic_model(
            cls) -> uwiz.models.StochasticSequential:
        pass

    @classmethod
    @abc.abstractmethod
    def _create_sample_bayesian_nn(
            cls) -> uwiz.models.StochasticSequential:
        pass

    @classmethod
    @abc.abstractmethod
    def _create_ensemble_model(
            cls) -> Union[None, uwiz.models.LazyEnsemble]:
        pass

    @classmethod
    @abc.abstractmethod
    def _evaluate_pp(cls,
                     src: str,
                     epoch: Optional[int],
                     true_labels: np.ndarray,
                     nn_outputs: np.ndarray,
                     thresholds: Dict[str, float]) -> Iterable[Result]:
        pass

    @classmethod
    @abc.abstractmethod
    def _evaluate_stochastic(cls,
                             src: str,
                             epoch: Optional[int],
                             true_labels:
                             np.ndarray, nn_outputs: np.ndarray,
                             thresholds: Dict[str, float],
                             model_type="stochastic") -> Iterable[Result]:
        pass

    @classmethod
    @abc.abstractmethod
    def _evaluate_ensemble(cls,
                           src: str,
                           epoch: Optional[int],
                           true_labels:
                           np.ndarray, nn_outputs: np.ndarray,
                           thresholds: Dict[str, float]) -> Iterable[Result]:
        pass

    @classmethod
    @abc.abstractmethod
    def get_batch_size(cls) -> int:
        pass

    @classmethod
    @abc.abstractmethod
    def num_ensemble_processes(cls):
        pass


class RegressionCaseStudy(CaseStudy, abc.ABC):
    def __init__(self):
        raise NotImplementedError("TODO")


def calc_s_scores(thresholds: Dict[str, float],
                  uncertainties: np.ndarray,
                  predictions: np.ndarray,
                  true_labels: np.ndarray,
                  quantifier_name: str,
                  target_fpr: int):
    threshold = thresholds[f"{quantifier_name}_tfpr_{target_fpr}"]
    index_of_seq_t = np.argwhere(uncertainties <= threshold)
    remaining_pred = predictions[index_of_seq_t].flatten()
    remaining_true = true_labels[index_of_seq_t].flatten()
    correctly_classified = remaining_pred == remaining_true
    s1_acceptantance_rate = remaining_pred.shape[0] / predictions.shape[0]
    s1_accuracy = np.sum(correctly_classified) / correctly_classified.shape[0]
    return s1_acceptantance_rate, s1_accuracy


class ClassificationCaseStudy(CaseStudy, abc.ABC):

    @classmethod
    def _eval_all(cls, nn_outputs, quantifiers, true_labels, sample_size: Union[int, None], model_type: str,
                  epoch: Optional[int], src: str, thresholds: Dict[str, float]):
        results = []
        for q_name, q in quantifiers.items():
            results.append(cls.eval_classification_quantifier(quantifier_name=q_name,
                                                              quantifier=q,
                                                              thresholds=thresholds,
                                                              true_labels=true_labels,
                                                              nn_outputs=nn_outputs,
                                                              sample_size=sample_size,
                                                              model_type=model_type,
                                                              epochs=epoch,
                                                              src=src,
                                                              case_study_id=cls._case_study_id()))
        return results

    @staticmethod
    def eval_classification_quantifier(quantifier_name: str,
                                       quantifier: uwiz.quantifiers.Quantifier,
                                       epochs: Optional[int],
                                       true_labels: np.ndarray,
                                       nn_outputs: np.ndarray,
                                       sample_size: Union[int, None],  # to store only, nn_outputs are already cut
                                       model_type: str,
                                       src: str,
                                       thresholds: Dict[str, float],
                                       case_study_id: str,
                                       ) -> Result:
        pred, quant = quantifier.calculate(nn_outputs=nn_outputs)
        uncertainties = quantifier.cast_conf_or_unc(as_confidence=False, superv_scores=quant)

        def s_scores(target_fpr: int) -> Tuple[float, float]:
            return calc_s_scores(thresholds,
                                 predictions=pred,
                                 true_labels=true_labels,
                                 uncertainties=uncertainties,
                                 quantifier_name=quantifier_name,
                                 target_fpr=target_fpr)

        s1_acceptantance_rate, s1_accuracy = s_scores(1)
        s5_acceptantance_rate, s5_accuracy = s_scores(5)
        s10_acceptantance_rate, s10_accuracy = s_scores(10)

        # calculate other (non-s) scores
        is_misclassified = true_labels.flatten() != pred
        num_correctly_classified = np.sum(np.logical_not(is_misclassified)).item()
        num_misclassified = np.sum(is_misclassified).item()
        assert true_labels.shape[0] == num_correctly_classified + num_misclassified
        point_biserial_r, point_biserial_p = pointbiserialr(is_misclassified, uncertainties)
        auc_roc = roc_auc_score(y_true=is_misclassified, y_score=uncertainties)
        avg_precision_score = average_precision_score(y_true=is_misclassified, y_score=uncertainties)
        return Result(
            study_id=case_study_id,
            model_type=model_type,
            epochs=epochs,
            src=src,
            num_samples=sample_size,
            num_inputs=true_labels.shape[0],
            num_misclassified=num_misclassified,
            num_correctly_classified=num_correctly_classified,
            metric=quantifier_name,
            point_biserial_r=point_biserial_r if point_biserial_r else None,
            point_biserial_p=point_biserial_p if point_biserial_r else None,
            auc_roc=auc_roc,
            avg_precision_score=avg_precision_score,
            s1_acceptantance_rate=s1_acceptantance_rate,
            s5_acceptantance_rate=s5_acceptantance_rate,
            s10_acceptantance_rate=s10_acceptantance_rate,
            s1_accuracy=s1_accuracy,
            s5_accuracy=s5_accuracy,
            s10_accuracy=s10_accuracy,
        )

    @classmethod
    def _evaluate_pp(cls,
                     src: str,
                     epoch: Optional[int],
                     true_labels: np.ndarray, nn_outputs: np.ndarray,
                     thresholds: Dict[str, float]) -> Iterable[Result]:
        return cls._eval_all(nn_outputs, PP_CLASSIFICATION_QUANTIFIERS, true_labels,
                             sample_size=None,
                             thresholds=thresholds,
                             model_type="point_pred",
                             epoch=epoch, src=src)

    @classmethod
    def _evaluate_stochastic(cls,
                             src: str,
                             epoch: Optional[int],
                             true_labels:
                             np.ndarray, nn_outputs: np.ndarray,
                             thresholds: Dict[str, float],
                             model_type="stochastic") -> Iterable[Result]:
        print(f"Calculating {src} quantifications for {model_type} model (epoch {epoch})")

        # Collect the results for increasing sample sizes, use multithreading to do so

        def worker(s_size: int):
            return cls._eval_all(nn_outputs[:, :s_size, :],
                                 quantifiers=BAYES_CLASSIFICATION_QUANTIFIERS,
                                 thresholds=thresholds,
                                 true_labels=true_labels,
                                 sample_size=s_size,
                                 model_type=model_type,
                                 epoch=epoch,
                                 src=src)

        with ThreadPoolExecutor(max_workers=MAX_QUANT_WORKERS) as executor:
            futures = [executor.submit(worker, sample_size) for sample_size in range(2, nn_outputs.shape[1])]
        list_of_lists = [f.result() for f in futures]
        return [item for sublist in list_of_lists for item in sublist]

    @classmethod
    def _evaluate_ensemble(cls,
                           src: str,
                           epoch: Optional[int],
                           true_labels:
                           np.ndarray, nn_outputs: np.ndarray,
                           thresholds: Dict[str, float]) -> Iterable[Result]:
        print(f"Calculating {src} quantifications for ensemble model (epoch {epoch})")

        def worker(s_size: int):
            return cls._eval_all(nn_outputs[:, :s_size, :],
                                 quantifiers=BAYES_CLASSIFICATION_QUANTIFIERS,
                                 true_labels=true_labels,
                                 sample_size=s_size,
                                 thresholds=thresholds,
                                 model_type="ensemble",
                                 epoch=epoch,
                                 src=src)

        with ThreadPoolExecutor(max_workers=MAX_QUANT_WORKERS) as executor:
            futures = [executor.submit(worker, sample_size) for sample_size in range(2, nn_outputs.shape[1])]
        list_of_lists = [f.result() for f in futures]
        return [item for sublist in list_of_lists for item in sublist]
