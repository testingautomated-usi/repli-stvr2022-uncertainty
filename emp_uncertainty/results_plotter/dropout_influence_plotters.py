import itertools
import os
import pickle
from statistics import harmonic_mean
from typing import List, Callable

import matplotlib.pyplot as plt

from emp_uncertainty.case_studies.case_study import BASE_OUTPUTS_SAVE_FOLDER
from emp_uncertainty.case_studies.result import Result

FS_RESULTS: List[Result] = []
for root, dir, files in os.walk(f"{BASE_OUTPUTS_SAVE_FOLDER}/dropout-experiments"):
    for f in files:
        if f.endswith('pickle'):
            with open(f"{root}/{f}", "rb") as openFile:
                FS_RESULTS.append(pickle.load(openFile))


def plot_for_case_study(case_study, objective_fun: Callable[[Result], float], objective):
    if not os.path.exists("/root/assets/plots/dropout-plots"):
        os.makedirs("/root/assets/plots/dropout-plots")

    font = {'size': 12}

    plt.rc('font', **font)

    res = [f for f in FS_RESULTS if f.study_id == case_study]
    sorted_by_src = sorted(res, key=lambda r: r.src)
    for src, res in itertools.groupby(sorted_by_src, lambda r: r.src):
        sorted_by_quantifier = sorted(res, key=lambda r: r.metric)
        for quantifier, res in itertools.groupby(sorted_by_quantifier, lambda r: r.metric):
            # Model type stands for string representation of dropout rate + some more text
            sorted_by_model_type = sorted(res, key=lambda r: r.model_type)

            mean_values = []
            for model_type, res in itertools.groupby(sorted_by_model_type, lambda r: r.model_type):
                dropout_rate = float(model_type.split("=")[1])
                values = [objective_fun(r) for r in res]
                assert len(values) == 10 # Number of runs
                avg_value = sum(values) / len(values)
                mean_values.append((dropout_rate, avg_value))

            sorted_by_model_type = sorted(mean_values, key=lambda t: t[0])
            keys = [s[0] for s in sorted_by_model_type]
            values = [s[1] for s in sorted_by_model_type]
            quantifier = quantifier.replace("mean_sm", "mean softmax")\
                .replace("mutu_info", "mutual info")\
                .replace("var_ratio", "variation ratio")\
                .replace("pred_entropy", "pred. entropy")
            plt.plot(keys, values, label=quantifier)



        plt.legend()
        # plt.title(f"{case_study}-{src}-{objective}")
        plt.savefig(f"/root/assets/plots/dropout-plots/dropout_influence_{case_study}-{src}-{objective}.png")
        plt.close()


def plot_for_all_cs(objective_fun: Callable[[Result], float], objective):
    plot_for_case_study("mnist", objective_fun, objective)
    plot_for_case_study("traffic", objective_fun, objective)
    plot_for_case_study("cifar10", objective_fun, objective)


if __name__ == '__main__':
    plot_for_all_cs(lambda x: x.auc_roc, "avg_pr")
