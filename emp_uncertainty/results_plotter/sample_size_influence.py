import sqlite3
from statistics import harmonic_mean

import matplotlib.pyplot as plt
import numpy as np

studies = {
    'efficientnet_b0':
        {
            'db_path': '/root/assets/efficientnet_b0-results.db',
            'sources': ['nominal'] + [f"ood-{i}" for i in range(1, 6)] + ["ood-all"],
            'combined_ood': "ood-all",
            'ensemble': False,
            'epochs': None
        },
    'efficientnet_b1':
        {
            'db_path': '/root/assets/efficientnet_b1-results.db',
            'sources': ['nominal'] + [f"ood-{i}" for i in range(1, 6)],
            'combined_ood': "ood-all",
            'ensemble': False,
            'epochs': None
        },
    'efficientnet_b2':
        {
            'db_path': '/root/assets/efficientnet_b2-results.db',
            'sources': ['nominal'] + [f"ood-{i}" for i in range(1, 6)],
            'combined_ood': "ood-all",
            'ensemble': False,
            'epochs': None
        },
    'efficientnet_b3':
        {
            'db_path': '/root/assets/efficientnet_b3-results.db',
            'sources': ['nominal'] + [f"ood-{i}" for i in range(1, 6)],
            'combined_ood': "ood-all",
            'ensemble': False,
            'epochs': None
        },
    'efficientnet_b4':
        {
            'db_path': '/root/assets/efficientnet_b4-results.db',
            'sources': ['nominal'] + [f"ood-{i}" for i in range(1, 6)],
            'combined_ood': "ood-all",
            'ensemble': False,
            'epochs': None
        },
    'efficientnet_b5':
        {
            'db_path': '/root/assets/efficientnet_b5-results.db',
            'sources': ['nominal'] + [f"ood-{i}" for i in range(1, 6)],
            'combined_ood': "ood-all",
            'ensemble': False,
            'epochs': None
        },
    'efficientnet_b6':
        {
            'db_path': '/root/assets/efficientnet_b6-results.db',
            'sources': ['nominal'] + [f"ood-{i}" for i in range(1, 6)],
            'combined_ood': "ood-all",
            'ensemble': False,
            'epochs': None
        },
    'efficientnet_b7':
        {
            'db_path': '/root/assets/efficientnet_b7-results.db',
            'sources': ['nominal'] + [f"ood-{i}" for i in range(1, 6)],
            'combined_ood': "ood-all",
            'ensemble': False,
            'epochs': None
        },
    'cifar10':
        {
            'db_path': '/root/assets/cifar10-results.db',
            'sources': ['nominal'] + [f"ood-{i}" for i in range(1, 6)],
            'combined_ood': "ood-all",
            'ensemble': True,
            'epochs': 200
        },
    'mnist':
        {
            'db_path': '/root/assets/mnist-results.db',
            'sources': ['nominal', 'ood-noseverity'],
            'combined_ood': "ood-noseverity",
            'ensemble': True,
            'epochs': 200
        },
    'traffic':
        {
            'db_path': '/root/assets/traffic-results.db',
            'sources': ['nominal'] + [f"ood-{i}" for i in range(1, 6)] + ["ood-all"],
            'combined_ood': "ood-all",
            'ensemble': True,
            'epochs': 200
        },
}


def plot_s_accuracy(include_effnet: bool,
                    model_type: str,
                    ood: bool,
                    metric: str,
                    epsilon_perc: int,
                    objective: str):
    font = {'size': 12}

    plt.rc('font', **font)
    if model_type == "stochastic" or model_type == "sampling_bnn":
        plt.figure(figsize=[6, 4.8])
        plt.xlabel("# samples")
    else:
        plt.figure(figsize=[6, 4.8])
        plt.xlabel("# atomic models")

    def get_for_case_study(src, case_study, db_path):
        conn = sqlite3.connect(db_path)
        epochs_clause = f"epochs='199' " if not case_study.startswith("eff") else "epochs IS NULL"
        acceptance_rate_column = f"s{epsilon_perc}_acceptantance_rate"
        accuracy_rate = f"s{epsilon_perc}_accuracy"
        res = conn.execute(f"select distinct num_samples, {accuracy_rate}, {acceptance_rate_column} "
                           f"from res "
                           f"where model_type='{model_type}' "
                           f"   and study_id='{case_study}' "
                           f"   and metric='{metric}' "  # metric does not matter, we just pick one
                           f"   and src='{src}' "
                           f"   and {epochs_clause} "
                           f"order by num_samples").fetchall()
        n_samples = []
        vals = []
        for i, entry in enumerate(res):
            n_samples.append(entry[0])
            s_accuracy = float(entry[1])
            acceptance_rate = float(entry[2])
            harmonic_s = harmonic_mean([s_accuracy, acceptance_rate])
            if objective == "s1":
                vals.append(harmonic_s)
            elif objective == "s_accuracy":
                vals.append(s_accuracy)
            elif objective == "acceptance_rate":
                vals.append(acceptance_rate)
            else:
                assert False, "objective must be 's1', 'acceptance_rate' or 's_accuracy' "
        return n_samples, vals

    num_samples = None
    for study, studies_info in studies.items():
        if not study.startswith("eff"):
            db = studies_info["db_path"]
            source = studies_info["combined_ood"] if ood else "nominal"
            ns, values = get_for_case_study(source, study, db)
            if num_samples is None:
                num_samples = ns
            else:
                assert num_samples == ns
            plt.plot(num_samples, values, label=study)

    if include_effnet:
        values_collections = []
        for effnet_number in range(8):
            study_name = f"efficientnet_b{effnet_number}"
            db = studies[study_name]['db_path']
            source = studies[study_name]['combined_ood'] if ood else "nominal"
            ns, values = get_for_case_study(source, study_name, db)
            assert num_samples == ns
            values_collections.append(values)
        values = np.array(values_collections)
        mean_values = np.mean(values, axis=0)
        plt.plot(num_samples, mean_values, label=f"imagenet (mean)")

    plt.legend(loc='lower right')
    if objective == "s_accuracy":
        ylabel = "supervised accuracy"
    elif objective == "s1":
        ylabel = "S1-Score"
    elif objective == "acceptance_rate":
        ylabel = "acceptance rate"
    else:
        assert False, f"Human readable name for {objective} not yet defined"
    plt.ylabel(ylabel)
    # plt.title(f"Dependency of {objective} on sample size")
    src = "ood" if ood else "nominal"
    plt.tight_layout()
    plt.savefig(f"/root/assets/plots/{objective}_by_samplessize_{model_type}_{src}_epsilon{epsilon_perc}")
    plt.close()


if __name__ == '__main__':

    for ood in [True, False]:
        for model_type in [
            "stochastic",
            "ensemble",
            "sampling_bnn"
        ]:
            include_effnet = (model_type == 'stochastic')
            epsilon_perc = 10
            metric = "mean_sm"
            plot_s_accuracy(include_effnet=include_effnet,
                            model_type=model_type,
                            ood=ood,
                            metric=metric,
                            epsilon_perc=epsilon_perc,
                            objective="s_accuracy")