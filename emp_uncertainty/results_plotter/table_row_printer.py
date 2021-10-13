import sqlite3
from concurrent.futures.process import ProcessPoolExecutor
from statistics import harmonic_mean
from typing import Union, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.stats import friedmanchisquare, rankdata

from emp_uncertainty.results_plotter.heatmap_helpers import heatmap, NonLinCdict

ROUND_DIGITS = 2
USE_PRE_FETCHED_GRID = True

if USE_PRE_FETCHED_GRID:
    print("Use pre-fetched grid")

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

BAYES_METRICS = [
    'var_ratio',
    'pred_entropy',
    'mutu_info',
    'mean_sm'
]

PP_METRICS = [
    'max_softmax',
    'pcs',
    'softmax_entropy'
]

FILTER_SIZE_X_EPOCHS = 5
FILTER_SIZE_Y_SAMPLES = 5
STRIDE_X_EPOCHS = 1
STRIDE_Y_SAMPLES = 1


def convolute(values, aggregation_function):
    num_epochs = values.shape[0]
    num_samples = values.shape[1]
    if num_epochs > 1 and num_samples > 1:
        convoluted_values = _convolute_2d(aggregation_function, num_epochs, num_samples, values)
    else:
        convoluted_values = _convolute_1d(aggregation_function, num_epochs, num_samples, values)
    return np.array(convoluted_values, dtype=float)


def _convolute_1d(aggregation_function, num_epochs, num_samples, values):
    assert num_epochs > 1 or num_samples > 1, "Either num_epochs or num_samples must be > 1"
    if num_epochs == 1:
        filter_size = FILTER_SIZE_X_EPOCHS
        stride = STRIDE_X_EPOCHS
        values = values[:, 1]
    elif num_samples == 1:
        filter_size = FILTER_SIZE_Y_SAMPLES
        stride = STRIDE_Y_SAMPLES
    else:
        assert False, "Array not one-dimensional"

    values = values.flatten()
    i = 0
    convoluted_values = []
    while i + filter_size < values.shape[0]:
        filter_values = values[i:i + filter_size]
        convoluted_values.append([aggregation_function(filter_values)])
        i += stride

    return convoluted_values


def _convolute_2d(aggregation_function, num_epochs, num_samples, values):
    x = 0
    convoluted_values = []
    while x + FILTER_SIZE_X_EPOCHS < num_epochs:
        y = 0
        convoluted_row_values = []
        while y + FILTER_SIZE_Y_SAMPLES < num_samples:
            filter_values = values[x:x + FILTER_SIZE_X_EPOCHS, y:y + FILTER_SIZE_Y_SAMPLES]
            convoluted_row_values.append(aggregation_function(filter_values))
            y += STRIDE_X_EPOCHS
        convoluted_values.append(convoluted_row_values)
        x += STRIDE_Y_SAMPLES
    return convoluted_values


def print_table_row(study, study_info: Dict, model_type, metric):
    print(f"Start with {study} {model_type} {metric}")
    conn = sqlite3.connect(study_info['db_path'])

    high_ps = 0
    low_ps = 0
    all_res = []

    results = dict()
    results["s1"] = select_optimal_model_on_val_set(1, conn, model_type, metric)
    results["s5"] = select_optimal_model_on_val_set(5, conn, model_type, metric)
    results["s10"] = select_optimal_model_on_val_set(10, conn, model_type, metric)

    for key, val_set_results in results.items():

        assert val_set_results is not None, f"val_set_not_fetched {study} {model_type} {metric}"
        epochs_from_val = val_set_results["epochs"]
        samples_from_val = val_set_results["num_samples"]

        for data_src in ["nominal", study_info["combined_ood"]]:
            res = {
                "study": study,
                "model_type": model_type,
                "metric": metric,
                "epsilon": key,
                "dataset": "nominal" if data_src == "nominal" else "ood",
            }
            # Probably not needed in the end. Using to allow debugging for now. Same for all entries in inner loop
            res["epochs"] = epochs_from_val if epochs_from_val else 'n.a.'
            res["samples"] = val_set_results["num_samples"]

            test_set_results = get_results_for_model(src=data_src,
                                                     epochs=epochs_from_val,
                                                     num_samples=samples_from_val,
                                                     connection=conn,
                                                     bayes_type=model_type,
                                                     metric=metric)

            # Calculate information about the perfomance of the model on the test (or ood) dataset
            accuracy = (test_set_results['num_correctly_classified'] /
                        (test_set_results['num_correctly_classified'] + test_set_results['num_misclassified']))
            s_accuracy = test_set_results[f"{key}_accuracy"]
            acceptance_rate = test_set_results[f"{key}_acceptantance_rate"]
            harmonic_s = harmonic_mean([s_accuracy, acceptance_rate])

            if model_type == 'point_pred' and study.startswith("eff"):
                instability_r = "n.a."
            else:
                np_grid = get_grid(conn=conn, model_type=model_type, s=key, study_info=study_info, source=data_src,
                                   min_epochs=0, min_samples=0, metric=metric, study=study)
                # min_epochs=50, min_samples=5, metric=metric, study=study)
                mean_values = convolute(np_grid, np.mean)
                standard_deviations = convolute(np_grid, np.std)
                instability_r, instability_p = stats.spearmanr(mean_values.flatten(), standard_deviations.flatten())
                if instability_p > 0.05:
                    high_ps += 1
                else:
                    low_ps += 1

            res["accuracy"] = accuracy
            res["s_accuracy"] = s_accuracy
            res["acceptance_rate"] = acceptance_rate
            res["harmonic_s"] = harmonic_s
            res["instability_r"] = instability_r
            all_res.append(res)
            print(f"Done with {study} {data_src} {model_type} {metric}")

    return all_res, high_ps, low_ps


def get_grid(conn, min_epochs, min_samples, model_type, s, study_info, source, metric, study):
    if USE_PRE_FETCHED_GRID:
        return np.load(f"/root/assets/grid_values/{study}-{s}-{model_type}-{metric}-{source}.npy")
    else:
        grid_values = get_all_values(objective=f"{s}_accuracy",
                                     min_epochs=min_epochs,
                                     min_samples=min_samples,
                                     connection=conn,
                                     model_type=model_type,
                                     study_info=study_info,
                                     source=source,
                                     metric=metric)
        # remove blank entries
        grid_values = [inner for inner in grid_values if inner]
        grid_values.reverse()
        np_grid = np.array(grid_values)
        np.save(f"/root/assets/grid_values/{study}-{s}-{model_type}-{metric}-{source}.npy", np_grid)
        return np_grid


def select_optimal_model_on_val_set(s: int, connection, bayes_type, metric):
    connection.row_factory = sqlite3.Row

    winner_row = connection.execute(f"select distinct * "
                                    f"from res "
                                    f"where model_type='{bayes_type}' "
                                    f"   and metric='{metric}' "
                                    f"   and src='val' "
                                    f"order by epochs desc, num_samples desc "
                                    f"limit 1").fetchone()
    return winner_row


def get_results_for_model(src: str, epochs: Optional[int], num_samples: Optional[int], connection, bayes_type, metric):
    connection.row_factory = sqlite3.Row
    epochs_clause = f"epochs='{epochs}' " if epochs else "epochs IS NULL"
    samples_clause = f"num_samples='{num_samples}' " if num_samples else "num_samples IS NULL"
    winner_row = connection.execute(f"select distinct * "
                                    f"from res "
                                    f"where model_type='{bayes_type}' "
                                    f"   and metric='{metric}' "
                                    f"   and src='{src}' "
                                    f"   and {epochs_clause} "
                                    f"   and {samples_clause} ").fetchall()
    assert len(winner_row) == 1
    return winner_row[0]


def get_all_values(objective, min_epochs, min_samples, connection, model_type, study_info: Dict, source: str,
                   metric: str):
    def get_for_num_epochs(epochs: Union[int, str]):
        epochs_clause = f"epochs='{epochs}' " if epochs else "epochs IS NULL"
        res = connection.execute(f"select distinct num_samples, {objective} "
                                 f"from res "
                                 f"where model_type='{model_type}' "
                                 f"   and {epochs_clause} "
                                 f"   and metric='{metric}' "
                                 f"   and src='{source}' "
                                 f"order by num_samples").fetchall()
        n_samples = []
        scores = []
        for i, entry in enumerate(res):
            assert i + 2 == entry[0] or (entry[0] is None and len(res) == 1), "Sample size not as expected"
            if entry[0] is None or entry[0] > min_samples:
                n_samples.append(entry[0])
                scores.append(entry[1])
        return n_samples, scores

    grid_values = []
    if study_info['epochs'] is None:
        grid_values.append(get_for_num_epochs(None))
    else:
        for epoch in range(min_epochs, 200):
            num_samples, scores = get_for_num_epochs(epoch)
            if scores:
                grid_values.append(scores)
    return grid_values


def get_rank_order(entries_from_db, selected_studies, include_ensembles):
    mo_types = ["point_pred", "stochastic", "ensemble", "sampling_bnn"] if include_ensembles else ["point_pred", "stochastic"]

    # List of list, for evey measurement (i.e., study - epsilon - ds combination), a list of length (num_metrics)
    measurements = []
    for study_name, study_i in selected_studies.items():
        for ds in ["nominal", "ood"]:
            for epsilon in ["s1", "s5", "s10"]:
                measures = []
                for mo in mo_types:
                    relevant_metrics = get_metrics(mo)
                    for rel_m in relevant_metrics:
                        def filter(val_dict):
                            return (
                                    val_dict["model_type"] == mo
                                    and val_dict["metric"] == rel_m
                                    and val_dict["epsilon"] == epsilon
                                    and val_dict["dataset"] == ds
                                    and val_dict["study"] == study_name
                            )

                        val_candidate = [v["harmonic_s"] for v in entries_from_db if filter(v)]
                        assert len(val_candidate) == 1
                        measures.append(val_candidate)
                measurements.append(measures)

    _, p = friedmanchisquare(*measurements)
    assert p < 10 ** (-5), "Friedmans-Test p is higher than E-5"
    rks = _avg_rank(measurements)
    rks_results = dict()
    i = 0
    for mo in mo_types:
        relevant_metrics = get_metrics(mo)
        rks_results[mo] = dict()
        for rel_m in relevant_metrics:
            rks_results[mo][rel_m] = round(rks[i], ROUND_DIGITS)
            i += 1
    return len(measurements), rks_results


def get_metrics(mo):
    if mo == 'point_pred':
        metrics = PP_METRICS
    else:
        metrics = BAYES_METRICS
    return metrics


def _avg_rank(measurements):
    arr = np.array(measurements)
    arr = np.reshape(arr, (arr.shape[0], arr.shape[1])) * -1
    ranks = []
    for i in range(arr.shape[0]):
        ranks.append(rankdata(arr[i], method='average'))
    mean_ranks = np.mean(np.array(ranks), axis=0)
    return mean_ranks


def rank_table():
    mni_n, mni_r = get_rank_order(values,
                                  selected_studies={"mnist": studies["mnist"]},
                                  include_ensembles=True)
    cif_n, cif_r = get_rank_order(values,
                                  selected_studies={"cifar10": studies["cifar10"]},
                                  include_ensembles=True)
    traf_n, traf_r = get_rank_order(values,
                                    selected_studies={"traffic": studies["traffic"]},
                                    include_ensembles=True)

    overa_n, overal_r = get_rank_order(values,
                                       selected_studies={
                                           "mnist": studies["mnist"],
                                           "cifar10": studies["cifar10"],
                                           "traffic": studies["traffic"],
                                       },
                                       include_ensembles=True)

    eff_studies = dict()
    for n, info in studies.items():
        if n.startswith("eff"):
            eff_studies[n] = info
    eff_n, eff_r = get_rank_order(values,
                                  selected_studies=eff_studies,
                                  include_ensembles=False)

    # TODO Print table with these values
    print("Breakpint")

    table = f"""
\\begin{{tabular}}{{@{{}}llcccccccc@{{}}}}
\\toprule
                                      &             &  \multicolumn{{3}}{{c}}{{Per Subject}}                                                                                          &&   \multicolumn{{2}}{{c}}{{Overall}}             \\\\
                                      &              & mnist                                    & cifar10                                  & traffic                                   &&  Full Studies                                & Pre-Trained                                                   \\\\
\multicolumn{{2}}{{c}}{{Technique}}                       & N={mni_n}                                & N={cif_n}                                & N={traf_n}                                &&  N={overa_n}                                 & N={eff_n}           \\\\ 
\\midrule
\multirow{{3}}{{*}}{{\\verti{{Point Pred.}}}}   & SM     & {mni_r["point_pred"]["max_softmax"]}     & {cif_r["point_pred"]["max_softmax"]}     & {traf_r["point_pred"]["max_softmax"]}     &&  {overal_r["point_pred"]["max_softmax"]}     & {eff_r["point_pred"]["max_softmax"]}             \\\\
                                      & PCS         & {mni_r["point_pred"]["pcs"]}             & {cif_r["point_pred"]["pcs"]}             & {traf_r["point_pred"]["pcs"]}             &&  {overal_r["point_pred"]["pcs"]}             & {eff_r["point_pred"]["pcs"]}                       \\\\
                                      & SME     & {mni_r["point_pred"]["softmax_entropy"]} & {cif_r["point_pred"]["softmax_entropy"]} & {traf_r["point_pred"]["softmax_entropy"]} &&  {overal_r["point_pred"]["softmax_entropy"]} & {eff_r["point_pred"]["softmax_entropy"]}  \\vspace{{\spacebetweenrankrows}}               \\\\
\multirow{{4}}{{*}}{{\\verti{{MC- Dropout}}}}      & VR  & {mni_r["stochastic"]['var_ratio']}       & {cif_r["stochastic"]['var_ratio']}       & {traf_r["stochastic"]['var_ratio']}       &&  {overal_r["stochastic"]['var_ratio']}       & {eff_r["stochastic"]['var_ratio']}             \\\\
                                                   & PE  & {mni_r["stochastic"]['pred_entropy']}    & {cif_r["stochastic"]['pred_entropy']}    & {traf_r["stochastic"]['pred_entropy']}    &&  {overal_r["stochastic"]['pred_entropy']}    & {eff_r["stochastic"]['pred_entropy']}             \\\\
                                                   & MI & {mni_r["stochastic"]['mutu_info']}       & {cif_r["stochastic"]['mutu_info']}       & {traf_r["stochastic"]['mutu_info']}       &&  {overal_r["stochastic"]['mutu_info']}       & {eff_r["stochastic"]['mutu_info']}             \\\\
                                                   & AS & {mni_r["stochastic"]['mean_sm']}         & {cif_r["stochastic"]['mean_sm']}         & {traf_r["stochastic"]['mean_sm']}         &&  {overal_r["stochastic"]['mean_sm']}         & {eff_r["stochastic"]['mean_sm']}   \\vspace{{\spacebetweenrankrows}}                \\\\
\multirow{{4}}{{*}}{{\\verti{{Ensem- ble}}}}       & VR  & {mni_r["ensemble"]['var_ratio']}         & {cif_r["ensemble"]['var_ratio']}         & {traf_r["ensemble"]['var_ratio']}         &&  {overal_r["ensemble"]['var_ratio']}         & n.a.                                                 \\\\
                                                   & PE  & {mni_r["ensemble"]['pred_entropy']}      & {cif_r["ensemble"]['pred_entropy']}      & {traf_r["ensemble"]['pred_entropy']}      &&  {overal_r["ensemble"]['pred_entropy']}      & n.a.                                                 \\\\
                                                   & MI & {mni_r["ensemble"]['mutu_info']}         & {cif_r["ensemble"]['mutu_info']}         & {traf_r["ensemble"]['mutu_info']}         &&  {overal_r["ensemble"]['mutu_info']}         & n.a.                                                 \\\\
                                                   & AS & {mni_r["ensemble"]['mean_sm']}           & {cif_r["ensemble"]['mean_sm']}           & {traf_r["ensemble"]['mean_sm']}           &&  {overal_r["ensemble"]['mean_sm']}           & n.a.                                     
\multirow{{4}}{{*}}{{\\verti{{Flip- out}}}}       & VR  & {mni_r["sampling_bnn"]['var_ratio']}         & {cif_r["sampling_bnn"]['var_ratio']}         & {traf_r["sampling_bnn"]['var_ratio']}         &&  {overal_r["sampling_bnn"]['var_ratio']}         & n.a.                                                 \\\\
                                                   & PE  & {mni_r["sampling_bnn"]['pred_entropy']}      & {cif_r["sampling_bnn"]['pred_entropy']}      & {traf_r["sampling_bnn"]['pred_entropy']}      &&  {overal_r["sampling_bnn"]['pred_entropy']}      & n.a.                                                 \\\\
                                                   & MI & {mni_r["sampling_bnn"]['mutu_info']}         & {cif_r["sampling_bnn"]['mutu_info']}         & {traf_r["sampling_bnn"]['mutu_info']}         &&  {overal_r["sampling_bnn"]['mutu_info']}         & n.a.                                                 \\\\
                                                   & AS & {mni_r["sampling_bnn"]['mean_sm']}           & {cif_r["sampling_bnn"]['mean_sm']}           & {traf_r["sampling_bnn"]['mean_sm']}           &&  {overal_r["sampling_bnn"]['mean_sm']}           & n.a.                                     
\end{{tabular}}
"""
    print(table)


def calc_avg_effnet_mean_lines(values_from_db):
    lines = []
    relevant_rows = [v for v in values_from_db if v["study"].startswith("eff")]
    assert len(relevant_rows) == 336, f"Expected 8 models * 7 quantifiers * 2 datasets * 3 epsilons, " \
                                      f"but found {len(relevant_rows)}"
    for mo in ["point_pred", "stochastic"]:
        mes = PP_METRICS if mo == 'point_pred' else BAYES_METRICS
        for m in mes:
            m_abbrev = abbreviate(m)
            line = ["effnet_avg", mo, m_abbrev]
            for d in ["nominal", "ood"]:
                for epsilon in ["s1",
                                "s5",
                                "s10"]:
                    def filter(val_dict):
                        return (
                                val_dict["model_type"] == mo
                                and val_dict["metric"] == m
                                and val_dict["epsilon"] == epsilon
                                and val_dict["dataset"] == d
                        )

                    v = [v for v in relevant_rows if filter(v)]
                    assert len(v) == 8, f"Expected 8 effnet studies, but found {len(v)}"
                    if epsilon == "s1":
                        line.append(sum([entry["accuracy"] for entry in v]) / 8)

                    line.append(sum([entry["s_accuracy"] for entry in v]) / 8)
                    line.append(sum([entry["acceptance_rate"] for entry in v]) / 8)
                    line.append(sum([entry["harmonic_s"] for entry in v]) / 8)
                    if mo == 'point_pred':  # We are in effnet method
                        line.append("n.a.")
                    else:
                        line.append(sum([entry["instability_r"] for entry in v]) / 8)
            lines.append(line)
    return lines


def abbreviate(m):
    if m == "max_softmax":
        return "SM"
    if m == "pcs":
        return "PCS"
    if m == "softmax_entropy":
        return "SME"
    if m == "var_ratio":
        return "VR"
    if m == "pred_entropy":
        return "PE"
    if m == "mutu_info":
        return "MI"
    if m == "mean_sm":
        return "MS"
    assert False, f"unknown quantifier {m}"


def huge_results_table_for_gdrive(values_from_db):
    print_lines = []
    for s, s_info in studies.items():
        model_types = _get_model_types(s_info)
        for mo in model_types:
            mes = PP_METRICS if mo == 'point_pred' else BAYES_METRICS
            for m in mes:
                m_abbrev = abbreviate(m)
                line = [s, mo, m_abbrev]
                for d in ["nominal", "ood"]:
                    for epsilon in ["s1",
                                    "s5",
                                    "s10"]:
                        def filter(val_dict):
                            return (
                                    val_dict["model_type"] == mo
                                    and val_dict["metric"] == m
                                    and val_dict["epsilon"] == epsilon
                                    and val_dict["dataset"] == d
                                    and val_dict["study"] == s
                            )

                        v = [v for v in values_from_db if filter(v)]
                        assert len(v) == 1
                        v = v[0]

                        if epsilon == "s1":
                            line.append(v["accuracy"])
                        line.append(v["s_accuracy"])
                        line.append(v["acceptance_rate"])
                        line.append(v["harmonic_s"])
                        line.append(v["instability_r"])
                print_lines.append(line)

    avg_effnet_values = calc_avg_effnet_mean_lines(values_from_db)
    print_lines += avg_effnet_values
    print("================ OVERALL RESULTS ================")
    print("================ OVERALL RESULTS ================")
    print("================ OVERALL RESULTS ================")
    for line in print_lines:
        print(",".join([str(l) for l in line]))


def _get_model_types(s_info):
    model_types = [
        'point_pred',
        'stochastic']
    if s_info['ensemble'] is True:
        model_types.append('ensemble')
        model_types.append('sampling_bnn')
    return model_types


def print_heatmaps():
    model_type = "ensemble"
    metric = "mutu_info"
    study = "traffic"
    source = "ood-all"
    s = "s1"

    grid = np.load(f"/root/assets/grid_values/{study}-{s}-{model_type}-{metric}-{source}.npy")
    mean_values = convolute(grid, np.mean)
    standard_deviations = convolute(grid, np.std)

    accuracy_hc = ['#e5e5ff', '#acacdf', '#7272bf', '#39399f', '#000080']
    accuracy_th = [0, 0.8, 0.9, 0.95, 1]
    accuracy_cdict = NonLinCdict(accuracy_th, accuracy_hc)
    accuracy_cm = LinearSegmentedColormap('test', accuracy_cdict)

    std_hc = ['#e5e5ff', '#acacdf', '#7272bf', '#39399f', '#000080', '#000080']
    std_th = [0, 0.04, 0.075, 0.15, 0.3, 1]  # Last two use same color
    std_cdict = NonLinCdict(std_th, std_hc)
    std_cm = LinearSegmentedColormap('test', std_cdict)

    _plot_heatmap(0, model_type, mean_values, accuracy_cm, metric, "supervised accuracy", source, study, "mean")
    _plot_heatmap(0, model_type, standard_deviations, std_cm, metric, "supervised accuracy", source, study, "std")


def _plot_heatmap(start_e, bayes_type, values, cm, metric, objective, src, study, aggregation):
    fig, ax = plt.subplots()
    if bayes_type == "ensemble":
        x_label = "# atomic models"
    elif bayes_type == "stochastic":
        x_label = "num_samples"
    elif bayes_type == "point_pred":
        x_label = "n.a."
    else:
        assert False, f"{bayes_type} not expected"
    im, cbar = heatmap(values, x_label, "epochs",
                       cm,
                       start_e,
                       vmin=0,
                       vmax=1,
                       ax=ax,
                       cmap="YlGn",
                       cbarlabel=f"{aggregation} {objective} - Filter Size: ({FILTER_SIZE_X_EPOCHS},{FILTER_SIZE_Y_SAMPLES})")
    # plt.title(f"{study}-{bayes_type}: MP using {metric} ")
    # fig.tight_layout()
    plt.savefig(f"/root/assets/plots/{objective}-{aggregation}-{study}-{bayes_type}-{metric}-{src}",
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    futures = []
    with ProcessPoolExecutor(max_workers=20) as poolexecutor:
        for study, study_info in studies.items():
            model_types = _get_model_types(study_info)
            for model_type in model_types:
                metrics = PP_METRICS if model_type == 'point_pred' else BAYES_METRICS
                for metric in metrics:
                    future = poolexecutor.submit(print_table_row,
                                                 study,
                                                 study_info,
                                                 model_type,
                                                 metric)
                    futures.append(future)

    values = []
    high_ps = 0
    low_ps = 0
    for future in futures:
        # values = future.result()
        # print(",".join([str(v) for v in values]))
        val, hps, lps = future.result()
        values += val
        high_ps += hps
        low_ps += lps

    rank_table()
    huge_results_table_for_gdrive(values)

    print("\n\n")
    print(f"In the sensitivity analysis, {high_ps} out of {low_ps + high_ps} entries hat a high p "
          f"(relative amount = {high_ps / (high_ps + low_ps)})")
    print_heatmaps()
