import pickle

from emp_uncertainty.traffic_bnn_search.settings import SETTINGS

ASSETS_PATH = "/root/assets"

RUNS = [
    "RERUN_0",
    "RERUN_1",
    "RERUN_2",
    "RERUN_3",
    "RERUN_4",
    "RERUN_5",
    "RERUN_6",
    "RERUN_7",
    "RERUN_8",
    "RERUN_9"
]


def read_values(rerun_id):
    histories = []
    for i in range(len(SETTINGS)):
        f = f"{ASSETS_PATH}/traffic_bnn_grid/{rerun_id}/training_histories/{i}-history.pickle"
        with open(f, "rb") as fb:
            histories.append(pickle.load(fb))
    return histories


def print_table():
    lines = []
    histories = [read_values(i) for i in RUNS]

    for setting_id, s in enumerate(SETTINGS):

        # line = f"{setting_id}" \
        #        f"&{s['cf']}  " \
        #        f"&{s['cd']}  " \
        #        f"&{s['opt']} " \
        #        f"&{s['lr']}  " \
        #        f"&{s['mom'] if s['mom'] != 0.0 else '-'} "
        line = f"{setting_id}" \
               f"&{s['cf']}  " \
               f"/ {s['cd']} " \
               f"&{s['opt']} / " \
               f"{s['lr']} / " \
               f"{s['mom'] if s['mom'] != 0.0 else '-'}  "
        for h in histories:
            val = h[setting_id]
            if val == "InvalidArgumentError()":
                print_val = "\hl{NaN}"
            else:
                print_val = round(val['val_accuracy'][-1], 2)
                if print_val < 0.9:
                    print_val = "\hl{" + str(print_val) + "}"
            line += f"& {print_val}"

        line += "\\\\"

        lines.append(line)

    for l in lines:
        print(l)


if __name__ == '__main__':
    print_table()
