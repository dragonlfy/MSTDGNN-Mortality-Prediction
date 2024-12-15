import os
from datetime import datetime
import numpy as np


def save_summary(n_fold, dataset, logkey, window, result_summary, model_args):
    current_date = datetime.now()
    string_date = current_date.strftime(r"%m%d%H%M%S")
    os.makedirs(f"logs/{dataset}/{logkey}", exist_ok=True)
    log_path = f"logs/{dataset}/{logkey}/{string_date}_{window}_Summary.csv"

    result_mean = {
        key: f"{np.mean(val_list):.3f}"
        for key, val_list in result_summary.items()
    }
    result_std = {
        key: f"{np.std(val_list):.3f}"
        for key, val_list in result_summary.items()
    }

    with open(log_path, 'w') as wfile:
        print('', *result_summary.keys(), sep=',', file=wfile)
        for i_fold in range(n_fold):
            val_list = [
                f"{result_summary[key][i_fold]:.3f}"
                for key in result_summary.keys()
            ]
            print(i_fold, *val_list, sep=',', file=wfile)

        print('Mean', *result_mean.values(), sep=',', file=wfile)
        print('std', *result_std.values(), sep=',', file=wfile)

    log_path = f"logs/{dataset}/{logkey}/{string_date}_{window}_ModelArgs.txt"
    with open(log_path, 'w') as wfile:
        print(model_args, file=wfile)


def save_summary_ml(n_fold, dataset, logkey, window, result_summary, model_args):
    current_date = datetime.now()
    string_date = current_date.strftime(r"%m%d%H%M%S")
    os.makedirs(f"logs/{dataset}/{logkey}", exist_ok=True)
    log_path = f"logs/{dataset}/{logkey}/{string_date}_{window}_Summary.csv"

    result_mean = {
        key: f"{np.mean(val_list):.3f}"
        for key, val_list in result_summary.items()
    }
    result_std = {
        key: f"{np.std(val_list):.3f}"
        for key, val_list in result_summary.items()
    }

    with open(log_path, 'w') as wfile:
        print('', *result_summary.keys(), sep=',', file=wfile)
        for i_fold in range(n_fold):
            val_list = [
                f"{result_summary[key][i_fold]:.3f}"
                for key in result_summary.keys()
            ]
            print(i_fold, *val_list, sep=',', file=wfile)

        print('Mean', *result_mean.values(), sep=',', file=wfile)
        print('std', *result_std.values(), sep=',', file=wfile)

    log_path = f"logs/{dataset}/{logkey}/{string_date}_{window}_ModelArgs.txt"
    with open(log_path, 'w') as wfile:
        print(model_args, file=wfile)
