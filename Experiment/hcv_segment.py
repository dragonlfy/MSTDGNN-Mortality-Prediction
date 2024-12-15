import pickle as pkl
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm
from Experiment.hcv_reader import HCV_Var_Impute_Value, read_simple_hcv_record, split_hcv

import Experiment.mimic4 as mimic4_config
from Experiment.config import ExperimentConfig as Exp_Config


def read_icustay_desc(icustay_desc_path):
    parse_dates = ["DEATHTIME", "INTIME", "OUTTIME", "DOD", "DISCHTIME"]
    icustay_desc = pd.read_csv(icustay_desc_path, parse_dates=parse_dates)
    return icustay_desc


def slide_stay(in_tuple):
    idx, row = in_tuple
    exp_config = Exp_Config('mimic4', 'DPGap4SPGap48Len24Str12Art2Grp8_hf')
    age = row.AGE
    gender = 0.0 if row.GENDER == "F" else 1.0
    mort_unit = row.MORTALITY_INUNIT
    mort_hosp = row.MORTALITY_INHOSPITAL
    # mort_in30days = row.MORTALITY_IN30DAYS
    subject_id = row.SUBJECT_ID
    icustay_id = row.ICUSTAY_ID
    # print(row)
    ts_fpath = f"{mimic4_config.stays_folder_path}"
    dm_fpath = ts_fpath + f"{subject_id}/episode_{icustay_id}.csv"
    try:
        dm = pd.read_csv(dm_fpath)
    except Exception:
        return None

    weight, height = dm.loc[0, ["Weight", "Height"]]
    if np.isnan(weight):
        weight = HCV_Var_Impute_Value["Weight"]
    if np.isnan(height):
        height = HCV_Var_Impute_Value["Height"]
    ts_fpath += f"{subject_id}/episode_{icustay_id}_timeseries.csv"
    flag, desc_dict, hcv = read_simple_hcv_record(ts_fpath)

    if not flag:
        return None

    if mort_unit:
        if isinstance(row.DEATHTIME, pd.Timestamp):
            los = row.DEATHTIME - row.INTIME
        else:
            los = row.DOD - row.INTIME

        out_hour = los.total_seconds() / 360.0
        window_end = out_hour - exp_config.death_pred_gap

    else:
        if isinstance(row.OUTTIME, pd.Timestamp):
            los = row.OUTTIME - row.INTIME
        else:
            los = row.DISCHTIME - row.INTIME
        out_hour = los.total_seconds() / 360.0

    window_end = out_hour - exp_config.survival_pred_gap
    death_hour = out_hour if mort_unit else (99 * 365 * 24)

    first_hour = hcv.Hours.to_numpy()[0]  # maybe less than 0
    window_begin = max(first_hour + exp_config.artifact_length, 0.0)

    if window_begin >= window_end - exp_config.segment_length:
        return None

    seg_begin_hour = window_end - exp_config.segment_length
    segment_list = []
    while seg_begin_hour > window_begin:
        seg_end_hour = seg_begin_hour + exp_config.segment_length
        seg_flag, _, data_dict = split_hcv(hcv, seg_begin_hour, seg_end_hour)
        if seg_flag:
            data_dict["LoStay"] = window_end - seg_end_hour
            data_dict["LoSurv"] = death_hour - seg_end_hour
            data_dict["age"] = age
            data_dict["gender"] = gender
            data_dict["weight"] = weight
            data_dict["height"] = height
            data_dict["mort_unit"] = mort_unit
            data_dict["mort_hosp"] = mort_hosp
            segment_list.append((seg_begin_hour, data_dict))
            seg_begin_hour -= exp_config.segment_stride
        else:
            seg_begin_hour -= 2.0

    if len(segment_list) == 0 or segment_list[-1][0] > 6.0:
        seg_end_hour = window_begin + exp_config.segment_length
        seg_flag, _, data_dict = split_hcv(hcv, window_begin, seg_end_hour)
        if seg_flag:
            data_dict["LoStay"] = window_end - seg_end_hour
            data_dict["LoSurv"] = death_hour - seg_end_hour
            data_dict["age"] = age
            data_dict["gender"] = gender
            data_dict["weight"] = weight
            data_dict["height"] = height
            data_dict["mort_unit"] = mort_unit
            data_dict["mort_hosp"] = mort_hosp
            segment_list.append((window_begin, data_dict))

    return segment_list


def slide_stay_with_error_handling(row):
    try:
        return slide_stay(row)
    except Exception as e:
        print(f"Error processing row {row[0]}: {e}")  # row[0] 是索引
        return None


def foreach_stay():
    # ages = icustay_desc.AGE
    # genders = icustay_desc.GENDER
    # mortality_inunit = icustay_desc.MORTALITY_INUNIT
    # mortality_inhospital = icustay_desc.MORTALITY_INHOSPITAL
    icustay_desc = read_icustay_desc(mimic4_config.icustay_desc_path)
    p = Pool(64)
    seg_list_list2 = list(
        tqdm(
            p.imap(slide_stay_with_error_handling, icustay_desc.iterrows()),
            # p.imap(slide_stay, icustay_desc.iterrows()),
            total=icustay_desc.shape[0],
        )
    )

    seg_list_list = []
    for seg_list in seg_list_list2:
        if seg_list is None or len(seg_list) == 0:
            continue
        seg_list_list.append(seg_list)

    exp_config = Exp_Config(dataset='mimic4')
    segs_group = [[] for _ in range(exp_config.num_groups)]
    for seg_list in seg_list_list:
        if seg_list is None or len(seg_list) == 0:
            continue

        grp_idx_end = min(exp_config.num_groups - 2, len(seg_list) - 1)
        for grp_idx in range(grp_idx_end):
            segs_group[grp_idx].append(seg_list[grp_idx])
        for seg_idx in range(exp_config.num_groups - 2, len(seg_list) - 1):
            segs_group[-2].append(seg_list[seg_idx])
        segs_group[-1].append(seg_list[-1])

    for grp_idx in range(exp_config.num_groups):
        print(grp_idx, len(segs_group[grp_idx]))

    # saving_path = f"{mimic4_config.segment_folder}/{exp_config._logkey}.pkl"
    saving_path = f"{mimic4_config.segment_folder}/DPGap4SPGap48Len24Str12Art2Grp8_hf.pkl"
    with open(saving_path, "wb") as wbf:
        pkl.dump(seg_list_list, wbf)


if __name__ == "__main__":
    foreach_stay()
