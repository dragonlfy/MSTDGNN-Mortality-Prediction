# from numpy importnp.nan

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Union
from scipy.interpolate import interp1d


def read_simple_hcv_record(filepath) -> Union[bool, Dict, Dict]:
    try:
        hcv_flag = True
        hcv = pd.read_csv(filepath, converters=HCV_Var_Converters)
        desc_dict = {'last_hour': f"{hcv.Hours.to_numpy()[-1]:.1f}"}
    except Exception as e:
        hcv_flag = False
        desc_dict = {'ErrorMsg': e.args}
        hcv = None

    return hcv_flag, desc_dict, hcv


def split_hcv(hcv, begin_hour, end_hour) -> Tuple[bool, Dict, Dict]:

    seg_desc_dict = {}
    if begin_hour < 0:
        error_msg = f"Begin hour ({begin_hour:.1f}) < 0."
        seg_desc_dict['ErrorMsg'] = error_msg
        return False, seg_desc_dict, None

    num_hours = int(end_hour - begin_hour + 0.1)
    cate_ts_list = []
    nume_ts_list = []

    valid_mask = np.logical_and(
        hcv['Hours'] > begin_hour - 4,
        hcv['Hours'] < end_hour,
    )
    valid_hcv = hcv[valid_mask].copy()
    # valid time series, whose hours < end_hour
    valid_hours = valid_hcv['Hours'].to_numpy()
    seg_all_hcv = valid_hcv[valid_hcv['Hours'] >= begin_hour].copy()
    # segment time series, whose begin_hour <= hours < end_hour
    hour_vals = np.arange(begin_hour + 0.5, end_hour, 1)
    # hours of seg_all_hcv for interpolating
    max_count_nonzero = 0
    for feat_name, is_cate in HCV_Is_Categorical_Channel.items():
        seg_vals = seg_all_hcv[feat_name].to_numpy()
        notnan_mask = ~np.isnan(seg_vals)
        count_nonzero = np.count_nonzero(notnan_mask)
        seg_desc_dict[feat_name] = count_nonzero
        max_count_nonzero = max(max_count_nonzero, count_nonzero)

    if max_count_nonzero < (num_hours // 2):
        error_msg = "No enough observations "
        error_msg += f"from {begin_hour:.1f}h to {end_hour:.1f}h."
        seg_desc_dict['ErrorMsg'] = error_msg
        return False, seg_desc_dict, None

    for feat_name, is_cate in HCV_Is_Categorical_Channel.items():
        seg_vals = seg_all_hcv[feat_name].to_numpy()
        notnan_mask = ~np.isnan(seg_vals)
        count_nonzero = seg_desc_dict[feat_name]
        if count_nonzero == 0:  # all values are none, using impute value
            impute_value = HCV_Var_Impute_Value[feat_name]
            if is_cate:
                impute_ts = np.ones((num_hours), dtype=np.int64)
                cate_ts_list.append(impute_value * impute_ts)
            else:
                impute_ts = np.ones((num_hours), dtype=np.float32)
                nume_ts_list.append(impute_value * impute_ts)
        elif count_nonzero < (num_hours // 2):  # using mean value
            notnan_vals = seg_vals[notnan_mask]
            val_mean = np.mean(notnan_vals)
            if is_cate:
                mean_ts = np.ones((num_hours), dtype=np.int64)
                mean_ts *= val_mean.astype(int)
                cate_ts_list.append(mean_ts)
            else:
                mean_ts = val_mean * np.ones((num_hours), dtype=np.float32)
                nume_ts_list.append(mean_ts)
        else:  # interpolating time series
            valid_vals = valid_hcv[feat_name].to_numpy()
            notnan_valid_mask = ~np.isnan(valid_vals)
            notnan_valid_hours = valid_hours[notnan_valid_mask]
            notnan_valid_vals = valid_vals[notnan_valid_mask]
            interpolate_func = interp1d(notnan_valid_hours,
                                        notnan_valid_vals, 'nearest',
                                        bounds_error=False,
                                        fill_value='extrapolate')
            interped_val = interpolate_func(hour_vals)
            # interpe time series for $hour_vals based on $notnan_vals
            # ranging from 0 to end_hour
            if is_cate:
                interped_val = np.around(interped_val, 0).astype(np.int64)
                num_possible = HCV_Num_Possible_Values[feat_name]
                interped_val[interped_val > (num_possible - 1)] = \
                    (num_possible - 1)  # max value
                interped_val[interped_val < 0] = 0  # min value
                cate_ts_list.append(interped_val)
            else:
                nume_ts_list.append(interped_val.astype(np.float32))

    cate_ts_arr = np.stack(cate_ts_list)
    nume_ts_arr = np.stack(nume_ts_list, dtype=np.float32)
    data_dict = {'category': cate_ts_arr, 'numeric': nume_ts_arr}
    return True, seg_desc_dict, data_dict


HCV_Var_Converters = {
    "Capillary refill rate":
        (lambda val: {0.0: 0, 1.0: 1}.get(val,np.nan)),
    "Diastolic blood pressure":
        (lambda val:np.nan if val == "" else float(val)),
    "Fraction inspired oxygen":
        (lambda val:np.nan if val == "" else float(val)),
    "Glascow coma scale eye opening":
        (lambda val: {
            "None": 0,
            "1 No Response": 1,
            "2 To pain": 2,
            "To Pain": 2,
            "3 To speech": 3,
            "To Speech": 3,
            "4 Spontaneously": 4,
            "Spontaneously": 4,
        }.get(val,np.nan)),
    "Glascow coma scale motor response":
        (lambda val: {
            "1 No Response": 1,
            "No response": 1,
            "2 Abnorm extensn": 2,
            "Abnormal extension": 2,
            "3 Abnorm flexion": 3,
            "Abnormal Flexion": 3,
            "4 Flex-withdraws": 4,
            "Flex-withdraws": 4,
            "5 Localizes Pain": 5,
            "Localizes Pain": 5,
            "6 Obeys Commands": 6,
            "Obeys Commands": 6
        }.get(val,np.nan) - 1),
    "Glascow coma scale total":
        (lambda val: {
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "10": 10,
            "11": 11,
            "12": 12,
            "13": 13,
            "14": 14,
            "15": 15,
        }.get(val,np.nan) - 3),
    "Glascow coma scale verbal response":
        (lambda val: {
            "No Response-ETT": 1,
            "No Response": 1,
            "1 No Response": 1,
            "1.0 ET/Trach": 1,
            "2 Incomp sounds": 2,
            "Incomprehensible sounds": 2 - 1,
            "3 Inapprop words": 3,
            "Inappropriate Words": 3,
            "4 Confused": 4,
            "Confused": 4,
            "5 Oriented": 5,
            "Oriented": 5
        }.get(val,np.nan) - 1),
    "Glucose":
        (lambda val:np.nan if val == "" else float(val)),
    "Heart Rate":
        (lambda val:np.nan if val == "" else float(val)),
    "Height":
        (lambda val:np.nan if val == "" else float(val)),
    "Mean blood pressure":
        (lambda val:np.nan if val == "" else float(val)),
    "Oxygen saturation":
        (lambda val:np.nan if val == "" else float(val)),
    "Respiratory rate":
        (lambda val:np.nan if val == "" else float(val)),
    "Systolic blood pressure":
        (lambda val:np.nan if val == "" else float(val)),
    "Temperature":
        (lambda val:np.nan if val == "" else float(val)),
    "Weight":
        (lambda val:np.nan if val == "" else float(val)),
    "pH":
        (lambda val:np.nan if val == "" else float(val)),
}


HCV_Is_Categorical_Channel = {
    'Capillary refill rate': True,
    'Diastolic blood pressure': False,
    'Fraction inspired oxygen': False,
    'Glascow coma scale eye opening': True,
    'Glascow coma scale motor response': True,
    'Glascow coma scale total': True,
    'Glascow coma scale verbal response': True,
    'Glucose': False,
    'Heart Rate': False,
    # 'Height': False,
    'Mean blood pressure': False,
    'Oxygen saturation': False,
    'Respiratory rate': False,
    'Systolic blood pressure': False,
    'Temperature': False,
    # 'Weight': False,
    'pH': False,
}

HCV_Var_Impute_Value = {
    'Capillary refill rate': 0,
    'Diastolic blood pressure': 59,
    'Fraction inspired oxygen': 0.21,
    'Glascow coma scale eye opening': 4,
    'Glascow coma scale motor response': 5,
    'Glascow coma scale total': 12,
    'Glascow coma scale verbal response': 4,
    'Glucose': 128,
    'Heart Rate': 86,
    'Height': 170,
    'Mean blood pressure': 77,
    'Oxygen saturation': 98,
    'Respiratory rate': 19,
    'Systolic blood pressure': 118,
    'Temperature': 36,
    'Weight': 81,
    'pH': 7.4,
}

HCV_Num_Possible_Values = {
    'Capillary refill rate': 2,
    'Glascow coma scale eye opening': 5,
    'Glascow coma scale motor response': 6,
    'Glascow coma scale total': 13,
    'Glascow coma scale verbal response': 5,
}
