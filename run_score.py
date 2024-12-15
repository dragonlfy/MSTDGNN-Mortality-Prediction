from Models.SAPS_II import calculate_saps2_score
import pandas as pd
import os
import numpy as np
from sklearn import metrics
from scipy.special import expit
from tqdm import tqdm

metrics_functions = {
   'ACC': lambda y, p: metrics.accuracy_score(y, p > 0),
   'AUROC': lambda y, p: metrics.roc_auc_score(y, expit(p)),
   'F1': lambda y, p: metrics.f1_score(y, p > 0, zero_division=0.),
   'Fbeta': lambda y, p: metrics.fbeta_score(y, p > 0, beta=np.sqrt(11.7), zero_division=0.),
   'Precision': lambda y, p: metrics.precision_score(y, p > 0, zero_division=0.),
   'Recall': lambda y, p: metrics.recall_score(y, p > 0, zero_division=0.),
   'AUPRC': lambda y, p: metrics.average_precision_score(y, expit(p)),
}

dataset = 'mimic3'
df = pd.read_csv(
    f'/home/rongqin/codes/icu-benchmarks/Datasets/{dataset}/icustay_desc_hf.csv')
patients = df['SUBJECT_ID'].to_list()
stay_path = f'/home/rongqin/codes/icu-benchmarks/Datasets/{dataset}/stays'


predictions = []
labels = []


def convert_score(score):
    return 0 if score < 80 else 1


for p in tqdm(patients):
    p_path = os.path.join(stay_path, str(p))
    time_file = [f for f in os.listdir(p_path) if 'timeseries' in f]
    stay_files = [f for f in os.listdir(p_path) if 'stays' in f]
    diagnoses_files = [f for f in os.listdir(p_path) if 'diagnoses' in f]
    if not time_file:
        continue
    if not stay_files:
        continue
    if not diagnoses_files:
        continue
    df_time = pd.read_csv(os.path.join(p_path, time_file[0]))
    df_stay = pd.read_csv(os.path.join(p_path, stay_files[0]))
    df_diagnoses = pd.read_csv(os.path.join(p_path, diagnoses_files[0]))
    age = df_stay['AGE'].iloc[0]
    if not df_time[df_time['Hours'] > 24].index.tolist():
        continue
    row = df_time[df_time['Hours'] > 24].index[0]
    heart_rate = df_time['Heart Rate'].iloc[row]
    systolic_bp = df_time['Systolic blood pressure'].iloc[row]
    temperature = df_time['Temperature'].iloc[row]
    paO2_FiO2_ratio = df_time['Oxygen saturation'].iloc[row]
    urine_output = 550
    blood_urea_nitrogen = 28
    white_blood_cell_count = 25
    potassium_level = 6
    sodium_level = 125
    bicarbonate_level = 15
    bilirubin_level = 6
    glasgow_coma_scale = df_time['Glascow coma scale total'].iloc[row]
    chronic_diseases = len(df_diagnoses)
    admission_type = 'unplanned'
    mechanical_ventilation = True
    ph_level = df_time['pH'].iloc[row]
    heart_rate = heart_rate or 86
    systolic_bp = systolic_bp or 118
    temperature = temperature or 36
    paO2_FiO2_ratio = paO2_FiO2_ratio or 98
    glasgow_coma_scale = glasgow_coma_scale or 12
    ph_level = ph_level or 7.4
    saps2_score = calculate_saps2_score(age, heart_rate, systolic_bp, temperature, paO2_FiO2_ratio, urine_output,
                                        blood_urea_nitrogen, white_blood_cell_count, potassium_level, sodium_level,
                                        bicarbonate_level, bilirubin_level, glasgow_coma_scale, chronic_diseases,
                                        admission_type, mechanical_ventilation, ph_level)
    
    label = df_stay['MORTALITY_INUNIT'].iloc[0]

    predictions.append(convert_score(saps2_score))  # Predicted class
    labels.append(label)

predictions = np.array(predictions)
labels = np.array(labels)


# for metric_name, metric_func in metrics_functions.items():
#     score = metric_func(labels, predictions)
#     print(f"{metric_name}: {score}")


with open(f"logs/{dataset}/SAPS_II.txt", "w") as file:
    for metric_name, metric_func in metrics_functions.items():
        score = metric_func(labels, predictions)
        file.write(f"{metric_name}: {score}\n")