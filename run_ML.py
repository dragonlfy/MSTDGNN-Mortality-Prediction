from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from Experiment.config import ExperimentConfig
from Experiment.args_parser import parse_args
import numpy as np
from sklearn import metrics
from scipy.special import expit
from sklearn.model_selection import KFold


metrics_functions = {
   'ACC': lambda y, p: metrics.accuracy_score(y, p > 0),
   'AUROC': lambda y, p: metrics.roc_auc_score(y, expit(p)),
   'F1': lambda y, p: metrics.f1_score(y, p > 0, zero_division=0.),
   'Fbeta': lambda y, p: metrics.fbeta_score(y, p > 0, beta=np.sqrt(11.7), zero_division=0.),
   'Precision': lambda y, p: metrics.precision_score(y, p > 0, zero_division=0.),
   'Recall': lambda y, p: metrics.recall_score(y, p > 0, zero_division=0.),
   'AUPRC': lambda y, p: metrics.average_precision_score(y, expit(p)),
}


args, model_args = parse_args()
cfg = ExperimentConfig(args.dataset, args.data, args.model, model_args, args.n_fold, 0)

# Get the datasets
train_dataset, val_dataset, test_dataset, _ = cfg.datamodule.get_dataloaders(-1)

# Unpack the datasets
X_train = []
y_train = []
# Iterate over the DataLoader
for batch in train_dataset:
    X_train.append(batch[0].numpy())  # Assuming the features are at index 0
    y_train.append(batch[-1].numpy())  # Assuming the labels are at the last index

# Convert the lists to NumPy arrays
X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)


X_val = []
y_val = []
# Iterate over the DataLoader
for batch in val_dataset:
    X_val.append(batch[0].numpy())  # Assuming the features are at index 0
    y_val.append(batch[-1].numpy())  # Assuming the labels are at the last index

# Convert the lists to NumPy arrays
X_val = np.concatenate(X_val)
y_val = np.concatenate(y_val)


# # Flatten the labels
X_train_flattened = X_train.reshape((X_train.shape[0], -1))
X_val_flattened = X_val.reshape((X_val.shape[0], -1))

X_combined = np.concatenate((X_train_flattened, X_val_flattened))
Y_combined = np.concatenate((y_train, y_val))

# Initialize the models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LinearRegression()
log_reg = LogisticRegression(random_state=42)
svm = SVC(random_state=42)
dt = DecisionTreeClassifier(random_state=42)
xgb = XGBClassifier(random_state=42)

# Train the model
rf.fit(X_train_flattened, y_train)
lr.fit(X_train_flattened, y_train)
log_reg.fit(X_train_flattened, y_train)
svm.fit(X_train_flattened, y_train)
dt.fit(X_train_flattened, y_train)
xgb.fit(X_train_flattened, y_train)

# Predict on the validation set
models = {
    'RandomForestClassifier': rf,
    'LinearRegression': lr,
    'Logistic Regression': log_reg,
    'SVM': svm,
    'Decision Tree': dt,
    'XGBoost': xgb
}


kf = KFold(n_splits=5, random_state=42, shuffle=True)
with open("ML_result_mimic4.txt", "w") as file:
    for model_name, model in models.items():
        file.write(f'\n{model_name} Metrics:\n')
        scores = {name: [] for name in metrics_functions.keys()}

        for fold, (train_index, val_index) in enumerate(kf.split(X_combined), 1):
            X_train_fold = X_combined[train_index]
            y_train_fold = Y_combined[train_index]
            X_val_fold = X_combined[val_index]
            y_val_fold = Y_combined[val_index]

            # Train the model
            model.fit(X_train_fold, y_train_fold)

            # Predict on the validation fold
            predictions = model.predict(X_val_fold)

            # Calculate and store the metrics
            for name, func in metrics_functions.items():
                score = func(y_val_fold, predictions)
                scores[name].append(score)

        # Print the average metrics across the folds
        for name, scores_list in scores.items():
            avg_score = np.mean(scores_list)
            std_score = np.std(scores_list)
            var_score = np.var(scores_list)
            file.write(f'{name}: {avg_score} ± {std_score} (variance: {var_score})\n')

# for model_name, model in models.items():
#     predictions = model.predict(X_val_flattened)
#     print(f'\n{model_name} Metrics:')
#     for name, func in metrics_functions.items():
#         score = func(y_val, predictions)
#         print(f'{name}: {score}')
