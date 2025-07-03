import os
import json
import argparse
import pathlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             accuracy_score, f1_score)


def parse_opt():
    parser = argparse.ArgumentParser(
        description='Random Forest model for age regression or gender classification using gait features')
    parser.add_argument('--gait_parameters', type=str, required=True, help='CSV with gait parameters')
    parser.add_argument('--patients_measures', type=str, required=True, help='CSV with patient measures')
    parser.add_argument('--partitions_path', type=str, required=True, help='Directory with train/val/test partitions')
    parser.add_argument('--features', nargs='+', required=True, help='Features used for training')
    parser.add_argument('--evaluation_path', type=str, required=True, help='Path to store evaluation results')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the forest')
    parser.add_argument('--seed', type=int, default=27, help='Random seed')
    parser.add_argument('--target', type=str, default='Age', help='Target column name')
    parser.add_argument('--task', choices=['age', 'gender'], default='age',
                        help='Prediction task: age regression or gender classification')
    parser.add_argument('--n_decimals', type=int, default=3, help='Decimals for reporting metrics')
    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def get_data(gait_path, measures_path, partitions_file, features, target, task):
    df1 = pd.read_csv(gait_path)
    df2 = pd.read_csv(measures_path)
    df = pd.merge(df1, df2, on='ID')

    with open(partitions_file, 'r') as f:
        partitions = json.load(f)

    train_ids = partitions['train']
    val_ids = partitions['validation']
    test_ids = partitions['test']

    def subset(ids):
        sub = df[df['ID'].isin(ids)]
        dtype = 'f' if task == 'age' else 'uint8'
        return sub[features].to_numpy(), sub[target].to_numpy(dtype=dtype)

    X_train, y_train = subset(train_ids)
    X_val, y_val = subset(val_ids)
    X_test, y_test = subset(test_ids)

    return {
        'train': {'data': X_train, 'target': y_train},
        'val': {'data': X_val, 'target': y_val},
        'test': {'data': X_test, 'target': y_test}
    }


def preprocess_data(data, scale_target=True):
    imp = SimpleImputer(strategy='mean')
    imp.fit(data['train']['data'])
    for split in ['train', 'val', 'test']:
        data[split]['data'] = imp.transform(data[split]['data'])

    scaler = StandardScaler()
    scaler.fit(data['train']['data'])
    for split in ['train', 'val', 'test']:
        data[split]['data'] = scaler.transform(data[split]['data'])

    if scale_target:
        target_scaler = MinMaxScaler()
        for split in ['train', 'val', 'test']:
            reshaped = data[split]['target'].reshape(-1, 1)
            if split == 'train':
                target_scaler.fit(reshaped)
            data[split]['target'] = target_scaler.transform(reshaped)
    else:
        target_scaler = None

    return data, target_scaler


def evaluate_model_by_age_groups(y_true, y_pred, age_groups):
    group_errors = {g: [] for g in age_groups}
    for true_age, pred_age in zip(y_true, y_pred):
        for group, (age_min, age_max) in age_groups.items():
            if age_min <= true_age < age_max:
                group_errors[group].append(abs(true_age - pred_age))
                break
    return {g: np.mean(errs) for g, errs in group_errors.items()}


def main(args):
    GAIT_PATH = args.gait_parameters
    MEASURES_PATH = args.patients_measures
    PARTITIONS_PATH = args.partitions_path
    FEATURES = args.features
    TARGET = args.target
    TASK = args.task
    EVAL_PATH = args.evaluation_path
    SEED = args.seed
    N_EST = args.n_estimators
    N_DECIMALS = args.n_decimals

    pathlib.Path(EVAL_PATH).mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    if TASK == 'age':
        results = {"mse": [], "rmse": [], "mae": [], "manual_mae": [], "std_mae": [], "partition_order": []}
        age_groups = {"18-34": (18, 35), "35-49": (35, 50), "50-64": (50, 65)}
        results_by_age = {"18-34": [], "35-49": [], "50-64": []}
    else:
        results = {"accuracy": [], "f1": [], "partition_order": []}
        age_groups = None
        results_by_age = None

    for partition_file in os.listdir(PARTITIONS_PATH):
        part_path = os.path.join(PARTITIONS_PATH, partition_file)
        data = get_data(GAIT_PATH, MEASURES_PATH, part_path, FEATURES, TARGET, TASK)
        data, target_scaler = preprocess_data(data, scale_target=(TASK == 'age'))

        X_train = np.concatenate([data['train']['data'], data['val']['data']])
        y_train = np.concatenate([data['train']['target'], data['val']['target']]).ravel()

        if TASK == 'age':
            model = RandomForestRegressor(n_estimators=N_EST, random_state=SEED)
            model.fit(X_train, y_train)

            y_preds = model.predict(data['test']['data'])
            y_preds = target_scaler.inverse_transform(y_preds.reshape(-1, 1)).flatten()
            y_true = target_scaler.inverse_transform(data['test']['target'].reshape(-1, 1)).flatten()

            age_errors = evaluate_model_by_age_groups(y_true, y_preds, age_groups)
            for k in age_groups:
                results_by_age[k].append(age_errors[k])

            total = np.abs(y_preds - y_true)
            mse = round(mean_squared_error(y_true, y_preds), N_DECIMALS)
            results['mse'].append(mse)
            results['rmse'].append(np.sqrt(mse))
            results['mae'].append(round(mean_absolute_error(y_true, y_preds), N_DECIMALS))
            results['manual_mae'].append(round(np.mean(total), N_DECIMALS))
            results['std_mae'].append(round(np.std(total), N_DECIMALS))
        else:
            model = RandomForestClassifier(n_estimators=N_EST, random_state=SEED)
            model.fit(X_train, y_train)

            y_preds = model.predict(data['test']['data'])
            y_true = data['test']['target']

            results['accuracy'].append(accuracy_score(y_true, y_preds))
            results['f1'].append(f1_score(y_true, y_preds))

        results['partition_order'].append(partition_file)

    if TASK == 'age':
        results['mean_mse'] = round(np.mean(results['mse']), N_DECIMALS)
        results['mean_mae'] = round(np.mean(results['mae']), N_DECIMALS)
        results['mean_std_mae'] = round(np.mean(results['std_mae']), N_DECIMALS)

        for k in age_groups:
            results_by_age[k] = np.mean(results_by_age[k])

        with open(os.path.join(EVAL_PATH, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        with open(os.path.join(EVAL_PATH, 'age_group_errors.json'), 'w', encoding='utf-8') as f:
            json.dump(results_by_age, f, ensure_ascii=False, indent=4)
    else:
        results['mean_accuracy'] = round(np.mean(results['accuracy']), N_DECIMALS)
        results['mean_f1'] = round(np.mean(results['f1']), N_DECIMALS)

        with open(os.path.join(EVAL_PATH, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    args = parse_opt()
    main(args)
