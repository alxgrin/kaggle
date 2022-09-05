import pandas as pd
from catboost import CatBoostClassifier, Pool, sum_models, to_classifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.preprocessing import StandardScaler


def cbm_v1(df: pd.DataFrame, selected_features: np.array, use_scaler=True):
    scaler = StandardScaler()
    skf = StratifiedKFold(n_splits=5)

    y = df[["target"]]
    X = df.drop("target", axis=1)

    if selected_features.size:
        X = X[selected_features]

    if use_scaler:
        cat_columns = [
            "B_30",
            "B_38",
            "D_114",
            "D_116",
            "D_117",
            "D_120",
            "D_126",
            "D_63",
            "D_64",
            "D_66",
            "D_68",
        ]

        num_columns = list(set(X.columns) - set(cat_columns))
        X[num_columns] = scaler.fit_transform(X[num_columns])

    ensemble = []

    for train_index, val_index in skf.split(X, y):
        X_sub_train, X_sub_valid = X.iloc[train_index], X.iloc[val_index]
        y_sub_train, y_sub_valid = y.iloc[train_index], y.iloc[val_index]

        train_pool = Pool(X_sub_train, y_sub_train)
        valid_pool = Pool(X_sub_valid, y_sub_valid)

        model = CatBoostClassifier()
        model.fit(train_pool, eval_set=valid_pool, verbose=False)

        ensemble.append(model)

    models_avrg = sum_models(ensemble, weights=[1.0 / len(ensemble)] * len(ensemble))
    return to_classifier(models_avrg)
