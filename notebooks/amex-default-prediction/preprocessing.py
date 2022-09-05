from sklearn import preprocessing
import pandas as pd
import numpy as np
import sqlite3
import tsfresh


def process_v1(customer_ids: list, conn: sqlite3.connect, test_mode=False):
    le = preprocessing.LabelEncoder()

    if test_mode:
        df = pd.read_sql_query(
            "SELECT * FROM test_data WHERE customer_ID IN ({seq})".format(
                seq="'" + "','".join(customer_ids) + "'"
            ),
            conn,
        ).set_index("customer_ID")
    else:
        df = pd.read_sql_query(
            "SELECT * FROM train_data WHERE customer_ID IN ({seq})".format(
                seq="'" + "','".join(customer_ids) + "'"
            ),
            conn,
        ).set_index("customer_ID")

    df.sort_values(["customer_ID", "S_2"], inplace=True)
    df.drop("S_2", axis=1, inplace=True)

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

    df[cat_columns] = df[cat_columns].astype(str)
    for cat_column in cat_columns:
        df[cat_column] = le.fit_transform(df[cat_column])

    categorical_df = df[cat_columns].groupby("customer_ID").max().copy()

    num_columns = list(set(df.columns) - set(cat_columns))

    df1 = (
        df[num_columns]
        .replace(r"^\s*$", np.nan, regex=True)
        .astype(float)
        .fillna(0)
        .copy()
    )
    df1 = df1.groupby("customer_ID").mean()

    df2 = tsfresh.extract_features(
        df1.reset_index(),
        column_id="customer_ID",
        default_fc_parameters=tsfresh.feature_extraction.MinimalFCParameters(),
        n_jobs=4,
        disable_progressbar=True,
    )

    if test_mode:
        return df2.join(categorical_df)

    label_df = pd.read_sql_query(
        "SELECT * FROM train_labels WHERE customer_ID IN ({seq})".format(
            seq="'" + "','".join(customer_ids) + "'"
        ),
        conn,
    ).set_index("customer_ID")

    return df2.join([categorical_df, label_df])


def process_v2(customer_ids: list, conn: sqlite3.connect, test_mode=False):
    le = preprocessing.LabelEncoder()

    tbl = "train_data"
    if test_mode:
        tbl = "test_data"

    df = (
        pd.read_sql_query(
            "SELECT * FROM {} WHERE customer_ID IN ({seq})".format(
                tbl, seq="'" + "','".join(customer_ids) + "'"
            ),
            conn,
        )
        .replace(r"^\s*$", np.nan, regex=True)
        .set_index("customer_ID")
    )

    if not test_mode:
        label_df = pd.read_sql_query(
            "SELECT * FROM train_labels WHERE customer_ID IN ({seq})".format(
                seq="'" + "','".join(customer_ids) + "'"
            ),
            conn,
        ).set_index("customer_ID")

    df.sort_values(["customer_ID", "S_2"], inplace=True)
    df.drop("S_2", axis=1, inplace=True)

    # Categorical
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

    for cat_column in cat_columns:
        df[cat_column] = le.fit_transform(df[cat_column])

    categorical_df = df[cat_columns].groupby("customer_ID").max().copy()

    categorical_df = (
        df[cat_columns].groupby("customer_ID").agg(["count", "last", "nunique"])
    )
    categorical_df.columns = ["_".join(x) for x in categorical_df.columns]

    # Numerical
    num_columns = list(set(df.columns) - set(cat_columns))

    numerical_df = df[num_columns].copy().fillna(0)
    numerical_df[numerical_df.columns] = numerical_df[numerical_df.columns].astype(
        float
    )

    numerical_df = numerical_df.groupby("customer_ID").agg(
        ["mean", "std", "min", "max", "last"]
    )
    numerical_df.columns = ["_".join(x) for x in numerical_df.columns]

    if test_mode:
        return numerical_df.join(categorical_df)

    return numerical_df.join([categorical_df, label_df])
