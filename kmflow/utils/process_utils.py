from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler


def apply_normalizer(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) == 0:
        return df
    scaler = Normalizer().fit(df[num_cols])
    df_out = df.copy()
    df_out[num_cols] = scaler.transform(df[num_cols])
    return df_out


def apply_standardization(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) == 0:
        return df
    scaler = StandardScaler().fit(df[num_cols])
    df_out = df.copy()
    df_out[num_cols] = scaler.transform(df[num_cols])
    return df_out


def apply_minmax(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) == 0:
        return df
    scaler = MinMaxScaler().fit(df[num_cols])
    df_out = df.copy()
    df_out[num_cols] = scaler.transform(df[num_cols])
    return df_out


def apply_log1p(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) == 0:
        return df
    df_out = df.copy()
    df_out[num_cols] = np.log1p(df[num_cols])
    return df_out


def apply_yeo_johnson(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) == 0:
        return df
    scaler = preprocessing.PowerTransformer(method="yeo-johnson").fit(df[num_cols])
    df_out = df.copy()
    df_out[num_cols] = scaler.transform(df[num_cols])
    return df_out
