from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
from pathlib import Path


def write_csv(dataframe: pd.DataFrame, prefix: str, suffix: str, output_dir: Path) -> Path:
    """Write `dataframe` to a CSV named {prefix}_{suffix}.csv in `output_dir`; return file path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{prefix}_{suffix}.csv"
    dataframe.to_csv(file_path, index=False)
    return file_path


def apply_normalizer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply L2 normalization to all numeric columns (row-wise)."""
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) == 0:
        return df
    scaler = Normalizer().fit(df[num_cols])
    df_out = df.copy()
    df_out[num_cols] = scaler.transform(df[num_cols])
    return df_out


def apply_standardization(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard scaling (zero mean, unit variance) to numeric columns."""
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) == 0:
        return df
    scaler = StandardScaler().fit(df[num_cols])
    df_out = df.copy()
    df_out[num_cols] = scaler.transform(df[num_cols])
    return df_out


def apply_minmax(df: pd.DataFrame) -> pd.DataFrame:
    """Rescale numeric columns to the [0, 1] range using Min-Max scaling."""
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) == 0:
        return df
    scaler = MinMaxScaler().fit(df[num_cols])
    df_out = df.copy()
    df_out[num_cols] = scaler.transform(df[num_cols])
    return df_out


def apply_log1p(df: pd.DataFrame) -> pd.DataFrame:
    """Apply natural log transform to numeric columns using log1p (log(1 + x))."""
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) == 0:
        return df
    df_out = df.copy()
    df_out[num_cols] = np.log1p(df[num_cols])
    return df_out


def apply_yeo_johnson(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Yeo-Johnson power transformation to numeric columns."""
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) == 0:
        return df
    scaler = preprocessing.PowerTransformer(method="yeo-johnson").fit(df[num_cols])
    df_out = df.copy()
    df_out[num_cols] = scaler.transform(df[num_cols])
    return df_out
