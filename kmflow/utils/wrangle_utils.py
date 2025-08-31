import pandas as pd


def find_iqr_outliers(df: pd.DataFrame) -> pd.Series:
    """Detect outliers in numeric columns using the IQR method; returns a stacked Series of outliers."""
    num_df = df.select_dtypes(include="number")
    q1 = num_df.quantile(0.25)
    q3 = num_df.quantile(0.75)
    iqr = q3 - q1
    lower_lim = q1 - 1.5 * iqr
    upper_lim = q3 + 1.5 * iqr
    outlier_mask = (num_df < lower_lim) | (num_df > upper_lim)
    iqr_outliers = num_df.where(outlier_mask).stack()
    return iqr_outliers


def drop_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return a copy of the DataFrame with the specified column(s) removed."""
    return df.drop(columns=[col])


def drop_row(df: pd.DataFrame, index_list: list[int]) -> pd.DataFrame:
    """Return a copy of the DataFrame with specified row(s) (by index) dropped and index reset."""
    return df.drop(index=index_list).reset_index(drop=True)


def dotless_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Rename the specified column header by removing all periods from its name."""
    new_col = col.replace(".", "")
    return df.rename(columns={col: new_col})
