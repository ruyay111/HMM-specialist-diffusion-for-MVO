# data_utils.py

import pandas as pd
import numpy as np
import datetime

def load_real_data(csv_path, test_start_year= '2013', intraday=False, use_active_month=False, benchmark = False):
    """
    Load real data. 
    For Futures CSV 
        has no header. Columns are:
          [timestamp, open, high, low, close, volume]
        If intraday=True, apply filtering logic:
          - Keep 09:30 to 16:00
          - Drop weekends
          - Possibly compute "close_60" if we want 60-second intervals
        If use_active_month=True, find most active month (volume) and keep only that range.
        Return a df with .index = timestamp, plus 'close' or 'close_60' for returns.
    For Benchmark csv:
        all data in daily frequency
        filter for data after test_start_year, default to 2013 for spring 2025 semester's notebooks  
        intraday and use_active_month don't apply to benchmark 
    """
    if benchmark:
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"as_of": "timestamp"}).assign(timestamp=lambda d: pd.to_datetime(d["timestamp"]))
        df =df.loc[lambda d: d["timestamp"] >= test_start_year+"-01-01"]
        df = df.set_index("timestamp").sort_index()  
        return df
    else:
        col_names = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.read_csv(csv_path, names=col_names)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df.sort_index()

    if intraday:
        # only want data from 09:30 to 16:00 on weekdays
        start_time = datetime.time(9, 30)
        end_time = datetime.time(16, 0)
        df.loc[df.index.time < start_time, "close"] = np.nan
        df.loc[df.index.time > end_time, "close"] = np.nan
        df.loc[df.index.weekday >= 5, "close"] = np.nan

        df["inter"] = df.index.to_series().diff().dt.total_seconds()
        # "close_60" is close if <60s intervals, else np.nan
        df["close_60"] = np.where(df["inter"] < 61, df["close"], np.nan)
        # compute returns from "close_60"
        df["returns"] = np.log(df["close_60"]).diff()
        df["returns"] = df["returns"].dropna()

    if use_active_month:
        # Resample by month to find the most active
        monthly_vol = df["volume"].resample("M").sum()
        best_month = monthly_vol.idxmax()
        # keep a one-month range around that best_month:
        start_of_best = best_month.replace(day=1)
        # end of that month = start_of_next - 1 day
        if start_of_best.month == 12:
            start_of_next = start_of_best.replace(year=start_of_best.year+1, month=1, day=1)
        else:
            start_of_next = start_of_best.replace(month=start_of_best.month+1, day=1)
        df = df.loc[(df.index >= start_of_best) & (df.index < start_of_next)]

    return df

def compute_returns(df, log = True, col = None ):
    """
    Compute returns (log or not log).
    Returns a new Series of returns.
    """
    if col:
        series = df[col].dropna()
    else:
        series = df
    if log:
        ret = np.log(series).diff().dropna()
    else:
        ret = series.diff().dropna()
    return ret

def load_synthetic_data(xlsx_path):
    """
    Load synthetic data from a .xlsx. 
    Returns a DataFrame with row=sample, col=TimeStep_1..TimeStep_12000
    """
    df = pd.read_excel(xlsx_path, index_col=0)
    return df
