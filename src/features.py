"""
RetailSense — Feature Engineering Module
=========================================
All feature transformations for the demand forecasting pipeline.
Designed to be importable in both notebook and Streamlit app.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


FEATURE_COLS = [
    'Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
    'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionAge',
    'Promo2', 'Promo2Active', 'Year', 'Month', 'Day', 'WeekOfYear', 'Quarter',
    'IsWeekend', 'IsMonthStart', 'IsMonthEnd',
    'sin_week', 'cos_week', 'sin_year', 'cos_year', 'sin_month', 'cos_month',
    'Sales_lag_7', 'Sales_lag_14', 'Sales_lag_28', 'Sales_lag_56',
    'Sales_roll_mean_7', 'Sales_roll_std_7', 'Sales_roll_max_7',
    'Sales_roll_mean_14', 'Sales_roll_std_14', 'Sales_roll_max_14',
    'Sales_roll_mean_28', 'Sales_roll_std_28', 'Sales_roll_max_28',
    'Sales_expanding_mean',
]


def clean_data(train: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """Merge, clean, and encode raw Rossmann data."""
    df = train.merge(store, on='Store', how='left')

    # Remove closed-store rows (no demand signal)
    df = df[(df['Open'] == 1) & (df['Sales'] > 0)].copy()

    # Impute missing values
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
    for col in ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
                'Promo2SinceWeek', 'Promo2SinceYear']:
        df[col].fillna(0, inplace=True)
    df['PromoInterval'].fillna('None', inplace=True)

    # Label-encode categoricals
    le = LabelEncoder()
    for col in ['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval']:
        df[col] = le.fit_transform(df[col].astype(str))

    df.sort_values(['Store', 'Date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features + Fourier terms."""
    df = df.copy()
    df['Year']         = df['Date'].dt.year
    df['Month']        = df['Date'].dt.month
    df['Day']          = df['Date'].dt.day
    df['WeekOfYear']   = df['Date'].dt.isocalendar().week.astype(int)
    df['Quarter']      = df['Date'].dt.quarter
    df['IsWeekend']    = (df['DayOfWeek'] >= 6).astype(int)
    df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
    df['IsMonthEnd']   = df['Date'].dt.is_month_end.astype(int)

    doy = df['Date'].dt.dayofyear
    df['sin_week']  = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['cos_week']  = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['sin_year']  = np.sin(2 * np.pi * doy / 365)
    df['cos_year']  = np.cos(2 * np.pi * doy / 365)
    df['sin_month'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['Month'] / 12)
    return df


def add_lag_features(df: pd.DataFrame, lags=(7, 14, 28, 56)) -> pd.DataFrame:
    """Sales lag features at key retail horizons (shift avoids leakage)."""
    df = df.copy()
    for lag in lags:
        df[f'Sales_lag_{lag}'] = df.groupby('Store')['Sales'].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows=(7, 14, 28)) -> pd.DataFrame:
    """Rolling mean, std, max — always shifted by 1 to prevent leakage."""
    df = df.copy()
    for w in windows:
        rolled = df.groupby('Store')['Sales'].shift(1).rolling(w)
        df[f'Sales_roll_mean_{w}'] = rolled.mean().reset_index(level=0, drop=True)
        df[f'Sales_roll_std_{w}']  = rolled.std().reset_index(level=0, drop=True)
        df[f'Sales_roll_max_{w}']  = rolled.max().reset_index(level=0, drop=True)
    df['Sales_expanding_mean'] = (
        df.groupby('Store')['Sales']
          .transform(lambda x: x.shift(1).expanding().mean())
    )
    return df


def add_competition_age(df: pd.DataFrame) -> pd.DataFrame:
    """Months since nearest competitor opened."""
    df = df.copy()
    df['CompetitionAge'] = (
        12 * (df['Year'] - df['CompetitionOpenSinceYear']) +
        (df['Month'] - df['CompetitionOpenSinceMonth'])
    ).clip(lower=0)
    return df


def add_promo2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Was ongoing promotional campaign (Promo2) active?"""
    df = df.copy()
    df['Promo2Active'] = (
        (df['Promo2'] == 1) & (df['Year'] >= df['Promo2SinceYear'])
    ).astype(int)
    return df


def build_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_competition_age(df)
    df = add_promo2_features(df)

    if drop_na:
        lag_cols = [c for c in df.columns if 'lag' in c or 'roll' in c]
        df.dropna(subset=lag_cols, inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df


def build_single_row(input_dict: dict) -> pd.DataFrame:
    """
    Build a single feature row for Streamlit what-if prediction.
    input_dict should contain all required raw fields.
    Returns a DataFrame with FEATURE_COLS columns.
    """
    row = pd.DataFrame([input_dict])
    # Ensure all feature columns exist (fill 0 for missing)
    for col in FEATURE_COLS:
        if col not in row.columns:
            row[col] = 0.0
    return row[FEATURE_COLS]
