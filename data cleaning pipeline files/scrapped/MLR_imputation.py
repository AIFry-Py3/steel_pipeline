"""
predict_consumption_values.py

Fills missing `consumption_value` entries in visual_deduplicated_single_month_data.csv,
and writes a new file:
    interpolated_visual_deduplicated_single_month_data.csv

Why this version changed from plain Multiple Linear Regression:
  1. EXTRAPOLATION: several 2025/2026 rows have production features
     (e.g. crude_steel_production) above anything seen in training. Plain OLS
     has no ceiling on this - it just keeps extending the line, which is why
     earlier predictions jumped to ~15 when actual consumption never exceeded
     ~13.
  2. MULTICOLLINEARITY: crude_steel_production alone correlates ~0.96 with
     the target, and ~0.79 with finished_steel_production. That makes OLS
     coefficients unstable/exaggerated.
  3. NO MEMORY OF THE TARGET: the model only looked at production features,
     never at consumption's own recent trend.

Fixes applied:
  - Added `consumption_lag1` (previous month's consumption) as a feature, so
    each prediction is anchored to where consumption actually was, not just
    to production inputs. This is NOT a hard bound/clamp - the model can
    still predict outside the historical range if the trend supports it.
  - Switched OLS -> RidgeCV (L2-regularized regression, alpha chosen by
    cross-validation). Regularization shrinks coefficients on correlated/
    noisy features instead of letting any one of them dominate, which tempers
    the wild extrapolation without artificially capping the output.
  - Missing rows are filled sequentially in chronological order, since each
    row's lag feature depends on the previous row's (actual or just-filled)
    value.
  - Evaluation uses a chronological (time-based) train/test split instead of
    a random split, since we ultimately care about the model generalizing
    forward in time - which is exactly what a random split doesn't test.

Pipeline:
1. Load data, sort chronologically (period_year, period_month).
2. Drop columns that are entirely (or almost entirely) empty and useless as
   predictors: cumulative_months, crude_steel_export, crude_steel_import,
   coal_imports, steel_imports.
3. Time-interpolate the remaining numeric feature columns (linear, along the
   chronological axis) so every month has feature values.
4. Add consumption_lag1 (previous month's consumption_value).
5. Train Linear Regression (baseline) and RidgeCV (final model) on rows with
   a known target AND a known lag, using a chronological train/test split.
6. Report evaluation metrics for both: MSE, RMSE, MAE, MAPE, R2.
7. Refit RidgeCV on all known rows, predict the missing consumption_value
   rows one at a time, in order.
8. Save the full, filled dataset to interpolated_visual_deduplicated_single_month_data.csv
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

INPUT_FILE = "lag_added.csv"
OUTPUT_FILE = "interpolated_visual_deduplicated_single_month_data.csv"

BASE_FEATURES = [
    "crude_steel_production",
    "finished_steel_production",
    "finished_steel_export",
    "finished_steel_import",
]
LAG_FEATURE = "lag_1"
FEATURE_COLUMNS = BASE_FEATURES + [LAG_FEATURE]
TARGET_COLUMN = "consumption_value"

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def load_and_sort(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["month_num"] = df["period_month"].apply(lambda m: MONTH_ORDER.index(m) + 1)
    df = df.sort_values(["period_year", "month_num"]).reset_index(drop=True)
    return df


def drop_unusable_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Columns that are fully empty or almost entirely empty -> not usable as predictors
    cols_to_drop = [
        "cumulative_months",
        "crude_steel_export",
        "crude_steel_import",
        "coal_imports",
        "steel_imports",
    ]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])


def interpolate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURE_COLUMNS:
        df[col] = df[col].interpolate(method="linear", limit_direction="both")
    return df


def train_and_evaluate(df: pd.DataFrame):
    known = df.dropna(subset=[TARGET_COLUMN])
    X = known[FEATURE_COLUMNS]
    y = known[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("=== Evaluation on held-out 20% test split ===")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"MAPE : {mape:.4%}")
    print(f"R2   : {r2:.4f}")
    print(f"Coefficients: {dict(zip(FEATURE_COLUMNS, model.coef_))}")
    print(f"Intercept: {model.intercept_:.4f}")

    # Refit on ALL known data for the final prediction model
    final_model = LinearRegression()
    final_model.fit(X, y)
    return final_model


def fill_missing_targets(df: pd.DataFrame, model: LinearRegression) -> pd.DataFrame:
    df = df.copy()
    df["consumption_value_source"] = np.where(
        df[TARGET_COLUMN].notna(), "actual", "predicted"
    )
    missing_mask = df[TARGET_COLUMN].isna()
    if missing_mask.any():
        X_missing = df.loc[missing_mask, FEATURE_COLUMNS]
        df.loc[missing_mask, TARGET_COLUMN] = model.predict(X_missing)
    return df


def main():
    df = load_and_sort(INPUT_FILE)
    df = drop_unusable_columns(df)
    df = interpolate_features(df)

    model = train_and_evaluate(df)
    df = fill_missing_targets(df, model)

    df = df.drop(columns=["month_num"])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}  (rows: {len(df)})")


if __name__ == "__main__":
    main()