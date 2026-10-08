import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

# ------------------------------------------------------------------------------
# 1. Define files and load data
# ------------------------------------------------------------------------------
CSV_PATH = "data/lagged_fdsm.csv"
OUTPUT_PATH = "data/ridge_lfdsm.csv"



df = pd.read_csv(CSV_PATH)

TARGET_COL = 'consumption_value'
FULL_FEATURE_COLS = [
    'crude_steel_production',
    'finished_steel_production',
    'finished_steel_export',
    'finished_steel_import'
]
FALLBACK_FEATURE_COLS = [
    'crude_steel_production',
    'finished_steel_export',
    'finished_steel_import'
]  # drops finished_steel_production -- used only for rows where it's missing,
   # so we never predict from an imputed feature (no stacked imputation)

alphas = np.logspace(-6, 3, 100)
loo = LeaveOneOut()


def fit_ridge_with_loocv(X, y, label):
    """Fit RidgeCV (LOOCV-tuned alpha) and report genuine LOOCV validation metrics."""
    model = RidgeCV(alphas=alphas, cv=None)
    model.fit(X, y)

    print(f"[{label}] Rows used for fitting: {len(X)}")
    print(f"[{label}] Best alpha (LOOCV): {model.alpha_:.6f}")
    if model.alpha_ <= alphas[1]:
        print(f"[{label}] NOTE: best alpha is at/near the search floor -- barely regularizing, "
              f"close to plain OLS.")
    print(f"[{label}] Coefficients: {dict(zip(X.columns, model.coef_))}")
    print(f"[{label}] Intercept: {model.intercept_:.4f}")

    y_loocv_pred = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        fold_model = Ridge(alpha=model.alpha_)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_loocv_pred[test_idx] = fold_model.predict(X.iloc[test_idx])

    mae = mean_absolute_error(y, y_loocv_pred)
    rmse = np.sqrt(mean_squared_error(y, y_loocv_pred))
    r2 = r2_score(y, y_loocv_pred)
    print(f"[{label}] LOOCV MAE:  {mae:.4f}")
    print(f"[{label}] LOOCV RMSE: {rmse:.4f}")
    print(f"[{label}] LOOCV R2:   {r2:.4f}\n")

    return model, {'mae': mae, 'rmse': rmse, 'r2': r2}


# ------------------------------------------------------------------------------
# 2. Fit the FULL 4-feature model (only on rows with all 4 features + target)
# ------------------------------------------------------------------------------
full_train = df.dropna(subset=FULL_FEATURE_COLS + [TARGET_COL]).copy()  
X_full = full_train[FULL_FEATURE_COLS] 
y_full = full_train[TARGET_COL]

full_model, full_metrics = fit_ridge_with_loocv(X_full, y_full, "4-FEATURE MODEL")

# ------------------------------------------------------------------------------
# 3. Fit the FALLBACK 3-feature model (drops finished_steel_production)
#    Trained only on rows where target + the 3 fallback features are present,
#    regardless of whether finished_steel_production happens to be there too --
#    this keeps the fallback model's training data as large as possible and
#    directly comparable in spirit to the full model.
# ------------------------------------------------------------------------------
fallback_train = df.dropna(subset=FALLBACK_FEATURE_COLS + [TARGET_COL]).copy()
X_fallback = fallback_train[FALLBACK_FEATURE_COLS]
y_fallback = fallback_train[TARGET_COL]

fallback_model, fallback_metrics = fit_ridge_with_loocv(X_fallback, y_fallback, "3-FEATURE FALLBACK MODEL")

# ------------------------------------------------------------------------------
# 4. Compare -- is dropping finished_steel_production actually costly?
# ------------------------------------------------------------------------------
print("[COMPARISON] 4-feature vs 3-feature fallback (LOOCV, genuine held-out performance)")
print(f"  MAE:  {full_metrics['mae']:.4f}  vs  {fallback_metrics['mae']:.4f}  "
      f"(delta: {fallback_metrics['mae'] - full_metrics['mae']:+.4f})")
print(f"  RMSE: {full_metrics['rmse']:.4f}  vs  {fallback_metrics['rmse']:.4f}  "
      f"(delta: {fallback_metrics['rmse'] - full_metrics['rmse']:+.4f})")
print(f"  R2:   {full_metrics['r2']:.4f}  vs  {fallback_metrics['r2']:.4f}  "
      f"(delta: {fallback_metrics['r2'] - full_metrics['r2']:+.4f})")
print("  (positive MAE/RMSE delta or negative R2 delta = fallback is worse than full model)\n")

# ------------------------------------------------------------------------------
# 5. Impute missing consumption_value rows
#    - Rows with all 4 features present -> use the 4-feature model
#    - Rows missing ONLY finished_steel_production but with the other 3 present
#      -> use the 3-feature fallback model (no stacked imputation: every
#      prediction is built only on real, reported feature values)
#    - Rows still missing a fallback feature -> left unresolved, reported below
# ------------------------------------------------------------------------------
output_df = df.copy()

if 'consumption_value_source' not in output_df.columns:
    output_df['consumption_value_source'] = np.where(
        output_df[TARGET_COL].notna(), 'actual', np.nan
    )

target_missing = output_df[TARGET_COL].isna()

full_eligible = target_missing & output_df[FULL_FEATURE_COLS].notna().all(axis=1)
n_full = full_eligible.sum()
if n_full > 0:
    output_df.loc[full_eligible, TARGET_COL] = full_model.predict(output_df.loc[full_eligible, FULL_FEATURE_COLS])
    output_df.loc[full_eligible, 'consumption_value_source'] = 'ridge_imputed_4feat'

# fallback eligible = target missing, NOT already filled by full model,
# and all 3 fallback features present (finished_steel_production may or may not be present --
# doesn't matter, we deliberately don't use it here)
still_missing_after_full = output_df[TARGET_COL].isna()
fallback_eligible = still_missing_after_full & output_df[FALLBACK_FEATURE_COLS].notna().all(axis=1)  # (axis=1) goes across columns.
n_fallback = fallback_eligible.sum()
if n_fallback > 0:
    output_df.loc[fallback_eligible, TARGET_COL] = fallback_model.predict(
        output_df.loc[fallback_eligible, FALLBACK_FEATURE_COLS]
    )
    output_df.loc[fallback_eligible, 'consumption_value_source'] = 'ridge_imputed_3feat_fallback'

print(f"Imputed via 4-feature model: {n_full} rows")
print(f"Imputed via 3-feature fallback model: {n_fallback} rows")

if n_full > 0:
    print("\n4-feature imputed rows:")
    print(output_df.loc[full_eligible, ['period_month', 'period_year', TARGET_COL, 'consumption_value_source']])
if n_fallback > 0:
    print("\n3-feature fallback imputed rows:")
    print(output_df.loc[fallback_eligible, ['period_month', 'period_year', TARGET_COL, 'consumption_value_source']])

# ------------------------------------------------------------------------------
# 6. Report any rows still unresolved (missing target AND a fallback feature)
# ------------------------------------------------------------------------------
still_missing = output_df[TARGET_COL].isna()
if still_missing.sum() > 0:
    print(f"\nWarning: {still_missing.sum()} rows STILL missing {TARGET_COL} "
          f"(missing one or more of the 3 fallback features too -- neither model can help):")
    diagnostic_cols = [c for c in ['period_month', 'period_year'] if c in output_df.columns] + FULL_FEATURE_COLS
    print(output_df.loc[still_missing, diagnostic_cols])
else:
    print("\nAll rows resolved.")

# ------------------------------------------------------------------------------
# 7. Save to new CSV -- original source file is never touched
# ------------------------------------------------------------------------------
output_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved output -> {OUTPUT_PATH}")