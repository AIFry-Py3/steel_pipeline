import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------------------
CSV_PATH = "data/visual_deduplicated_single_month_filled.csv"  # <-- change to your actual file

df = pd.read_csv(CSV_PATH)

# ------------------------------------------------------------------------------
# 2. Select columns for correlation: base features + target
#    Update to match your actual column names
# ------------------------------------------------------------------------------
COLS = [
    'consumption_value',           # target
    'crude_steel_production',
    'finished_steel_production',
    'finished_steel_export',
    'finished_steel_import',
]

corr_data = df[COLS].dropna()
print(f"Rows used for correlation (complete cases across these columns): {len(corr_data)}")

# ------------------------------------------------------------------------------
# 3. Compute correlation matrix
# ------------------------------------------------------------------------------
corr_matrix = corr_data.corr()
print("\nCorrelation matrix:")
print(corr_matrix.round(3))

print("\nCorrelation with target (consumption_value), sorted:")
print(corr_matrix['consumption_value'].drop('consumption_value').sort_values(ascending=False).round(3))

# ------------------------------------------------------------------------------
# 4. Plot heatmap
# ------------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5
)
plt.title('Correlation Heatmap: Base Features vs Consumption Value')
plt.tight_layout()
plt.savefig('EDA/correlation_heatmap.png', dpi=150)
plt.show()

print("\nSaved -> correlation_heatmap.png")