# The remove_columns.py file performs two functions:
# 1. Copy the INPUT_CSV file into the final_data directory as "data_non_stationary.csv".
# 2. Save a different copy of the data with limited columns, to be used for model training, 
#    as "final_data_non_stationary.csv" in the final_data directory.


import pandas as pd

# ------------------------------------------------------------------------------
# Config -- update these as needed
# ------------------------------------------------------------------------------
INPUT_CSV = "data/ridge_data/ridge_imputed_ladsmf.csv"  
COPY_CSV = "final_data/data_non_stationary.csv"
OUTPUT_CSV = "final_data/final_data_non_stationary.csv" 

# 1. Copying the file.
copy_df = pd.read_csv(INPUT_CSV)
copy_df.to_csv(COPY_CSV, index=False)


# 2. Removal of Columns.
KEEP_COLS = [
    'period_month',
    'period_year',
    'consumption_value',
    'consumption_unit',
    'consumption_value_source',
    'lag_1',
    'lag_2'
]

# ------------------------------------------------------------------------------
# Load, filter, save -- original input file is never modified
# ------------------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)

missing = [c for c in KEEP_COLS if c not in df.columns]
if missing:
    print(f"Warning: these columns were not found in the CSV and will be skipped: {missing}")

cols_present = [c for c in KEEP_COLS if c in df.columns]
trimmed_df = df[cols_present].copy()

trimmed_df.to_csv(OUTPUT_CSV, index=False)
print(f"Kept {len(cols_present)} columns: {cols_present}")
print(f"Rows: {len(trimmed_df)}")
print(f"Saved -> {OUTPUT_CSV}")