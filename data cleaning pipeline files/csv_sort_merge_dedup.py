import csv_clean_utils
from csv_clean_utils import *


month_dict = {  "January": 1, "February": 2, "March":3, "April": 4, "May": 5, "June":6, "July":7,
                "August": 8, "September": 9, "October": 10, "November": 11, "December": 12  }


single_month_data_csv = "data/raw_sm.csv"
sorted_data_csv = "data/intermediate_sorted.csv"
deduplicated_data_csv = "data/deduplicated_sm.csv"

removed_missing_values_csv = "data/visual_removed_missing_values.csv"



with open(single_month_data_csv, newline="") as f:
    single_month_data = list(csv.DictReader(f))

sorted_single_month_data = sort_by_month_year(single_month_data)
rewrite_csv(sorted_data_csv, sorted_single_month_data)

# --- Group + merge, guarding against empty data ---
merged_rows = []
data = sorted_single_month_data

if len(data) == 0:
    pass  # nothing to merge
elif len(data) == 1:
    merged_rows.append(data[0])
else:
    file_recency = build_file_recency_map(data)
    i, j = 1, 0
    while i < len(data):
        if natural_key(data[i]) == natural_key(data[j]):
            i += 1
            continue
        merged_rows.append(merge_rows(data[j:i], file_recency))
        j = i
        i += 1
    merged_rows.append(merge_rows(data[j:i], file_recency))

# --- Dedup guard on write: skip rows whose (month, year) key is already on disk ---
existing_keys = load_existing_keys(deduplicated_data_csv)
for row in merged_rows:
    key = natural_key(row)
    if key in existing_keys:
        continue
    append_to_csv(deduplicated_data_csv, row)
    existing_keys.add(key)

# --- Filter out rows missing consumption_value, write survivors to a new CSV ---
# I am commenting this out because the rows still contain some data and it would be lost if they were 
# eliminated.
filekeys = load_existing_keys(removed_missing_values_csv)
for row in merged_rows:
    if row["consumption_value"] == "":
        continue
    key = natural_key(row)
    if key in filekeys:
        continue
    append_to_csv(removed_missing_values_csv, row)
    filekeys.add(key)