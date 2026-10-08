import csv
import os


raw_steel_data = "data/raw_steel_data.csv"

csvpath_singlemonth = "data/raw_sm.csv"  # stands for raw_single_month.csv
csvpath_cumulative = "data/raw_cumulative.csv"


rows = list(csv.DictReader(open(raw_steel_data)))  # rows of steel_data.csv

def natural_key(row):
    """Logical identity of a row: same source file + same reporting period."""
    return (row["source_file"], row["period_month"], row["cumulative_months"])#, row["period_year"])

def write_csv(csv_path, rows):
    """Fully overwrite csv_path with the given rows (regenerate-from-source pattern)."""
    if not rows:
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

# Collect into in-memory lists first. Dedup within this run via a seen-keys set
# per bucket, since steel_data.csv itself could still contain repeats.
# year_rows, 
single_month_rows, cumulative_rows = [], []
# year_keys, 
single_month_keys, cumulative_keys = set(), set()

for row in rows:
    key = natural_key(row)
    if row["period_month"] and row["cumulative_months"] == "":
        if key in single_month_keys:
            continue
        single_month_rows.append(row)
        single_month_keys.add(key)

    elif row["period_month"] == "" and row["cumulative_months"]:
        if key in cumulative_keys:
            continue
        cumulative_rows.append(row)
        cumulative_keys.add(key)

# write_csv(csvpath_year, year_rows)
write_csv(csvpath_singlemonth, single_month_rows)
write_csv(csvpath_cumulative, cumulative_rows)

# print("len(year_rows) =", len(year_rows))
print("len(single_month_rows) =", len(single_month_rows))
print("len(cumulative_rows) =", len(cumulative_rows))





# print("len(only_year) =", len(only_year))
# print("len(cumulative) =", len(cumulative))
# print("len(single_month) =", len(single_month))

# current outputs:
# len(only_year) = 13
# len(cumulative) = 129
# len(single_month) = 231