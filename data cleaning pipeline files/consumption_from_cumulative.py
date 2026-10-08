"""
fill_missing_consumption.py

Some newer Ministry of Steel reports only publish consumption_value as a
fiscal-year-to-date CUMULATIVE figure (e.g. "April-September"), not as a
standalone monthly number, even though production/import/export are still
reported per month. This script recovers a monthly consumption_value for
those months by differencing two consecutive cumulative totals in the same
fiscal year (e.g. April-September minus April-August = September).

Because different reports can carry revised/provisional prior-period figures,
a raw difference is NOT trusted blindly: it's only accepted if it falls
within a plausible range of real observed monthly consumption values.
Anything outside that range is left blank rather than silently injected.
"""

import csv
import os

month_dict = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
              "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
month_names = {v: k for k, v in month_dict.items()}




# The goal is to fill in the empty consumption_values in the 'deduplicated_single_month.csv'
# using the 'raw_cumulative.csv' data.

# Quoted from above:
# 'by differencing two consecutive cumulative totals in the same fiscal year (e.g. April-September minus April-August = September).'

dedup_data_csv = "data/deduplicated_sm.csv"
cumulative_data_csv = "data/raw_cumulative.csv"
filled_output_csv = "data/filled_dsm.csv"
derived_output_csv = "data/secondary_sm.csv"



# Plausible range for a real monthly consumption_value (MT). Generously wide
# to allow for the 2020 COVID dip; tune these if your data's range differs.
PLAUSIBLE_MIN = 0.5
PLAUSIBLE_MAX = 18.0


def parse_cumulative_range(cumulative_months, period_year):
    """
    Returns (start_month_num, end_month_num, start_year, end_year).
    Fiscal-year convention observed in this dataset: period_year is the
    calendar year of the END month. If start_month > end_month (e.g.
    April..January), the start month belongs to the PRIOR calendar year.
    """
    start_name, end_name = [m.strip() for m in cumulative_months.split("-")]
    start_m, end_m = month_dict[start_name], month_dict[end_name]
    end_year = int(period_year)
    start_year = end_year - 1 if start_m > end_m else end_year
    return start_m, end_m, start_year, end_year


def load_cumulative_rows():
    with open(cumulative_data_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    parsed = []
    for row in rows:
        if not row.get("cumulative_months") or row.get("consumption_value") in (None, ""):
            continue
        try:
            start_m, end_m, start_y, end_y = parse_cumulative_range(
                row["cumulative_months"], row["period_year"]
            )
        except (KeyError, ValueError):
            continue
        parsed.append({
            "start_m": start_m, "end_m": end_m,
            "start_y": start_y, "end_y": end_y,
            "value": float(row["consumption_value"]),
            "source_file": row.get("source_file", ""),
        })
    return parsed


def derive_monthly_values(cumulative_rows):
    """
    Group by (start_m, start_y) — i.e. the same fiscal-year window — then
    for each pair of entries whose end months are exactly one month apart,
    difference them to recover that single month's value.
    """
    by_window = {}
    for row in cumulative_rows:
        key = (row["start_m"], row["start_y"])
        by_window.setdefault(key, []).append(row)

    derived = []
    for key, entries in by_window.items():
        # sort by end month's distance from the fiscal-year start month
        def fiscal_offset(e):
            return (e["end_m"] - e["start_m"]) % 12
        entries.sort(key=fiscal_offset)

        for i in range(1, len(entries)):
            prev, curr = entries[i - 1], entries[i]
            if fiscal_offset(curr) - fiscal_offset(prev) != 1:
                continue  # not consecutive months, can't safely difference
            derived_value = curr["value"] - prev["value"]
            if not (PLAUSIBLE_MIN <= derived_value <= PLAUSIBLE_MAX):
                continue  # fails sanity check, discard
            derived.append({
                "period_month": month_names[curr["end_m"]],
                "period_year": str(curr["end_y"]),
                "consumption_value": round(derived_value, 3),
                "derived_from": f"{prev['source_file']} | {curr['source_file']}",
            })
    return derived


def derived_key(row):
    """Identity of a derived row: same month/year AND same source pair,
    so re-running the script won't duplicate an already-recorded derivation."""
    return (row["period_month"], row["period_year"], row["derived_from"])


def load_existing_derived_keys(csv_path):
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="") as f:
        return {derived_key(r) for r in csv.DictReader(f)}


def append_derived_row(csv_path, row):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_derived_csv(derived_rows):
    if not derived_rows:
        print("No derived values passed the sanity check.")
        return

    existing_keys = load_existing_derived_keys(derived_output_csv)
    new_count = 0
    for row in derived_rows:
        key = derived_key(row)
        if key in existing_keys:
            continue
        append_derived_row(derived_output_csv, row)
        existing_keys.add(key)
        new_count += 1

    print(f"Appended {new_count} new derived values to {derived_output_csv} "
          f"({len(derived_rows) - new_count} already present, skipped).")


def fill_into_single_month_data(derived_rows):
    """
    Fills consumption_value into a COPY of the single-month dataset
    wherever that (month, year) is currently empty. Does not overwrite
    the original deduplicated file.
    """
    with open(dedup_data_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    derived_lookup = {(d["period_month"], d["period_year"]): d["consumption_value"] for d in derived_rows}

    filled_count = 0
    for row in rows:
        key = (row["period_month"], row["period_year"])
        if row.get("consumption_value") in (None, "") and key in derived_lookup:
            row["consumption_value"] = derived_lookup[key]
            filled_count += 1

    if rows:
        with open(filled_output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print(f"Filled {filled_count} previously-empty consumption_value cells.")
    print(f"Wrote result to {filled_output_csv}")


if __name__ == "__main__":
    cumulative_rows = load_cumulative_rows()
    derived_rows = derive_monthly_values(cumulative_rows)
    write_derived_csv(derived_rows)
    fill_into_single_month_data(derived_rows)
