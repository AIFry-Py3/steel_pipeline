import csv
import os

month_dict = {  "January": 1, "February": 2, "March":3, "April": 4, "May": 5, "June":6, "July":7,
                "August": 8, "September": 9, "October": 10, "November": 11, "December": 12  }


def natural_key(row):
    """Logical identity of a row: same reporting period."""
    return (row["period_month"], row["period_year"])

def load_existing_keys(csv_path):
    """Read a CSV once and return the set of natural keys already written to it."""
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="") as f:
        return {natural_key(r) for r in csv.DictReader(f)}

def append_to_csv(csv_path, row):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def sort_by_month_year(data):
    return sorted(data, key=lambda row: (int(row["period_year"]), month_dict[row["period_month"]]))

def rewrite_csv(csv_path, rows):
    """Fully overwrite csv_path with rows (regenerate-from-source pattern)."""
    if not rows:
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def build_file_recency_map(data):
    """Map each source_file -> the latest (year, month) it reports anywhere in its rows.
    Used as a proxy for how recently that file was published."""
    recency = {}
    for row in data:
        fname = row["source_file"]
        period = (int(row["period_year"]), month_dict[row["period_month"]])
        if fname not in recency or period > recency[fname]:
            recency[fname] = period
    return recency

def is_own_headline_month(row, file_recency):
    """True if this row's period is the file's own max period — i.e. the file's
    own reporting month, not a back-reference to some other month."""
    fname = row["source_file"]
    period = (int(row["period_year"]), month_dict[row["period_month"]])
    return period == file_recency[fname]

def merge_rows(rows, file_recency):
    if len(rows) == 1:
        return rows[0]

    merged = {}
    for key in rows[0].keys():
        non_empty_rows = [row for row in rows if row[key] != ""]

        if len(non_empty_rows) == 0:
            merged[key] = ""
        elif len({row[key] for row in non_empty_rows}) == 1:
            merged[key] = non_empty_rows[0][key]
        elif key == "source_file":
            merged[key] = "|".join(row["source_file"] for row in non_empty_rows)
        else:
            # Conflict: prefer the row(s) whose file actually authored this month
            # (own headline month) over any file merely back-referencing it.
            own_month_rows = [r for r in non_empty_rows if is_own_headline_month(r, file_recency)]
            candidates = own_month_rows if own_month_rows else non_empty_rows
            # Among remaining candidates, break ties by most recent source file.
            latest_row = max(candidates, key=lambda r: file_recency[r["source_file"]])
            merged[key] = latest_row[key]

    return merged

