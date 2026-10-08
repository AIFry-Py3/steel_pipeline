"""
prophet_data_customization.py

Converts the filled single-month dataset into Prophet's required format:
two columns, 'ds' (date) and 'y' (target value).

Design choices:
- ds is set to the LAST calendar day of each (period_month, period_year),
  not the 1st. Prophet only needs a real, monotonic date per row - it
  doesn't require even day-of-month spacing - so this is a style choice,
  not a functional requirement.
- Rows with no consumption_value are skipped entirely (not interpolated
  or filled with 0), preserving the "pure, non-interfered data" gaps.
  Prophet handles missing months natively by simply not seeing those dates.
"""

import csv
import calendar

month_dict = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
              "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}

input_csv = "current_final_data/lagged_after_interpolation.csv"
output_csv = "data/prophet_data/interpolation_lagroll_prophet_data.csv" 


def end_of_month_date(period_month, period_year):
    month_num = month_dict[period_month]
    year = int(period_year)
    last_day = calendar.monthrange(year, month_num)[1]
    return f"{year:04d}-{month_num:02d}-{last_day:02d}"


def main():
    with open(input_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    prophet_rows = []
    skipped = 0
    for row in rows:
        if not row.get("period_month") or row.get("consumption_value") in (None, ""):
            skipped += 1
            continue
        ds = end_of_month_date(row["period_month"], row["period_year"])
        prophet_rows.append({"ds": ds, "y": row["consumption_value"]})

    # Prophet expects ds sorted chronologically, ascending.
    prophet_rows.sort(key=lambda r: r["ds"])

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ds", "y"])
        writer.writeheader()
        writer.writerows(prophet_rows)

    print(f"Wrote {len(prophet_rows)} rows to {output_csv} ({skipped} rows skipped - no consumption_value).")


if __name__ == "__main__":
    main()