"""
llm_visual_extractor_local.py
Extracts steel-sector data from PDF pages (tables AND charts) with a LOCAL vision
model served by Ollama. No API key needed. Each PDF page is rendered to a JPEG and
sent as an image. Rows are appended to a per-machine CSV (no git merge conflicts).

Setup:  install Ollama, then
    ollama pull qwen3-vl:8b-instruct
    pip install ollama pymupdf
Run:    python llm_visual_extractor_local.py
Edit the constants below to change behaviour (MAX_FILES=1 for a test run).
"""

import csv
import glob
import json
import os
import re
import socket
import sys
import time

import httpx
import ollama
import pymupdf

# ---------- config (edit here) ----------
MODEL_CHAIN = ["qwen3-vl:8b-instruct", "qwen3-vl:4b"]   # tried in order
MAX_FILES = 0               # 0 = all pending PDFs
PAGES_PER_CALL = 3
RENDER_ZOOM = 2.0           # 1.0 = 72 dpi
JPEG_QUALITY = 85
NUM_CTX = 16384             # raise if output is empty/truncated
MAX_OUTPUT_TOKENS = 4096
USE_SCHEMA = True           # set False if schema-constrained output fails
CALL_TIMEOUT_S = 600
MAX_RETRIES = 2
RETRY_WAIT_S = 5

PDF_DIR = "pdfs"
HOST = socket.gethostname()
CSV_PATH = f"data/raw_steel_data_{HOST}.csv"
DONE_LOG = f"data/processed_files_{HOST}.txt"
EMPTY_LOG = f"data/empty_files_{HOST}.txt"

TONNAGE_FIELDS = [
    "consumption_value", "crude_steel_production", "finished_steel_production",
    "crude_steel_export", "crude_steel_import",
    "finished_steel_export", "finished_steel_import",
    "coal_imports", "steel_imports",
]
FIELDNAMES = (
    ["source_file", "period_month", "cumulative_months", "period_year"]
    + TONNAGE_FIELDS
    + ["value_source", "extra_notes", "model_used", "needs_review"]
)
MONTHS = {
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
}

client = ollama.Client(timeout=CALL_TIMEOUT_S)
dead_models = set()         # models that failed hard this run

# ---------- output schema ----------
_NUM = {"anyOf": [{"type": "number"}, {"type": "null"}]}
_STR = {"anyOf": [{"type": "string"}, {"type": "null"}]}
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"records": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "period_month": _STR,
            "cumulative_months": _STR,
            "period_year": _STR,
            **{f: _NUM for f in TONNAGE_FIELDS},
            "value_source": {"type": "string", "enum": ["table", "text", "chart", "mixed"]},
        },
        "required": ["period_month", "cumulative_months", "period_year",
                     *TONNAGE_FIELDS, "value_source"],
    }}},
    "required": ["records"],
}

SYSTEM_PROMPT = """
You extract steel sector data from Indian government reports.
You receive page images from ONE report. Read tables AND bar/line charts/graphs,
not just body text.

Return ONLY a JSON object of the form {"records": [ ... ]}, with no markdown fences
and no extra text. One record per month (or per cumulative period when no monthly
breakdown exists). Every record must have all of these keys: period_month,
cumulative_months, period_year, consumption_value, crude_steel_production,
finished_steel_production, crude_steel_export, crude_steel_import,
finished_steel_export, finished_steel_import, coal_imports, steel_imports,
value_source. If nothing relevant is on the pages, return {"records": []}.

GENERAL RULES
- Never invent or estimate data that is not on the page. Use null if a field
  cannot be found in a table, chart or text.
- Chart-only values (no printed number): read them off the axis and set
  value_source to "chart". Printed values: "table" or "text". Mixed: "mixed".
- Placeholders ("-", "--", "NA", "n.a.", blank) mean null. Use 0 only if 0 is printed.
- Plain decimal numbers, dot separator, no thousands separators, no units, no %.
- Percentages (growth, YoY, MoM, share) are never tonnage values.
- If provisional and revised figures both exist for a month, use the revised one.

UNITS: every tonnage field must be in MILLION TONNES.
- Read the unit from each table's / chart's own header, caption, footnote or axis.
  Different tables in one report can use different units; check each separately.
- Conversions: "million tonnes"/"MT" -> as is; "thousand tonnes"/"'000 tonnes"/"KT"
  -> divide by 1,000; "lakh tonnes" -> divide by 10; plain "tonnes" -> divide by 1,000,000.
- Sanity guide (not a filter): monthly consumption, crude and finished production
  are typically 2-16 Mt; monthly finished steel exports/imports typically 0.1-2 Mt.
  A value above 100 is almost certainly still in thousand tonnes: re-check and
  convert (586 -> 0.586) unless the table clearly states million tonnes.

TIME PERIODS
- period_year is the CALENDAR year of the month, not the fiscal year. Indian FY is
  April-March: Jan/Feb/Mar 2024 belong to FY 2023-24 but have period_year "2024";
  Apr-Dec 2023 have "2023".
- Single month: fill period_month (full month name) + period_year,
  cumulative_months = null.
- Cumulative period with no monthly breakdown (e.g. "January-April"): fill
  cumulative_months (full month names) + period_year, period_month = null.
- Never put a cumulative/YTD value into a single-month record or vice versa.
- Trend tables/charts show several months and previous-year columns: create one
  record per month with its own correct month and year. Never assign a
  previous-year or previous-month value to the current month.

FIELD DEFINITIONS
- consumption_value: finished steel consumption for the month.
- crude_steel_production: total crude steel production.
- finished_steel_production: TOTAL finished steel (alloy/stainless + non-alloy).
  If only a narrower variant exists (non-alloy only, "for sale"), return null.
- finished_steel_export / finished_steel_import: total FINISHED steel trade only.
  Not crude, semis, pig iron, sponge iron, or combined "total steel" rows.
- crude_steel_export / crude_steel_import: crude steel or semis lines only.
- steel_imports: only a separately labelled figure distinct from
  finished_steel_import. Never copy one into the other.
- coal_imports: only an explicit coal import tonnage. Never put steel here.
- Never swap exports/imports or production/consumption. Verify each row label.

DATA QUALITY
- Every value must come from its own labelled row/column/bar for that month and
  field. Never copy from another field, month or neighbouring column.
"""


class FatalError(Exception):
    pass


# ---------- PDF -> images ----------
def render_pages(pdf_path):
    """List of JPEG bytes, one per page."""
    matrix = pymupdf.Matrix(RENDER_ZOOM, RENDER_ZOOM)
    with pymupdf.open(pdf_path) as doc:
        return [p.get_pixmap(matrix=matrix, alpha=False).tobytes("jpeg", jpg_quality=JPEG_QUALITY)
                for p in doc]


# ---------- model calls ----------
def parse_records(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("records")
    if not isinstance(data, list):
        raise ValueError("no 'records' array in JSON")
    return data


def try_model(model, images, first_page):
    """Returns list[dict], or None if this model should be dropped."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "images": images,
         "content": f"These are pages {first_page}-{first_page + len(images) - 1} "
                    f"of one report. Extract the steel sector data."},
    ]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t = time.monotonic()
            print(f"  call -> {model} ({len(images)} pages, attempt {attempt}/{MAX_RETRIES})")
            resp = client.chat(
                model=model,
                messages=messages,
                format=RESPONSE_SCHEMA if USE_SCHEMA else "json",
                options={"temperature": 0, "num_ctx": NUM_CTX, "num_predict": MAX_OUTPUT_TOKENS},
            )
            print(f"  done in {time.monotonic() - t:.1f}s")
            text = resp["message"]["content"]
            if not text:
                raise ValueError("empty output")
            return parse_records(text)

        except (json.JSONDecodeError, ValueError) as e:
            print(f"  {model}: bad output ({e})")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_S)
        except ConnectionError as e:
            raise FatalError(f"Cannot reach Ollama. Is it running? ({e})") from e
        except httpx.TimeoutException:
            print(f"  {model}: no response after {CALL_TIMEOUT_S}s")
            return None
        except ollama.ResponseError as e:
            hint = f" Run: ollama pull {model}" if e.status_code == 404 else ""
            print(f"  {model}: [{e.status_code}] {str(e)[:200]}.{hint}")
            return None
    return None


def call_model(images, first_page):
    """Returns (records, model) or (None, None)."""
    for model in MODEL_CHAIN:
        if model in dead_models:
            continue
        data = try_model(model, images, first_page)
        if data is not None:
            return data, model
        dead_models.add(model)
        print(f"  {model} dropped; trying next model.")
    return None, None


# ---------- merge / validate ----------
def merge_rows(rows):
    """Same period can appear in several page-chunks: fill nulls, keep first values."""
    merged = {}
    for r in rows:
        key = (r.get("period_month"), r.get("cumulative_months"), str(r.get("period_year")))
        if key not in merged:
            merged[key] = dict(r)
        else:
            for k, v in r.items():
                if merged[key].get(k) is None and v is not None:
                    merged[key][k] = v
    return list(merged.values())


def validate_rows(rows, filename, model_used):
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        r["source_file"] = filename
        issues = []
        for f in TONNAGE_FIELDS:
            if r.get(f) is None:
                continue
            try:
                r[f] = float(r[f])
            except (TypeError, ValueError):
                issues.append(f"{f}:non-numeric")
                r[f] = None
                continue
            if r[f] < 0:
                issues.append(f"{f}:negative")
            if r[f] > 100:
                issues.append(f"{f}:>100(unit?)")
        pm, cm, py = r.get("period_month"), r.get("cumulative_months"), r.get("period_year")
        if (pm is None) == (cm is None):
            issues.append("month/cumulative: need exactly one")
        if pm is not None and pm not in MONTHS:
            issues.append("bad period_month")
        if not (py and str(py).isdigit() and 1990 <= int(py) <= 2100):
            issues.append("bad period_year")
        r["model_used"] = model_used
        r["needs_review"] = "; ".join(issues)
        out.append(r)
    return out


# ---------- files ----------
def append_rows(rows):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    write_header = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore", restval="")
        if write_header:
            w.writeheader()
        w.writerows(rows)


def append_line(path, line):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_done():
    """Files already handled on ANY machine (all per-machine CSVs/logs after a git pull)."""
    done = set()
    for path in glob.glob("data/raw_steel_data*.csv"):
        with open(path, newline="", encoding="utf-8") as f:
            done.update(r["source_file"] for r in csv.DictReader(f))
    for path in glob.glob("data/processed_files*.txt"):
        with open(path, encoding="utf-8") as f:
            done.update(line.strip() for line in f if line.strip())
    return done


def process_file(name):
    """Returns validated rows (maybe empty), or None if any page-chunk failed."""
    try:
        pages = render_pages(os.path.join(PDF_DIR, name))
    except Exception as e:
        print(f"  Could not render {name}: {e}")
        return None
    print(f"  {len(pages)} pages rendered")

    rows, model_used = [], None
    for i in range(0, len(pages), PAGES_PER_CALL):
        data, model_used = call_model(pages[i:i + PAGES_PER_CALL], i + 1)
        if data is None:
            return None             # file stays pending; no partial writes
        rows.extend(data)
    return validate_rows(merge_rows(rows), name, model_used)


def main():
    pdfs = sorted(f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))
    done = load_done()
    pending = [f for f in pdfs if f not in done]
    print(f"{len(pdfs)} PDFs, {len(pending)} pending. host={HOST}, models={MODEL_CHAIN}")

    failed = 0
    for n, name in enumerate(pending, 1):
        if MAX_FILES and n > MAX_FILES:
            print("MAX_FILES reached; stopping.")
            break
        print(f"\nFile {n}: {name}")
        try:
            rows = process_file(name)
        except FatalError as e:
            print(f"Fatal: {e}")
            sys.exit(2)

        if rows is None:
            failed += 1
            print("  File failed; stays pending for the next run.")
            if all(m in dead_models for m in MODEL_CHAIN):
                print("All models failed. Stopping; check Ollama and installed models.")
                break
            continue

        if rows:
            append_rows(rows)
        else:
            append_line(EMPTY_LOG, name)
            print(f"  WARNING: no rows (logged in {EMPTY_LOG})")
        append_line(DONE_LOG, name)
        flagged = sum(1 for r in rows if r["needs_review"])
        print(f"  Wrote {len(rows)} rows ({flagged} flagged needs_review).")

    if failed:
        print(f"{failed} file(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()