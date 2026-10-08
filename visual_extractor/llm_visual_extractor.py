"""
llm_visual_extractor.py
Reads steel-sector data from PDF pages (tables AND charts) via Gemini's
Interactions API, with model fallback. Appends validated rows to
data/raw_steel_data.csv.

Test-run knobs (PowerShell):
  $env:BATCH_SIZE=1; $env:MAX_BATCHES=1; python .\\visual_extractor\\llm_visual_extractor.py
  $env:USE_SCHEMA=0     # if the schema is suspected
  $env:THINKING_LEVEL=  # empty = don't send thinking_level
"""

import os
import re
import sys
import csv
import json
import time
import threading
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ---------- config ----------
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "3"))
MAX_BATCHES = int(os.getenv("MAX_BATCHES", "0"))          # 0 = no limit
USE_SCHEMA = os.getenv("USE_SCHEMA", "1") == "1"
THINKING_LEVEL = os.getenv("THINKING_LEVEL", "low")        # "" = omit

MODEL_CHAIN = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]
MODEL_COOLDOWN_S = 600           # after repeated bad output / non-retryable error
QUOTA_BLOCK_S = 24 * 3600        # after a quota 429: skip model for the rest of the run
_blocked_until = {}

CALL_DEADLINE_S = 150
MAX_RETRIES = 2                  # per model
RETRY_WAIT_SECONDS = 10
MAX_429_WAIT_S = 90
MIN_SECONDS_BETWEEN_CALLS = 12
UPLOAD_TIMEOUT_S = 300
TRANSIENT_CODES = {500, 503, 504}
FATAL_CODES = {401, 403}

PDF_DIR = "pdfs"
CSV_PATH = "data/raw_steel_data.csv"
DONE_LOG = "data/processed_files.txt"
EMPTY_LOG = "data/empty_files.txt"

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(
        timeout=CALL_DEADLINE_S * 1000,
        retry_options=types.HttpRetryOptions(attempts=1),   # no hidden SDK retries
    ),
)

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

# ---------- structured output schema ----------
_NUM = {"type": ["number", "null"]}
_STR = {"type": ["string", "null"]}
RECORD_SCHEMA = {
    "type": "object",
    "properties": {
        "source_file": {"type": "string"},
        "period_month": _STR,
        "cumulative_months": _STR,
        "period_year": _STR,
        **{f: _NUM for f in TONNAGE_FIELDS},
        "value_source": {"type": "string", "enum": ["table", "text", "chart", "mixed"]},
    },
    "required": ["source_file", "period_month", "cumulative_months", "period_year",
                 *TONNAGE_FIELDS, "value_source"],
}
RESPONSE_FORMAT = {
    "type": "text",
    "mime_type": "application/json",
    "schema": {"type": "array", "items": RECORD_SCHEMA},
}

SYSTEM_PROMPT = """
You extract steel sector data from Indian government reports.
You receive one or more PDF reports as visual input, each preceded by a text label
giving its source_file ID (for example doc_1). Copy that ID EXACTLY into every
record's source_file. Read tables AND bar/line charts/graphs, not just body text.
Treat every document independently.

Return ONE JSON array covering all documents, with no markdown fences and no
extra text. One object per month per document (or per cumulative period when no
monthly breakdown exists). Every object must have all of these keys: source_file,
period_month, cumulative_months, period_year, consumption_value,
crude_steel_production, finished_steel_production, crude_steel_export,
crude_steel_import, finished_steel_export, finished_steel_import, coal_imports,
steel_imports, value_source.

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
- Single month: fill period_month + period_year, cumulative_months = null.
- Cumulative period with no monthly breakdown (e.g. "January-April"): fill
  cumulative_months (full month names) + period_year, period_month = null.
- Never put a cumulative/YTD value into a single-month record or vice versa.
- Trend tables/charts show several months and previous-year columns: create one
  object per month with its own correct month and year. Never assign a
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


class FatalAPIError(Exception):
    pass


class CallTimeout(Exception):
    pass


# ---------- helpers ----------
def _status(exc):
    for attr in ("status_code", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    return None


def _is_transient(exc):
    code = _status(exc)
    if code in TRANSIENT_CODES:
        return True
    return code is None and any(s in type(exc).__name__ for s in ("Connection", "Timeout"))


def create_with_deadline(deadline_s, **kwargs):
    """Run interactions.create in a thread; give up after deadline_s."""
    box = {}

    def worker():
        try:
            box["result"] = client.interactions.create(**kwargs)
        except BaseException as e:
            box["error"] = e

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    th.join(deadline_s)
    if th.is_alive():
        raise CallTimeout(f"no response after {deadline_s}s")
    if "error" in box:
        raise box["error"]
    return box["result"]


def _parse_json_array(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("top-level JSON is not an array")
    return data


def _all_blocked():
    now = time.monotonic()
    return all(_blocked_until.get(m, 0) > now for m in MODEL_CHAIN)


# ---------- model calls ----------
def _try_model(model, blocks, label):
    """Returns list[dict], or None if this model should be abandoned."""
    kwargs = dict(model=model, input=blocks, system_instruction=SYSTEM_PROMPT, store=False)
    if USE_SCHEMA:
        kwargs["response_format"] = RESPONSE_FORMAT
    if THINKING_LEVEL:
        kwargs["generation_config"] = {"thinking_level": THINKING_LEVEL}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t = time.monotonic()
            print(f"  call -> {model} ({len(blocks) // 2} docs, attempt {attempt}/{MAX_RETRIES})")
            interaction = create_with_deadline(CALL_DEADLINE_S, **kwargs)
            print(f"  done in {time.monotonic() - t:.1f}s")
            text = interaction.output_text
            if not text:
                raise ValueError(f"empty output_text (status={getattr(interaction, 'status', None)!r})")
            return _parse_json_array(text)

        except CallTimeout as e:
            # Don't pile more requests on a model that hangs; it may still be counting quota.
            print(f"  {model}: {e}. Abandoning this model.")
            return None

        except (json.JSONDecodeError, ValueError) as e:
            print(f"  {model}: bad output ({e})")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_SECONDS)

        except Exception as e:
            code = _status(e)
            if code in FATAL_CODES:
                raise FatalAPIError(f"[{code}] {e}") from e
            if code == 429:
                m = re.search(r"retry in ([\d.]+)s", str(e))
                delay = float(m.group(1)) if m else None
                if delay is not None and delay <= MAX_429_WAIT_S and attempt < MAX_RETRIES:
                    print(f"  {model}: 429, waiting {delay + 2:.0f}s")
                    time.sleep(delay + 2)
                    continue
                print(f"  {model}: 429 quota hit ({str(e)[:160]}). Blocking it for this run.")
                _blocked_until[model] = time.monotonic() + QUOTA_BLOCK_S
                return None
            if _is_transient(e):
                print(f"  {model}: [{code}] {str(e)[:160]}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_WAIT_SECONDS * attempt)
            else:
                print(f"  {model}: non-retryable (code={code}): {str(e)[:300]}")
                return None
    return None


def call_gemini(uploaded, label):
    """uploaded: list of (uri, mime, source_filename). Returns (rows, model) or (None, None)."""
    blocks = []
    for i, (uri, mime, _name) in enumerate(uploaded, 1):
        blocks.append({"type": "text", "text": f'source_file = "doc_{i}"'})
        blocks.append({"type": "document", "uri": uri, "mime_type": mime})
    blocks.append({"type": "text",
                   "text": "Extract the steel sector data from each of these reports."})

    now = time.monotonic()
    for model in MODEL_CHAIN:
        if _blocked_until.get(model, 0) > now:
            continue
        data = _try_model(model, blocks, label)
        if data is not None:
            return data, model
        if _blocked_until.get(model, 0) <= time.monotonic():
            _blocked_until[model] = time.monotonic() + MODEL_COOLDOWN_S
        print(f"  {model} abandoned; trying next model.")
    return None, None


# ---------- files / validation / csv ----------
def upload_and_wait(pdf_path, name):
    try:
        f = client.files.upload(file=pdf_path)
    except Exception as e:
        print(f"Upload failed for {name}: {e}")
        return None
    deadline = time.monotonic() + UPLOAD_TIMEOUT_S
    while f.state.name == "PROCESSING" and time.monotonic() < deadline:
        time.sleep(5)
        f = client.files.get(name=f.name)
    if f.state.name != "ACTIVE":
        print(f"File not usable for {name} (state={f.state.name}).")
        return None
    return f


def validate_rows(rows, alias_map, model_used):
    """Map doc_N back to real filenames; coerce types; flag problems."""
    out, dropped = [], 0
    for r in rows:
        name = alias_map.get(str(r.get("source_file", "")).strip())
        if name is None:
            dropped += 1
            print(f"  Dropping row with unknown source_file: {r.get('source_file')!r}")
            continue
        r["source_file"] = name
        issues = []
        for f in TONNAGE_FIELDS:
            v = r.get(f)
            if v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                issues.append(f"{f}:non-numeric")
                r[f] = None
                continue
            if v < 0:
                issues.append(f"{f}:negative")
            if v > 100:
                issues.append(f"{f}:>100(unit?)")
            r[f] = v
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
    if dropped:
        print(f"  WARNING: dropped {dropped} row(s) with unknown source_file")
    return out


def append_rows(rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    write_header = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore", restval="")
        if write_header:
            w.writeheader()
        w.writerows(rows)


def _append_lines(path, lines):
    if not lines:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def load_done():
    done = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            done.update(r["source_file"] for r in csv.DictReader(f))
    if os.path.exists(DONE_LOG):
        with open(DONE_LOG, encoding="utf-8") as f:
            done.update(line.strip() for line in f if line.strip())
    return done


def chunk(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def process_batch(filenames):
    """Returns (rows, files_with_rows, files_without_rows) or None on failure."""
    uploaded, ok_names = [], []
    for name in filenames:
        f = upload_and_wait(os.path.join(PDF_DIR, name), name)
        if f is not None:
            uploaded.append((f.uri, f.mime_type or "application/pdf", name))
            ok_names.append(name)
    if not uploaded:
        return None

    parsed, model_used = call_gemini(uploaded, "batch")
    if parsed is None:
        return None

    alias_map = {f"doc_{i}": n for i, n in enumerate(ok_names, 1)}
    rows = validate_rows(parsed, alias_map, model_used)
    got = sorted({r["source_file"] for r in rows})
    empty = [n for n in ok_names if n not in got]
    return rows, got, empty


def main():
    pdfs = sorted(f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))
    done = load_done()
    pending = [f for f in pdfs if f not in done]
    print(f"{len(pdfs)} PDFs, {len(pending)} pending. batch={BATCH_SIZE}, "
          f"max_batches={MAX_BATCHES or 'all'}, schema={USE_SCHEMA}, thinking={THINKING_LEVEL or 'default'}")

    failed, ran = 0, 0
    for names in chunk(pending, BATCH_SIZE):
        if MAX_BATCHES and ran >= MAX_BATCHES:
            print("MAX_BATCHES reached; stopping.")
            break
        ran += 1
        print(f"\nBatch {ran}: {names}")
        t0 = time.monotonic()
        try:
            result = process_batch(names)
        except FatalAPIError as e:
            print(f"Fatal API error, aborting: {e}")
            sys.exit(2)

        wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - t0)
        if wait > 0:
            time.sleep(wait)

        if result is None:
            failed += 1
            print("Batch failed; files stay pending for the next run.")
            if _all_blocked():
                print("All models are blocked (quota or repeated failures). Stopping; rerun later.")
                break
            continue

        rows, got, empty = result
        append_rows(rows)
        _append_lines(DONE_LOG, got + empty)
        if empty:
            _append_lines(EMPTY_LOG, empty)
            print(f"  WARNING: no rows for {empty} (logged in {EMPTY_LOG}, skipped from now on)")
        flagged = sum(1 for r in rows if r["needs_review"])
        print(f"Wrote {len(rows)} rows ({flagged} flagged needs_review).")

    if failed:
        print(f"{failed} batch(es) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()