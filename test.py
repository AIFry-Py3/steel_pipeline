"""
llm_visual_extractor.py
Reads steel-sector data from PDF pages (tables AND charts) via Gemini vision,
using the stable generate_content endpoint (same path as llm_extractor.py,
which is known to work). Model fallback + validation. Appends to
data/raw_steel_data.csv.

Test run (PowerShell):
  $env:BATCH_SIZE=1; $env:MAX_BATCHES=1; python .\\visual_extractor\\llm_visual_extractor.py
  $env:THINKING_LEVEL=""     # omit thinking config entirely
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
from google.genai import types, errors

load_dotenv()

# ---------- config ----------
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "3"))
MAX_BATCHES = int(os.getenv("MAX_BATCHES", "0"))           # 0 = no limit
THINKING_LEVEL = os.getenv("THINKING_LEVEL", "high")         # "" = omit

MODEL_CHAIN = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
]

MODEL_COOLDOWN_S = 600
QUOTA_BLOCK_S = 24 * 3600
_blocked_until = {}

CALL_DEADLINE_S = 180
MAX_RETRIES = 3                  # per model
RETRY_WAIT_SECONDS = 15
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
    + ["value_source", "model_used", "needs_review"]
)
MONTHS = {
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
}

SYSTEM_PROMPT = """
HI extract data from the report pls
"""


class FatalAPIError(Exception):
    pass


class CallTimeout(Exception):
    pass


# ---------- helpers ----------
def _status(exc):
    for attr in ("code", "status_code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    return None


def _make_config():
    kw = dict(
        max_output_tokens=65536,
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
    )
    if THINKING_LEVEL:
        try:
            kw["thinking_config"] = types.ThinkingConfig(thinking_level=THINKING_LEVEL)
        except Exception as e:
            print(f"thinking_level not accepted by this SDK ({e}); continuing without it.")
    return types.GenerateContentConfig(**kw)


def generate_with_deadline(deadline_s, **kwargs):
    """Run generate_content in a thread; give up after deadline_s."""
    box = {}

    def worker():
        try:
            box["result"] = client.models.generate_content(**kwargs)
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
def _try_model(model, contents, config):
    """Returns list[dict], or None if this model should be abandoned."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t = time.monotonic()
            print(f"  call -> {model} (attempt {attempt}/{MAX_RETRIES})")
            response = generate_with_deadline(
                CALL_DEADLINE_S, model=model, contents=contents, config=config)
            print(f"  done in {time.monotonic() - t:.1f}s")
            text = response.text
            if not text:
                fr = response.candidates[0].finish_reason if response.candidates else None
                raise ValueError(f"empty response (finish_reason={fr})")
            return _parse_json_array(text)

        except CallTimeout as e:
            print(f"  {model}: {e}. Abandoning this model.")
            return None

        except (json.JSONDecodeError, ValueError) as e:
            print(f"  {model}: bad output ({e})")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_SECONDS)

        except errors.APIError as e:
            code = _status(e)
            msg = str(getattr(e, "message", e))
            if code in FATAL_CODES:
                raise FatalAPIError(f"[{code}] {msg}") from e
            if code == 429:
                m = re.search(r"retry in ([\d.]+)s", msg)
                delay = float(m.group(1)) if m else None
                if delay is not None and delay <= MAX_429_WAIT_S and attempt < MAX_RETRIES:
                    print(f"  {model}: 429, waiting {delay + 2:.0f}s")
                    time.sleep(delay + 2)
                    continue
                print(f"  {model}: 429 quota hit ({msg[:160]}). Blocking it for this run.")
                _blocked_until[model] = time.monotonic() + QUOTA_BLOCK_S
                return None
            if code in TRANSIENT_CODES:
                print(f"  {model}: [{code}] {msg[:160]}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_WAIT_SECONDS * attempt)
            else:
                print(f"  {model}: non-retryable [{code}]: {msg[:300]}")
                return None

        except Exception as e:
            print(f"  {model}: unexpected {type(e).__name__}: {str(e)[:300]}")
            return None
    return None


def call_gemini(uploaded):
    """uploaded: list of (uri, mime, source_filename). Returns (rows, model) or (None, None)."""
    contents = []
    for i, (uri, mime, _name) in enumerate(uploaded, 1):
        contents.append(f'source_file = "doc_{i}"')
        contents.append(types.Part.from_uri(file_uri=uri, mime_type=mime))
    contents.append("Extract the steel sector data from each of these reports.")

    config = _make_config()
    now = time.monotonic()
    for model in MODEL_CHAIN:
        if _blocked_until.get(model, 0) > now:
            continue
        data = _try_model(model, contents, config)
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
        if not isinstance(r, dict):
            dropped += 1
            continue
        key = r.get("source_file", r.get("source_filename", ""))
        name = alias_map.get(str(key).strip())
        if name is None:
            dropped += 1
            print(f"  Dropping row with unknown source_file: {key!r}")
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
        print(f"  WARNING: dropped {dropped} row(s)")
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

    parsed, model_used = call_gemini(uploaded)
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
          f"max_batches={MAX_BATCHES or 'all'}, thinking={THINKING_LEVEL or 'default'}")

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
                print("All models blocked (quota or repeated failures). Stopping; rerun later.")
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