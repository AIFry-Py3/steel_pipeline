"""
llm_extractor.py
Uses Gemini to pull structured, monthly steel sector data out of
messy, inconsistently-formatted PDF text.
"""

import os
import re
import json
import time
from google import genai
from google.genai import types, errors
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "models/gemini-3.6-flash"

MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 10

SYSTEM_PROMPT = """You extract steel sector data from Indian government reports.

The input is a JSON array of {source_file, text} objects. 

For EACH month explicitly mentioned in the text part of the objects, 
process each object's text independently with respect to the source_file (including monthly figures inside tables),
return a JSON array of objects with this exact schema:

{
  "source_filename":"<source_file_name, as given in the input JSON>",
  "period_month":"<month name, e.g. February, or null if not stated>",
  "cumulative_months": "<range of month names, e.g. January-April, April-August, or null if not stated>",
  "period_year": "<4-digit year as string, or null>",
  "consumption_value": "<numeric value only, e.g. 9.26, or null if not found>",
  "consumption_unit": "<e.g. 'million tonnes' or 'MT', or null>",
  "crude_steel_production": "<numeric value, or null>",
  "finished_steel_production": "<numeric value, or null>",
  "crude_steel_export": "<numeric value, or null>",
  "crude_steel_import":"<numeric value, or null>",
  "finished_steel_export":"<numeric value, or null>",
  "finished_steel_import":"<numeric value, or null>",
  "coal_imports":"<numeric value, or null>",
  "steel_imports":"<numeric value, or null>"
}

Return a JSON array of objects containing inside each a response in the above format. 
An object for every month found in the text. 

Rules:
- Return ONLY the JSON array as given above. No explanation, no markdown code fences.
- If a field cannot be found for a given month, use null.
- Normalize numeric values as plain numbers (no commas, no units inside the number).
- If report covers a single month (e.g. "September") only, enter it simply into "period_month",
  enter it as given (e.g. period_month:"September", period_year: "2023"). Let "cumulative_months" be null.
- If the report covers a cumulative period (e.g. "April-January") with no monthly
  breakdown, enter it as given in "cumulative_months", with full month names 
  and capitalized first letters (e.g. cumulative_months: "January-April", period_year: "2026"). 
  Let "period_month" be null.
- If the report contains some values for which year is mentioned but not month (e.g. "2025"),
  enter details respectively into csv with "period_month" and "cumulative_months" as null.
- Do not manipulate any data or create your own, only use data that is given.
"""



def _call_gemini_with_retry(contents, context_label):
    """
    Shared call+retry logic used by both single-file and batch extraction.
    Returns parsed JSON (list[dict]) or None on permanent failure.
    """
    config = types.GenerateContentConfig(
        max_output_tokens=65536,
        system_instruction=SYSTEM_PROMPT,
        response_mime_type='application/json',
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=config,
            )
            cleaned = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)  # list[dict]

        except errors.APIError as e:
            transient = e.code in (429, 503)
            if transient and attempt < MAX_RETRIES:
                wait_time = RETRY_WAIT_SECONDS * attempt  # 10, 20, 30, 40...
                match = re.search(r"retry in ([\d.]+)s", str(e.message))
                if match:
                    wait_time = float(match.group(1)) + 2  # small buffer
                print(f"Attempt {attempt} failed for {context_label}: [{e.code}] {e.message}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            elif transient:
                print(f"LLM extraction failed for {context_label} after {MAX_RETRIES} attempts: [{e.code}] {e.message}")
                return None
            else:
                print(f"LLM extraction failed for {context_label} (non-retryable): [{e.code}] {e.message}")
                return None

        except json.JSONDecodeError as e:
            print(f"LLM returned invalid JSON for {context_label}: {e}")
            return None


def extract_steel_data(text, source_filename=""):
    """
    Single-PDF extraction (legacy path, still usable). Returns list of dicts, or None.
    """
    contents = [
        json.dumps([{"source_file": source_filename, "text": text}]),
        "Extract the steel sector data from this report.",
    ]
    parsed = _call_gemini_with_retry(contents, source_filename)
    if parsed is None:
        return None
    for entry in parsed:
        if "source_file" not in entry and "source_filename" in entry:
            entry["source_file"] = entry.pop("source_filename")
        entry.setdefault("source_file", source_filename)
    return parsed


def extract_steel_data_batch(files):
    """
    Batched extraction. `files` is a list of (source_filename, text) tuples.
    Sends them as one JSON array of {source_file, text} objects in a single
    API call, cutting request count by up to len(files)-fold. Returns a flat
    list of dicts (each tagged with its source_file), or None on permanent failure.
    """
    input_array = [{"source_file": name, "text": text} for name, text in files]
    contents = [
        json.dumps(input_array),
        "Extract the steel sector data from each of these reports.",
    ]

    label = f"batch({', '.join(name for name, _ in files)})"
    parsed = _call_gemini_with_retry(contents, label)
    if parsed is None:
        return None

    for entry in parsed:
        if "source_file" not in entry and "source_filename" in entry:
            entry["source_file"] = entry.pop("source_filename")

    return parsed