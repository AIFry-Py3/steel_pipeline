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
MODEL = "models/gemini-3.5-flash"

MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 10

SYSTEM_PROMPT = """You extract steel sector data from Indian government reports.

The input is a JSON array of {source_file, text} objects. 

For EACH month explicitly mentioned in the text part of the objects, 
process each object's text independently with respect to the source_file (including monthly figures inside tables),
return a JSON array of objects with this exact schema:

{
  "source_filename":"<source_file_name, as given in the input JSON>",
  "period_month": "<month name, e.g. January, or null if not stated>",
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
  "iron_ore_used":"<numeric value, or null>"
}

Return a JSON array of objects containing inside each a response in the above format. 
An object for every month found in the text. 

Rules:
- Return ONLY the JSON array as given above. No explanation, no markdown code fences.
- If a field cannot be found for a given month, use null.
- Normalize numeric values as plain numbers (no commas, no units inside the number).
- If the report covers a cumulative period (e.g. "April-January") with no monthly
  breakdown, use the report's own month/year label (e.g. "Monthly Economic Report
  for January 2026" -> period_month: "January", period_year: "2026").
"""



def extract_steel_data(text, source_filename=""):
    """
    Sends PDF text to Gemini and returns a list of dicts matching the schema above,
    one dict per month found in the report. Retries on APIError up to MAX_RETRIES
    times. Returns None if the call fails permanently or the response isn't valid JSON.
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
                contents=[text, "Extract the steel sector data from this report."],
                config=config,
            )

            cleaned = response.text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(cleaned)  # list[dict]

            for entry in parsed:
                entry["source_file"] = source_filename

            return parsed

        except errors.APIError as e:
            transient = e.code in (429, 503)
            if transient and attempt < MAX_RETRIES:
                wait_time = RETRY_WAIT_SECONDS * attempt  # 10, 20, 30, 40...
                # Gemini's 429 message embeds its own suggested retry delay in seconds
                match = re.search(r"retry in ([\d.]+)s", str(e.message))
                if match:
                    wait_time = float(match.group(1)) + 2  # small buffer
                print(f"Attempt {attempt} failed for {source_filename}: [{e.code}] {e.message}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            elif transient:
                print(f"LLM extraction failed for {source_filename} after {MAX_RETRIES} attempts: [{e.code}] {e.message}")
                return None
            else:
                # Non-transient error (404 model not found, 400 bad request, etc.) — retrying won't help
                print(f"LLM extraction failed for {source_filename} (non-retryable): [{e.code}] {e.message}")
                return None

        except json.JSONDecodeError as e:
            print(f"LLM returned invalid JSON for {source_filename}: {e}")
            return None