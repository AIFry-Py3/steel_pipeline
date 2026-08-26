"""
Steel data pipeline: steel.gov.in/monthly-summary

1. Scan paginated pages for PDF links
2. Download PDFs not already stored locally
3. Extract text from each PDF
4. Parse monthly steel data out of the text via Gemini (llm_extractor.py)
5. Append each month's row to steel_data.csv, skipping files already logged
"""

import os
import csv
import time
import requests
from bs4 import BeautifulSoup
import pdfplumber
from llm_extractor import extract_steel_data

BASE_URL = "https://steel.gov.in"
PAGE_URL = "https://steel.gov.in/monthly-summary"
PDF_DIR = "pdfs"
CSV_PATH = "steel_data.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (research/academic data collection)"}

os.makedirs(PDF_DIR, exist_ok=True)


def get_pdf_links_from_page(page_num):
    """Return all PDF URLs found on a single paginated results page."""
    url = f"{PAGE_URL}?page={page_num}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            full_url = href if href.startswith("http") else BASE_URL + href
            links.append(full_url)
    return list(set(links))


def get_all_pdf_links(max_pages=500, no_growth_limit=5):
    """
    Loop through ?page=0,1,2...
    Stops after `no_growth_limit` consecutive pages add zero NEW links
    (tolerates recurring static docs, e.g. a policy PDF appearing on every page).
    """
    all_links = set()
    consecutive_no_growth = 0

    for page_num in range(max_pages):
        links = get_pdf_links_from_page(page_num)
        before_count = len(all_links)
        all_links.update(links)
        new_count = len(all_links) - before_count

        print(f"Page {page_num}: {len(links)} links found, {new_count} new.")

        if new_count == 0:
            consecutive_no_growth += 1
            if consecutive_no_growth >= no_growth_limit:
                print(f"No new links for {no_growth_limit} consecutive pages — stopping.")
                break
        else:
            consecutive_no_growth = 0

    return list(all_links)


def already_downloaded(url):
    filename = os.path.basename(url.split("?")[0])
    return os.path.exists(os.path.join(PDF_DIR, filename)), filename


def download_pdf(url, filename):
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    path = os.path.join(PDF_DIR, filename)
    with open(path, "wb") as f:
        f.write(resp.content)
    return path


def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def append_to_csv(row):
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def already_in_csv(source_filename):
    if not os.path.exists(CSV_PATH):
        return False
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        return any(r["source_file"] == source_filename for r in reader)


def main():
    pdf_links = get_all_pdf_links()
    print(f"Found {len(pdf_links)} total PDF links across all pages.")

    for url in pdf_links:
        exists, filename = already_downloaded(url)

        if not exists:
            print(f"Downloading: {filename}")
            path = download_pdf(url, filename)
        else:
            print(f"Already downloaded: {filename}")
            path = os.path.join(PDF_DIR, filename)

        if already_in_csv(filename):
            print(f"Already in CSV, skipping: {filename}")
            continue

        text = extract_text(path)
        parsed_data = extract_steel_data(text, filename)
        time.sleep(6)

        if not parsed_data:
            print(f"Skipping {filename} — extraction failed or returned nothing.")
            continue

        for row in parsed_data:
            append_to_csv(row)
            print(f"Appended: {row}")


if __name__ == "__main__":
    main()