# Anthony Wert
# CMPSC 445 Project 1
# Collect historical vehicle MSRP reference data from The Car Connection
# Save verified prices and missing combinations for later matching

import argparse
import csv
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


MODEL_SLUGS = {
    ("Ford", "F-150"): "ford_f-150",
    ("Ford", "Explorer"): "ford_explorer",
    ("Ford", "Escape"): "ford_escape",
    ("Chevrolet", "Silverado 1500"): "chevrolet_silverado-1500",
    ("Chevrolet", "Equinox"): "chevrolet_equinox",
    ("Chevrolet", "Malibu"): "chevrolet_malibu",
    ("Toyota", "Camry"): "toyota_camry",
    ("Toyota", "Corolla"): "toyota_corolla",
    ("Toyota", "RAV4"): "toyota_rav4",
}

REFERENCE_COLUMNS = [
    "make", "model", "model_year", "configuration", "original_msrp",
    "reference_type", "source_url", "collected_at"
]


def find_integrated_csv(project):
    candidates = [
        project / "data" / "processed" / "vehicles_integrated_clean.csv",
        project / "data" / "vehicles_integrated_clean.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Cannot find vehicles_integrated_clean.csv. Use --input to specify it.")


def page_url(make, model, year):
    slug = MODEL_SLUGS[(make, model)]
    return f"https://www.thecarconnection.com/specifications/{slug}_{year}"


def number_from_price(value):
    match = re.search(r"\$\s*([\d,]+)", value)
    if not match:
        return None
    price = int(match.group(1).replace(",", ""))
    return price if 1000 <= price <= 300000 else None


def parse_msrp_page(html, make, model, year, url):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.select("script, style, noscript"):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    expected = f"{year} {make} {model}".lower()
    if expected not in re.sub(r"\s+", " ", title.lower()):
        return [], "unexpected page title"

    # Read only the MSRP section so equipment and accessory prices are excluded
    lines = [re.sub(r"\s+", " ", text).strip()
             for text in soup.get_text("\n", strip=True).splitlines()]
    lines = [line for line in lines if line]
    start = next((i for i, line in enumerate(lines) if line.startswith("MSRP based on")), None)
    if start is None:
        # Some pages contain a selected MSRP but no complete trim table
        return [], "no multi-trim MSRP table"

    end = next((i for i in range(start + 1, len(lines))
                if lines[i] in ("Quick Specs", "### Quick Specs", "Compare")), len(lines))
    section = lines[start:end]

    # Configuration and price may be together or on adjacent lines
    extracted = []
    seen = set()
    price_only = re.compile(r"^\$\s*[\d,]{4,7}$")
    inline_price = re.compile(r"^(.{3,100}?)\s+\$\s*([\d,]{4,7})$")
    for index, line in enumerate(section):
        match = inline_price.match(line)
        if match:
            config = match.group(1)
            value = number_from_price("$" + match.group(2))
        elif price_only.match(line) and index > 0:
            config = section[index - 1]
            value = number_from_price(line)
        else:
            continue
        config = re.sub(r"^[•*\-]\s*", "", config).strip()
        if not value or len(config) < 3 or config.lower() in ("msrp", "msrp based on"):
            continue
        if any(phrase in config.lower() for phrase in ("camry solara", "corolla cross", "equinox ev")):
            continue
        key = (config.casefold(), value)
        if key not in seen:
            seen.add(key)
            extracted.append({
                "make": make, "model": model, "model_year": year,
                "configuration": config, "original_msrp": value,
                "reference_type": "listed_configuration", "source_url": url,
                "collected_at": datetime.now().astimezone().isoformat(timespec="seconds")
            })
    return extracted, "ok" if extracted else "MSRP table found but prices not parsed"


def fetch_html(session, url, timeout, browser=None):
    if browser is not None:
        browser.get(url)
        time.sleep(1)
        return browser.page_source
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def main():
    parser = argparse.ArgumentParser(description="Build a sourced historical MSRP reference table")
    parser.add_argument("--input", type=Path, help="Clean integrated vehicle CSV")
    parser.add_argument("--output", type=Path, default=Path("data/msrp"))
    parser.add_argument("--test", action="store_true", help="Check four model-year pages first")
    parser.add_argument("--browser", action="store_true", help="Use Chrome Selenium instead of requests")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    project = Path(__file__).resolve().parent
    input_path = args.input or find_integrated_csv(project)
    vehicles = pd.read_csv(input_path)
    required = {"make", "model", "model_year"}
    if not required.issubset(vehicles.columns):
        parser.error("Input CSV must include make, model, and model_year")

    combinations = (vehicles[["make", "model", "model_year"]]
                    .dropna().drop_duplicates())
    combinations["model_year"] = combinations["model_year"].astype(int)
    combinations = [tuple(row) for row in combinations.itertuples(index=False, name=None)
                    if (row[0], row[1]) in MODEL_SLUGS and 2005 <= row[2] <= 2026]
    combinations.sort(key=lambda row: (row[0], row[1], row[2]))
    if args.test:
        samples = [
            ("Toyota", "Camry", 2005),
            ("Ford", "F-150", 2005),
            ("Chevrolet", "Silverado 1500", 2022),
            ("Toyota", "RAV4", 2025),
        ]
        combinations = samples

    output = args.output if args.output.is_absolute() else project / args.output
    output.mkdir(parents=True, exist_ok=True)
    reference_path = output / ("msrp_test.csv" if args.test else "msrp_reference.csv")
    status_path = output / ("msrp_test_status.csv" if args.test else "msrp_status.csv")

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; AcademicResearch/1.0)"})
    browser = None
    if args.browser:
        from selenium import webdriver
        browser = webdriver.Chrome()
        browser.set_page_load_timeout(args.timeout)

    # Keep successful results when a long collection is interrupted
    collected_keys = set()
    if reference_path.exists() and not args.test:
        prior = pd.read_csv(reference_path)
        for row in prior.itertuples(index=False):
            collected_keys.add((str(row.make), str(row.model), int(row.model_year),
                                str(row.configuration), int(row.original_msrp)))
    finished = set()
    if status_path.exists() and not args.test:
        prior_status = pd.read_csv(status_path)
        for row in prior_status.itertuples(index=False):
            if row.status == "ok":
                finished.add((str(row.make), str(row.model), int(row.model_year)))

    try:
        with reference_path.open("a", newline="", encoding="utf-8") as ref_file, \
             status_path.open("a", newline="", encoding="utf-8") as status_file:
            ref_writer = csv.DictWriter(ref_file, fieldnames=REFERENCE_COLUMNS)
            status_writer = csv.DictWriter(status_file, fieldnames=[
                "make", "model", "model_year", "source_url", "configurations", "status"])
            if ref_file.tell() == 0:
                ref_writer.writeheader()
            if status_file.tell() == 0:
                status_writer.writeheader()

            for index, (make, model, year) in enumerate(combinations, start=1):
                if (make, model, year) in finished:
                    continue
                url = page_url(make, model, year)
                try:
                    html = fetch_html(session, url, args.timeout, browser)
                    entries, status = parse_msrp_page(html, make, model, year, url)
                except Exception as error:
                    entries, status = [], f"request failed: {str(error)[:100]}"
                new_count = 0
                for entry in entries:
                    key = (make, model, year, entry["configuration"], entry["original_msrp"])
                    if key not in collected_keys:
                        ref_writer.writerow(entry)
                        collected_keys.add(key)
                        new_count += 1
                status_writer.writerow({"make": make, "model": model, "model_year": year,
                                        "source_url": url, "configurations": new_count, "status": status})
                ref_file.flush()
                status_file.flush()
                print(f"[{index}/{len(combinations)}] {year} {make} {model}: "
                      f"{len(entries)} configurations | {status}", flush=True)
                time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\nInterrupted. Completed results are saved.")
    finally:
        if browser is not None:
            browser.quit()
        session.close()
    print("\nReference:", reference_path)
    print("Status report:", status_path)
    print("Review failed or incomplete pages before matching MSRP to vehicles")


if __name__ == "__main__":
    main()
