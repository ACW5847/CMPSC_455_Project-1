# Anthony Wert
# CMPSC 445 Project 1
# Collect a balanced dataset of used vehicle listings from Craigslist
# Save raw records and a usable preview for later integration with OfferUp

'''
to begin the collection process, paste this into the terminal and enter:
python collect_craigslist.py --per-model 75 --max-details-per-model 160 --pages 1 --search-wait 12
do not run the code normally, as it wont collect the data with the parameters we need for a balanced set
'''

import argparse
import math
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.support.ui import WebDriverWait


VEHICLES = {
    "Ford": ["F-150", "Explorer", "Escape"],
    "Chevrolet": ["Silverado 1500", "Equinox", "Malibu"],
    "Toyota": ["Camry", "Corolla", "RAV4"],
}

# Craigslist has separate regional sites rather than one nationwide inventory
REGIONS = [
    "philadelphia", "newyork", "southjersey", "newjersey", "washingtondc",
    "boston", "chicago", "atlanta", "dallas", "houston", "austin",
    "losangeles", "sfbay", "sandiego", "seattle", "phoenix", "miami", "denver",
]

MIN_YEAR = 2005
MAX_YEAR = 2026
COLUMNS = [
    "source", "listing_url", "collected_at", "title", "make", "model",
    "model_year", "trim", "listing_price", "price_text", "mileage",
    "mileage_text", "condition", "fuel_type", "transmission", "drivetrain",
    "body_style", "exterior_color", "interior_color", "engine", "cylinders",
    "title_status", "seller_type", "location", "vin", "description",
    "posted_at", "stock_number", "validation_issues",
]


def text_of(element):
    return element.get_text(" ", strip=True) if element else ""


def number(value):
    match = re.search(r"\d[\d,]*(?:\.\d+)?", str(value)) if value is not None else None
    return float(match.group().replace(",", "")) if match else None


def clean_url(url):
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def model_pattern(model):
    if model == "F-150":
        return r"\bf[\s-]?150\b"
    if model == "Silverado 1500":
        return r"\bsilverado[\s-]*1500\b"
    if model == "RAV4":
        return r"\brav[\s-]?4\b"
    return r"\b" + re.escape(model).replace(r"\ ", r"[\s-]+") + r"\b"


def matches_identity(value, make, model):
    value = value or ""
    if model == "Explorer" and re.search(r"sport\s+trac", value, re.I):
        return False
    if model == "Camry" and re.search(r"\bsolara\b", value, re.I):
        return False
    if model == "Corolla" and re.search(r"\bcross\b", value, re.I):
        return False
    if model == "Equinox" and re.search(r"\bequinox\s+ev\b", value, re.I):
        return False
    make_pattern = r"\b(?:chevrolet|chevy)\b" if make == "Chevrolet" else r"\b" + re.escape(make) + r"\b"
    return bool(re.search(make_pattern, value, re.I) and re.search(model_pattern(model), value, re.I))


def search_url(region, make, model, offset, first_year=MIN_YEAR, last_year=MAX_YEAR):
    query = {"query": f"{make} {model}", "min_auto_year": first_year,
             "max_auto_year": last_year, "s": offset}
    return f"https://{region}.craigslist.org/search/cta?{urlencode(query)}"


def candidate_urls(html, base_url, make=None, model=None):
    soup = BeautifulSoup(html, "html.parser")
    found = []
    seen = set()

    # Current Craigslist search cards link to /view/d/... rather than /cto/d/...
    for card in soup.select(".cl-search-result"):
        title = card.get("title", "") or text_of(card.select_one(".posting-title"))
        if make and model and not matches_identity(title, make, model):
            continue
        link = card.select_one('a.posting-title[href], a.main[href]')
        if not link:
            continue
        url = clean_url(urljoin(base_url, link.get("href", "")))
        if not re.search(r"/(?:view/d/|(?:ctd|cto)/d/|d/[^/]+/\d+\.html$)", urlsplit(url).path, re.I):
            continue
        if url not in seen:
            seen.add(url)
            found.append(url)
    return found


def embedded_result_count(html):
    soup = BeautifulSoup(html, "html.parser")
    script = soup.select_one("#ld_searchpage_results")
    if not script:
        return 0
    try:
        return len(json.loads(script.get_text()).get("itemListElement", []))
    except (ValueError, TypeError, AttributeError):
        return 0


def get_search_html(driver, url, delay, wait_seconds, make, model):
    if delay:
        time.sleep(delay)
    driver.get(url)
    # Do not mistake the temporary 'retrieving' screen for a completed search
    try:
        WebDriverWait(driver, wait_seconds, poll_frequency=0.5).until(
            lambda d: candidate_urls(d.page_source, url, make, model)
            or 'retrieving-results' not in (d.find_element('tag name', 'main').get_attribute('class') or '')
        )
    except (TimeoutException, WebDriverException):
        pass
    return driver.page_source


def attributes_from_page(soup):
    attrs = {}
    for block in soup.select(".attrgroup"):
        for item in block.select(".attr"):
            label = text_of(item.select_one(".labl")).lower().strip().rstrip(":")
            value = text_of(item.select_one(".valu"))
            if label and value:
                attrs[label] = value
        for span in block.find_all("span", recursive=False):
            value = text_of(span)
            if ":" in value:
                key, val = value.split(":", 1)
                attrs.setdefault(key.strip().lower(), val.strip())
    for label in soup.select("dt"):
        key = text_of(label).lower().rstrip(":")
        value = label.find_next_sibling("dd")
        if key and value:
            attrs.setdefault(key, text_of(value))
    return attrs


def parse_listing(html, url, make, model):
    soup = BeautifulSoup(html, "html.parser")
    attrs = attributes_from_page(soup)
    title = text_of(soup.select_one("#titletextonly")) or text_of(soup.select_one("h1"))
    identity = text_of(soup.select_one(".makemodel")) or attrs.get("make / model", "")
    if not identity:
        identity = text_of(soup.select_one(".attrgroup b")) or title
    year_text = text_of(soup.select_one(".year")) or attrs.get("year", "")
    year_match = re.search(r"\b(?:19|20)\d{2}\b", year_text or identity or title)
    year = int(year_match.group()) if year_match else None
    price_text = text_of(soup.select_one(".postingtitletext .price"))
    if not price_text:
        price_text = text_of(soup.select_one(".price"))
    mileage_text = attrs.get("odometer", attrs.get("mileage", ""))
    description_node = soup.select_one("#postingbody")
    if description_node:
        for item in description_node.select(".print-information"):
            item.decompose()
    description = text_of(description_node).removeprefix("QR Code Link to This Post").strip()
    vin = attrs.get("vin", "").strip().upper()
    if not vin:
        match = re.search(r"\bVIN\s*[:#]?\s*([A-HJ-NPR-Z0-9]{17})\b", description, re.I)
        vin = match.group(1).upper() if match else ""
    if not re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", vin):
        vin = ""
    full_text = soup.get_text(" ", strip=True).lower()
    seller_type = ""
    if "cars & trucks - by dealer" in full_text or "for sale by dealer" in full_text:
        seller_type = "dealer"
    elif "cars & trucks - by owner" in full_text or "for sale by owner" in full_text:
        seller_type = "private"
    region_match = re.search(r"\(([^()]*)\)\s*$", text_of(soup.select_one(".postingtitletext")))
    location = region_match.group(1) if region_match else urlsplit(url).netloc.split(".")[0]
    posted = soup.select_one("time[datetime]")
    trim = attrs.get("trim", "")
    if not trim:
        model_hit = re.search(model_pattern(model), identity, re.I)
        if model_hit:
            trim = identity[model_hit.end():].strip(" -")
    record = dict.fromkeys(COLUMNS, "")
    record.update({
        "source": "craigslist", "listing_url": clean_url(url),
        "collected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "title": title, "make": make, "model": model, "model_year": year,
        "trim": trim, "listing_price": number(price_text), "price_text": price_text,
        "mileage": number(mileage_text), "mileage_text": mileage_text,
        "condition": attrs.get("condition", ""), "fuel_type": attrs.get("fuel", ""),
        "transmission": attrs.get("transmission", ""), "drivetrain": attrs.get("drive", ""),
        "body_style": attrs.get("type", ""), "exterior_color": attrs.get("paint color", ""),
        "interior_color": attrs.get("interior color", ""), "engine": attrs.get("engine", ""),
        "cylinders": attrs.get("cylinders", ""), "title_status": attrs.get("title status", ""),
        "seller_type": seller_type, "location": location, "vin": vin,
        "description": description, "posted_at": posted.get("datetime", "") if posted else "",
    })
    errors = []
    if not matches_identity(identity, make, model):
        errors.append("make/model mismatch")
    if year is None or not MIN_YEAR <= year <= MAX_YEAR:
        errors.append("year missing/out of range")
    if record["listing_price"] is None or record["listing_price"] <= 0:
        errors.append("price missing/invalid")
    if record["mileage"] is None or record["mileage"] < 0:
        errors.append("mileage missing/invalid")
    if not description:
        errors.append("description missing")
    title_year = re.search(r"\b(?:19|20)\d{2}\b", title)
    if year and title_year and int(title_year.group()) != year:
        errors.append("conflicting year")
    record["validation_issues"] = "; ".join(errors)
    return record


def get_html(driver, url, delay):
    if delay:
        time.sleep(delay)
    driver.get(url)
    return driver.page_source


def write_report(folder):
    path = folder / "craigslist_raw.csv"
    data = pd.read_csv(path, dtype=str).fillna("")
    data = data.drop_duplicates(subset=["listing_url"], keep="first")
    good = data[data["validation_issues"].eq("")].copy()
    valid_vin = good["vin"].str.fullmatch(r"[A-HJ-NPR-Z0-9]{17}")
    duplicate_vin = good["vin"].duplicated()
    good = good.loc[~(valid_vin & duplicate_vin)]
    data.to_csv(path, index=False)
    good.to_csv(folder / "craigslist_usable_preview.csv", index=False)
    stats = data.groupby(["make", "model"]).agg(
        parsed=("listing_url", "size"),
        initially_usable=("validation_issues", lambda s: s.eq("").sum()),
    )
    stats.to_csv(folder / "model_counts.csv")
    data.replace(r"^\s*$", pd.NA, regex=True).notna().sum().rename("nonmissing").to_csv(
        folder / "attribute_coverage.csv", header=True)
    print("\nFINAL SUMMARY")
    print(stats.to_string())
    print("Parsed:", len(data), "Usable after VIN dedup:", len(good), flush=True)
    print("Saved to:", folder.resolve(), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Balanced Craigslist used vehicle collection")
    parser.add_argument("--per-model", type=int, default=75)
    parser.add_argument("--max-details-per-model", type=int, default=160)
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--search-wait", type=int, default=12, help="Seconds to wait for listing URLs")
    parser.add_argument("--regions", nargs="+", default=REGIONS)
    parser.add_argument("--model", help='Optional pilot, e.g. "Toyota:Camry"')
    args = parser.parse_args()
    if args.per_model < 1 or args.max_details_per_model < 1 or args.pages < 1 or args.delay < 0 or args.search_wait < 1:
        parser.error("Invalid collection limits")
    vehicles = [(make, model) for make, models in VEHICLES.items() for model in models]
    if args.model:
        vehicles = [(mk, md) for mk, md in vehicles if (mk + ":" + md).lower() == args.model.lower()]
        if not vehicles:
            parser.error("Unknown model")
    folder = Path(__file__).resolve().parent / "craigslist_results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    raw_path = folder / "craigslist_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=COLUMNS).writeheader()
    seen_urls = set()
    seen_vins = set()
    driver = webdriver.Chrome()
    driver.set_page_load_timeout(30)
    # Search four year ranges in rotation before allowing a general fallback.
    # This prevents the first region or earliest model years from dominating.
    year_groups = [(2005, 2009), (2010, 2014), (2015, 2019), (2020, 2026)]
    empty_searches = 0
    try:
        for make, model in vehicles:
            usable = 0
            attempts = 0
            year_counts = {group: 0 for group in year_groups}
            per_group_target = math.ceil(args.per_model / len(year_groups))

            # First pass: collect up to the group quota from each year range.
            # Second pass: fill any remaining spots using the full year range.
            for balanced in (True, False):
                if usable >= args.per_model or attempts >= args.max_details_per_model:
                    break
                groups = year_groups if balanced else [(MIN_YEAR, MAX_YEAR)]
                for first_year, last_year in groups:
                    if usable >= args.per_model or attempts >= args.max_details_per_model:
                        break
                    group = (first_year, last_year)
                    attempts_at_group_start = attempts
                    group_attempt_cap = max(10, args.max_details_per_model // 5)
                    if balanced and year_counts[group] >= per_group_target:
                        continue
                    for region in args.regions:
                        if usable >= args.per_model or attempts >= args.max_details_per_model:
                            break
                        if balanced and (year_counts[group] >= per_group_target or
                                         attempts - attempts_at_group_start >= group_attempt_cap):
                            break
                        for page in range(args.pages):
                            if usable >= args.per_model or attempts >= args.max_details_per_model:
                                break
                            if balanced and (year_counts[group] >= per_group_target or
                                             attempts - attempts_at_group_start >= group_attempt_cap):
                                break
                            url = search_url(region, make, model, page * 120, first_year, last_year)
                            print("\nSEARCH", make, model, region, first_year, "to", last_year,
                                  "page", page + 1, flush=True)
                            try:
                                html = get_search_html(driver, url, args.delay, args.search_wait, make, model)
                                urls = candidate_urls(html, url, make, model)
                            except (TimeoutException, WebDriverException) as error:
                                print("Search failed:", str(error)[:120], flush=True)
                                continue
                            fresh = [u for u in urls if u not in seen_urls]
                            print("Candidates:", len(urls), "new:", len(fresh),
                                  "embedded search entries:", embedded_result_count(driver.page_source), flush=True)
                            if not urls:
                                empty_searches += 1
                            else:
                                empty_searches = 0
                            if not urls:
                                diagnostic = folder / (f"search_{make}_{model}_{region}_{first_year}_{last_year}_{page}.html".replace(" ", "_"))
                                diagnostic.write_text(driver.page_source, encoding="utf-8")
                                if empty_searches >= 3:
                                    raise RuntimeError("Three searches returned zero listing URLs; review the saved search HTML")
                            for listing_url in fresh:
                                if usable >= args.per_model or attempts >= args.max_details_per_model:
                                    break
                                if balanced and (year_counts[group] >= per_group_target or
                                                 attempts - attempts_at_group_start >= group_attempt_cap):
                                    break
                                seen_urls.add(listing_url)
                                attempts += 1
                                try:
                                    row = parse_listing(get_html(driver, listing_url, args.delay), listing_url, make, model)
                                except (TimeoutException, WebDriverException) as error:
                                    print("Detail failed:", str(error)[:120], flush=True)
                                    continue
                                with raw_path.open("a", newline="", encoding="utf-8") as file:
                                    writer = csv.DictWriter(file, fieldnames=COLUMNS)
                                    writer.writerow(row)
                                if not row["validation_issues"]:
                                    year = int(row["model_year"])
                                    # Check the actual year, not just the search filter.
                                    if balanced and not first_year <= year <= last_year:
                                        print(" ", year, row["title"][:40], "| outside searched year range")
                                        continue
                                    vin = row["vin"]
                                    if vin and vin in seen_vins:
                                        print("  duplicate VIN", flush=True)
                                        continue
                                    if vin:
                                        seen_vins.add(vin)
                                    usable += 1
                                    for bucket in year_groups:
                                        if bucket[0] <= year <= bucket[1]:
                                            year_counts[bucket] += 1
                                            break
                                print(" ", row["model_year"], row["title"][:40], "|",
                                      row["validation_issues"] or "USABLE", flush=True)
                            if not urls:
                                break
            print("COMPLETED", make, model, "usable:", usable, "/", args.per_model,
                  "year groups:", year_counts, "detail attempts:", attempts, flush=True)
    except (KeyboardInterrupt, RuntimeError) as error:
        print("\nStopped:", str(error) or "Interrupted", "| CSV records preserved", flush=True)
    finally:
        driver.quit()
        write_report(folder)


if __name__ == "__main__":
    main()
