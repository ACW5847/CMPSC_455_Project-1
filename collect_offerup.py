# Anthony Wert
# CMPSC 445 Project 1
# Collect a balanced dataset of used car listings from OfferUp
# Save raw listings and a validated preview for later preprocessing

'''
to begin the collection process, paste this into the terminal and enter:
python collect_offerup_final.py --per-model 75 --max-details-per-model 350 --scrolls 3
do not run the code normally, as it wont collect the data with the parameters we need for a balanced set
'''
import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit, urlencode

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, WebDriverException

VEHICLES = {
    "Ford": ["F-150", "Explorer", "Escape"],
    "Chevrolet": ["Silverado 1500", "Equinox", "Malibu"],
    "Toyota": ["Camry", "Corolla", "RAV4"],
}
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


def numeric(value):
    if value is None:
        return None
    match = re.search(r"\d[\d,]*(?:\.\d+)?", str(value))
    return float(match.group().replace(",", "")) if match else None


def clean_url(url):
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def model_regex(model):
    if model == "F-150":
        return r"\bf[\s-]?150\b"
    if model == "Silverado 1500":
        return r"\bsilverado[\s-]*1500\b"
    if model == "RAV4":
        return r"\brav[\s-]?4\b"
    return r"\b" + re.escape(model).replace(r"\ ", r"[\s-]+") + r"\b"


def identity_matches(title, make, model):
    if make == "Ford" and model == "Explorer" and re.search(r"\bsport\s+trac\b", title, re.I):
        return False
    return bool(re.search(r"\b" + re.escape(make) + r"\b", title, re.I)
                and re.search(model_regex(model), title, re.I))


def jsonld_data(soup):
    items = []
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.get_text())
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            items.append(data)
        elif isinstance(data, list):
            items.extend(x for x in data if isinstance(x, dict))
    return items


def candidate_urls(html, make, model):
    soup = BeautifulSoup(html, "html.parser")
    candidates = []
    seen = set()
    for data in jsonld_data(soup):
        if data.get("@type") != "ItemList":
            continue
        for entry in data.get("itemListElement", []):
            item = entry.get("item", {})
            name, url = str(item.get("name", "")), item.get("url", "")
            if not url or not identity_matches(name, make, model):
                continue
            url = clean_url(urljoin("https://offerup.com", url))
            if urlsplit(url).path.startswith("/item/detail/") and url not in seen:
                candidates.append(url)
                seen.add(url)
    # Some search page variants omit JSON-LD
    if not candidates:
        for link in soup.select('a[href*="/item/detail/"]'):
            title = text_of(link) or str(link.get("aria-label", ""))
            if not identity_matches(title, make, model):
                continue
            url = clean_url(urljoin("https://offerup.com", link.get("href", "")))
            if url not in seen:
                candidates.append(url)
                seen.add(url)
    return candidates


def detail_attributes(soup):
    attrs = {}
    for term in soup.select("dt"):
        key = text_of(term).strip().lower().rstrip(":")
        sibling = term.find_next_sibling("dd")
        if key and sibling:
            attrs[key] = text_of(sibling)
    # VIN sometimes appears separately outside the definition list
    for el in soup.select(".ou-grid-item"):
        parts = el.find_all("p", recursive=False)
        if len(parts) == 2 and text_of(parts[0]).strip().lower() == "vin":
            attrs["vin"] = text_of(parts[1])
    return attrs


def vehicle_data(soup):
    script = soup.select_one("script#__NEXT_DATA__")
    if not script:
        return {}
    try:
        data = json.loads(script.get_text())
        state = data.get("props", {}).get("pageProps", {}).get("initialApolloState", {})
        root = state.get("ROOT_QUERY", {})
        for key, listing in root.items():
            if key.startswith("listing(") and isinstance(listing, dict):
                attributes = listing.get("vehicleAttributes", {})
                if isinstance(attributes, dict):
                    return attributes
    except (ValueError, TypeError, AttributeError):
        pass
    return {}


def parse_listing(html, url, make, model):
    soup = BeautifulSoup(html, "html.parser")
    product = next((x for x in jsonld_data(soup) if x.get("@type") == "Product"), {})
    attrs = detail_attributes(soup)
    vehicle = vehicle_data(soup)
    name = str(product.get("name", "") or text_of(soup.select_one("h1")))
    offers = product.get("offers", {})
    if not isinstance(offers, dict):
        offers = {}
    description = BeautifulSoup(str(product.get("description", "")), "html.parser").get_text(" ", strip=True)
    # Prefer explicit vehicle year; otherwise use the listing's own title
    year_text = vehicle.get("vehicleYear") or attrs.get("year", "")
    year_match = re.search(r"\b(?:19|20)\d{2}\b", str(year_text) if year_text else name)
    year = int(year_match.group()) if year_match else None
    price = numeric(offers.get("price"))
    if price is None:
        price_meta = soup.select_one('meta[property="og:price:amount"]')
        price = numeric(price_meta.get("content")) if price_meta else None
    mileage_text = str(vehicle.get("vehicleMiles") or next(
        (attrs[k] for k in ("mileage", "odometer", "miles") if attrs.get(k)), ""))
    mileage = numeric(mileage_text)
    vin = str(vehicle.get("vehicleVin") or attrs.get("vin", "")).strip().upper()
    if not vin:
        full_text = soup.get_text(" ", strip=True)
        hit = re.search(r"\bVIN\s*[:#]?\s*([A-HJ-NPR-Z0-9]{17})\b", full_text, re.I)
        vin = hit.group(1).upper() if hit else ""
    if vin and not re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", vin):
        vin = ""
    location = ""
    meta_title = soup.select_one('meta[property="og:title"]')
    page_title = meta_title.get("content", "") if meta_title else text_of(soup.title)
    match = re.search(r"for Sale in (.*?)\s*-\s*OfferUp", page_title, re.I)
    if match:
        location = match.group(1)
    recorded = {
        "source": "offerup", "listing_url": clean_url(url),
        "collected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "title": name, "make": make, "model": model, "model_year": year,
        "trim": vehicle.get("vehicleTrim") or attrs.get("trim", ""), "listing_price": price,
        "price_text": str(offers.get("price", "")),
        "mileage": mileage, "mileage_text": mileage_text,
        "condition": attrs.get("condition", ""),
        "fuel_type": vehicle.get("vehicleFuelType") or attrs.get("fuel type", attrs.get("fuel", "")),
        "transmission": vehicle.get("vehicleTransmissionClean") or attrs.get("transmission", ""),
        "drivetrain": vehicle.get("vehicleDriveTrain") or attrs.get("drive train", attrs.get("drivetrain", attrs.get("drive", ""))),
        "body_style": vehicle.get("vehicleBody") or attrs.get("body type", attrs.get("body style", "")),
        "exterior_color": vehicle.get("vehicleColor") or attrs.get("color", attrs.get("exterior color", "")),
        "interior_color": attrs.get("interior color", ""),
        "engine": attrs.get("engine", ""),
        "cylinders": vehicle.get("vehicleEngineCylinders") or attrs.get("engine cylinders", attrs.get("cylinders", "")),
        "title_status": vehicle.get("vehicleTitleStatus") or attrs.get("title status", ""),
        "seller_type": "", "location": location, "vin": vin,
        "description": description, "posted_at": "", "stock_number": "",
        "validation_issues": "",
    }
    reasons = []
    if not identity_matches(name, make, model):
        reasons.append("title does not match requested make/model")
    if year is None or not MIN_YEAR <= year <= MAX_YEAR:
        reasons.append(f"year missing or outside {MIN_YEAR}-{MAX_YEAR}")
    if price is None or price <= 0:
        reasons.append("price missing or invalid")
    if mileage is None or mileage < 0:
        reasons.append("mileage missing or invalid")
    if not description:
        reasons.append("description missing")
    recorded["validation_issues"] = "; ".join(reasons)
    return recorded


def get_html(driver, url, delay):
    time.sleep(delay)
    driver.get(url)
    try:
        WebDriverWait(driver, 15).until(
            lambda d: d.find_elements(By.CSS_SELECTOR, 'script[type="application/ld+json"], #__NEXT_DATA__')
        )
    except TimeoutException:
        pass
    return driver.page_source


def find_search_urls(driver, query, make, model, delay, scrolls):
    url = "https://offerup.com/search?" + urlencode({
        "q": query, "distance": "5000", "cid": "5.1"
    })
    print("\nSEARCH", query, flush=True)
    get_html(driver, url, delay)
    discovered = set()
    stagnant = 0
    for step in range(scrolls + 1):
        found = set(candidate_urls(driver.page_source, make, model))
        new = found - discovered
        discovered.update(found)
        print("  Scroll", step, "new URLs:", len(new), "total:", len(discovered), flush=True)
        if step == scrolls:
            break
        stagnant = stagnant + 1 if not new else 0
        if stagnant >= 3:
            break
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(delay)
    return sorted(discovered)


def save_reports(folder, rows, queries):
    raw = pd.DataFrame(rows, columns=COLUMNS).drop_duplicates(subset="listing_url")
    raw.to_csv(folder / "offerup_raw.csv", index=False)
    valid = raw.loc[raw["validation_issues"].fillna("").eq("")].copy()
    vin = valid["vin"].fillna("").str.upper()
    valid_vin = vin.str.fullmatch(r"[A-HJ-NPR-Z0-9]{17}")
    valid = valid.loc[~(valid_vin & vin.duplicated())].copy()
    valid.to_csv(folder / "offerup_usable_preview.csv", index=False)

    models = [(make, model) for make, values in VEHICLES.items() for model in values]
    summary = []
    for make, model in models:
        group = raw.loc[(raw["make"] == make) & (raw["model"] == model)]
        usable = valid.loc[(valid["make"] == make) & (valid["model"] == model)]
        summary.append({"make": make, "model": model,
                        "parsed": len(group), "usable": len(usable)})
    pd.DataFrame(summary).to_csv(folder / "model_counts.csv", index=False)
    pd.DataFrame(queries).to_csv(folder / "query_summary.csv", index=False)
    fields = [c for c in COLUMNS if c not in
              ("source", "listing_url", "collected_at", "validation_issues")]
    coverage = raw[fields].replace(r"^\s*$", pd.NA, regex=True).notna().sum()
    coverage.rename("nonmissing").to_csv(folder / "attribute_coverage.csv")

    print("\nFINAL RESULTS")
    print(pd.DataFrame(summary).to_string(index=False))
    print("Parsed:", len(raw), "Initially usable after VIN deduplication:", len(valid))
    print("Output folder:", folder.resolve())
    print("MSRP matching and additional price quality checks are still required")


def main():
    parser = argparse.ArgumentParser(description="Collect balanced OfferUp vehicle listings")
    parser.add_argument("--per-model", type=int, default=75, help="Initially usable vehicles per model")
    parser.add_argument("--max-details-per-model", type=int, default=350)
    parser.add_argument("--scrolls", type=int, default=3)
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()
    if args.per_model < 1 or args.max_details_per_model < 1 or args.scrolls < 0 or args.delay < 0:
        parser.error("Invalid numeric option")

    folder = (Path(__file__).resolve().parent / "offerup_results" /
              datetime.now().strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(parents=True, exist_ok=True)
    raw_file = folder / "offerup_raw.csv"
    pd.DataFrame(columns=COLUMNS).to_csv(raw_file, index=False)

    rows = []
    queries = []
    seen_urls = set()
    seen_vins = set()
    driver = webdriver.Chrome()
    driver.set_page_load_timeout(40)
    try:
        for make, models in VEHICLES.items():
            for model in models:
                usable = 0
                attempts = 0
                # General query first, then each model year
                search_terms = [f"{make} {model}"] + [
                    f"{year} {make} {model}" for year in range(MIN_YEAR, MAX_YEAR + 1)
                ]
                for query in search_terms:
                    if usable >= args.per_model or attempts >= args.max_details_per_model:
                        break
                    try:
                        links = find_search_urls(driver, query, make, model,
                                                 args.delay, args.scrolls)
                    except WebDriverException as error:
                        print("  Search failed:", str(error)[:160])
                        queries.append({"make": make, "model": model, "query": query,
                                        "candidates": 0, "new_urls": 0, "new_usable": 0})
                        continue
                    new_links = [url for url in links if url not in seen_urls]
                    added = 0
                    for url in new_links:
                        if usable >= args.per_model or attempts >= args.max_details_per_model:
                            break
                        seen_urls.add(url)
                        attempts += 1
                        try:
                            html = get_html(driver, url, args.delay)
                            row = parse_listing(html, url, make, model)
                        except (WebDriverException, ValueError) as error:
                            print("  Detail failed:", str(error)[:160])
                            continue
                        rows.append(row)
                        pd.DataFrame([row], columns=COLUMNS).to_csv(
                            raw_file, mode="a", header=False, index=False)
                        vin = str(row["vin"]).upper()
                        good_vin = bool(re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", vin))
                        if not row["validation_issues"] and not (good_vin and vin in seen_vins):
                            usable += 1
                            added += 1
                            if good_vin:
                                seen_vins.add(vin)
                        print(" ", row["model_year"], str(row["title"])[:35],
                              "|", "USABLE" if not row["validation_issues"] else row["validation_issues"],
                              flush=True)
                    queries.append({"make": make, "model": model, "query": query,
                                    "candidates": len(links), "new_urls": len(new_links),
                                    "new_usable": added})
                print("COMPLETED", make, model, "usable:", usable, "/", args.per_model,
                      "detail attempts:", attempts, flush=True)
                if usable < args.per_model:
                    print("WARNING: model did not reach its collection quota", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted; already collected records are preserved", flush=True)
    finally:
        driver.quit()
        save_reports(folder, rows, queries)


if __name__ == "__main__":
    main()