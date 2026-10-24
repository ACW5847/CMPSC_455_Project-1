# Anthony Wert
# CMPSC 445 Project 1
# Clean and combine the Craigslist and OfferUp vehicle datasets
# Save the integrated dataset and a report of removed or flagged listings

import argparse
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

VEHICLES = {
    "Ford": ["F-150", "Explorer", "Escape"],
    "Chevrolet": ["Silverado 1500", "Equinox", "Malibu"],
    "Toyota": ["Camry", "Corolla", "RAV4"],
}

MIN_YEAR = 2005
MAX_YEAR = 2026
CURRENT_YEAR = 2026

# These attributes were collected directly from the listing pages
SOURCE_COLUMNS = [
    "source", "listing_url", "collected_at", "title", "make", "model",
    "model_year", "trim", "listing_price", "price_text", "mileage",
    "mileage_text", "condition", "fuel_type", "transmission", "drivetrain",
    "body_style", "exterior_color", "interior_color", "engine", "cylinders",
    "title_status", "seller_type", "location", "vin", "description",
    "posted_at", "stock_number", "validation_issues",
]

# Different vehicle nameplates must not share the MSRP of the parent model
EXCLUDED_VARIANTS = {
    ("Ford", "Explorer"): [r"\bsport\s*trac\b"],
    ("Chevrolet", "Equinox"): [r"\bequinox\s+ev\b"],
    ("Toyota", "Camry"): [r"\bsolara\b"],
    ("Toyota", "Corolla"): [r"\bcorolla\s+cross\b"],
}


def locate_csv(parent, source):
    # Prefer the latest completed run with the most usable records
    matches = list(parent.glob(f"{source}_results/*/{source}_usable_preview.csv"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {source}_usable_preview.csv in {source}_results"
        )
    matches.sort(key=lambda path: (path.parent.name, path.stat().st_mtime), reverse=True)
    return matches[0]


def read_source(path, source):
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"listing_url", "make", "model", "model_year", "listing_price", "mileage"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    for column in SOURCE_COLUMNS:
        if column not in data.columns:
            data[column] = ""
    data = data[SOURCE_COLUMNS].copy()
    data["source"] = source
    return data


def parse_number(value):
    text = str(value).replace(",", "").replace("$", "").strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def normalize_url(value):
    text = str(value).strip()
    if not text:
        return ""
    parts = urlsplit(text)
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_fuel(value):
    # Standardize fuel types from both sources
    text = normalize_text(value).lower().replace("_", " ")

    if not text:
        return ""
    if "hybrid" in text:
        return "hybrid"
    if "flex" in text or text.startswith("ffv"):
        return "flex fuel"
    if "diesel" in text:
        return "diesel"
    if "electric" in text or text == "ev":
        return "electric"
    if any(word in text for word in ("gas", "unleaded", "premium", "regular")):
        return "gasoline"

    return text


def normalize_drivetrain(value):
    # Standardize drivetrain descriptions
    text = normalize_text(value).lower()

    if not text:
        return ""
    if "awd" in text or "all wheel" in text or "all-wheel" in text:
        return "awd"
    if any(word in text for word in ("4wd", "4x4", "four-wheel", "4-wheel")):
        return "4wd"
    if "fwd" in text or "front wheel" in text or "front-wheel" in text:
        return "fwd"
    if "rwd" in text or "rear wheel" in text or "rear-wheel" in text:
        return "rwd"
    if "2wd" in text or "4x2" in text:
        return "2wd"

    return text


def normalize_condition(value):
    # Combine similar vehicle condition labels
    text = normalize_text(value).lower()

    mapping = {
        "new": "new",
        "like new": "like new",
        "excellent": "excellent",
        "very good": "excellent",
        "good": "good",
        "fair": "fair",
        "for parts": "for parts"
    }

    return mapping.get(text, text)


def normalize_body_style(value):
    # Standardize vehicle body styles
    text = normalize_text(value).lower()

    if not text:
        return ""
    if "suv" in text or "sport utility" in text:
        return "suv"
    if "sedan" in text:
        return "sedan"
    if "hatchback" in text:
        return "hatchback"
    if any(word in text for word in (
        "pickup", "truck", "crew cab", "double cab",
        "extended cab", "regular cab", "supercrew", "supercab"
    )):
        return "pickup"
    if text in ("coupe", "wagon", "van", "convertible"):
        return text
    if text == "other":
        return "other"

    return ""


def model_variant_issue(row):
    key = (row["make"], row["model"])
    text = row["title"] + " " + row["trim"]
    for pattern in EXCLUDED_VARIANTS.get(key, []):
        if re.search(pattern, text, re.IGNORECASE):
            return "different vehicle model variant"
    return ""


def main():
    parser = argparse.ArgumentParser(description="Clean and integrate two used car datasets")
    parser.add_argument("--offerup", type=Path, help="Path to OfferUp usable CSV")
    parser.add_argument("--craigslist", type=Path, help="Path to Craigslist usable CSV")
    parser.add_argument("--output", type=Path, default=Path("data") / "processed")
    args = parser.parse_args()

    project_folder = Path(__file__).resolve().parent
    offerup_path = args.offerup or locate_csv(project_folder, "offerup")
    craigslist_path = args.craigslist or locate_csv(project_folder, "craigslist")
    output = args.output
    if not output.is_absolute():
        output = project_folder / output
    output.mkdir(parents=True, exist_ok=True)

    print("OfferUp input:", offerup_path)
    print("Craigslist input:", craigslist_path)
    offerup = read_source(offerup_path, "offerup")
    craigslist = read_source(craigslist_path, "craigslist")
    data = pd.concat([offerup, craigslist], ignore_index=True)
    print("OfferUp records:", len(offerup))
    print("Craigslist records:", len(craigslist))
    print("Combined records:", len(data))

    # Preserve the original scraped text while cleaning whitespace
    text_columns = [col for col in SOURCE_COLUMNS if col not in (
        "listing_price", "mileage", "model_year")]
    for column in text_columns:
        data[column] = data[column].map(normalize_text)
    data["listing_url"] = data["listing_url"].map(normalize_url)
    data["vin"] = data["vin"].str.upper().str.strip()
    data["model_year"] = data["model_year"].map(parse_number)
    data["listing_price"] = data["listing_price"].map(parse_number)
    data["mileage"] = data["mileage"].map(parse_number)

    # Standardize categorical values from both sources
    data["fuel_type"] = data["fuel_type"].map(normalize_fuel)
    data["drivetrain"] = data["drivetrain"].map(normalize_drivetrain)
    data["condition"] = data["condition"].map(normalize_condition)
    data["body_style"] = data["body_style"].map(normalize_body_style)

    # Standardize transmission values
    data["transmission"] = data["transmission"].fillna("").str.strip().str.lower()


    # Accept minor spelling and capitalization differences in the make and model
    recognized = {(make.lower(), model.lower()): (make, model)
                  for make, models in VEHICLES.items() for model in models}
    for index, row in data.iterrows():
        pair = (row["make"].lower(), row["model"].lower())
        if pair in recognized:
            data.at[index, "make"], data.at[index, "model"] = recognized[pair]

    # Record the reason for rejecting a listing instead of silently dropping it
    rejected_reasons = []
    flags = []
    for _, row in data.iterrows():
        reasons = []
        warnings = []
        if (row["make"], row["model"]) not in [(m, n) for m, ns in VEHICLES.items() for n in ns]:
            reasons.append("unknown make or model")
        if not row["listing_url"]:
            reasons.append("missing listing URL")
        if pd.isna(row["model_year"]) or not MIN_YEAR <= row["model_year"] <= MAX_YEAR:
            reasons.append("invalid model year")
        if pd.isna(row["listing_price"]) or row["listing_price"] <= 0:
            reasons.append("missing or invalid listing price")
        if pd.isna(row["mileage"]) or row["mileage"] < 0:
            reasons.append("missing or invalid mileage")
        if not row["description"]:
            reasons.append("missing description")
        if row["validation_issues"]:
            reasons.append("source validation issue")
        variant_issue = model_variant_issue(row)
        if variant_issue:
            reasons.append(variant_issue)

        # Keep questionable prices and odometers visible for later inspection
        if pd.notna(row["listing_price"]) and row["listing_price"] < 1000:
            warnings.append("price below 1000")
        if pd.notna(row["listing_price"]) and row["listing_price"] > 150000:
            warnings.append("price above 150000")
        if pd.notna(row["mileage"]) and row["mileage"] > 500000:
            warnings.append("mileage above 500000")
        if pd.notna(row["mileage"]) and row["mileage"] < 1000 and pd.notna(row["model_year"]):
            if row["model_year"] <= CURRENT_YEAR - 3:
                warnings.append("unusually low mileage for vehicle age")
        if re.search(r"\b(down payment|weekly payment|monthly payment|per month|\$/mo)\b",
                     row["title"] + " " + row["description"], re.IGNORECASE):
            warnings.append("possible financing advertisement")
        if re.search(r"\b(salvage|rebuilt|flood|parts only|not running)\b",
                     row["title"] + " " + row["title_status"] + " " + row["description"], re.IGNORECASE):
            warnings.append("possible damaged or special title")

        rejected_reasons.append("; ".join(reasons))
        flags.append("; ".join(warnings))

    data["removal_reason"] = rejected_reasons
    data["quality_flags"] = flags
    rejected = data[data["removal_reason"] != ""].copy()
    cleaned = data[data["removal_reason"] == ""].copy()

    # Remove repeated listing URLs and then repeated valid VINs across both sources
    duplicate_urls = cleaned.duplicated("listing_url", keep="first")
    duplicate_rows = cleaned[duplicate_urls].copy()
    duplicate_rows["removal_reason"] = "duplicate listing URL"
    cleaned = cleaned[~duplicate_urls].copy()
    valid_vins = cleaned["vin"].str.fullmatch(r"[A-HJ-NPR-Z0-9]{17}")
    repeated_vins = valid_vins & cleaned.loc[valid_vins, "vin"].reindex(cleaned.index).duplicated()
    vin_duplicates = cleaned[repeated_vins].copy()
    vin_duplicates["removal_reason"] = "duplicate VIN"
    cleaned = cleaned[~repeated_vins].copy()
    rejected = pd.concat([rejected, duplicate_rows, vin_duplicates], ignore_index=True)

    # Retain genuine missing values so that later ML pipelines can impute or encode them
    numeric_columns = ["model_year", "listing_price", "mileage"]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    for column in SOURCE_COLUMNS:
        if column not in numeric_columns:
            cleaned[column] = cleaned[column].replace("", pd.NA)

    # This file does not invent MSRP values or create the regression target
    cleaned = cleaned.drop(columns=["removal_reason"])
    cleaned.to_csv(output / "vehicles_integrated_clean.csv", index=False)
    rejected.to_csv(output / "removed_listings.csv", index=False)
    cleaned[cleaned["quality_flags"] != ""].to_csv(
        output / "flagged_listings.csv", index=False)

    counts = cleaned.groupby(["source", "make", "model"], dropna=False).size()
    counts.rename("count").to_csv(output / "model_counts.csv")
    year_counts = cleaned.groupby(["source", "model_year"]).size()
    year_counts.rename("count").to_csv(output / "year_counts.csv")
    attribute_columns = [col for col in SOURCE_COLUMNS if col not in (
        "source", "listing_url", "collected_at", "validation_issues")]
    coverage = cleaned[attribute_columns].notna().sum().rename("nonmissing")
    coverage.to_csv(output / "attribute_coverage.csv", header=True)

    print("\nFINAL SUMMARY")
    print("Removed rows:", len(rejected))
    print("Integrated clean rows:", len(cleaned))
    print("Flagged for review:", int((cleaned["quality_flags"] != "").sum()))
    print("\nBy source:")
    print(cleaned["source"].value_counts().to_string())
    print("\nBy model:")
    print(counts.to_string())
    print("\nSaved to:", output.resolve())
    if len(cleaned) < 1000:
        print("WARNING: Fewer than 1000 records remain before MSRP matching")
    else:
        print("At least 1000 records remain before MSRP matching")


if __name__ == "__main__":
    main()
