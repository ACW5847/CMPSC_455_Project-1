# Anthony Wert
# CMPSC 445 Project 1
# Match used vehicle listings to historical MSRP values
# Calculate retention ratios and save a dataset for machine learning

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parent
VEHICLES_FILE = PROJECT_FOLDER / "data" / "processed" / "vehicles_integrated_clean.csv"
MSRP_FILE = PROJECT_FOLDER / "data" / "msrp" / "msrp_reference.csv"
OUTPUT_FOLDER = PROJECT_FOLDER / "data" / "processed"

# Longer trim names come first to avoid matching LE inside XLE
TRIM_NAMES = [
    "High Country", "King Ranch", "Custom Trail Boss", "LT Trail Boss",
    "Limited", "Platinum", "Lariat", "Trail Boss", "Work Truck",
    "TRD Off Road", "TRD Sport", "Nightshade", "Adventure", "Woodland",
    "Premium", "Touring", "Sport", "Hybrid", "Custom", "Reserve",
    "XLE", "XSE", "XLT", "LTZ", "RST", "STX", "FX4", "FX2",
    "ZR2", "Z71", "SR5", "TRD", "SEL", "SE", "LE", "LS", "LT",
    "XL", "L", "S", "CE", "SR", "XLS", "Base", "Limited Edition",
]


def normalize_text(value):
    if pd.isna(value):
        return ""
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def has_phrase(text, phrase):
    return bool(re.search(r"(?:^|\s)" + re.escape(normalize_text(phrase)) + r"(?:\s|$)", text))


def get_trim(listing):
    # Use the listed trim or vehicle title before searching the description
    for field in (listing.get("trim", ""), listing.get("title", "")):
        text = normalize_text(field)
        for trim in sorted(TRIM_NAMES, key=lambda x: -len(normalize_text(x))):
            if has_phrase(text, trim):
                return trim

    # Short labels in descriptions can be ordinary words rather than trims
    description = normalize_text(listing.get("description", ""))
    for trim in sorted(TRIM_NAMES, key=lambda x: -len(normalize_text(x))):
        if len(normalize_text(trim)) >= 3 and has_phrase(description, trim):
            return trim
    return ""


def check_drivetrain(listing):
    drivetrain = normalize_text(listing.get("drivetrain", ""))
    if any(has_phrase(drivetrain, x) for x in ("4wd", "4x4", "awd", "four wheel", "all wheel")):
        return "4wd"
    if any(has_phrase(drivetrain, x) for x in ("2wd", "4x2", "fwd", "rwd", "front wheel", "rear wheel")):
        return "2wd"
    return ""


def reference_drivetrain(configuration):
    text = normalize_text(configuration)
    if any(has_phrase(text, x) for x in ("4wd", "awd", "4x4")):
        return "4wd"
    if any(has_phrase(text, x) for x in ("2wd", "fwd", "rwd", "4x2")):
        return "2wd"
    return ""


def select_reference(listing, references):
    # Only use rows from the same manufacturer, model and model year
    if references.empty:
        return None, "unmatched", "", "", 0

    trim = get_trim(listing)
    drive = check_drivetrain(listing)
    candidates = references.copy()
    method = "model_year_base"

    if trim:
        names = candidates["configuration"].map(normalize_text)
        matching = candidates.loc[names.map(lambda x: has_phrase(x, trim))].copy()
        if not matching.empty:
            candidates = matching
            method = "trim_approx"

    # Use drivetrain only when it is explicitly available in both sources
    if drive and method == "trim_approx":
        matching = candidates.loc[candidates["configuration"].map(reference_drivetrain).eq(drive)]
        if not matching.empty:
            candidates = matching.copy()
            method = "trim_drivetrain_approx"

    # Several configurations can share a trim with different original prices
    # Use the lowest verified price and label it as an approximation
    chosen = candidates.sort_values(["original_msrp", "configuration"]).iloc[0]
    return (float(chosen["original_msrp"]), method, str(chosen["configuration"]),
            str(chosen["source_url"]), len(candidates))


def load_reference(path, output_folder):
    msrp = pd.read_csv(path, low_memory=False)
    needed = {"make", "model", "model_year", "configuration", "original_msrp", "source_url"}
    if not needed.issubset(msrp.columns):
        raise ValueError("MSRP reference CSV is missing required columns")

    msrp["model_year"] = pd.to_numeric(msrp["model_year"], errors="coerce")
    msrp["original_msrp"] = pd.to_numeric(msrp["original_msrp"], errors="coerce")
    msrp = msrp.dropna(subset=["make", "model", "model_year", "configuration", "original_msrp"])
    msrp = msrp[msrp["original_msrp"].between(1000, 250000)].copy()
    msrp["make"] = msrp["make"].astype(str).str.strip()
    msrp["model"] = msrp["model"].astype(str).str.strip()
    msrp["configuration"] = msrp["configuration"].astype(str).str.strip()
    msrp["model_year"] = msrp["model_year"].astype(int)

    keys = ["make", "model", "model_year", "configuration"]
    groups = msrp.groupby(keys)["original_msrp"].nunique()
    conflicts = groups[groups > 1].reset_index()[keys]
    conflicts = msrp.merge(conflicts, on=keys, how="inner")
    conflicts.to_csv(output_folder / "msrp_conflicts.csv", index=False)

    # Do not use a configuration if its reference rows disagree on the MSRP
    if not conflicts.empty:
        bad = conflicts[keys].drop_duplicates().assign(conflict=True)
        msrp = msrp.merge(bad, on=keys, how="left")
        msrp = msrp[msrp["conflict"].isna()].drop(columns="conflict")

    msrp = msrp.drop_duplicates(subset=keys, keep="first")
    lookup = {key: group.copy() for key, group in msrp.groupby(["make", "model", "model_year"])}
    return lookup, len(conflicts), len(msrp)


def training_issue(row):
    reasons = []
    price = row.get("listing_price")
    ratio = row.get("value_retention_ratio")
    if pd.isna(price) or price < 1000:
        reasons.append("price below 1000 or missing")
    if "financing advertisement" in str(row.get("quality_flags", "")).lower():
        reasons.append("possible down payment advertisement")
    if "damaged or special title" in str(row.get("quality_flags", "")).lower():
        reasons.append("possible damaged vehicle or special title")
    if "unusually low mileage" in str(row.get("quality_flags", "")).lower():
        reasons.append("unusually low mileage")
    if "mileage above 500000" in str(row.get("quality_flags", "")).lower():
        reasons.append("extremely high mileage")
    if pd.isna(ratio) or ratio <= 0:
        reasons.append("missing or invalid MSRP ratio")
    elif ratio > 1.5:
        reasons.append("retention ratio above 1.5 needs review")
    return "; ".join(reasons)


def main():
    parser = argparse.ArgumentParser(description="Match vehicle listings to historical MSRP")
    parser.add_argument("--vehicles", type=Path, default=VEHICLES_FILE)
    parser.add_argument("--msrp", type=Path, default=MSRP_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FOLDER)
    args = parser.parse_args()

    # Output files are kept separate from the original scraped data
    output_folder = args.output
    output_folder.mkdir(parents=True, exist_ok=True)

    vehicles = pd.read_csv(args.vehicles, low_memory=False)
    required = {"make", "model", "model_year", "listing_price", "mileage", "title", "trim"}
    if not required.issubset(vehicles.columns):
        raise ValueError("Clean vehicle CSV is missing required columns")

    vehicles["model_year"] = pd.to_numeric(vehicles["model_year"], errors="coerce")
    vehicles["listing_price"] = pd.to_numeric(vehicles["listing_price"], errors="coerce")
    vehicles["mileage"] = pd.to_numeric(vehicles["mileage"], errors="coerce")
    vehicles["make"] = vehicles["make"].astype(str).str.strip()
    vehicles["model"] = vehicles["model"].astype(str).str.strip()

    lookup, conflicting_rows, usable_references = load_reference(args.msrp, output_folder)
    results = []
    for _, listing in vehicles.iterrows():
        year = listing["model_year"]
        key = (listing["make"], listing["model"], int(year)) if pd.notna(year) else None
        candidates = lookup.get(key, pd.DataFrame())
        price, method, config, url, options = select_reference(listing, candidates)
        results.append({"original_msrp": price, "msrp_match_method": method,
                        "matched_configuration": config, "msrp_source_url": url,
                        "msrp_matching_options": options, "detected_trim": get_trim(listing)})

    matched = pd.concat([vehicles.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    matched["value_retention_ratio"] = matched["listing_price"] / matched["original_msrp"]
    matched["training_exclusion_reason"] = matched.apply(training_issue, axis=1)
    ready = matched[matched["training_exclusion_reason"].eq("")].copy()
    excluded = matched[matched["training_exclusion_reason"].ne("")].copy()

    matched.to_csv(output_folder / "vehicles_with_msrp.csv", index=False)
    ready.to_csv(output_folder / "vehicles_model_ready.csv", index=False)
    excluded.to_csv(output_folder / "vehicles_excluded_from_training.csv", index=False)

    matched.groupby("msrp_match_method", dropna=False).size().rename("listings").to_csv(
        output_folder / "msrp_match_summary.csv", header=True)
    ready.groupby(["source", "make", "model"], dropna=False).size().rename("listings").to_csv(
        output_folder / "model_ready_counts.csv", header=True)

    print("\nMSRP MATCHING SUMMARY")
    print("Input listings:", len(vehicles))
    print("Reference configurations without conflicts:", usable_references)
    print("Reference rows involving conflicting configurations:", conflicting_rows)
    print("Listings with MSRP:", matched["original_msrp"].notna().sum())
    print("Matching methods:\n", matched["msrp_match_method"].value_counts().to_string())
    print("Model ready listings:", len(ready))
    print("Excluded for review:", len(excluded))
    print("Saved results to:", output_folder.resolve())
    if len(ready) < 1000:
        print("WARNING: Fewer than 1000 model ready records. Review excluded listings before modeling.")
    print("MSRP fallbacks are approximate and should be documented in the report")


if __name__ == "__main__":
    main()
