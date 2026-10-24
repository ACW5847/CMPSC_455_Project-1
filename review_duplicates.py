# Anthony Wert
# CMPSC 445 Project 1
# Review possible duplicate vehicles before removing any records

from pathlib import Path

import pandas as pd


DATA_FILE = Path(__file__).resolve().parent / "data" / "processed" / "vehicles_model_ready.csv"
OUTPUT_FILE = Path(__file__).resolve().parent / "results" / "exploration" / "possible_duplicates.csv"
KEY_COLUMNS = ["make", "model", "model_year", "listing_price", "mileage"]


def main():
    # Load the final modeling dataset
    data = pd.read_csv(DATA_FILE)
    for column in ["model_year", "listing_price", "mileage"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Identify records with identical basic vehicle measurements
    duplicates = data[data.duplicated(KEY_COLUMNS, keep=False)].copy()
    duplicates = duplicates.sort_values(KEY_COLUMNS)
    duplicates.insert(0, "review_group", duplicates.groupby(KEY_COLUMNS, dropna=False).ngroup() + 1)

    # Preserve identifying details to help decide whether two ads show one car
    review_columns = [
        "review_group", "source", "make", "model", "model_year", "listing_price",
        "mileage", "vin", "location", "title", "trim", "listing_url", "description"
    ]
    duplicates = duplicates[[col for col in review_columns if col in duplicates.columns]]
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    duplicates.to_csv(OUTPUT_FILE, index=False)

    groups = duplicates["review_group"].nunique() if not duplicates.empty else 0
    print("\nPOSSIBLE DUPLICATE REVIEW")
    print("Groups with matching make/model/year/price/mileage:", groups)
    print("Records requiring review:", len(duplicates))
    print("Potential extra records:", len(duplicates) - groups)
    print("No rows were automatically deleted")
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()