# Anthony Wert
# CMPSC 445 Project 1
# Explore vehicle data statistics and visualize data characteristics

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_FOLDER = Path(__file__).resolve().parent / "data" / "processed"
RESULT_FOLDER = Path(__file__).resolve().parent / "results" / "exploration"


def main():
    # Load the integrated dataset with original scraped attributes
    integrated = pd.read_csv(DATA_FOLDER / "vehicles_integrated_clean.csv")
    ready = pd.read_csv(DATA_FOLDER / "vehicles_model_ready.csv")
    RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Convert measurements to numeric values
    for column in ["listing_price", "mileage", "model_year"]:
        integrated[column] = pd.to_numeric(integrated[column], errors="coerce")
    ready["value_retention_ratio"] = pd.to_numeric(
        ready["value_retention_ratio"], errors="coerce"
    )

    # Summarize the numerical measurements across the sources
    measures = ["listing_price", "mileage", "model_year"]
    overall = integrated[measures].describe().round(2)
    by_source = integrated.groupby("source")[measures].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).round(2)
    by_make = integrated.groupby("make")[measures].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).round(2)
    retention = ready.groupby("source")["value_retention_ratio"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).round(3)

    overall.to_csv(RESULT_FOLDER / "attribute_statistics_overall.csv")
    by_source.to_csv(RESULT_FOLDER / "attribute_statistics_by_source.csv")
    by_make.to_csv(RESULT_FOLDER / "attribute_statistics_by_make.csv")
    retention.to_csv(RESULT_FOLDER / "retention_statistics_by_source.csv")

    print("\nOVERALL ATTRIBUTE STATISTICS")
    print(overall.to_string())
    print("\nSTATISTICS BY SOURCE")
    print(by_source.to_string())
    print("\nRETENTION BY SOURCE")
    print(retention.to_string())

    # Show the asking price distribution using the model-ready dataset
    ready["listing_price"] = pd.to_numeric(ready["listing_price"], errors="coerce")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ready["listing_price"].dropna(), bins=35, edgecolor="black")
    ax.set(title="Used Vehicle Listing Price Distribution",
           xlabel="Listing price ($)", ylabel="Number of listings")
    fig.tight_layout()
    fig.savefig(RESULT_FOLDER / "price_distribution.png", dpi=160)
    plt.close(fig)

    # Show the distribution of vehicle mileage
    ready["mileage"] = pd.to_numeric(ready["mileage"], errors="coerce")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ready["mileage"].dropna(), bins=35, edgecolor="black")
    ax.set(title="Used Vehicle Mileage Distribution",
           xlabel="Mileage (miles)", ylabel="Number of listings")
    fig.tight_layout()
    fig.savefig(RESULT_FOLDER / "mileage_distribution.png", dpi=160)
    plt.close(fig)

    # Compare model frequencies between the two listing sources
    counts = integrated.groupby(["model", "source"]).size().unstack(fill_value=0)
    counts.to_csv(RESULT_FOLDER / "model_frequency.csv")
    ax = counts.plot(kind="bar", figsize=(10, 5), rot=35)
    ax.set(title="Vehicle Model Counts by Data Source",
           xlabel="Vehicle model", ylabel="Number of listings")
    ax.legend(title="Source")
    ax.figure.tight_layout()
    ax.figure.savefig(RESULT_FOLDER / "model_frequency.png", dpi=160)
    plt.close(ax.figure)

    print("\nSaved statistics and graphs to:", RESULT_FOLDER)


if __name__ == "__main__":
    main()
