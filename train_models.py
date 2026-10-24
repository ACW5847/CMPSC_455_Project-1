# Anthony Wert
# CMPSC 445 Project 1
# Train regression models to predict used car value retention
# Compare model accuracy and analyze important vehicle features

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CURRENT_YEAR = 2026
NUMERIC_FEATURES = ["vehicle_age", "mileage", "one_owner", "accident_mentioned", "clean_title_mentioned"]
CATEGORICAL_FEATURES = ["make", "model", "source", "trim", "condition", "fuel_type", "transmission", "drivetrain", "body_style", "exterior_color", "title_status"]


# create features from the listing descriptions
def description_features(description):
    # Extract simple features using the sellers description
    text = str(description).lower() if pd.notna(description) else ""
    one_owner = bool(re.search(r"\b(?:one|1)[ -]?owner\b|\bsingle owner\b", text))
    accident = bool(re.search(r"\baccident\b|\bcollision\b|\bcrash\b|\bwreck\b", text))
    clean_title = bool(re.search(r"\bclean title\b|\bclear title\b", text))
    return int(one_owner), int(accident), int(clean_title)


# prepare the test input
def prepare_features(data):
    # Calculate age and create description based features
    features = data.copy()
    features["model_year"] = pd.to_numeric(features["model_year"], errors="coerce")
    features["mileage"] = pd.to_numeric(features["mileage"], errors="coerce")
    features["vehicle_age"] = (CURRENT_YEAR - features["model_year"]).clip(lower=0)
    extracted = features["description"].apply(description_features).tolist()
    features[["one_owner", "accident_mentioned", "clean_title_mentioned"]] = pd.DataFrame(
        extracted, index=features.index
    )

    # Standardize text categories and keep unknown values as missing
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype("string").str.strip().str.lower()
        features[column] = features[column].replace({"": np.nan, "nan": np.nan, "none": np.nan})
        features[column] = features[column].astype(object).where(features[column].notna(), np.nan)
    return features


# set up the preprocessing
def make_preprocessor():
    # Fit all imputers and encoders using training data only
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        # Group rare trim categories instead of creating hundreds of columns
        ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


# save graphs and feature rankings
def save_charts(results, predictions, forest, test_x, test_y, dataset, output):
    # Compare accuracy for the three regression methods
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(results["model"], results["rmse"])
    ax.set_title("Regression Model Comparison")
    ax.set_ylabel("Test RMSE (retention ratio)")
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=160)
    plt.close(fig)

    # Show actual and predicted retention values for the best model
    best_name = results.loc[results["rmse"].idxmin(), "model"]
    best = predictions[predictions["model"] == best_name]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(best["actual"], best["predicted"], alpha=0.45, s=14)
    bound = max(1.0, float(best[["actual", "predicted"]].max().max()))
    ax.plot([0, bound], [0, bound], linestyle="--")
    ax.set(xlabel="Actual retention ratio", ylabel="Predicted retention ratio",
           title="Actual vs Predicted: " + best_name)
    fig.tight_layout()
    fig.savefig(output / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)


    # Measure the importance of each original input feature
    importance = permutation_importance(
        forest,
        test_x,
        test_y,
        n_repeats=5,
        random_state=42,
        scoring="r2",
        n_jobs=-1
    )

    # Rank the features by their effect on prediction accuracy
    ranked = pd.DataFrame({
        "feature": test_x.columns,
        "importance": importance.importances_mean,
        "std": importance.importances_std
    })

    ranked = ranked.sort_values(
        "importance", ascending=False
    )

    ranked.to_csv(
        output / "feature_importance.csv",
        index=False
    )

    # Plot the ten most important original features
    top = ranked.head(10).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"])

    ax.set(
        xlabel="Mean decrease in R2 when shuffled",
        title="Random Forest Feature Importance"
    )

    fig.tight_layout()
    fig.savefig(output / "feature_importance.png", dpi=160)
    plt.close(fig)


    # Compare observed retention by vehicle age
    age_table = dataset.groupby("vehicle_age")["value_retention_ratio"].agg(["median", "count"]).reset_index()
    age_table.to_csv(output / "retention_by_age.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(age_table["vehicle_age"], age_table["median"], marker="o")
    ax.set(xlabel="Vehicle age (years)", ylabel="Median retention ratio",
           title="Observed Value Retention by Vehicle Age")
    fig.tight_layout()
    fig.savefig(output / "retention_by_age.png", dpi=160)
    plt.close(fig)


# Load data and train the regression models
def main():
    parser = argparse.ArgumentParser(description="Train regression models for vehicle value retention")
    parser.add_argument("--input", type=Path, default=Path("data/processed/vehicles_model_ready.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/ml"))
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error("Input CSV was not found: " + str(args.input))

    args.output.mkdir(parents=True, exist_ok=True)
    # Load the cleaned dataset with matched original MSRP values
    data = pd.read_csv(args.input)
    data["value_retention_ratio"] = pd.to_numeric(data["value_retention_ratio"], errors="coerce")
    data = data.dropna(subset=["value_retention_ratio", "model_year", "mileage"]).copy()
    data = data[data["value_retention_ratio"].between(0.01, 1.5)].reset_index(drop=True)
    data = prepare_features(data)
    print("\nVALIDATED DATASET")
    print("Records:", len(data))

    # Use an 80 percent training and 20 percent testing approach
    # Listing price and MSRP determine the target so they cannot be inputs
    x = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data["value_retention_ratio"]
    stratify = data["model"] if data["model"].value_counts().min() >= 2 else None
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=stratify
    )

    # Compare three regression algorithms using the same train and test split
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1, min_samples_leaf=2),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    scores = []
    predictions = []
    fitted = {}
    for name, estimator in models.items():
        # Fit preprocessing on training data only for each model
        # This prevents the test data from influencing the training process
        pipeline = Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])
        pipeline.fit(train_x, train_y)
        predicted = pipeline.predict(test_x)
        mae = mean_absolute_error(test_y, predicted)
        rmse = np.sqrt(mean_squared_error(test_y, predicted))
        r2 = r2_score(test_y, predicted)
        scores.append({"model": name, "mae": mae, "rmse": rmse, "r2": r2})
        predictions.append(pd.DataFrame({
            "model": name, "actual": test_y.to_numpy(), "predicted": predicted,
        }))
        fitted[name] = pipeline
        print(f"{name}: MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

    # Save the model scores and test predictions for the report
    results = pd.DataFrame(scores).sort_values("rmse").reset_index(drop=True)
    prediction_table = pd.concat(predictions, ignore_index=True)
    results.to_csv(args.output / "model_comparison.csv", index=False)
    prediction_table.to_csv(args.output / "test_predictions.csv", index=False)
    save_charts(results, prediction_table, fitted["Random Forest"], test_x, test_y, data, args.output)
    print("\nTraining rows:", len(train_x), "Test rows:", len(test_x))
    print("Best model:", results.iloc[0]["model"])
    print("Results saved to:", args.output.resolve())


if __name__ == "__main__":
    main()
