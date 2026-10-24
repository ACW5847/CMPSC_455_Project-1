# CMPSC 445 Project 1
## Predicting Used Car Value Retention Ratio Using Multi-Source Data and Machine Learning

This project uses vehicle listings collected from Craigslist and OfferUp to predict used vehicle value retention and identify important factors affecting depreciation.

The value retention ratio is calculated as:

**Value Retention Ratio = Current Listing Price / Original MSRP**

### Dataset

The project examines three vehicle makes and nine models:

- Ford: F-150, Explorer, Escape
- Chevrolet: Silverado 1500, Equinox, Malibu
- Toyota: Camry, Corolla, RAV4

A total of 1,350 initially usable vehicle listings were collected from Craigslist and OfferUp. After integration and cleaning, 1,205 records were retained for machine learning.

Historical MSRP information was collected from The Car Connection and matched to each listing.

### Project Files

- `collect_craigslist.py`: Collects vehicle listings from Craigslist
- `collect_offerup.py`: Collects vehicle listings from OfferUp
- `preprocess_data.py`: Cleans and integrates the datasets
- `build_msrp_reference.py`: Collects historical MSRP information
- `match_msrp.py`: Matches MSRP values and calculates retention ratios
- `explore_data.py`: Produces descriptive statistics and data visualizations
- `review_duplicates.py`: Identifies potentially duplicate advertisements
- `train_models.py`: Trains and evaluates regression models

### Machine Learning

Three regression algorithms were compared:

1. Linear Regression
2. Random Forest Regression
3. Gradient Boosting Regression

The dataset was divided into 80% training data and 20% testing data.

Models were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R².

### Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.0918 | 0.1262 | 0.7441 |
| Random Forest | 0.0827 | 0.1198 | 0.7697 |
| Gradient Boosting | 0.0815 | 0.1172 | 0.7794 |

Gradient Boosting achieved the best results on the held-out test dataset.

Vehicle age and mileage were among the strongest predictors of value retention.

### Running the Project

Install the required packages:

```bash
pip install -r requirements.txt
```

The scripts are organized in the following order:

1. `collect_craigslist.py` and `collect_offerup.py`
2. `preprocess_data.py`
3. `build_msrp_reference.py`
4. `match_msrp.py`
5. `explore_data.py` and `review_duplicates.py`
6. `train_models.py`

The included CSV files allow the preprocessing and machine learning stages to be reproduced without repeating the original scraping.

### Limitations

Some historical MSRP values use model-year or trim-level approximations when exact vehicle configurations are unavailable. Listing prices are advertised asking prices rather than confirmed transaction prices. Some advertisements may contain incomplete or inaccurate vehicle information.

### Course Information

CMPSC 445 - Project 1

Anthony Wert