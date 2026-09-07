# Food Delivery ETA Prediction & Delay Analysis

An end-to-end machine learning project for predicting food-delivery ETA and analyzing long-delivery risk using operational, temporal, geographic, and delivery-context features.

## Project Highlights

* End-to-end ETA prediction pipeline
* Geographic data quality auditing and cleaning
* Temporal validation to reduce evaluation leakage
* Tail-aware modeling for long-delivery cases
* Permutation feature importance
* Operational scenario testing
* Long-delivery risk analysis
* Model error analysis across important delivery segments
* Reproducible prediction scripts

## Final Model

The final trained model is:

`models/eta_model_final_tail_aware.joblib`

## Example Prediction

A demonstration prediction produced:

* **Predicted ETA:** 24.99 minutes
* **Traffic:** Jam
* **Weather:** Cloudy
* **Multiple deliveries:** 1
* **Vehicle:** Motorcycle
* **Distance:** 4.52 km

The operational long-delivery threshold is **45 minutes**.

> The 45-minute threshold represents a predicted long delivery, not a promised-ETA lateness classification.

## Operational Scenario Testing

The final scenario tests include:

* Baseline
* Heavy traffic
* Low traffic
* Sunny weather
* Stormy weather
* Multiple deliveries
* Festival
* Short distance
* Longer distance
* Peak + jam + multiple deliveries

All basic operational sanity checks passed.

## Important Interpretation

Scenario testing evaluates **model behavior**, not causal effects.

For example, if changing a feature does not change a prediction, this does not prove that the feature has no real-world effect. It only describes the behavior of the trained model for the tested inputs.

## Repository Structure

```text
food-delivery-eta-delay-analysis/
│
├── data/
│
├── models/
│   └── eta_model_final_tail_aware.joblib
│
├── reports/
│   ├── FINAL_PROJECT_REPORT.md
│   ├── operational_scenario_predictions.csv
│   ├── model_comparison.csv
│   ├── permutation_importance.csv
│   ├── tail_aware_model_comparison.csv
│   └── ...
│
├── src/
│   ├── predict.py
│   ├── test_scenarios.py
│   └── generate_final_report.py
│
├── tests/
├── .gitignore
└── README.md
```

## Running the Project

Activate the virtual environment and install the required dependencies.

### Generate a demonstration prediction

```bash
python src/predict.py
```

### Run operational scenario testing

```bash
python src/test_scenarios.py
```

### Generate the final project report

```bash
python src/generate_final_report.py
```

## Outputs

Operational scenario predictions are saved to:

`reports/operational_scenario_predictions.csv`

The final project report is saved to:

`reports/FINAL_PROJECT_REPORT.md`

## Skills Demonstrated

* Python
* Pandas
* NumPy
* Scikit-learn
* Machine Learning
* Regression
* Feature Engineering
* Model Evaluation
* Temporal Validation
* Geographic Data Quality
* Error Analysis
* Explainability
* Operational ML
* Git/GitHub

## Project Status

**Completed — portfolio ready after final repository review.**
