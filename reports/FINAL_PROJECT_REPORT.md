# Food Delivery ETA Prediction & Delay Analysis

## 1. Project Overview

This project develops an end-to-end machine-learning system for predicting food-delivery ETA in minutes and identifying potentially long deliveries.

The project focuses on **operational prediction and model behavior**, rather than claiming causal relationships between individual delivery features and delivery time.

The workflow includes:

* Data quality and geographic anomaly handling
* Feature engineering
* Gradient-boosted tree modeling
* Model comparison
* Tail-aware training for long deliveries
* Error analysis
* Operational scenario testing
* Model-behavior sanity checks
* Reproducible prediction scripts

---

## 2. Problem Definition

The primary prediction target is:

`Time_taken(min)`

The system predicts delivery time in minutes from delivery, vehicle, traffic, weather, geographic, and temporal information.

A secondary operational objective is to improve prediction performance for **long deliveries**, defined in this project as deliveries with predicted/observed delivery time at or above the **45-minute operational threshold** used during tail-aware modeling.

The 45-minute threshold should not be interpreted as a promised ETA or as a formal lateness classification label.

---

## 3. Data Quality and Geographic Cleaning

The modeling pipeline includes geographic data-quality screening.

The final geographic cleaning rule used for the selected model was:

* Delivery distance ≤ 50 km
* Coordinate differences ≤ 5 degrees

Using this cleaning process, **431 geographic anomalies were removed** from the modeling data.

The final model configuration reports:

* Training rows: **45,162**
* Removed geographic anomalies: **431**
* Long-delivery training rows: **1,781**

Geographic cleaning was treated as a **data-quality intervention**, not as a causal modeling assumption.

---

## 4. Features

The final model uses the following 15 features:

1. `Delivery_person_Age`
2. `Delivery_person_Ratings`
3. `Weatherconditions`
4. `Road_traffic_density`
5. `Vehicle_condition`
6. `Type_of_order`
7. `Type_of_vehicle`
8. `multiple_deliveries`
9. `Festival`
10. `City`
11. `distance_km`
12. `order_hour`
13. `is_peak_hour`
14. `order_dayofweek`
15. `is_weekend`

The feature set combines:

* Delivery-person characteristics
* Weather conditions
* Traffic conditions
* Vehicle information
* Order characteristics
* Delivery distance
* City
* Time-of-day information
* Day-of-week information
* Peak-period indicators

---

## 5. Model Development

Several regression models were evaluated.

### Overall model comparison

| Model             | MAE (min) | RMSE (min) |     R² |
| ----------------- | --------: | ---------: | -----: |
| **Tuned XGBoost** |     3.140 |      3.946 | 0.8224 |
| XGBoost           |     3.144 |      3.949 | 0.8222 |
| Random Forest     |     3.182 |      3.997 | 0.8178 |
| Linear Regression |     4.878 |      6.179 | 0.5645 |

The tree-based models substantially outperform the linear baseline on the reported evaluation metrics.

The tuned XGBoost configuration was therefore investigated further.

---

## 6. Tail-Aware Modeling

Overall average error does not necessarily reflect performance on operationally important long deliveries.

To address this, a tail-aware XGBoost model was trained with increased weight assigned to long-delivery examples.

Final configuration:

| Parameter                   |   Value |
| --------------------------- | ------: |
| Model                       | XGBoost |
| Tail weight                 |     2.0 |
| Long-delivery threshold     |  45 min |
| Estimators                  |     500 |
| Maximum depth               |       6 |
| Learning rate               |    0.03 |
| Minimum child weight        |       3 |
| Subsample                   |     0.9 |
| Column subsampling          |     1.0 |
| Training rows               |  45,162 |
| Long-delivery training rows |   1,781 |

The objective was not simply to minimize average error, but to improve prediction quality in the long-delivery tail.

---

## 7. Overall vs. Long-Delivery Performance

The baseline and tail-aware XGBoost models produced:

| Metric                     | Baseline XGBoost | Tail-Aware XGBoost |
| -------------------------- | ---------------: | -----------------: |
| Overall MAE                |    **3.060 min** |          3.080 min |
| Overall RMSE               |    **3.853 min** |          3.863 min |
| Overall R²                 |       **0.8333** |             0.8325 |
| Long-delivery MAE          |        4.703 min |      **4.045 min** |
| Long-delivery RMSE         |        5.944 min |      **5.174 min** |
| Long-delivery signed error |       -4.106 min |     **-3.280 min** |

### Interpretation

The baseline XGBoost has slightly better overall predictive accuracy.

However, the tail-aware model produces a substantial reduction in long-delivery error:

* Long-delivery MAE decreases from **4.703 to 4.045 minutes**
* Long-delivery RMSE decreases from **5.944 to 5.174 minutes**
* Long-delivery signed error becomes less negative, indicating reduced underprediction in this segment

The overall MAE increases only slightly from **3.060 to 3.080 minutes**.

Therefore, the tail-aware model represents an **operational trade-off**:

> A small deterioration in average accuracy is accepted in exchange for substantially better performance on long deliveries.

This is why the tail-aware model is retained as the operational model for this project.

Importantly, the project does **not** claim that the tail-aware model is the best model on every metric.

---

## 8. Final Model

The selected operational model is:

`models/eta_model_final_tail_aware.joblib`

The corresponding feature metadata is:

`models/feature_metadata.joblib`

The final model is an XGBoost regression model configured with a tail weight of 2.0 and a 45-minute long-delivery threshold.

---

## 9. Demonstration Prediction

The prediction pipeline successfully generated the following demonstration:

* Predicted ETA: **24.99 minutes**
* Traffic: **Jam**
* Weather: **Cloudy**
* Multiple deliveries: **1**
* Vehicle: **Motorcycle**
* Distance: **4.52 km**
* Operational flag: **NORMAL PREDICTED DELIVERY**

The 45-minute threshold represents a **predicted long-delivery operational threshold**. It is not a promised ETA and does not represent a formal lateness classification without an appropriate delivery-promise reference.

---

## 10. Operational Scenario Testing

The prediction pipeline was tested against several controlled operational scenarios.

| Scenario              | Predicted ETA |
| --------------------- | ------------: |
| Baseline              |     24.99 min |
| Heavy Traffic         |     24.99 min |
| Low Traffic           |     24.99 min |
| Sunny Weather         |     24.99 min |
| Stormy Weather        |     24.99 min |
| Multiple Deliveries   |     36.99 min |
| Festival              |     24.99 min |
| Short Distance        |     23.13 min |
| Longer Distance       |     26.61 min |
| Peak + Jam + Multiple |     36.99 min |

These scenarios are intended to evaluate whether the trained model behaves sensibly under selected operational input changes.

---

## 11. Behavioral Sanity Checks

The scenario-testing pipeline includes basic monotonicity-style sanity checks:

* Low traffic ≤ baseline: **PASS**
* Heavy traffic ≥ low traffic: **PASS**
* Multiple deliveries ≥ baseline: **PASS**
* Longer distance ≥ short distance: **PASS**

All defined basic operational sanity checks passed.

These checks are useful for detecting obviously unexpected model behavior in the tested scenarios.

However, they should not be interpreted as proof that the model has learned causal relationships.

---

## 12. Important Modeling Interpretation

Scenario testing evaluates **model behavior**, not causality.

For example, if changing weather in a controlled scenario does not change the model prediction, this does not prove that weather has no real-world effect on delivery time.

Likewise, if changing traffic produces a particular prediction change, that should not automatically be interpreted as a causal effect of traffic.

The scenario framework answers a narrower question:

> How does the trained model respond to the tested input configurations?

This distinction is important when interpreting machine-learning systems used for operational decision support.

---

## 13. Operational Insights

Several useful model-behavior observations emerged from the scenario analysis.

### Multiple deliveries

The baseline scenario predicts approximately **24.99 minutes**.

The multiple-delivery scenario predicts approximately **36.99 minutes**, an increase of approximately **12 minutes**.

This feature therefore produces a substantial change in the tested model scenarios.

### Distance

The short-distance scenario predicts approximately **23.13 minutes**, while the longer-distance scenario predicts approximately **26.61 minutes**.

The model therefore produces higher predicted ETA in the tested longer-distance scenario.

### Long-delivery tail

The most important modeling improvement comes from the tail-aware objective.

Although overall MAE changes only slightly, long-delivery MAE decreases by approximately **14%**, and long-delivery RMSE decreases by approximately **13%** relative to the baseline XGBoost.

This demonstrates why evaluating only overall MAE can hide important operational behavior.

---

## 14. Explainability and Error Analysis

The project also includes permutation-importance and grouped error-analysis outputs.

These analyses are intended to answer questions such as:

* Which features are associated with predictive importance?
* Where does the model make larger errors?
* How does error vary across traffic conditions?
* How does error vary across weather conditions?
* How does error vary across vehicle types?
* How does error vary across distance?
* Which groups contain higher long-delivery risk?
* Which combinations produce larger prediction errors?

These analyses complement aggregate metrics by examining model behavior across operational segments.

---

## 15. Project Outputs

### Prediction

`src/predict.py`

Loads the final model and generates a demonstration ETA prediction.

### Scenario testing

`src/test_scenarios.py`

Runs controlled operational scenarios and performs basic behavioral sanity checks.

### Final report

`reports/FINAL_PROJECT_REPORT.md`

Contains the final project methodology, model comparison, operational interpretation, and results.

### Scenario predictions

`reports/operational_scenario_predictions.csv`

Contains scenario-level prediction results.

### Final model

`models/eta_model_final_tail_aware.joblib`

Contains the selected operational ETA regression model.

### Feature metadata

`models/feature_metadata.joblib`

Stores the feature metadata required by the prediction pipeline.

---

## 16. Reproducibility

The main prediction workflow can be executed with:

```bash
python src\predict.py
```

Operational scenario testing can be executed with:

```bash
python src\test_scenarios.py
```

The final report can be regenerated with:

```bash
python src\generate_final_report.py
```

---

## 17. Limitations

This project has several important limitations.

### Dataset limitations

The reported results depend on the available dataset and its underlying data-generation process. Model performance on this dataset should not automatically be interpreted as performance on a production food-delivery platform.

### Operational threshold limitation

The 45-minute threshold is a project-defined operational threshold for identifying long deliveries. It is not necessarily equivalent to a customer's promised delivery time.

### Scenario limitation

Controlled scenario testing evaluates model behavior. It does not establish causal effects.

### Generalization limitation

The reported metrics should not be assumed to generalize to unseen cities, delivery platforms, time periods, or operational environments without additional external validation.

### Model-selection trade-off

The tail-aware model does not achieve the best overall MAE, RMSE, or R². It was selected because its long-delivery performance is substantially better, which is the project's operational priority.

---

## 18. Conclusion

The Food Delivery ETA Prediction & Delay Analysis project demonstrates an end-to-end machine-learning workflow for operational ETA prediction.

The project combines:

* Data-quality auditing
* Geographic anomaly handling
* Feature engineering
* XGBoost regression
* Model comparison
* Tail-aware modeling
* Long-delivery error analysis
* Permutation-based feature analysis
* Operational scenario testing
* Behavioral sanity checks
* Responsible interpretation of model behavior

The strongest modeling result is the trade-off between average predictive accuracy and long-delivery performance.

The baseline XGBoost achieves an overall MAE of **3.060 minutes**, while the selected tail-aware XGBoost achieves an overall MAE of **3.080 minutes** but reduces long-delivery MAE from **4.703 to 4.045 minutes**.

Thus, the final model is selected not because it is universally superior, but because it better aligns the model objective with the project's operational focus on long deliveries.

This provides a more realistic machine-learning decision-making framework than selecting a model solely on aggregate average error.
