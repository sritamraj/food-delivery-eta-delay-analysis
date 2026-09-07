# Food Delivery ETA Prediction & Delay Analysis

## 1. Project Overview

This project develops a machine-learning system for predicting food-delivery
ETA in minutes and identifying potentially long deliveries.

The project focuses on operational prediction rather than claiming causal
relationships between individual delivery features and ETA.

## 2. Model

Final model:

`models/eta_model_final_tail_aware.joblib`

The model is evaluated using operational scenario testing and basic behavioral
sanity checks.

## 3. Demonstration Prediction

The prediction script successfully generated:

- Predicted ETA: **24.99 minutes**
- Traffic: Jam
- Weather: Cloudy
- Multiple deliveries: 1
- Vehicle: Motorcycle
- Distance: 4.52 km
- Operational flag: **NORMAL PREDICTED DELIVERY**

The 45-minute threshold represents a predicted long delivery, not a promised
ETA lateness classification.

## 4. Operational Scenario Testing

The scenario testing produced the following results:

| Scenario | Predicted ETA |
|---|---:|
| Baseline | 24.99 min |
| Heavy Traffic | 24.99 min |
| Low Traffic | 24.99 min |
| Sunny Weather | 24.99 min |
| Stormy Weather | 24.99 min |
| Multiple Deliveries | 36.99 min |
| Festival | 24.99 min |
| Short Distance | 23.13 min |
| Longer Distance | 26.61 min |
| Peak + Jam + Multiple | 36.99 min |

## 5. Sanity Checks

All basic operational sanity checks passed:

- Low traffic <= baseline: **PASS**
- Heavy traffic >= low traffic: **PASS**
- Multiple deliveries >= baseline: **PASS**
- Longer distance >= short distance: **PASS**

## 6. Key Observations

The model predicts approximately 24.99 minutes for the baseline scenario.

The multiple-delivery scenario increases predicted ETA to approximately
36.99 minutes, representing an increase of about 12 minutes relative to the
baseline.

The short-distance scenario produces a lower ETA of approximately
23.13 minutes.

The longer-distance scenario produces an ETA of approximately
26.61 minutes.

## 7. Important Modeling Interpretation

These scenario tests are behavioral sanity checks.

They should **not** be interpreted as causal-effect estimates.

For example, the fact that changing a feature in a scenario does not change
the prediction does not prove that the corresponding real-world factor has no
effect on delivery time. It only describes the behavior of this trained model
under the tested inputs.

## 8. Project Outputs

### Prediction

`src/predict.py`

Generates a demonstration ETA prediction using the trained model.

### Operational Scenario Testing

`src/test_scenarios.py`

Tests realistic delivery scenarios and verifies basic model behavior.

### Scenario Results

`reports/operational_scenario_predictions.csv`

Contains the generated scenario-level predictions.

### Final Model

`models/eta_model_final_tail_aware.joblib`

Contains the trained model used for prediction.

## 9. Conclusion

The Food Delivery ETA Prediction project provides an end-to-end machine
learning workflow for operational ETA prediction.

The final system successfully loads the trained model, generates predictions,
tests operational scenarios, and passes the defined sanity checks.

The project is designed to demonstrate practical machine-learning skills
including model development, prediction pipelines, operational evaluation,
scenario testing, and responsible interpretation of model behavior.
