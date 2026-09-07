from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from data_prep import load_dataset


# ============================================================
# FINAL MODEL CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

TARGET = "Time_taken(min)"

LONG_DELIVERY_THRESHOLD = 45

TAIL_WEIGHT = 2.0

FEATURES = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Weatherconditions",
    "Road_traffic_density",
    "Vehicle_condition",
    "Type_of_order",
    "Type_of_vehicle",
    "multiple_deliveries",
    "Festival",
    "City",
    "distance_km",
    "order_hour",
    "is_peak_hour",
    "order_dayofweek",
    "is_weekend",
]

XGB_PARAMS = {
    "subsample": 0.9,
    "n_estimators": 500,
    "min_child_weight": 3,
    "max_depth": 6,
    "learning_rate": 0.03,
    "colsample_bytree": 1.0,
}


# ============================================================
# GEOGRAPHIC CLEANING
# ============================================================

def apply_geographic_cleaning(df):

    distance = pd.to_numeric(
        df["distance_km"],
        errors="coerce",
    )

    latitude_difference = (
        df["Restaurant_latitude"]
        - df["Delivery_location_latitude"]
    ).abs()

    longitude_difference = (
        df["Restaurant_longitude"]
        - df["Delivery_location_longitude"]
    ).abs()

    geographic_outlier = (
        (distance > 50)
        | (latitude_difference > 5)
        | (longitude_difference > 5)
    )

    cleaned_df = df.loc[
        ~geographic_outlier
    ].copy()

    return cleaned_df


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(X):

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )


# ============================================================
# MODEL
# ============================================================

def build_model(X):

    preprocessor = build_preprocessor(X)

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **XGB_PARAMS,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "FINAL TAIL-AWARE XGBOOST MODEL TRAINING"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_dataset()

    print(
        f"\nOriginal rows: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Apply validated geographic cleaning
    # --------------------------------------------------------

    original_rows = len(df)

    df = apply_geographic_cleaning(
        df
    )

    removed_rows = (
        original_rows
        - len(df)
    )

    print(
        f"Geographic anomalies removed: "
        f"{removed_rows:,}"
    )

    print(
        f"Final training rows: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Prepare features and target
    # --------------------------------------------------------

    X = df[
        FEATURES
    ].copy()

    y = df[
        TARGET
    ].copy()

    # --------------------------------------------------------
    # Tail-aware sample weights
    # --------------------------------------------------------

    long_mask = (
        y
        >= LONG_DELIVERY_THRESHOLD
    )

    sample_weight = np.where(
        long_mask,
        TAIL_WEIGHT,
        1.0,
    )

    print(
        f"\nLong-delivery threshold: "
        f"{LONG_DELIVERY_THRESHOLD} minutes"
    )

    print(
        f"Long-delivery training rows: "
        f"{long_mask.sum():,}"
    )

    print(
        f"Normal training rows: "
        f"{(~long_mask).sum():,}"
    )

    print(
        f"Long-delivery sample weight: "
        f"{TAIL_WEIGHT:.1f}x"
    )

    # --------------------------------------------------------
    # Train final model on all cleaned data
    # --------------------------------------------------------

    print(
        "\nTraining final model..."
    )

    model = build_model(X)

    model.fit(
        X,
        y,
        model__sample_weight=sample_weight,
    )

    print(
        "Final model training complete."
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODELS_DIR
        / "eta_model_final_tail_aware.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"\nSaved model:"
    )

    print(
        f"  {model_path}"
    )

    # --------------------------------------------------------
    # Save training configuration
    # --------------------------------------------------------

    config = pd.DataFrame(
        [
            {
                "model": "XGBoost",
                "target": TARGET,
                "tail_weight": TAIL_WEIGHT,
                "long_delivery_threshold": LONG_DELIVERY_THRESHOLD,
                "n_estimators": XGB_PARAMS["n_estimators"],
                "max_depth": XGB_PARAMS["max_depth"],
                "learning_rate": XGB_PARAMS["learning_rate"],
                "min_child_weight": XGB_PARAMS["min_child_weight"],
                "subsample": XGB_PARAMS["subsample"],
                "colsample_bytree": XGB_PARAMS["colsample_bytree"],
                "training_rows": len(df),
                "removed_geographic_anomalies": removed_rows,
                "long_delivery_training_rows": int(
                    long_mask.sum()
                ),
            }
        ]
    )

    config_path = (
        REPORTS_DIR
        / "final_model_configuration.csv"
    )

    config.to_csv(
        config_path,
        index=False,
    )

    print(
        f"  {config_path}"
    )

    # --------------------------------------------------------
    # Save feature list
    # --------------------------------------------------------

    feature_config = pd.DataFrame(
        {
            "feature": FEATURES
        }
    )

    feature_path = (
        REPORTS_DIR
        / "final_model_features.csv"
    )

    feature_config.to_csv(
        feature_path,
        index=False,
    )

    print(
        f"  {feature_path}"
    )

    print("\n" + "=" * 70)
    print(
        "FINAL MODEL READY"
    )
    print("=" * 70)

    print(
        "\nModel:"
    )

    print(
        "  XGBoost + geographic cleaning"
    )

    print(
        "  + 2x weighting for observed "
        "long deliveries (>=45 min)"
    )

    print(
        "\nValidation reference:"
    )

    print(
        "  Unseen-person baseline MAE: 3.0602 min"
    )

    print(
        "  Unseen-person tail-aware MAE: 3.0802 min"
    )

    print(
        "  Long-delivery MAE: 4.0447 min"
    )

    print(
        "  Long-delivery improvement: 14.00%"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "These validation metrics come from "
        "the held-out unseen-person experiment."
    )

    print(
        "The final model was then retrained "
        "on all cleaned observations."
    )


if __name__ == "__main__":
    main()