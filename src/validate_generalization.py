from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from data_prep import load_dataset


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "Time_taken(min)"

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
        (
            df["Restaurant_latitude"]
            - df["Delivery_location_latitude"]
        )
        .abs()
    )

    longitude_difference = (
        (
            df["Restaurant_longitude"]
            - df["Delivery_location_longitude"]
        )
        .abs()
    )

    geographic_outlier = (
        (distance > 50)
        | (latitude_difference > 5)
        | (longitude_difference > 5)
    )

    cleaned_df = df.loc[
        ~geographic_outlier
    ].copy()

    return cleaned_df, geographic_outlier


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
# BUILD MODEL
# ============================================================

def build_model(X_train):

    preprocessor = build_preprocessor(
        X_train
    )

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
# TEMPORAL SPLIT
# ============================================================

def chronological_split(df):

    working = df.copy()

    working["Order_Date"] = pd.to_datetime(
        working["Order_Date"],
        errors="coerce",
        dayfirst=True,
    )

    working = working.dropna(
        subset=["Order_Date"]
    )

    unique_dates = sorted(
        working["Order_Date"]
        .dt.normalize()
        .unique()
    )

    split_index = int(
        len(unique_dates) * 0.80
    )

    train_dates = set(
        unique_dates[:split_index]
    )

    test_dates = set(
        unique_dates[split_index:]
    )

    train_df = working[
        working["Order_Date"]
        .dt.normalize()
        .isin(train_dates)
    ].copy()

    test_df = working[
        working["Order_Date"]
        .dt.normalize()
        .isin(test_dates)
    ].copy()

    return (
        train_df,
        test_df,
        train_dates,
        test_dates,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TEMPORAL GENERALIZATION — GEOGRAPHICALLY CLEANED DATA")
    print("=" * 70)

    # --------------------------------------------------------
    # Load original data
    # --------------------------------------------------------

    df = load_dataset()

    print(
        f"\nOriginal rows: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Geographic cleaning
    # --------------------------------------------------------

    cleaned_df, geographic_outlier = (
        apply_geographic_cleaning(df)
    )

    print(
        f"Geographic anomalies removed: "
        f"{geographic_outlier.sum():,}"
    )

    print(
        f"Cleaned rows: "
        f"{len(cleaned_df):,}"
    )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    (
        train_df,
        test_df,
        train_dates,
        test_dates,
    ) = chronological_split(
        cleaned_df
    )

    print("\nTemporal split:")

    print(
        f"  Train rows: "
        f"{len(train_df):,}"
    )

    print(
        f"  Test rows : "
        f"{len(test_df):,}"
    )

    print(
        f"  Train dates: "
        f"{min(train_dates).date()} "
        f"to "
        f"{max(train_dates).date()}"
    )

    print(
        f"  Test dates : "
        f"{min(test_dates).date()} "
        f"to "
        f"{max(test_dates).date()}"
    )

    cutoff_date = min(test_dates)

    print(
        f"  Temporal cutoff: "
        f"{cutoff_date.date()}"
    )

    # --------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------

    date_overlap = (
        set(train_dates)
        & set(test_dates)
    )

    if len(date_overlap) == 0:

        print(
            "\nDate leakage check: PASS"
        )

    else:

        print(
            "\nDate leakage check: FAIL"
        )

        raise RuntimeError(
            "Train and test dates overlap."
        )

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X_train = train_df[
        FEATURES
    ].copy()

    y_train = train_df[
        TARGET
    ].copy()

    X_test = test_df[
        FEATURES
    ].copy()

    y_test = test_df[
        TARGET
    ].copy()

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    print(
        "\nTraining cleaned temporal model..."
    )

    model = build_model(
        X_train
    )

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    print("\n" + "=" * 70)
    print("CLEANED TEMPORAL VALIDATION RESULTS")
    print("=" * 70)

    print(
        f"\nMAE : "
        f"{mae:.4f}"
    )

    print(
        f"RMSE: "
        f"{rmse:.4f}"
    )

    print(
        f"R²  : "
        f"{r2:.4f}"
    )

    # --------------------------------------------------------
    # Prediction report
    # --------------------------------------------------------

    prediction_report = test_df[
        [
            "ID",
            "Delivery_person_ID",
            "Order_Date",
            TARGET,
        ]
    ].copy()

    prediction_report[
        "predicted"
    ] = predictions

    prediction_report[
        "error"
    ] = (
        prediction_report["predicted"]
        - prediction_report[TARGET]
    )

    prediction_report[
        "absolute_error"
    ] = (
        prediction_report["error"]
        .abs()
    )

    prediction_report[
        "is_long_delivery"
    ] = (
        prediction_report[TARGET]
        >= 45
    )

    prediction_report.to_csv(
        REPORTS_DIR
        / "cleaned_temporal_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Long-delivery performance
    # --------------------------------------------------------

    long_mask = (
        y_test.to_numpy()
        >= 45
    )

    if long_mask.sum() > 0:

        long_mae = mean_absolute_error(
            y_test.to_numpy()[long_mask],
            predictions[long_mask],
        )

        long_rmse = np.sqrt(
            mean_squared_error(
                y_test.to_numpy()[long_mask],
                predictions[long_mask],
            )
        )

        long_signed_error = (
            predictions[long_mask]
            - y_test.to_numpy()[long_mask]
        ).mean()

        print(
            "\nLong-delivery performance "
            "(observed delivery time >=45 min):"
        )

        print(
            f"  Orders: "
            f"{long_mask.sum():,}"
        )

        print(
            f"  MAE: "
            f"{long_mae:.4f}"
        )

        print(
            f"  RMSE: "
            f"{long_rmse:.4f}"
        )

        print(
            f"  Signed error: "
            f"{long_signed_error:+.4f}"
        )

    # --------------------------------------------------------
    # Overall signed error
    # --------------------------------------------------------

    signed_error = (
        predictions
        - y_test.to_numpy()
    ).mean()

    print(
        f"\nOverall signed error: "
        f"{signed_error:+.4f}"
    )

    # --------------------------------------------------------
    # Save validation summary
    # --------------------------------------------------------

    validation_summary = pd.DataFrame(
        [
            {
                "dataset": "Geographically Cleaned",
                "validation": "Temporal",
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_start":
                    min(train_dates).date(),
                "train_end":
                    max(train_dates).date(),
                "test_start":
                    min(test_dates).date(),
                "test_end":
                    max(test_dates).date(),
                "cutoff_date":
                    cutoff_date.date(),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "overall_signed_error":
                    signed_error,
                "long_delivery_MAE":
                    long_mae
                    if long_mask.sum() > 0
                    else np.nan,
                "long_delivery_RMSE":
                    long_rmse
                    if long_mask.sum() > 0
                    else np.nan,
                "long_delivery_signed_error":
                    long_signed_error
                    if long_mask.sum() > 0
                    else np.nan,
            }
        ]
    )

    validation_summary.to_csv(
        REPORTS_DIR
        / "cleaned_temporal_validation.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODELS_DIR
        / "eta_model_cleaned_temporal.joblib",
    )

    print("\nSaved:")

    print(
        "  reports\\cleaned_temporal_validation.csv"
    )

    print(
        "  reports\\cleaned_temporal_predictions.csv"
    )

    print(
        "  models\\eta_model_cleaned_temporal.joblib"
    )

    print("\n" + "=" * 70)
    print("CLEANED TEMPORAL VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()