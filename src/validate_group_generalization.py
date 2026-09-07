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

GROUP_COLUMN = "Delivery_person_ID"

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
# GROUP SPLIT
# ============================================================

def group_split(df):

    groups = (
        df[GROUP_COLUMN]
        .dropna()
        .unique()
    )

    rng = np.random.RandomState(42)

    shuffled_groups = rng.permutation(
        groups
    )

    split_index = int(
        len(shuffled_groups) * 0.80
    )

    train_groups = set(
        shuffled_groups[:split_index]
    )

    test_groups = set(
        shuffled_groups[split_index:]
    )

    train_df = df[
        df[GROUP_COLUMN]
        .isin(train_groups)
    ].copy()

    test_df = df[
        df[GROUP_COLUMN]
        .isin(test_groups)
    ].copy()

    return (
        train_df,
        test_df,
        train_groups,
        test_groups,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "GROUP GENERALIZATION — "
        "GEOGRAPHICALLY CLEANED DATA"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
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
    # Delivery-person groups
    # --------------------------------------------------------

    total_groups = (
        cleaned_df[GROUP_COLUMN]
        .nunique()
    )

    print(
        f"Unique delivery persons: "
        f"{total_groups:,}"
    )

    # --------------------------------------------------------
    # Group split
    # --------------------------------------------------------

    (
        train_df,
        test_df,
        train_groups,
        test_groups,
    ) = group_split(
        cleaned_df
    )

    print("\nGroup split:")

    print(
        f"  Train groups: "
        f"{len(train_groups):,}"
    )

    print(
        f"  Test groups : "
        f"{len(test_groups):,}"
    )

    print(
        f"  Train rows: "
        f"{len(train_df):,}"
    )

    print(
        f"  Test rows : "
        f"{len(test_df):,}"
    )

    # --------------------------------------------------------
    # Group leakage check
    # --------------------------------------------------------

    group_overlap = (
        train_groups
        & test_groups
    )

    if len(group_overlap) == 0:

        print(
            "\nDelivery-person leakage check: PASS"
        )

    else:

        print(
            "\nDelivery-person leakage check: FAIL"
        )

        raise RuntimeError(
            "Delivery-person groups overlap."
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
    # Confirm group ID is not a feature
    # --------------------------------------------------------

    if GROUP_COLUMN in FEATURES:

        raise RuntimeError(
            "Delivery_person_ID must not be "
            "included as a model feature."
        )

    print(
        "\nDelivery_person_ID used only "
        "for splitting: PASS"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "\nTraining cleaned group-generalization model..."
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
    print(
        "CLEANED GROUP VALIDATION RESULTS"
    )
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

    else:

        long_mae = np.nan
        long_rmse = np.nan
        long_signed_error = np.nan

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
        / "cleaned_group_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Validation summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "dataset":
                    "Geographically Cleaned",
                "validation":
                    "Delivery Person Group",
                "train_groups":
                    len(train_groups),
                "test_groups":
                    len(test_groups),
                "train_rows":
                    len(train_df),
                "test_rows":
                    len(test_df),
                "MAE":
                    mae,
                "RMSE":
                    rmse,
                "R2":
                    r2,
                "overall_signed_error":
                    signed_error,
                "long_delivery_MAE":
                    long_mae,
                "long_delivery_RMSE":
                    long_rmse,
                "long_delivery_signed_error":
                    long_signed_error,
            }
        ]
    )

    summary.to_csv(
        REPORTS_DIR
        / "cleaned_group_validation.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODELS_DIR
        / "eta_model_cleaned_group.joblib",
    )

    print("\nSaved:")

    print(
        "  reports\\cleaned_group_validation.csv"
    )

    print(
        "  reports\\cleaned_group_predictions.csv"
    )

    print(
        "  models\\eta_model_cleaned_group.joblib"
    )

    print("\n" + "=" * 70)
    print(
        "CLEANED GROUP VALIDATION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()