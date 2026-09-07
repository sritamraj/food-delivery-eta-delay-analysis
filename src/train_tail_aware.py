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

LONG_DELIVERY_THRESHOLD = 45

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
# MODEL
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
        df["Delivery_person_ID"]
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
        df["Delivery_person_ID"]
        .isin(train_groups)
    ].copy()

    test_df = df[
        df["Delivery_person_ID"]
        .isin(test_groups)
    ].copy()

    return (
        train_df,
        test_df,
        train_groups,
        test_groups,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_predictions(
    y_true,
    predictions,
):

    error = (
        predictions
        - y_true.to_numpy()
    )

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    r2 = r2_score(
        y_true,
        predictions,
    )

    bias = error.mean()

    long_mask = (
        y_true.to_numpy()
        >= LONG_DELIVERY_THRESHOLD
    )

    if long_mask.sum() > 0:

        long_mae = mean_absolute_error(
            y_true.to_numpy()[long_mask],
            predictions[long_mask],
        )

        long_rmse = np.sqrt(
            mean_squared_error(
                y_true.to_numpy()[long_mask],
                predictions[long_mask],
            )
        )

        long_bias = (
            predictions[long_mask]
            - y_true.to_numpy()[long_mask]
        ).mean()

    else:

        long_mae = np.nan
        long_rmse = np.nan
        long_bias = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "overall_signed_error": bias,
        "long_delivery_count": int(
            long_mask.sum()
        ),
        "long_delivery_MAE": long_mae,
        "long_delivery_RMSE": long_rmse,
        "long_delivery_signed_error": long_bias,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "TAIL-AWARE XGBOOST EXPERIMENT"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load and clean data
    # --------------------------------------------------------

    df = load_dataset()

    print(
        f"\nOriginal rows: "
        f"{len(df):,}"
    )

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

    print(
        f"\nTrain groups: "
        f"{len(train_groups):,}"
    )

    print(
        f"Test groups: "
        f"{len(test_groups):,}"
    )

    print(
        f"Train rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Test rows: "
        f"{len(test_df):,}"
    )

    overlap = (
        train_groups
        & test_groups
    )

    if overlap:

        raise RuntimeError(
            "Delivery-person leakage detected."
        )

    print(
        "Delivery-person leakage check: PASS"
    )

    # --------------------------------------------------------
    # Features
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
    # Create tail weights
    # --------------------------------------------------------

    long_mask_train = (
        y_train
        >= LONG_DELIVERY_THRESHOLD
    )

    long_count = int(
        long_mask_train.sum()
    )

    normal_count = int(
        (~long_mask_train).sum()
    )

    print("\nTraining target distribution:")

    print(
        f"  Normal deliveries (<45): "
        f"{normal_count:,}"
    )

    print(
        f"  Long deliveries (>=45): "
        f"{long_count:,}"
    )

    # --------------------------------------------------------
    # Weighting strategy
    #
    # Long deliveries receive 2x weight.
    # Normal deliveries receive 1x weight.
    # --------------------------------------------------------

    sample_weight = np.where(
        long_mask_train,
        2.0,
        1.0,
    )

    print(
        "\nTail weighting:"
    )

    print(
        "  Normal delivery weight: 1.0"
    )

    print(
        "  Long delivery weight  : 2.0"
    )

    # --------------------------------------------------------
    # Train baseline model
    # --------------------------------------------------------

    print(
        "\nTraining BASELINE XGBoost..."
    )

    baseline_model = build_model(
        X_train
    )

    baseline_model.fit(
        X_train,
        y_train,
    )

    baseline_predictions = (
        baseline_model.predict(
            X_test
        )
    )

    baseline_metrics = (
        evaluate_predictions(
            y_test,
            baseline_predictions,
        )
    )

    # --------------------------------------------------------
    # Train tail-aware model
    # --------------------------------------------------------

    print(
        "Training TAIL-AWARE XGBoost..."
    )

    tail_model = build_model(
        X_train
    )

    tail_model.fit(
        X_train,
        y_train,
        model__sample_weight=sample_weight,
    )

    tail_predictions = (
        tail_model.predict(
            X_test
        )
    )

    tail_metrics = (
        evaluate_predictions(
            y_test,
            tail_predictions,
        )
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "BASELINE VS TAIL-AWARE RESULTS"
    )
    print("=" * 70)

    results = pd.DataFrame(
        [
            {
                "model":
                    "Baseline XGBoost",
                **baseline_metrics,
            },
            {
                "model":
                    "Tail-Aware XGBoost",
                **tail_metrics,
            },
        ]
    )

    print(
        results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Improvement calculations
    # --------------------------------------------------------

    baseline_mae = (
        baseline_metrics["MAE"]
    )

    tail_mae = (
        tail_metrics["MAE"]
    )

    baseline_rmse = (
        baseline_metrics["RMSE"]
    )

    tail_rmse = (
        tail_metrics["RMSE"]
    )

    baseline_long_mae = (
        baseline_metrics[
            "long_delivery_MAE"
        ]
    )

    tail_long_mae = (
        tail_metrics[
            "long_delivery_MAE"
        ]
    )

    print("\n" + "=" * 70)
    print(
        "IMPROVEMENT ANALYSIS"
    )
    print("=" * 70)

    print(
        f"\nOverall MAE change: "
        f"{tail_mae - baseline_mae:+.4f}"
    )

    print(
        f"Overall MAE improvement: "
        f"{(baseline_mae - tail_mae) / baseline_mae * 100:+.2f}%"
    )

    print(
        f"Overall RMSE change: "
        f"{tail_rmse - baseline_rmse:+.4f}"
    )

    print(
        f"Overall RMSE improvement: "
        f"{(baseline_rmse - tail_rmse) / baseline_rmse * 100:+.2f}%"
    )

    print(
        f"\nLong-delivery MAE change: "
        f"{tail_long_mae - baseline_long_mae:+.4f}"
    )

    print(
        f"Long-delivery MAE improvement: "
        f"{(baseline_long_mae - tail_long_mae) / baseline_long_mae * 100:+.2f}%"
    )

    # --------------------------------------------------------
    # Decision logic
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "MODEL SELECTION DIAGNOSTIC"
    )
    print("=" * 70)

    overall_mae_change_pct = (
        (tail_mae - baseline_mae)
        / baseline_mae
        * 100
    )

    long_mae_change_pct = (
        (tail_long_mae - baseline_long_mae)
        / baseline_long_mae
        * 100
    )

    if (
        long_mae_change_pct < 0
        and overall_mae_change_pct <= 1.0
    ):

        print(
            "\nRESULT: Tail-aware model is promising."
        )

        print(
            "It improves long-delivery error "
            "without materially damaging "
            "overall MAE."
        )

    elif long_mae_change_pct < 0:

        print(
            "\nRESULT: Tail-aware model improves "
            "the long-delivery tail, but "
            "overall performance has a trade-off."
        )

    else:

        print(
            "\nRESULT: Tail weighting did not "
            "improve the long-delivery tail."
        )

        print(
            "Keep the baseline model for now."
        )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    results[
        "geographic_cleaning"
    ] = "distance<=50km and coordinate differences<=5deg"

    results.to_csv(
        REPORTS_DIR
        / "tail_aware_model_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_report = test_df[
        [
            "ID",
            "Delivery_person_ID",
            TARGET,
        ]
    ].copy()

    prediction_report[
        "baseline_prediction"
    ] = baseline_predictions

    prediction_report[
        "tail_aware_prediction"
    ] = tail_predictions

    prediction_report[
        "baseline_error"
    ] = (
        baseline_predictions
        - prediction_report[TARGET]
    )

    prediction_report[
        "tail_aware_error"
    ] = (
        tail_predictions
        - prediction_report[TARGET]
    )

    prediction_report[
        "is_long_delivery"
    ] = (
        prediction_report[TARGET]
        >= LONG_DELIVERY_THRESHOLD
    )

    prediction_report.to_csv(
        REPORTS_DIR
        / "tail_aware_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save candidate model
    # --------------------------------------------------------

    joblib.dump(
        tail_model,
        MODELS_DIR
        / "eta_model_tail_aware_candidate.joblib",
    )

    print("\nSaved:")

    print(
        "  reports\\tail_aware_model_comparison.csv"
    )

    print(
        "  reports\\tail_aware_predictions.csv"
    )

    print(
        "  models\\eta_model_tail_aware_candidate.joblib"
    )

    print("\n" + "=" * 70)
    print(
        "TAIL-AWARE EXPERIMENT COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()