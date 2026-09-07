from pathlib import Path

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

REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "Time_taken(min)"

LONG_DELIVERY_THRESHOLD = 45

GROUP_COLUMN = "Delivery_person_ID"

WEIGHTS = [
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
    2.50,
    3.00,
]

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
# EVALUATION
# ============================================================

def evaluate(
    y_true,
    predictions,
):

    actual = y_true.to_numpy()

    error = (
        predictions
        - actual
    )

    mae = mean_absolute_error(
        actual,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predictions,
        )
    )

    r2 = r2_score(
        actual,
        predictions,
    )

    bias = error.mean()

    long_mask = (
        actual
        >= LONG_DELIVERY_THRESHOLD
    )

    if long_mask.sum() > 0:

        long_mae = mean_absolute_error(
            actual[long_mask],
            predictions[long_mask],
        )

        long_rmse = np.sqrt(
            mean_squared_error(
                actual[long_mask],
                predictions[long_mask],
            )
        )

        long_bias = (
            predictions[long_mask]
            - actual[long_mask]
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
        "TAIL-WEIGHT SWEEP — "
        "UNSEEN DELIVERY PERSON VALIDATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load and clean
    # --------------------------------------------------------

    df = load_dataset()

    print(
        f"\nOriginal rows: "
        f"{len(df):,}"
    )

    df = apply_geographic_cleaning(
        df
    )

    print(
        f"Cleaned rows: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Same deterministic group split
    # --------------------------------------------------------

    (
        train_df,
        test_df,
        train_groups,
        test_groups,
    ) = group_split(
        df
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
    # Prepare data
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

    long_train_mask = (
        y_train
        >= LONG_DELIVERY_THRESHOLD
    )

    long_test_mask = (
        y_test
        >= LONG_DELIVERY_THRESHOLD
    )

    print(
        f"\nTraining long deliveries: "
        f"{long_train_mask.sum():,}"
    )

    print(
        f"Test long deliveries: "
        f"{long_test_mask.sum():,}"
    )

    # --------------------------------------------------------
    # Sweep
    # --------------------------------------------------------

    all_results = []

    print("\n" + "=" * 70)
    print(
        "RUNNING WEIGHT SWEEP"
    )
    print("=" * 70)

    for weight in WEIGHTS:

        print(
            f"\nTraining long-delivery weight = "
            f"{weight:.2f}x"
        )

        model = build_model(
            X_train
        )

        sample_weight = np.where(
            long_train_mask,
            weight,
            1.0,
        )

        model.fit(
            X_train,
            y_train,
            model__sample_weight=sample_weight,
        )

        predictions = model.predict(
            X_test
        )

        metrics = evaluate(
            y_test,
            predictions,
        )

        result = {
            "long_delivery_weight":
                weight,
            **metrics,
        }

        all_results.append(
            result
        )

        print(
            f"  Overall MAE: "
            f"{metrics['MAE']:.4f}"
        )

        print(
            f"  Overall RMSE: "
            f"{metrics['RMSE']:.4f}"
        )

        print(
            f"  R2: "
            f"{metrics['R2']:.4f}"
        )

        print(
            f"  Long MAE: "
            f"{metrics['long_delivery_MAE']:.4f}"
        )

        print(
            f"  Long bias: "
            f"{metrics['long_delivery_signed_error']:+.4f}"
        )

    results = pd.DataFrame(
        all_results
    )

    # --------------------------------------------------------
    # Baseline reference
    # --------------------------------------------------------

    baseline = results[
        results[
            "long_delivery_weight"
        ] == 1.0
    ].iloc[0]

    results[
        "MAE_change_vs_baseline"
    ] = (
        results["MAE"]
        - baseline["MAE"]
    )

    results[
        "RMSE_change_vs_baseline"
    ] = (
        results["RMSE"]
        - baseline["RMSE"]
    )

    results[
        "long_MAE_change_vs_baseline"
    ] = (
        results["long_delivery_MAE"]
        - baseline["long_delivery_MAE"]
    )

    results[
        "long_MAE_improvement_pct"
    ] = (
        (
            baseline["long_delivery_MAE"]
            - results["long_delivery_MAE"]
        )
        / baseline["long_delivery_MAE"]
        * 100
    )

    # --------------------------------------------------------
    # Print complete comparison
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "COMPLETE WEIGHT SWEEP"
    )
    print("=" * 70)

    display_columns = [
        "long_delivery_weight",
        "MAE",
        "RMSE",
        "R2",
        "overall_signed_error",
        "long_delivery_MAE",
        "long_delivery_RMSE",
        "long_delivery_signed_error",
        "long_MAE_improvement_pct",
    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Candidate selection
    # --------------------------------------------------------

    # Constraint:
    # Do not accept a model whose overall MAE
    # is more than 1% worse than baseline.

    baseline_mae = baseline[
        "MAE"
    ]

    acceptable = results[
        results["MAE"]
        <= baseline_mae * 1.01
    ].copy()

    if len(acceptable) > 0:

        best = acceptable.loc[
            acceptable[
                "long_delivery_MAE"
            ].idxmin()
        ]

        print("\n" + "=" * 70)
        print(
            "BEST TRADE-OFF CANDIDATE"
        )
        print("=" * 70)

        print(
            f"\nSelected weight: "
            f"{best['long_delivery_weight']:.2f}x"
        )

        print(
            f"Overall MAE: "
            f"{best['MAE']:.4f}"
        )

        print(
            f"Overall RMSE: "
            f"{best['RMSE']:.4f}"
        )

        print(
            f"R2: "
            f"{best['R2']:.4f}"
        )

        print(
            f"Long-delivery MAE: "
            f"{best['long_delivery_MAE']:.4f}"
        )

        print(
            f"Long-delivery improvement: "
            f"{best['long_MAE_improvement_pct']:+.2f}%"
        )

    else:

        best = None

        print(
            "\nNo weight satisfies the "
            "overall-MAE constraint."
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results.to_csv(
        REPORTS_DIR
        / "tail_weight_sweep.csv",
        index=False,
    )

    if best is not None:

        pd.DataFrame(
            [best]
        ).to_csv(
            REPORTS_DIR
            / "tail_weight_selected_candidate.csv",
            index=False,
        )

    print("\nSaved:")

    print(
        "  reports\\tail_weight_sweep.csv"
    )

    if best is not None:

        print(
            "  reports\\tail_weight_selected_candidate.csv"
        )

    print("\n" + "=" * 70)
    print(
        "TAIL-WEIGHT SWEEP COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()