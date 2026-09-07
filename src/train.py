from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
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
# FEATURES
# ============================================================

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

TARGET = "Time_taken(min)"


# ============================================================
# MODEL PARAMETERS
# ============================================================

TUNED_XGB_PARAMS = {
    "subsample": 0.9,
    "n_estimators": 500,
    "min_child_weight": 3,
    "max_depth": 6,
    "learning_rate": 0.03,
    "colsample_bytree": 1.0,
}


# ============================================================
# METRICS
# ============================================================

def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)

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

    return {
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }, predictions


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

    preprocessor = ColumnTransformer(
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

    return preprocessor


# ============================================================
# BUILD XGBOOST
# ============================================================

def build_xgb(preprocessor):

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **TUNED_XGB_PARAMS,
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
# TRAIN ONE DATASET VERSION
# ============================================================

def train_experiment(
    df,
    dataset_name,
    random_state=42,
):

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(
        X_train
    )

    results = []

    # --------------------------------------------------------
    # Linear Regression
    # --------------------------------------------------------

    linear_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )

    linear_model.fit(
        X_train,
        y_train,
    )

    result, _ = evaluate_model(
        "Linear Regression",
        linear_model,
        X_test,
        y_test,
    )

    result["dataset"] = dataset_name
    results.append(result)

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    rf_model = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    X_train
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=18,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    rf_model.fit(
        X_train,
        y_train,
    )

    result, _ = evaluate_model(
        "Random Forest",
        rf_model,
        X_test,
        y_test,
    )

    result["dataset"] = dataset_name
    results.append(result)

    # --------------------------------------------------------
    # Tuned XGBoost
    # --------------------------------------------------------

    xgb_model = build_xgb(
        build_preprocessor(
            X_train
        )
    )

    xgb_model.fit(
        X_train,
        y_train,
    )

    result, predictions = evaluate_model(
        "Tuned XGBoost",
        xgb_model,
        X_test,
        y_test,
    )

    result["dataset"] = dataset_name
    results.append(result)

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions_df = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted": predictions,
        }
    )

    predictions_df["error"] = (
        predictions_df["predicted"]
        - predictions_df["actual"]
    )

    predictions_df["absolute_error"] = (
        predictions_df["error"].abs()
    )

    predictions_df["dataset"] = dataset_name

    predictions_df.to_csv(
        REPORTS_DIR
        / f"{dataset_name.lower()}_test_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Permutation importance
    # --------------------------------------------------------

    permutation = permutation_importance(
        xgb_model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance":
                permutation.importances_mean,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance_df[
        "dataset"
    ] = dataset_name

    importance_df.to_csv(
        REPORTS_DIR
        / f"{dataset_name.lower()}_permutation_importance.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        xgb_model,
        MODELS_DIR
        / f"eta_model_{dataset_name.lower()}.joblib",
    )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ETA MODEL — GEOGRAPHIC CLEANING EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load original dataset
    # --------------------------------------------------------

    df = load_dataset()

    print(
        f"\nOriginal dataset: "
        f"{len(df):,} rows"
    )

    # --------------------------------------------------------
    # Apply geographic cleaning
    # --------------------------------------------------------

    cleaned_df, geographic_outlier = (
        apply_geographic_cleaning(df)
    )

    print(
        f"Geographic anomalies removed: "
        f"{geographic_outlier.sum():,}"
    )

    print(
        f"Cleaned dataset: "
        f"{len(cleaned_df):,} rows"
    )

    # --------------------------------------------------------
    # Train baseline
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("EXPERIMENT 1: ORIGINAL DATASET")
    print("-" * 70)

    original_results = train_experiment(
        df,
        "Original",
    )

    for result in original_results:

        print(
            f"{result['model']}: "
            f"MAE={result['MAE']:.4f}, "
            f"RMSE={result['RMSE']:.4f}, "
            f"R²={result['R2']:.4f}"
        )

    # --------------------------------------------------------
    # Train cleaned model
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("EXPERIMENT 2: GEOGRAPHICALLY CLEANED DATASET")
    print("-" * 70)

    cleaned_results = train_experiment(
        cleaned_df,
        "Cleaned",
    )

    for result in cleaned_results:

        print(
            f"{result['model']}: "
            f"MAE={result['MAE']:.4f}, "
            f"RMSE={result['RMSE']:.4f}, "
            f"R²={result['R2']:.4f}"
        )

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    all_results = (
        original_results
        + cleaned_results
    )

    comparison = pd.DataFrame(
        all_results
    )

    comparison.to_csv(
        REPORTS_DIR
        / "geo_cleaning_model_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Calculate improvement
    # --------------------------------------------------------

    baseline_xgb = comparison[
        (
            comparison["dataset"]
            == "Original"
        )
        & (
            comparison["model"]
            == "Tuned XGBoost"
        )
    ].iloc[0]

    cleaned_xgb = comparison[
        (
            comparison["dataset"]
            == "Cleaned"
        )
        & (
            comparison["model"]
            == "Tuned XGBoost"
        )
    ].iloc[0]

    mae_change = (
        (
            cleaned_xgb["MAE"]
            - baseline_xgb["MAE"]
        )
        / baseline_xgb["MAE"]
        * 100
    )

    rmse_change = (
        (
            cleaned_xgb["RMSE"]
            - baseline_xgb["RMSE"]
        )
        / baseline_xgb["RMSE"]
        * 100
    )

    r2_change = (
        cleaned_xgb["R2"]
        - baseline_xgb["R2"]
    )

    print("\n" + "=" * 70)
    print("ORIGINAL vs CLEANED — TUNED XGBOOST")
    print("=" * 70)

    print(
        f"\nOriginal MAE : "
        f"{baseline_xgb['MAE']:.4f}"
    )

    print(
        f"Cleaned MAE  : "
        f"{cleaned_xgb['MAE']:.4f}"
    )

    print(
        f"MAE change   : "
        f"{mae_change:+.2f}%"
    )

    print(
        f"\nOriginal RMSE: "
        f"{baseline_xgb['RMSE']:.4f}"
    )

    print(
        f"Cleaned RMSE : "
        f"{cleaned_xgb['RMSE']:.4f}"
    )

    print(
        f"RMSE change  : "
        f"{rmse_change:+.2f}%"
    )

    print(
        f"\nOriginal R²  : "
        f"{baseline_xgb['R2']:.4f}"
    )

    print(
        f"Cleaned R²   : "
        f"{cleaned_xgb['R2']:.4f}"
    )

    print(
        f"R² change    : "
        f"{r2_change:+.4f}"
    )

    # --------------------------------------------------------
    # Final interpretation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if cleaned_xgb["MAE"] < baseline_xgb["MAE"]:

        print(
            "\nRESULT: Geographic cleaning improved "
            "test MAE."
        )

    elif cleaned_xgb["MAE"] > baseline_xgb["MAE"]:

        print(
            "\nRESULT: Geographic cleaning increased "
            "test MAE."
        )

    else:

        print(
            "\nRESULT: Geographic cleaning produced "
            "the same test MAE."
        )

    print(
        "\nThis experiment does not yet establish "
        "the final model."
    )

    print(
        "Temporal and delivery-person group validation "
        "must be repeated on the cleaned dataset."
    )

    print("\nSaved:")
    print(
        "  reports\\geo_cleaning_model_comparison.csv"
    )
    print(
        "  models\\eta_model_original.joblib"
    )
    print(
        "  models\\eta_model_cleaned.joblib"
    )

    print("\n" + "=" * 70)
    print("GEOGRAPHIC CLEANING MODEL EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()