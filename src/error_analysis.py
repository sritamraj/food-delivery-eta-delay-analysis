from pathlib import Path

import numpy as np
import pandas as pd

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

PREDICTIONS_FILE = (
    REPORTS_DIR
    / "cleaned_group_predictions.csv"
)


# ============================================================
# HELPERS
# ============================================================

def print_section(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def summarize_error(
    df,
    group_column,
):

    summary = (
        df.groupby(
            group_column,
            dropna=False,
        )
        .agg(
            orders=(TARGET, "size"),
            actual_mean=(TARGET, "mean"),
            predicted_mean=("predicted", "mean"),
            mae=("absolute_error", "mean"),
            signed_error=("error", "mean"),
        )
        .reset_index()
    )

    summary["long_delivery_rate"] = (
        df.groupby(
            group_column,
            dropna=False,
        )["is_long_delivery"]
        .mean()
        .values
    )

    return summary.sort_values(
        "mae",
        ascending=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_section(
        "LONG-DELIVERY ERROR ANALYSIS"
    )

    # --------------------------------------------------------
    # Load cleaned group predictions
    # --------------------------------------------------------

    if not PREDICTIONS_FILE.exists():

        raise FileNotFoundError(
            "Missing cleaned group predictions: "
            f"{PREDICTIONS_FILE}"
        )

    predictions = pd.read_csv(
        PREDICTIONS_FILE
    )

    print(
        f"Prediction rows: "
        f"{len(predictions):,}"
    )

    # --------------------------------------------------------
    # Load original cleaned dataset
    # --------------------------------------------------------

    data = load_dataset()

    # Recreate the same geographic cleaning
    distance = pd.to_numeric(
        data["distance_km"],
        errors="coerce",
    )

    latitude_difference = (
        data["Restaurant_latitude"]
        - data["Delivery_location_latitude"]
    ).abs()

    longitude_difference = (
        data["Restaurant_longitude"]
        - data["Delivery_location_longitude"]
    ).abs()

    geographic_outlier = (
        (distance > 50)
        | (latitude_difference > 5)
        | (longitude_difference > 5)
    )

    data = data.loc[
        ~geographic_outlier
    ].copy()

    # --------------------------------------------------------
    # Merge prediction metadata
    # --------------------------------------------------------

    merge_columns = [
        "ID",
        "Delivery_person_ID",
        "Order_Date",
        TARGET,
        "predicted",
        "error",
        "absolute_error",
        "is_long_delivery",
    ]

    prediction_columns = [
        c
        for c in merge_columns
        if c in predictions.columns
    ]

    pred = predictions[
        prediction_columns
    ].copy()

    # --------------------------------------------------------
    # Attach engineered/raw features
    # --------------------------------------------------------

    feature_columns = [
        "ID",
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

    available_features = [
        c
        for c in feature_columns
        if c in data.columns
    ]

    feature_data = data[
        available_features
    ].copy()

    df = pred.merge(
        feature_data,
        on="ID",
        how="left",
        suffixes=("", "_data"),
    )

    # --------------------------------------------------------
    # Recreate long-delivery flag
    # --------------------------------------------------------

    df["is_long_delivery"] = (
        df[TARGET]
        >= LONG_DELIVERY_THRESHOLD
    )

    # --------------------------------------------------------
    # Overall performance
    # --------------------------------------------------------

    print_section(
        "OVERALL GROUP-VALIDATION ERROR"
    )

    overall_mae = df[
        "absolute_error"
    ].mean()

    overall_rmse = np.sqrt(
        np.mean(
            df["error"] ** 2
        )
    )

    overall_bias = df[
        "error"
    ].mean()

    print(
        f"MAE : {overall_mae:.4f}"
    )

    print(
        f"RMSE: {overall_rmse:.4f}"
    )

    print(
        f"Bias: {overall_bias:+.4f}"
    )

    # --------------------------------------------------------
    # Long vs normal
    # --------------------------------------------------------

    print_section(
        "LONG VS NORMAL DELIVERY ERROR"
    )

    long_summary = (
        df.groupby(
            "is_long_delivery"
        )
        .agg(
            orders=(TARGET, "size"),
            actual_mean=(TARGET, "mean"),
            predicted_mean=("predicted", "mean"),
            mae=("absolute_error", "mean"),
            rmse=(
                "error",
                lambda x: np.sqrt(
                    np.mean(x ** 2)
                ),
            ),
            signed_error=("error", "mean"),
        )
        .reset_index()
    )

    print(
        long_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Delivery-time buckets
    # --------------------------------------------------------

    print_section(
        "ERROR BY ACTUAL DELIVERY TIME"
    )

    bins = [
        0,
        15,
        25,
        35,
        45,
        np.inf,
    ]

    labels = [
        "<=15",
        "16-25",
        "26-35",
        "36-45",
        ">45",
    ]

    df["delivery_time_bucket"] = pd.cut(
        df[TARGET],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    bucket_summary = summarize_error(
        df,
        "delivery_time_bucket",
    )

    print(
        bucket_summary.to_string(
            index=False
        )
    )

    bucket_summary.to_csv(
        REPORTS_DIR
        / "cleaned_group_error_by_delivery_time.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Traffic
    # --------------------------------------------------------

    print_section(
        "LONG-TAIL ERROR BY TRAFFIC"
    )

    traffic_summary = summarize_error(
        df,
        "Road_traffic_density",
    )

    print(
        traffic_summary.to_string(
            index=False
        )
    )

    traffic_summary.to_csv(
        REPORTS_DIR
        / "cleaned_group_error_by_traffic.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    print_section(
        "LONG-TAIL ERROR BY WEATHER"
    )

    weather_summary = summarize_error(
        df,
        "Weatherconditions",
    )

    print(
        weather_summary.to_string(
            index=False
        )
    )

    weather_summary.to_csv(
        REPORTS_DIR
        / "cleaned_group_error_by_weather.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Multiple deliveries
    # --------------------------------------------------------

    print_section(
        "LONG-TAIL ERROR BY MULTIPLE DELIVERIES"
    )

    multiple_summary = summarize_error(
        df,
        "multiple_deliveries",
    )

    print(
        multiple_summary.to_string(
            index=False
        )
    )

    multiple_summary.to_csv(
        REPORTS_DIR
        / "cleaned_group_error_by_multiple_deliveries.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Festival
    # --------------------------------------------------------

    print_section(
        "LONG-TAIL ERROR BY FESTIVAL"
    )

    festival_summary = summarize_error(
        df,
        "Festival",
    )

    print(
        festival_summary.to_string(
            index=False
        )
    )

    festival_summary.to_csv(
        REPORTS_DIR
        / "cleaned_group_error_by_festival.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Vehicle
    # --------------------------------------------------------

    print_section(
        "LONG-TAIL ERROR BY VEHICLE"
    )

    vehicle_summary = summarize_error(
        df,
        "Type_of_vehicle",
    )

    print(
        vehicle_summary.to_string(
            index=False
        )
    )

    vehicle_summary.to_csv(
        REPORTS_DIR
        / "cleaned_group_error_by_vehicle.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Vehicle condition
    # --------------------------------------------------------

    print_section(
        "LONG-TAIL ERROR BY VEHICLE CONDITION"
    )

    vehicle_condition_summary = summarize_error(
        df,
        "Vehicle_condition",
    )

    print(
        vehicle_condition_summary.to_string(
            index=False
        )
    )

    vehicle_condition_summary.to_csv(
        REPORTS_DIR
        / "cleaned_group_error_by_vehicle_condition.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Distance
    # --------------------------------------------------------

    print_section(
        "LONG-TAIL ERROR BY DISTANCE"
    )

    distance_bins = [
        0,
        3,
        6,
        10,
        15,
        21,
    ]

    distance_labels = [
        "<=3 km",
        "3-6 km",
        "6-10 km",
        "10-15 km",
        ">15 km",
    ]

    df["distance_bucket"] = pd.cut(
        df["distance_km"],
        bins=distance_bins,
        labels=distance_labels,
        include_lowest=True,
    )

    distance_summary = summarize_error(
        df,
        "distance_bucket",
    )

    print(
        distance_summary.to_string(
            index=False
        )
    )

    distance_summary.to_csv(
        REPORTS_DIR
        / "cleaned_group_error_by_distance.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Peak hour
    # --------------------------------------------------------

    print_section(
        "LONG-TAIL ERROR BY PEAK HOUR"
    )

    peak_summary = summarize_error(
        df,
        "is_peak_hour",
    )

    print(
        peak_summary.to_string(
            index=False
        )
    )

    peak_summary.to_csv(
        REPORTS_DIR
        / "cleaned_group_error_by_peak_hour.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Find the most underpredicted long deliveries
    # --------------------------------------------------------

    print_section(
        "WORST UNDERPREDICTIONS — LONG DELIVERIES"
    )

    long_df = df[
        df["is_long_delivery"]
    ].copy()

    worst_long = (
        long_df.sort_values(
            "error",
            ascending=True,
        )
        .head(100)
    )

    worst_columns = [
        "ID",
        "Delivery_person_ID",
        TARGET,
        "predicted",
        "error",
        "absolute_error",
        "distance_km",
        "Road_traffic_density",
        "Weatherconditions",
        "multiple_deliveries",
        "Festival",
        "Type_of_vehicle",
        "Vehicle_condition",
        "City",
        "order_hour",
        "is_peak_hour",
    ]

    worst_columns = [
        c
        for c in worst_columns
        if c in worst_long.columns
    ]

    worst_long[
        worst_columns
    ].to_csv(
        REPORTS_DIR
        / "cleaned_group_worst_long_predictions.csv",
        index=False,
    )

    print(
        worst_long[
            worst_columns
        ].head(20).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Identify systematic underprediction
    # --------------------------------------------------------

    print_section(
        "LONG-TAIL BIAS DIAGNOSTIC"
    )

    long_bias = (
        long_df["error"]
        .mean()
    )

    long_actual = (
        long_df[TARGET]
        .mean()
    )

    long_predicted = (
        long_df["predicted"]
        .mean()
    )

    print(
        f"Long-delivery orders: "
        f"{len(long_df):,}"
    )

    print(
        f"Actual mean: "
        f"{long_actual:.4f}"
    )

    print(
        f"Predicted mean: "
        f"{long_predicted:.4f}"
    )

    print(
        f"Mean signed error: "
        f"{long_bias:+.4f}"
    )

    if long_bias < 0:

        print(
            "\nDIAGNOSTIC: "
            "The model systematically "
            "underpredicts long deliveries."
        )

    else:

        print(
            "\nDIAGNOSTIC: "
            "No overall long-delivery "
            "underprediction detected."
        )

    # --------------------------------------------------------
    # High-risk combination analysis
    # --------------------------------------------------------

    print_section(
        "HIGH-RISK COMBINATION ANALYSIS"
    )

    combination = (
        df.groupby(
            [
                "Road_traffic_density",
                "multiple_deliveries",
                "is_peak_hour",
            ],
            dropna=False,
        )
        .agg(
            orders=(TARGET, "size"),
            actual_mean=(TARGET, "mean"),
            predicted_mean=("predicted", "mean"),
            mae=("absolute_error", "mean"),
            signed_error=("error", "mean"),
            long_delivery_rate=(
                "is_long_delivery",
                "mean",
            ),
        )
        .reset_index()
    )

    combination = combination.sort_values(
        [
            "long_delivery_rate",
            "orders",
        ],
        ascending=[False, False],
    )

    print(
        combination.head(20).to_string(
            index=False
        )
    )

    combination.to_csv(
        REPORTS_DIR
        / "cleaned_group_high_risk_combinations.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print_section(
        "ERROR ANALYSIS COMPLETE"
    )

    print(
        "Reports saved to:"
    )

    print(
        "  reports\\cleaned_group_error_by_delivery_time.csv"
    )

    print(
        "  reports\\cleaned_group_error_by_traffic.csv"
    )

    print(
        "  reports\\cleaned_group_error_by_weather.csv"
    )

    print(
        "  reports\\cleaned_group_error_by_multiple_deliveries.csv"
    )

    print(
        "  reports\\cleaned_group_error_by_distance.csv"
    )

    print(
        "  reports\\cleaned_group_worst_long_predictions.csv"
    )

    print(
        "  reports\\cleaned_group_high_risk_combinations.csv"
    )


if __name__ == "__main__":
    main()