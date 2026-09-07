from pathlib import Path

import numpy as np
import pandas as pd

from data_prep import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ============================================================
# GEOGRAPHIC CLEANING EXPERIMENT
# ============================================================
#
# We do NOT blindly delete records.
#
# The audit found 431 records with distances >50 km.
# These records also contain extreme coordinate differences.
#
# This script:
#   1. Creates a geographic-quality flag.
#   2. Compares original vs cleaned data.
#   3. Saves the suspicious records.
#   4. Tests whether removing the suspicious geographic
#      records changes the target distribution.
#
# IMPORTANT:
# This is an experiment. It does not overwrite the original
# dataset.
# ============================================================


def main():

    print("=" * 70)
    print("GEOGRAPHIC CLEANING EXPERIMENT")
    print("=" * 70)

    df = load_dataset()

    print(f"\nOriginal rows: {len(df):,}")

    # --------------------------------------------------------
    # Coordinate columns
    # --------------------------------------------------------

    coordinate_columns = [
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    ]

    for column in coordinate_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    distance = pd.to_numeric(
        df["distance_km"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Geographic anomaly signals
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Primary cleaning rule
    # --------------------------------------------------------
    #
    # A distance above 50 km is treated as suspicious.
    #
    # Why 50 km?
    #
    # The audit showed:
    #   99th percentile ≈ 20.97 km
    #   99.5th percentile ≈ 5888 km
    #
    # There is an enormous discontinuity.
    #
    # We therefore use >50 km as a conservative anomaly
    # threshold rather than aggressively trimming normal
    # 15-25 km deliveries.
    # --------------------------------------------------------

    geo_outlier = distance > 50

    # --------------------------------------------------------
    # Additional coordinate anomaly flags
    # --------------------------------------------------------

    extreme_latitude_difference = (
        latitude_difference > 5
    )

    extreme_longitude_difference = (
        longitude_difference > 5
    )

    # --------------------------------------------------------
    # Combined geographic-quality flag
    # --------------------------------------------------------

    df["geo_outlier"] = geo_outlier

    df["extreme_latitude_difference"] = (
        extreme_latitude_difference
    )

    df["extreme_longitude_difference"] = (
        extreme_longitude_difference
    )

    # --------------------------------------------------------
    # Print counts
    # --------------------------------------------------------

    print("\nGeographic anomaly counts:")

    print(
        f"  Distance >50 km: "
        f"{geo_outlier.sum():,}"
    )

    print(
        f"  Latitude difference >5°: "
        f"{extreme_latitude_difference.sum():,}"
    )

    print(
        f"  Longitude difference >5°: "
        f"{extreme_longitude_difference.sum():,}"
    )

    combined_anomaly = (
        geo_outlier
        | extreme_latitude_difference
        | extreme_longitude_difference
    )

    print(
        f"  Combined geographic anomalies: "
        f"{combined_anomaly.sum():,}"
    )

    # --------------------------------------------------------
    # Save suspicious records
    # --------------------------------------------------------

    suspicious_columns = [
        "ID",
        "Delivery_person_ID",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
        "distance_km",
        "Order_Date",
        "Time_taken(min)",
        "Road_traffic_density",
        "Weatherconditions",
        "multiple_deliveries",
        "City",
        "geo_outlier",
        "extreme_latitude_difference",
        "extreme_longitude_difference",
    ]

    suspicious_columns = [
        column
        for column in suspicious_columns
        if column in df.columns
    ]

    suspicious = (
        df.loc[
            combined_anomaly,
            suspicious_columns,
        ]
        .sort_values(
            "distance_km",
            ascending=False,
        )
    )

    suspicious.to_csv(
        REPORTS_DIR
        / "geo_cleaning_candidates.csv",
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        "  reports\\geo_cleaning_candidates.csv"
    )

    # --------------------------------------------------------
    # Original dataset statistics
    # --------------------------------------------------------

    original_stats = {
        "rows": len(df),
        "mean_distance_km": distance.mean(),
        "median_distance_km": distance.median(),
        "p95_distance_km": distance.quantile(0.95),
        "p99_distance_km": distance.quantile(0.99),
        "max_distance_km": distance.max(),
        "mean_delivery_minutes":
            df["Time_taken(min)"].mean(),
        "median_delivery_minutes":
            df["Time_taken(min)"].median(),
    }

    # --------------------------------------------------------
    # Clean experiment
    # --------------------------------------------------------
    #
    # Remove ONLY records flagged as geographic anomalies.
    #
    # The original dataframe is not modified.
    # --------------------------------------------------------

    clean_df = df.loc[
        ~combined_anomaly
    ].copy()

    clean_distance = clean_df[
        "distance_km"
    ]

    clean_stats = {
        "rows": len(clean_df),
        "mean_distance_km": clean_distance.mean(),
        "median_distance_km": clean_distance.median(),
        "p95_distance_km":
            clean_distance.quantile(0.95),
        "p99_distance_km":
            clean_distance.quantile(0.99),
        "max_distance_km":
            clean_distance.max(),
        "mean_delivery_minutes":
            clean_df["Time_taken(min)"].mean(),
        "median_delivery_minutes":
            clean_df["Time_taken(min)"].median(),
    }

    # --------------------------------------------------------
    # Comparison table
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "metric": "Rows",
                "original": original_stats["rows"],
                "cleaned": clean_stats["rows"],
            },
            {
                "metric": "Mean distance (km)",
                "original":
                    original_stats[
                        "mean_distance_km"
                    ],
                "cleaned":
                    clean_stats[
                        "mean_distance_km"
                    ],
            },
            {
                "metric": "Median distance (km)",
                "original":
                    original_stats[
                        "median_distance_km"
                    ],
                "cleaned":
                    clean_stats[
                        "median_distance_km"
                    ],
            },
            {
                "metric": "95th percentile distance (km)",
                "original":
                    original_stats[
                        "p95_distance_km"
                    ],
                "cleaned":
                    clean_stats[
                        "p95_distance_km"
                    ],
            },
            {
                "metric": "99th percentile distance (km)",
                "original":
                    original_stats[
                        "p99_distance_km"
                    ],
                "cleaned":
                    clean_stats[
                        "p99_distance_km"
                    ],
            },
            {
                "metric": "Maximum distance (km)",
                "original":
                    original_stats[
                        "max_distance_km"
                    ],
                "cleaned":
                    clean_stats[
                        "max_distance_km"
                    ],
            },
            {
                "metric": "Mean delivery time (min)",
                "original":
                    original_stats[
                        "mean_delivery_minutes"
                    ],
                "cleaned":
                    clean_stats[
                        "mean_delivery_minutes"
                    ],
            },
            {
                "metric": "Median delivery time (min)",
                "original":
                    original_stats[
                        "median_delivery_minutes"
                    ],
                "cleaned":
                    clean_stats[
                        "median_delivery_minutes"
                    ],
            },
        ]
    )

    comparison.to_csv(
        REPORTS_DIR
        / "geo_cleaning_comparison.csv",
        index=False,
    )

    print(
        "\nOriginal vs cleaned:"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Delivery-time comparison for removed records
    # --------------------------------------------------------

    removed_df = df.loc[
        combined_anomaly
    ].copy()

    if len(removed_df) > 0:

        removed_delivery_summary = pd.DataFrame(
            [
                {
                    "group": "Original dataset",
                    "rows": len(df),
                    "mean_delivery_minutes":
                        df["Time_taken(min)"].mean(),
                    "median_delivery_minutes":
                        df["Time_taken(min)"].median(),
                },
                {
                    "group": "Geographic anomalies",
                    "rows": len(removed_df),
                    "mean_delivery_minutes":
                        removed_df[
                            "Time_taken(min)"
                        ].mean(),
                    "median_delivery_minutes":
                        removed_df[
                            "Time_taken(min)"
                        ].median(),
                },
                {
                    "group": "Cleaned dataset",
                    "rows": len(clean_df),
                    "mean_delivery_minutes":
                        clean_df[
                            "Time_taken(min)"
                        ].mean(),
                    "median_delivery_minutes":
                        clean_df[
                            "Time_taken(min)"
                        ].median(),
                },
            ]
        )

        removed_delivery_summary.to_csv(
            REPORTS_DIR
            / "geo_removed_delivery_summary.csv",
            index=False,
        )

        print(
            "\nDelivery-time comparison:"
        )

        print(
            removed_delivery_summary.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Geographic anomaly rate
    # --------------------------------------------------------

    anomaly_rate = (
        combined_anomaly.mean()
        * 100
    )

    print(
        f"\nGeographic anomaly rate: "
        f"{anomaly_rate:.2f}%"
    )

    # --------------------------------------------------------
    # Important conclusion
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIMENT CONCLUSION")
    print("=" * 70)

    print(
        "\nThe original dataset has NOT been modified."
    )

    print(
        "The experiment only identifies and evaluates "
        "geographic anomalies."
    )

    print(
        "\nNext step:"
    )

    print(
        "Retrain the ETA model using the cleaned geographic "
        "data and compare random, temporal, and group validation."
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()