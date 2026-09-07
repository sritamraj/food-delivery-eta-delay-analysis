from pathlib import Path

import numpy as np
import pandas as pd

from data_prep import load_dataset


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(exist_ok=True)


# ============================================================
# CONSTANTS
# ============================================================

LATITUDE_COLUMNS = [
    "Restaurant_latitude",
    "Delivery_location_latitude",
]

LONGITUDE_COLUMNS = [
    "Restaurant_longitude",
    "Delivery_location_longitude",
]

DISTANCE_COLUMN = "distance_km"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("GEOGRAPHIC DATA QUALITY AUDIT")
    print("=" * 60)

    # --------------------------------------------------------
    # Load engineered dataset
    # --------------------------------------------------------

    df = load_dataset()

    print(
        f"\nTotal rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # Convert coordinates to numeric
    # --------------------------------------------------------

    coordinate_columns = (
        LATITUDE_COLUMNS
        + LONGITUDE_COLUMNS
    )

    for column in coordinate_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Coordinate summary
    # --------------------------------------------------------

    print("\nCoordinate ranges:")

    for column in coordinate_columns:

        print(
            f"\n{column}:"
        )

        print(
            f"  Min : "
            f"{df[column].min():.6f}"
        )

        print(
            f"  Max : "
            f"{df[column].max():.6f}"
        )

        print(
            f"  Mean: "
            f"{df[column].mean():.6f}"
        )

        print(
            f"  Missing: "
            f"{df[column].isna().sum():,}"
        )

    # --------------------------------------------------------
    # Basic geographic validity checks
    # --------------------------------------------------------

    invalid_latitude = (
        (df["Restaurant_latitude"].abs() > 90)
        | (df["Delivery_location_latitude"].abs() > 90)
    )

    invalid_longitude = (
        (df["Restaurant_longitude"].abs() > 180)
        | (df["Delivery_location_longitude"].abs() > 180)
    )

    invalid_coordinates = (
        invalid_latitude
        | invalid_longitude
    )

    print("\nBasic coordinate validity:")

    print(
        f"  Invalid latitude rows : "
        f"{invalid_latitude.sum():,}"
    )

    print(
        f"  Invalid longitude rows: "
        f"{invalid_longitude.sum():,}"
    )

    print(
        f"  Invalid coordinate rows: "
        f"{invalid_coordinates.sum():,}"
    )

    # --------------------------------------------------------
    # Distance statistics
    # --------------------------------------------------------

    distance = pd.to_numeric(
        df[DISTANCE_COLUMN],
        errors="coerce",
    )

    print("\nDistance statistics:")

    print(
        f"  Mean   : "
        f"{distance.mean():.4f} km"
    )

    print(
        f"  Median : "
        f"{distance.median():.4f} km"
    )

    print(
        f"  Std    : "
        f"{distance.std():.4f} km"
    )

    print(
        f"  Minimum: "
        f"{distance.min():.4f} km"
    )

    print(
        f"  Maximum: "
        f"{distance.max():.4f} km"
    )

    print(
        f"  >15 km: "
        f"{(distance > 15).sum():,}"
    )

    print(
        f"  >50 km: "
        f"{(distance > 50).sum():,}"
    )

    print(
        f"  >100 km: "
        f"{(distance > 100).sum():,}"
    )

    print(
        f"  >200 km: "
        f"{(distance > 200).sum():,}"
    )

    # --------------------------------------------------------
    # Distance quantiles
    # --------------------------------------------------------

    quantiles = distance.quantile(
        [
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            0.995,
            0.999,
        ]
    )

    print("\nDistance quantiles:")

    for percentile, value in quantiles.items():

        print(
            f"  {percentile * 100:6.1f}% : "
            f"{value:.4f} km"
        )

    # --------------------------------------------------------
    # Distance buckets
    # --------------------------------------------------------

    df["distance_bucket"] = pd.cut(
        distance,
        bins=[
            -np.inf,
            3,
            6,
            10,
            15,
            25,
            50,
            100,
            200,
            np.inf,
        ],
        labels=[
            "<=3 km",
            "3-6 km",
            "6-10 km",
            "10-15 km",
            "15-25 km",
            "25-50 km",
            "50-100 km",
            "100-200 km",
            ">200 km",
        ],
    )

    distance_distribution = (
        df.groupby(
            "distance_bucket",
            observed=False,
        )
        .agg(
            orders=(
                DISTANCE_COLUMN,
                "size",
            ),
            mean_distance_km=(
                DISTANCE_COLUMN,
                "mean",
            ),
            median_distance_km=(
                DISTANCE_COLUMN,
                "median",
            ),
        )
        .reset_index()
    )

    distance_distribution[
        "percentage"
    ] = (
        distance_distribution["orders"]
        / len(df)
        * 100
    )

    distance_distribution.to_csv(
        REPORTS_DIR
        / "geo_distance_distribution.csv",
        index=False,
    )

    print("\nDistance distribution:")
    print(
        distance_distribution.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Inspect extreme-distance records
    # --------------------------------------------------------

    extreme_columns = [
        "ID",
        "Delivery_person_ID",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
        DISTANCE_COLUMN,
        "Order_Date",
        "Time_taken(min)",
        "Road_traffic_density",
        "Weatherconditions",
        "multiple_deliveries",
        "City",
    ]

    extreme_columns = [
        column
        for column in extreme_columns
        if column in df.columns
    ]

    extreme_records = (
        df[
            distance > 50
        ][extreme_columns]
        .sort_values(
            DISTANCE_COLUMN,
            ascending=False,
        )
        .head(200)
    )

    extreme_records.to_csv(
        REPORTS_DIR
        / "geo_extreme_distance_records.csv",
        index=False,
    )

    print(
        "\nTop extreme-distance records:"
    )

    if len(extreme_records) == 0:

        print(
            "  No records above 50 km."
        )

    else:

        print(
            extreme_records.head(20)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # Compare normal vs extreme-distance deliveries
    # --------------------------------------------------------

    df["distance_group"] = np.select(
        [
            distance <= 15,
            (distance > 15)
            & (distance <= 50),
            distance > 50,
        ],
        [
            "0-15 km",
            "15-50 km",
            ">50 km",
        ],
        default="Unknown",
    )

    distance_impact = (
        df.groupby(
            "distance_group",
            dropna=False,
        )
        .agg(
            orders=(
                "Time_taken(min)",
                "size",
            ),
            mean_distance_km=(
                DISTANCE_COLUMN,
                "mean",
            ),
            median_distance_km=(
                DISTANCE_COLUMN,
                "median",
            ),
            mean_delivery_minutes=(
                "Time_taken(min)",
                "mean",
            ),
        )
        .reset_index()
    )

    distance_impact[
        "percentage"
    ] = (
        distance_impact["orders"]
        / len(df)
        * 100
    )

    distance_impact.to_csv(
        REPORTS_DIR
        / "geo_distance_impact.csv",
        index=False,
    )

    print(
        "\nDistance impact:"
    )

    print(
        distance_impact.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Check coordinate-pair differences
    # --------------------------------------------------------

    df["latitude_difference"] = np.abs(
        df["Restaurant_latitude"]
        - df["Delivery_location_latitude"]
    )

    df["longitude_difference"] = np.abs(
        df["Restaurant_longitude"]
        - df["Delivery_location_longitude"]
    )

    coordinate_difference = (
        df[
            [
                "latitude_difference",
                "longitude_difference",
            ]
        ]
        .describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .transpose()
    )

    coordinate_difference.to_csv(
        REPORTS_DIR
        / "geo_coordinate_difference.csv"
    )

    print(
        "\nCoordinate-pair differences:"
    )

    print(
        coordinate_difference.to_string()
    )

    # --------------------------------------------------------
    # Potential outlier flag
    #
    # We do NOT automatically delete these rows.
    # This audit only identifies them.
    # --------------------------------------------------------

    df["potential_geo_outlier"] = (
        distance > 50
    )

    outlier_count = (
        df["potential_geo_outlier"]
        .sum()
    )

    outlier_percentage = (
        outlier_count
        / len(df)
        * 100
    )

    print(
        "\nPotential geographic outliers:"
    )

    print(
        f"  Distance >50 km: "
        f"{outlier_count:,} rows "
        f"({outlier_percentage:.2f}%)"
    )

    # --------------------------------------------------------
    # Save audit summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "total_rows": len(df),
                "invalid_coordinate_rows":
                    int(
                        invalid_coordinates.sum()
                    ),
                "mean_distance_km":
                    distance.mean(),
                "median_distance_km":
                    distance.median(),
                "max_distance_km":
                    distance.max(),
                "distance_over_15km":
                    int(
                        (distance > 15).sum()
                    ),
                "distance_over_50km":
                    int(
                        (distance > 50).sum()
                    ),
                "distance_over_100km":
                    int(
                        (distance > 100).sum()
                    ),
                "distance_over_200km":
                    int(
                        (distance > 200).sum()
                    ),
                "potential_outlier_percentage":
                    outlier_percentage,
            }
        ]
    )

    summary.to_csv(
        REPORTS_DIR
        / "geo_audit_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Final methodology note
    # --------------------------------------------------------

    print("\nMethodology note:")
    print(
        "  This audit identifies suspicious geographic "
        "records but does NOT automatically remove them."
    )

    print(
        "  Any filtering decision will be made only "
        "after inspecting the resulting records."
    )

    print("\nSaved geographic audit reports:")

    print(
        "  reports\\geo_audit_summary.csv"
    )

    print(
        "  reports\\geo_distance_distribution.csv"
    )

    print(
        "  reports\\geo_extreme_distance_records.csv"
    )

    print(
        "  reports\\geo_distance_impact.csv"
    )

    print(
        "  reports\\geo_coordinate_difference.csv"
    )

    print("\n" + "=" * 60)
    print("GEOGRAPHIC DATA QUALITY AUDIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()