from pathlib import Path

import pandas as pd

from data_prep import TARGET, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# A delivery at or above this value is treated as a "long delivery".
LONG_DELIVERY_THRESHOLD = 45


def analyze_numeric_relationship(df):
    result = (
        df.groupby("distance_bucket", observed=True)[TARGET]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )

    return result


def group_analysis(df, column):
    """
    Calculate delivery-time statistics for a categorical factor.
    """

    result = (
        df.groupby(column, dropna=False)[TARGET]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
        )
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    return result


def main():

    print("=" * 70)
    print("FOOD DELIVERY DELAY ANALYSIS")
    print("=" * 70)

    df = load_dataset()

    print(f"\nRows analyzed: {len(df):,}")

    # ---------------------------------------------------------
    # Long-delivery label
    # ---------------------------------------------------------

    df["is_long_delivery"] = (
        df[TARGET] >= LONG_DELIVERY_THRESHOLD
    ).astype(int)

    long_rate = df["is_long_delivery"].mean()

    print(
        f"\nLong-delivery threshold: "
        f"{LONG_DELIVERY_THRESHOLD} minutes"
    )

    print(
        f"Long deliveries: "
        f"{df['is_long_delivery'].sum():,} "
        f"({long_rate:.2%})"
    )

    # ---------------------------------------------------------
    # Distance buckets
    # ---------------------------------------------------------

    df["distance_bucket"] = pd.cut(
        df["distance_km"],
        bins=[
            -float("inf"),
            3,
            6,
            10,
            15,
            float("inf"),
        ],
        labels=[
            "<=3 km",
            "3-6 km",
            "6-10 km",
            "10-15 km",
            ">15 km",
        ],
    )

    distance_analysis = analyze_numeric_relationship(df)

    distance_analysis.to_csv(
        REPORT_DIR / "delay_by_distance.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Categorical analyses
    # ---------------------------------------------------------

    factors = [
        "Road_traffic_density",
        "Weatherconditions",
        "Type_of_vehicle",
        "Vehicle_condition",
        "multiple_deliveries",
        "Festival",
        "City",
        "is_peak_hour",
    ]

    for factor in factors:

        if factor not in df.columns:
            continue

        result = group_analysis(
            df,
            factor,
        )

        filename = (
            f"delay_by_"
            f"{factor.lower()}.csv"
        )

        result.to_csv(
            REPORT_DIR / filename,
            index=False,
        )

        print(
            f"\n--- {factor} ---"
        )

        print(
            result.to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Long-delivery rate by factor
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("LONG-DELIVERY RISK ANALYSIS")
    print("=" * 70)

    risk_factors = [
        "Road_traffic_density",
        "Weatherconditions",
        "Type_of_vehicle",
        "Vehicle_condition",
        "multiple_deliveries",
        "Festival",
        "City",
        "is_peak_hour",
    ]

    for factor in risk_factors:

        if factor not in df.columns:
            continue

        risk = (
            df.groupby(factor, dropna=False)[
                "is_long_delivery"
            ]
            .agg(
                orders="count",
                long_deliveries="sum",
                long_delivery_rate="mean",
            )
            .reset_index()
            .sort_values(
                "long_delivery_rate",
                ascending=False,
            )
        )

        filename = (
            f"long_delivery_risk_"
            f"{factor.lower()}.csv"
        )

        risk.to_csv(
            REPORT_DIR / filename,
            index=False,
        )

        print(
            f"\n--- Long-delivery risk: {factor} ---"
        )

        print(
            risk.to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Peak-hour analysis
    # ---------------------------------------------------------

    peak_analysis = (
        df.groupby("is_peak_hour")[TARGET]
        .agg(
            orders="count",
            mean_minutes="mean",
            median_minutes="median",
            long_delivery_rate=(
                "mean"
            ),
        )
        .reset_index()
    )

    # Correct long-delivery rate
    peak_risk = (
        df.groupby("is_peak_hour")[
            "is_long_delivery"
        ]
        .mean()
        .reset_index(
            name="long_delivery_rate"
        )
    )

    peak_analysis = peak_analysis.drop(
        columns=["long_delivery_rate"]
    ).merge(
        peak_risk,
        on="is_peak_hour",
    )

    peak_analysis.to_csv(
        REPORT_DIR / "peak_hour_analysis.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Important methodological note
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("METHODOLOGY NOTE")
    print("=" * 70)

    print(
        "\nThe dataset does not contain a promised ETA "
        "or promised delivery deadline."
    )

    print(
        "Therefore, 'long delivery' is defined using "
        f"an observed delivery-time threshold of "
        f"{LONG_DELIVERY_THRESHOLD} minutes."
    )

    print(
        "This should NOT be described as true "
        "'late delivery' classification."
    )

    print("\nAnalysis reports saved to:")
    print(REPORT_DIR)

    print("\nDELAY ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()