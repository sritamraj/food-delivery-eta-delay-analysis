from pathlib import Path

import pandas as pd

from predict import predict_eta


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(exist_ok=True)


# ============================================================
# BASE ORDER
# ============================================================

BASE_ORDER = {
    "Delivery_person_Age": 30,
    "Delivery_person_Ratings": 4.7,
    "Restaurant_latitude": 22.5726,
    "Restaurant_longitude": 88.3639,
    "Delivery_location_latitude": 22.5958,
    "Delivery_location_longitude": 88.4000,
    "Order_Date": "10-03-2022",
    "Time_Orderd": "20:30",
    "Weatherconditions": "Cloudy",
    "Road_traffic_density": "Medium",
    "Vehicle_condition": 2,
    "Type_of_order": "Meal",
    "Type_of_vehicle": "motorcycle",
    "multiple_deliveries": 1,
    "Festival": "No",
    "City": "Metropolitian",
}


# ============================================================
# SCENARIOS
# ============================================================

SCENARIOS = [
    {
        "scenario": "Baseline",
        "description": (
            "Typical evening order with medium traffic"
        ),
        "changes": {},
    },
    {
        "scenario": "Heavy Traffic",
        "description": (
            "Same order under jam traffic"
        ),
        "changes": {
            "Road_traffic_density": "Jam",
        },
    },
    {
        "scenario": "Low Traffic",
        "description": (
            "Same order under low traffic"
        ),
        "changes": {
            "Road_traffic_density": "Low",
        },
    },
    {
        "scenario": "Sunny Weather",
        "description": (
            "Same order under sunny weather"
        ),
        "changes": {
            "Weatherconditions": "Sunny",
        },
    },
    {
        "scenario": "Stormy Weather",
        "description": (
            "Same order under stormy weather"
        ),
        "changes": {
            "Weatherconditions": "Stormy",
        },
    },
    {
        "scenario": "Multiple Deliveries",
        "description": (
            "Rider is handling three deliveries"
        ),
        "changes": {
            "multiple_deliveries": 3,
        },
    },
    {
        "scenario": "Festival",
        "description": (
            "Order occurs during a festival"
        ),
        "changes": {
            "Festival": "Yes",
        },
    },
    {
        "scenario": "Short Distance",
        "description": (
            "Shorter restaurant-to-customer distance"
        ),
        "changes": {
            "Delivery_location_latitude": 22.5800,
            "Delivery_location_longitude": 88.3750,
        },
    },
    {
        "scenario": "Longer Distance",
        "description": (
            "Longer restaurant-to-customer distance"
        ),
        "changes": {
            "Delivery_location_latitude": 22.7000,
            "Delivery_location_longitude": 88.5000,
        },
    },
    {
        "scenario": "Peak + Jam + Multiple",
        "description": (
            "Combined high-risk operational conditions"
        ),
        "changes": {
            "Road_traffic_density": "Jam",
            "multiple_deliveries": 3,
        },
    },
]


# ============================================================
# BUILD SCENARIO DATA
# ============================================================

def build_scenario(
    scenario,
):

    order = BASE_ORDER.copy()

    order.update(
        scenario["changes"]
    )

    order["scenario"] = (
        scenario["scenario"]
    )

    order["description"] = (
        scenario["description"]
    )

    return order


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "ETA MODEL — OPERATIONAL SCENARIO TEST"
    )
    print("=" * 75)

    scenario_orders = [
        build_scenario(
            scenario
        )
        for scenario in SCENARIOS
    ]

    input_df = pd.DataFrame(
        scenario_orders
    )

    prediction_input = input_df.drop(
        columns=[
            "scenario",
            "description",
        ]
    )

    results = predict_eta(
        prediction_input
    )

    results[
        "scenario"
    ] = input_df[
        "scenario"
    ].values

    results[
        "description"
    ] = input_df[
        "description"
    ].values

    # --------------------------------------------------------
    # Reorder useful columns
    # --------------------------------------------------------

    output_columns = [
        "scenario",
        "description",
        "Road_traffic_density",
        "Weatherconditions",
        "multiple_deliveries",
        "Festival",
        "distance_km",
        "predicted_eta_min",
        "predicted_long_delivery",
    ]

    results = results[
        output_columns
    ]

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print(
        "SCENARIO RESULTS"
    )
    print("=" * 75)

    print(
        results[
            [
                "scenario",
                "Road_traffic_density",
                "Weatherconditions",
                "multiple_deliveries",
                "Festival",
                "distance_km",
                "predicted_eta_min",
                "predicted_long_delivery",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Baseline comparison
    # --------------------------------------------------------

    baseline_eta = results.loc[
        results["scenario"]
        == "Baseline",
        "predicted_eta_min",
    ].iloc[0]

    results[
        "eta_change_vs_baseline"
    ] = (
        results[
            "predicted_eta_min"
        ]
        - baseline_eta
    )

    print("\n" + "=" * 75)
    print(
        "ETA CHANGE VS BASELINE"
    )
    print("=" * 75)

    print(
        results[
            [
                "scenario",
                "predicted_eta_min",
                "eta_change_vs_baseline",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Basic monotonicity checks
    # --------------------------------------------------------

    baseline = results.loc[
        results["scenario"]
        == "Baseline",
        "predicted_eta_min",
    ].iloc[0]

    heavy_traffic = results.loc[
        results["scenario"]
        == "Heavy Traffic",
        "predicted_eta_min",
    ].iloc[0]

    low_traffic = results.loc[
        results["scenario"]
        == "Low Traffic",
        "predicted_eta_min",
    ].iloc[0]

    multiple = results.loc[
        results["scenario"]
        == "Multiple Deliveries",
        "predicted_eta_min",
    ].iloc[0]

    longer_distance = results.loc[
        results["scenario"]
        == "Longer Distance",
        "predicted_eta_min",
    ].iloc[0]

    short_distance = results.loc[
        results["scenario"]
        == "Short Distance",
        "predicted_eta_min",
    ].iloc[0]

    print("\n" + "=" * 75)
    print(
        "SANITY CHECKS"
    )
    print("=" * 75)

    checks = {
        "Low traffic <= baseline":
            low_traffic <= baseline,

        "Heavy traffic >= low traffic":
            heavy_traffic >= low_traffic,

        "Multiple deliveries >= baseline":
            multiple >= baseline,

        "Longer distance >= short distance":
            longer_distance >= short_distance,
    }

    all_passed = True

    for name, passed in checks.items():

        status = (
            "PASS"
            if passed
            else "REVIEW"
        )

        print(
            f"{name}: {status}"
        )

        if not passed:
            all_passed = False

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_path = (
        REPORTS_DIR
        / "operational_scenario_predictions.csv"
    )

    results.to_csv(
        report_path,
        index=False,
    )

    print("\n" + "=" * 75)

    if all_passed:

        print(
            "ALL BASIC SANITY CHECKS PASSED"
        )

    else:

        print(
            "SOME SANITY CHECKS REQUIRE REVIEW"
        )

    print("=" * 75)

    print(
        f"\nSaved:"
    )

    print(
        f"  {report_path}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "These are model-behavior sanity checks, "
        "not causal-effect estimates."
    )


if __name__ == "__main__":
    main()