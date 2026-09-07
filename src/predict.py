from pathlib import Path

import numpy as np
import pandas as pd
import joblib


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "eta_model_final_tail_aware.joblib"
)


# ============================================================
# MODEL FEATURES
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


# ============================================================
# GEOGRAPHIC FEATURE
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Calculate great-circle distance
    between two coordinate pairs in km.
    """

    earth_radius_km = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    c = 2 * np.arcsin(
        np.sqrt(a)
    )

    return earth_radius_km * c


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input(data):

    required_columns = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
        "Order_Date",
        "Time_Orderd",
        "Weatherconditions",
        "Road_traffic_density",
        "Vehicle_condition",
        "Type_of_order",
        "Type_of_vehicle",
        "multiple_deliveries",
        "Festival",
        "City",
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:

        raise ValueError(
            "Missing required input columns: "
            + ", ".join(missing)
        )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def prepare_features(data):

    df = data.copy()

    validate_input(df)

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
        "Vehicle_condition",
        "multiple_deliveries",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Geographic distance
    # --------------------------------------------------------

    df["distance_km"] = haversine_distance(
        df["Restaurant_latitude"],
        df["Restaurant_longitude"],
        df["Delivery_location_latitude"],
        df["Delivery_location_longitude"],
    )

    # --------------------------------------------------------
    # Order hour
    # --------------------------------------------------------

    order_time = pd.to_datetime(
        df["Time_Orderd"]
        .astype(str)
        .str.strip(),
        format="%H:%M",
        errors="coerce",
    )

    df["order_hour"] = (
        order_time.dt.hour
    )

    # --------------------------------------------------------
    # Peak hour
    # --------------------------------------------------------

    df["is_peak_hour"] = (
        df["order_hour"]
        .isin(
            [
                12,
                13,
                14,
                19,
                20,
                21,
                22,
            ]
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    order_date = pd.to_datetime(
        df["Order_Date"],
        errors="coerce",
        dayfirst=True,
    )

    df["order_dayofweek"] = (
        order_date.dt.dayofweek
    )

    df["is_weekend"] = (
        df["order_dayofweek"]
        >= 5
    ).astype(int)

    # --------------------------------------------------------
    # Return model features
    # --------------------------------------------------------

    X = df[
        FEATURES
    ].copy()

    return X


# ============================================================
# PREDICTION
# ============================================================

def predict_eta(data):

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    X = prepare_features(
        data
    )

    predictions = model.predict(
        X
    )

    predictions = np.maximum(
        predictions,
        0,
    )

    result = data.copy()

    result[
        "predicted_eta_min"
    ] = np.round(
        predictions,
        2,
    )

    result[
        "predicted_long_delivery"
    ] = (
        result[
            "predicted_eta_min"
        ]
        >= 45
    )

    # Add engineered distance so the
    # caller can inspect it.
    result[
        "distance_km"
    ] = X[
        "distance_km"
    ].values

    return result


# ============================================================
# DEMO ORDER
# ============================================================

def create_demo_order():

    return pd.DataFrame(
        [
            {
                "Delivery_person_Age": 30,
                "Delivery_person_Ratings": 4.7,
                "Restaurant_latitude": 22.5726,
                "Restaurant_longitude": 88.3639,
                "Delivery_location_latitude": 22.5958,
                "Delivery_location_longitude": 88.4000,
                "Order_Date": "10-03-2022",
                "Time_Orderd": "20:30",
                "Weatherconditions": "Cloudy",
                "Road_traffic_density": "Jam",
                "Vehicle_condition": 2,
                "Type_of_order": "Meal",
                "Type_of_vehicle": "motorcycle",
                "multiple_deliveries": 1,
                "Festival": "No",
                "City": "Metropolitian",
            }
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "FOOD DELIVERY ETA PREDICTION"
    )
    print("=" * 70)

    print(
        f"\nLoading model:"
    )

    print(
        f"  {MODEL_PATH}"
    )

    demo_order = create_demo_order()

    prediction = predict_eta(
        demo_order
    )

    row = prediction.iloc[0]

    print("\n" + "=" * 70)
    print(
        "DEMO PREDICTION"
    )
    print("=" * 70)

    print(
        f"\nPredicted ETA: "
        f"{row['predicted_eta_min']:.2f} minutes"
    )

    if row[
        "predicted_long_delivery"
    ]:

        print(
            "Operational flag: "
            "PREDICTED LONG DELIVERY"
        )

    else:

        print(
            "Operational flag: "
            "NORMAL PREDICTED DELIVERY"
        )

    print("\nInput summary:")

    print(
        f"  Traffic: "
        f"{row['Road_traffic_density']}"
    )

    print(
        f"  Weather: "
        f"{row['Weatherconditions']}"
    )

    print(
        f"  Multiple deliveries: "
        f"{row['multiple_deliveries']}"
    )

    print(
        f"  Vehicle: "
        f"{row['Type_of_vehicle']}"
    )

    print(
        f"  Distance: "
        f"{row['distance_km']:.2f} km"
    )

    print("\n" + "=" * 70)
    print(
        "PREDICTION COMPLETE"
    )
    print("=" * 70)

    print(
        "\nNote:"
    )

    print(
        "The 45-minute flag represents a "
        "predicted long delivery, not a "
        "promised-ETA lateness classification."
    )


if __name__ == "__main__":
    main()