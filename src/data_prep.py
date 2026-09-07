import re
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "Time_taken(min)"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Food delivery.csv"


def clean_numeric(value):
    """
    Convert messy numeric strings such as '(min) 24' into 24.0.
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    match = re.search(r"-?\d+(?:\.\d+)?", text)

    if match:
        return float(match.group())

    return np.nan


def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Calculate great-circle distance between restaurant
    and customer coordinates in kilometers.
    """

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    a = np.clip(a, 0, 1)

    c = 2 * np.arcsin(np.sqrt(a))

    earth_radius_km = 6371.0

    return earth_radius_km * c


def extract_hour(time_value):
    """
    Extract hour from values such as:
    '11:30'
    '11:30 AM'
    '11:30:00'
    """

    if pd.isna(time_value):
        return np.nan

    text = str(time_value).strip()

    match = re.search(r"(\d{1,2})", text)

    if not match:
        return np.nan

    hour = int(match.group(1))

    if 0 <= hour <= 23:
        return hour

    return np.nan


def build_features(df):
    """
    Clean raw food-delivery data and create ML features.
    """

    df = df.copy()

    # ---------------------------------------------------------
    # Clean target
    # ---------------------------------------------------------

    df[TARGET] = df[TARGET].apply(clean_numeric)

    # ---------------------------------------------------------
    # Clean numeric columns
    # ---------------------------------------------------------

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
        if column in df.columns:
            df[column] = df[column].apply(clean_numeric)

    # ---------------------------------------------------------
    # Haversine distance
    # ---------------------------------------------------------

    df["distance_km"] = haversine_distance(
        df["Restaurant_latitude"],
        df["Restaurant_longitude"],
        df["Delivery_location_latitude"],
        df["Delivery_location_longitude"],
    )

    df["distance_km"] = df["distance_km"].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # ---------------------------------------------------------
    # Order hour
    # ---------------------------------------------------------

    df["order_hour"] = df["Time_Orderd"].apply(extract_hour)

    # ---------------------------------------------------------
    # Peak-hour feature
    # ---------------------------------------------------------

    df["is_peak_hour"] = df["order_hour"].isin(
        [12, 13, 14, 19, 20, 21, 22]
    ).astype(int)

    # ---------------------------------------------------------
    # Date features
    # ---------------------------------------------------------

    order_date = pd.to_datetime(
        df["Order_Date"],
        errors="coerce",
        dayfirst=True,
    )

    df["order_dayofweek"] = order_date.dt.dayofweek

    df["is_weekend"] = (
        order_date.dt.dayofweek >= 5
    ).astype(int)

    # ---------------------------------------------------------
    # Pickup delay proxy
    # ---------------------------------------------------------

    order_time = pd.to_datetime(
        df["Time_Orderd"].astype(str).str.strip(),
        format="%H:%M",
        errors="coerce",
    )

    pickup_time = pd.to_datetime(
        df["Time_Order_picked"].astype(str).str.strip(),
        format="%H:%M",
        errors="coerce",
    )

    pickup_delay = (
        pickup_time - order_time
    ).dt.total_seconds() / 60.0

    # Handle crossing midnight
    pickup_delay = pickup_delay.where(
        pickup_delay >= 0,
        pickup_delay + 24 * 60,
    )

    df["pickup_delay_min"] = pickup_delay

    # ---------------------------------------------------------
    # Remove invalid target rows
    # ---------------------------------------------------------

    df = df.dropna(subset=[TARGET])

    df = df[df[TARGET] > 0]

    return df


def load_dataset():
    """
    Load raw CSV and apply feature engineering.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    df = build_features(df)

    return df


if __name__ == "__main__":
    data = load_dataset()

    print("=" * 60)
    print("FOOD DELIVERY ETA DATA PREPARATION")
    print("=" * 60)

    print(f"Rows after preprocessing: {len(data):,}")
    print(f"Columns after feature engineering: {len(data.columns)}")

    print("\nTarget statistics:")
    print(data[TARGET].describe())

    print("\nEngineered features:")

    for column in [
        "distance_km",
        "order_hour",
        "is_peak_hour",
        "order_dayofweek",
        "is_weekend",
        "pickup_delay_min",
    ]:
        print(f"  - {column}")

    print("\nSample:")

    print(
        data[
            [
                TARGET,
                "distance_km",
                "order_hour",
                "is_peak_hour",
                "pickup_delay_min",
            ]
        ].head()
    )