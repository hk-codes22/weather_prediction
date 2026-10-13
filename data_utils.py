"""
data_utils.py
Functions for loading, checking and cleaning the weather data.
"""
import os
import pandas as pd

# Folder paths (works on any laptop because they are built from this file's location)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "sample_weather.csv")

# Columns that every weather CSV must have
REQUIRED_COLUMNS = ["date", "temperature", "humidity", "pressure", "wind_speed", "rainfall"]


def check_columns(df):
    """Return a list of required columns that are missing from the DataFrame."""
    missing = []
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            missing.append(column)
    return missing


def clean_data(df):
    """
    Clean the weather data:
    1. make column names lowercase,
    2. convert 'date' to real dates,
    3. convert the other columns to numbers,
    4. remove duplicate rows,
    5. fill missing numbers with the column median,
    6. sort by date.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])                 # remove rows with bad dates

    for column in REQUIRED_COLUMNS[1:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].fillna(df[column].median())

    df = df.drop_duplicates()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_weather_data(path=DATA_PATH):
    """
    Read a CSV file and return a clean DataFrame.
    Raises ValueError with a friendly message if something is wrong.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise ValueError("Data file not found: " + path)
    except Exception:
        raise ValueError("Could not read the file. Please upload a valid CSV.")

    df.columns = [c.strip().lower() for c in df.columns]
    missing = check_columns(df)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    return clean_data(df)


def dashboard_stats(df):
    """Calculate the numbers shown on the home dashboard (nothing is hardcoded)."""
    return {
        "total_records": len(df),
        "average_rainfall": round(df["rainfall"].mean(), 2),
        "max_rainfall": round(df["rainfall"].max(), 1),
        "latest_date": df["date"].max().strftime("%d %b %Y"),
    }