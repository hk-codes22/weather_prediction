"""
convert_noaa.py
Converts a NOAA "Global Summary of the Day" (GSOD) CSV file into the format used by this project.

Usage:
    python convert_noaa.py downloaded_file.csv data/my_station.csv

What it does:
    temperature : TEMP  (Fahrenheit)  -> degrees Celsius
    humidity    : calculated from TEMP and DEWP (dew point) with the Magnus formula
    pressure    : SLP (sea level pressure, millibars = hPa); STP is used only if SLP is empty
    wind_speed  : WDSP  (knots)       -> km/h
    rainfall    : PRCP  (inches)      -> millimetres
Use a file for ONE station only. Days with no rainfall value are removed.
"""
import sys

import numpy as np
import pandas as pd

# NOAA writes these numbers when a value is missing
MISSING_CODES = {"TEMP": 9999.9, "DEWP": 9999.9, "SLP": 9999.9,
                 "STP": 9999.9, "WDSP": 999.9, "PRCP": 99.99}


def fahrenheit_to_celsius(f):
    return (f - 32) * 5 / 9


def humidity_from_dew_point(temp_c, dew_c):
    """Relative humidity in % from air temperature and dew point (Magnus formula)."""
    a, b = 17.625, 243.04
    rh = 100 * np.exp(a * dew_c / (b + dew_c)) / np.exp(a * temp_c / (b + temp_c))
    return rh.clip(0, 100)


def convert(df):
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]

    needed = ["DATE", "TEMP", "DEWP", "WDSP", "PRCP"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError("This does not look like a GSOD file. Missing columns: " + ", ".join(missing))
    if "SLP" not in df.columns and "STP" not in df.columns:
        raise ValueError("The file has no pressure column (SLP or STP).")
    if "STATION" in df.columns and df["STATION"].nunique() > 1:
        raise ValueError("The file contains " + str(df["STATION"].nunique())
                         + " stations. Please use a file with one station only.")

    for column, code in MISSING_CODES.items():
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").replace(code, np.nan)

    pressure = df["SLP"] if "SLP" in df.columns else df["STP"]
    pressure_name = "SLP"
    if pressure.notna().sum() == 0 and "STP" in df.columns:
        pressure, pressure_name = df["STP"], "STP"

    temp_c = fahrenheit_to_celsius(df["TEMP"])
    dew_c = fahrenheit_to_celsius(df["DEWP"])

    result = pd.DataFrame({
        "date": pd.to_datetime(df["DATE"], errors="coerce").dt.strftime("%Y-%m-%d"),
        "temperature": temp_c.round(1),
        "humidity": humidity_from_dew_point(temp_c, dew_c).round(1),
        "pressure": pressure.round(1),
        "wind_speed": (df["WDSP"] * 1.852).round(1),
        "rainfall": (df["PRCP"] * 25.4).round(1),
    })
    result = result.dropna(subset=["date", "rainfall"])      # rainfall is the target: it must exist
    return result, pressure_name


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_noaa.py input.csv [output.csv]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/converted_weather.csv"
    try:
        converted, used = convert(pd.read_csv(input_path))
    except FileNotFoundError:
        print("File not found:", input_path)
        sys.exit(1)
    except ValueError as error:
        print("Conversion failed:", error)
        sys.exit(1)
    converted.to_csv(output_path, index=False)
    print("Saved", len(converted), "rows to", output_path, "(pressure column used:", used + ")")
    print("Empty cells per column (they are filled with the median when the data is loaded):")
    print(converted.isna().sum().to_string())