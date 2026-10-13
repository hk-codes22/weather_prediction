"""
app.py
Main Flask application: it connects the web pages (templates) with Python code.
Run with:  python app.py
"""
import os

import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

import data_utils

app = Flask(__name__)
app.secret_key = "change-this-secret-key"             # needed for flash messages
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024    # uploads limited to 5 MB

# If a user uploads a CSV, it is saved here and used instead of the sample data
UPLOAD_PATH = os.path.join(data_utils.BASE_DIR, "data", "uploaded_weather.csv")

NUMBER_COLUMNS = ["temperature", "humidity", "pressure", "wind_speed", "rainfall"]


# ---------------------------------------------------------------- helper functions
def using_upload():
    """True if the user has uploaded their own CSV."""
    return os.path.exists(UPLOAD_PATH)


def active_data_path():
    """Return the path of the CSV file that the app should use right now."""
    if using_upload():
        return UPLOAD_PATH
    return data_utils.DATA_PATH


def missing_values_list(path):
    """Count missing (empty) cells in each column of the ORIGINAL file."""
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    result = []
    for column in raw.columns:
        result.append({"column": column, "missing": int(raw[column].isna().sum())})
    return result


def statistics_rows(df):
    """Mean, median, standard deviation, min and max of each weather column."""
    rows = []
    for column in NUMBER_COLUMNS:
        rows.append({
            "name": column,
            "mean": round(float(np.nan_to_num(df[column].mean())), 2),
            "median": round(float(np.nan_to_num(df[column].median())), 2),
            "std": round(float(np.nan_to_num(df[column].std())), 2),
            "min": round(float(np.nan_to_num(df[column].min())), 2),
            "max": round(float(np.nan_to_num(df[column].max())), 2),
        })
    return rows


# ---------------------------------------------------------------- page 1: dashboard
@app.route("/")
def index():
    stats = None
    try:
        df = data_utils.load_weather_data(active_data_path())
        stats = data_utils.dashboard_stats(df)
    except ValueError as error:
        flash(str(error), "danger")
    # 'chart' will be filled in Step 8 (Matplotlib)
    return render_template("index.html", stats=stats, chart=None, using_upload=using_upload())


# ---------------------------------------------------------------- page 2: weather data
@app.route("/weather-data")
def weather_data():
    try:
        df = data_utils.load_weather_data(active_data_path())
        columns_info = missing_values_list(active_data_path())
    except ValueError as error:
        flash(str(error), "danger")
        return render_template("weather_data.html", df_ok=False)

    # --- date filter (values come from the form in the URL) ---
    start = request.args.get("start", "").strip()
    end = request.args.get("end", "").strip()
    filtered = df

    if start:
        start_date = pd.to_datetime(start, errors="coerce")
        if pd.isna(start_date):
            flash("Start date is not valid.", "warning")
        else:
            filtered = filtered[filtered["date"] >= start_date]
    if end:
        end_date = pd.to_datetime(end, errors="coerce")
        if pd.isna(end_date):
            flash("End date is not valid.", "warning")
        else:
            filtered = filtered[filtered["date"] <= end_date]

    # --- prepare the table rows (show at most 100 rows to keep the page fast) ---
    table = filtered.head(100).copy()
    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    rows = table.to_dict("records")

    preview = None
    if using_upload():
        first_rows = df.head(5).copy()
        first_rows["date"] = first_rows["date"].dt.strftime("%Y-%m-%d")
        preview = first_rows.to_dict("records")

    return render_template(
        "weather_data.html",
        df_ok=True,
        rows=rows,
        shown=len(rows),
        total=len(filtered),
        stats_rows=statistics_rows(filtered) if len(filtered) > 0 else [],
        columns_info=columns_info,
        preview=preview,
        start=start,
        end=end,
        min_date=df["date"].min().strftime("%Y-%m-%d"),
        max_date=df["date"].max().strftime("%Y-%m-%d"),
        using_upload=using_upload(),
    )


@app.route("/upload", methods=["POST"])
def upload_csv():
    file = request.files.get("csv_file")

    if file is None or file.filename == "":
        flash("Please choose a CSV file first.", "warning")
        return redirect(url_for("weather_data"))

    if not file.filename.lower().endswith(".csv"):
        flash("Only .csv files are allowed.", "danger")
        return redirect(url_for("weather_data"))

    temp_path = UPLOAD_PATH + ".tmp"
    file.save(temp_path)                       # save to a temporary file first

    try:
        df = data_utils.load_weather_data(temp_path)
        if df.empty:
            raise ValueError("The file has no valid rows. Check the date column.")
    except ValueError as error:
        os.remove(temp_path)                   # throw away the bad file
        flash(str(error), "danger")
        return redirect(url_for("weather_data"))

    os.replace(temp_path, UPLOAD_PATH)         # the file is good, so keep it
    flash("Upload successful. " + str(len(df)) + " records loaded.", "success")
    return redirect(url_for("weather_data"))


@app.route("/reset-data", methods=["POST"])
def reset_data():
    if using_upload():
        os.remove(UPLOAD_PATH)
    flash("Switched back to the sample dataset.", "info")
    return redirect(url_for("weather_data"))


# ---------------------------------------------------------------- pages built later
@app.route("/prediction")
def prediction():
    return render_template("prediction.html")


@app.route("/analysis")
def analysis():
    return render_template("analysis.html")


@app.route("/satellite")
def satellite():
    return render_template("satellite.html")


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)