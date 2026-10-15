"""
app.py
Main Flask application: it connects the web pages (templates) with Python code.
Run with:  python app.py
"""
import io
import math
import os
import time

import numpy as np
import pandas as pd
from flask import (Flask, Response, flash, redirect, render_template,
                   request, session, url_for)
from PIL import Image

import charts
import data_utils
import train_model
from analysis_utils import WeatherAnalyzer

app = Flask(__name__)
app.secret_key = "change-this-secret-key"             # needed for flash messages
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024    # uploads limited to 5 MB

# If a user uploads a CSV, it is saved here and used instead of the sample data
UPLOAD_PATH = os.path.join(data_utils.BASE_DIR, "data", "uploaded_weather.csv")

NUMBER_COLUMNS = ["temperature", "humidity", "pressure", "wind_speed", "rainfall"]

# The 4 prediction inputs: allowed range is only used to catch typing mistakes
INPUT_FIELDS = [
    {"name": "temperature", "label": "Temperature (°C)", "min": -30, "max": 60, "step": "0.1"},
    {"name": "humidity", "label": "Humidity (%)", "min": 0, "max": 100, "step": "0.1"},
    {"name": "pressure", "label": "Atmospheric pressure (hPa)", "min": 850, "max": 1100, "step": "0.1"},
    {"name": "wind_speed", "label": "Wind speed (km/h)", "min": 0, "max": 200, "step": "0.1"},
]

# Badge colour for each rainfall category
CATEGORY_COLOURS = {
    "No rain": "secondary",
    "Light rain": "info",
    "Moderate rain": "primary",
    "Heavy rain": "danger",
}


# Satellite image settings
IMAGE_FOLDER = os.path.join(data_utils.BASE_DIR, "static", "images")
ALLOWED_IMAGE_TYPES = {"PNG": "png", "JPEG": "jpg"}     # image format -> file ending
Image.MAX_IMAGE_PIXELS = 40_000_000                     # refuse extremely large images


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
    chart = None
    try:
        df = data_utils.load_weather_data(active_data_path())
        stats = data_utils.dashboard_stats(df)
        chart = charts.monthly_total_chart(WeatherAnalyzer(df))
    except ValueError as error:
        flash(str(error), "danger")
    return render_template("index.html", stats=stats, chart=chart, using_upload=using_upload())


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


# ---------------------------------------------------------------- page 3: prediction
@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    bundle = train_model.load_model()          # None if the model is not trained yet
    entered = {}                               # what the user typed (to refill the form)
    result = None

    if bundle is not None and request.method == "POST":
        values = {}
        errors = []
        for field in INPUT_FIELDS:
            text = request.form.get(field["name"], "").strip()
            entered[field["name"]] = text
            try:
                number = float(text)
                if not math.isfinite(number):
                    raise ValueError
            except ValueError:
                errors.append(field["label"] + " must be a number.")
                continue
            if number < field["min"] or number > field["max"]:
                errors.append(field["label"] + " must be between "
                              + str(field["min"]) + " and " + str(field["max"]) + ".")
                continue
            values[field["name"]] = number

        if errors:
            for message in errors:
                flash(message, "danger")
        else:
            millimetres = train_model.predict_rainfall(bundle, values)
            category = train_model.rainfall_category(millimetres)
            result = {
                "mm": millimetres,
                "category": category,
                "colour": CATEGORY_COLOURS[category],
            }

    return render_template(
        "prediction.html",
        bundle=bundle,
        fields=INPUT_FIELDS,
        entered=entered,
        result=result,
        trained_on_sample=(bundle is not None and bundle["data_file"] == "sample_weather.csv"),
    )


@app.route("/train-model", methods=["POST"])
def train_model_route():
    """Train the model again on the data currently in use (sample or uploaded CSV)."""
    try:
        bundle = train_model.train_and_save(active_data_path())
    except ValueError as error:
        flash("Training failed: " + str(error), "danger")
        return redirect(url_for("prediction"))
    flash("Model trained on " + bundle["data_file"] + ". Test MAE = "
          + str(bundle["metrics"]["mae"]) + " mm, RMSE = " + str(bundle["metrics"]["rmse"]) + " mm.", "success")
    return redirect(url_for("prediction"))


# ---------------------------------------------------------------- page 4: analysis
@app.route("/analysis")
def analysis():
    try:
        df = data_utils.load_weather_data(active_data_path())
    except ValueError as error:
        flash(str(error), "danger")
        return render_template("analysis.html", ok=False)

    analyzer = WeatherAnalyzer(df)

    chart_images = {
        "rainfall": charts.trend_chart(analyzer, "rainfall", "Rainfall over time", "Rainfall (mm)"),
        "temperature": charts.trend_chart(analyzer, "temperature", "Temperature over time", "Temperature (°C)"),
        "humidity": charts.trend_chart(analyzer, "humidity", "Humidity over time", "Humidity (%)"),
        "monthly": charts.monthly_average_chart(analyzer),
        "heatmap": charts.heatmap_chart(analyzer),
    }

    trend_sentences = []
    for column in ["rainfall", "temperature", "humidity"]:
        trend_sentences.append(analyzer.trend_text(column))

    correlation_rows = []
    for name, value in analyzer.rainfall_correlations().items():
        correlation_rows.append({"name": name, "value": 0.0 if pd.isna(value) else float(value)})

    return render_template(
        "analysis.html",
        ok=True,
        charts=chart_images,
        summary=analyzer.summary(),
        trends=trend_sentences,
        correlations=correlation_rows,
        days=len(df),
    )


@app.route("/download-analysis")
def download_analysis():
    """Send the analysed data as a CSV file (cleaned data + year, month and 30-day averages)."""
    try:
        df = data_utils.load_weather_data(active_data_path())
    except ValueError as error:
        flash(str(error), "danger")
        return redirect(url_for("analysis"))

    analyzer = WeatherAnalyzer(df)
    result = df.copy()
    result["year"] = result["date"].dt.year
    result["month"] = result["date"].dt.month
    for column in ["rainfall", "temperature", "humidity"]:
        result[column + "_30day_avg"] = analyzer.moving_average(column).round(2)
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")

    return Response(
        result.to_csv(index=False),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=weather_analysis.csv"},
    )


# ---------------------------------------------------------------- page 5: satellite images
@app.route("/satellite", methods=["GET", "POST"])
def satellite():
    if request.method == "POST":
        file = request.files.get("image_file")
        if file is None or file.filename == "":
            flash("Please choose an image first.", "warning")
            return redirect(url_for("satellite"))

        ending = os.path.splitext(file.filename)[1].lower()
        if ending not in (".png", ".jpg", ".jpeg"):
            flash("Only PNG and JPEG images are supported.", "danger")
            return redirect(url_for("satellite"))

        data = file.read()
        try:
            Image.open(io.BytesIO(data)).verify()        # is it really an image?
            image = Image.open(io.BytesIO(data))         # open again after verify()
            image_format = image.format
            width, height = image.size
            colour_mode = image.mode
            grey = image.convert("L")                    # grey-scale copy for brightness
        except Exception:
            flash("This file could not be read as an image. It may be damaged.", "danger")
            return redirect(url_for("satellite"))

        if image_format not in ALLOWED_IMAGE_TYPES:
            flash("Only PNG and JPEG images are supported.", "danger")
            return redirect(url_for("satellite"))

        os.makedirs(IMAGE_FOLDER, exist_ok=True)
        for old_name in ("satellite_upload.png", "satellite_upload.jpg"):
            old_path = os.path.join(IMAGE_FOLDER, old_name)
            if os.path.exists(old_path):
                os.remove(old_path)

        saved_name = "satellite_upload." + ALLOWED_IMAGE_TYPES[image_format]
        with open(os.path.join(IMAGE_FOLDER, saved_name), "wb") as saved_file:
            saved_file.write(data)

        session["satellite"] = {
            "original_name": os.path.basename(file.filename),
            "saved_name": saved_name,
            "width": width,
            "height": height,
            "format": image_format,
            "mode": colour_mode,
            "size_kb": round(len(data) / 1024, 1),
            "brightness": round(float(np.asarray(grey).mean()), 1),
            "version": int(time.time()),                 # makes the browser reload the new image
        }
        flash("Image uploaded successfully.", "success")
        return redirect(url_for("satellite"))

    info = session.get("satellite")
    if info and not os.path.exists(os.path.join(IMAGE_FOLDER, info["saved_name"])):
        info = None
    return render_template("satellite.html", info=info)


# ---------------------------------------------------------------- page 6 (built in Step 9)


@app.route("/about")
def about():
    return render_template("about.html")


@app.errorhandler(413)
def file_too_large(error):
    flash("The file is too large. The limit is 5 MB.", "danger")
    return redirect(request.referrer or url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)