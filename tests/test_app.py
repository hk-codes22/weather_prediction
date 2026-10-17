"""
test_app.py
Automatic tests for the Weather Prediction project.

Run from the main project folder:
    pip install pytest        (only needed once)
    python -m pytest -v

The tests use their OWN temporary data, model and upload folders,
so they never change your real files in data/, models/ or static/images/.
"""
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

# Let Python find app.py, data_utils.py ... in the main folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module
import data_utils
import train_model
from analysis_utils import WeatherAnalyzer

ROWS = 200


# ---------------------------------------------------------------- helpers
def make_weather_dataframe(rows=ROWS):
    """Small fake weather table. Rain is more likely when humidity is high."""
    rng = np.random.default_rng(0)
    humidity = rng.uniform(30, 100, rows)
    rain = np.where(humidity > 75, (humidity - 75) * rng.uniform(0.2, 1.0, rows), 0)
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=rows, freq="D").strftime("%Y-%m-%d"),
        "temperature": rng.normal(28, 4, rows),
        "humidity": humidity,
        "pressure": rng.normal(1010, 4, rows),
        "wind_speed": rng.uniform(2, 20, rows),
        "rainfall": rain,
    })
    return df.round(1)


def make_image_bytes(image_format, size=(320, 200)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (120, 160, 200)).save(buffer, image_format)
    return buffer.getvalue()


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "sample_weather.csv"
    make_weather_dataframe().to_csv(path, index=False)
    return str(path)


@pytest.fixture
def client(tmp_path, csv_path, monkeypatch):
    """A fake browser. All files are redirected to a temporary folder."""
    monkeypatch.setattr(data_utils, "DATA_PATH", csv_path)
    monkeypatch.setattr(app_module, "UPLOAD_PATH", str(tmp_path / "uploaded_weather.csv"))
    monkeypatch.setattr(app_module, "IMAGE_FOLDER", str(tmp_path / "images"))
    monkeypatch.setattr(train_model, "MODEL_PATH", str(tmp_path / "model.joblib"))
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def upload_csv(client, content, filename="data.csv"):
    return client.post("/upload", data={"csv_file": (io.BytesIO(content), filename)},
                       content_type="multipart/form-data", follow_redirects=True).get_data(as_text=True)


def upload_image(client, content, filename):
    return client.post("/satellite", data={"image_file": (io.BytesIO(content), filename)},
                       content_type="multipart/form-data", follow_redirects=True).get_data(as_text=True)


GOOD_INPUT = {"temperature": "30", "humidity": "90", "pressure": "1000", "wind_speed": "14"}


# ---------------------------------------------------------------- data_utils
def test_check_columns_finds_missing_columns():
    df = pd.DataFrame({"date": [], "temperature": []})
    missing = data_utils.check_columns(df)
    assert "rainfall" in missing and "date" not in missing


def test_clean_data_removes_duplicates_and_fills_missing():
    df = pd.DataFrame({
        "Date": ["2023-01-02", "2023-01-01", "2023-01-01", "not a date"],
        "Temperature": [20, 10, 10, 5],
        "Humidity": [50, None, None, 60],
        "Pressure": [1010, 1011, 1011, 1012],
        "Wind_Speed": [5, 6, 6, 7],
        "Rainfall": [1.0, 0.0, 0.0, 2.0],
    })
    clean = data_utils.clean_data(df)
    assert len(clean) == 2                                   # duplicate and bad date removed
    assert clean["date"].is_monotonic_increasing             # sorted by date
    assert clean["humidity"].isna().sum() == 0               # missing value filled


def test_load_weather_data_reports_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Missing required columns"):
        data_utils.load_weather_data(str(path))


def test_load_weather_data_reports_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        data_utils.load_weather_data(str(tmp_path / "nothing.csv"))


def test_dashboard_stats_are_calculated_from_data():
    df = data_utils.clean_data(make_weather_dataframe(10))
    stats = data_utils.dashboard_stats(df)
    assert stats["total_records"] == 10
    assert stats["max_rainfall"] == round(df["rainfall"].max(), 1)
    assert stats["latest_date"] == "10 Jan 2023"


# ---------------------------------------------------------------- analysis_utils
def test_monthly_average_has_twelve_months(csv_path):
    analyzer = WeatherAnalyzer(data_utils.load_weather_data(csv_path))
    monthly = analyzer.monthly_average_rainfall()
    assert len(monthly) == 12 and list(monthly.index)[0] == "Jan"


def test_correlation_table_is_square_and_diagonal_is_one(csv_path):
    table = WeatherAnalyzer(data_utils.load_weather_data(csv_path)).correlation_table()
    assert table.shape == (5, 5)
    assert all(table.iloc[i, i] == 1.0 for i in range(5))


def test_trend_per_year_detects_a_rising_line():
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    df = pd.DataFrame({"date": dates, "temperature": np.linspace(20, 30, 365),
                       "humidity": 50.0, "pressure": 1010.0, "wind_speed": 5.0, "rainfall": 0.0})
    change = WeatherAnalyzer(df).trend_per_year("temperature")
    assert 9.5 < change < 10.5                                # about +10 degrees in one year


# ---------------------------------------------------------------- train_model
@pytest.mark.parametrize("mm, expected", [
    (0.0, "No rain"), (0.4, "No rain"), (0.5, "Light rain"), (7.4, "Light rain"),
    (7.5, "Moderate rain"), (34.9, "Moderate rain"), (35.0, "Heavy rain"), (80.0, "Heavy rain"),
])
def test_rainfall_category_thresholds(mm, expected):
    assert train_model.rainfall_category(mm) == expected


def test_chronological_split_keeps_time_order(csv_path):
    df = data_utils.load_weather_data(csv_path)
    train, test = train_model.chronological_split(df)
    assert len(train) + len(test) == len(df)
    assert train["date"].max() < test["date"].min()           # train = past, test = future


def test_training_saves_a_model_that_can_predict(client, csv_path):
    bundle = train_model.train_and_save(csv_path)
    assert os.path.exists(train_model.MODEL_PATH)
    assert bundle["model_name"] == "Random Forest Regressor"
    assert bundle["metrics"]["mae"] >= 0 and bundle["metrics"]["rmse"] >= bundle["metrics"]["mae"]
    loaded = train_model.load_model()
    wet = train_model.predict_rainfall(loaded, {"temperature": 28, "humidity": 95, "pressure": 1008, "wind_speed": 10})
    assert isinstance(wet, float) and wet >= 0


def test_training_refuses_a_tiny_dataset(client, tmp_path):
    path = tmp_path / "tiny.csv"
    make_weather_dataframe(10).to_csv(path, index=False)
    with pytest.raises(ValueError, match="rows"):
        train_model.train_and_save(str(path))


def test_small_dataset_uses_the_baseline(client, tmp_path):
    path = tmp_path / "small.csv"
    make_weather_dataframe(50).to_csv(path, index=False)
    bundle = train_model.train_and_save(str(path))
    assert bundle["model_name"].startswith("Baseline")


def test_load_model_returns_none_when_not_trained(client):
    assert train_model.load_model() is None


# ---------------------------------------------------------------- pages
@pytest.mark.parametrize("url", ["/", "/weather-data", "/prediction", "/analysis", "/satellite", "/about"])
def test_every_page_opens(client, url):
    assert client.get(url).status_code == 200


def test_dashboard_shows_calculated_numbers_and_chart(client):
    html = client.get("/").get_data(as_text=True)
    assert "Total records" in html and str(ROWS) in html
    assert "data:image/png;base64," in html


def test_pages_do_not_load_anything_from_the_internet(client):
    for url in ["/", "/analysis", "/about"]:
        assert "cdn.jsdelivr" not in client.get(url).get_data(as_text=True)


# ---------------------------------------------------------------- weather data page
def test_date_filter_limits_the_records(client):
    html = client.get("/weather-data?start=2023-01-01&end=2023-01-10").get_data(as_text=True)
    assert "showing 10 of 10" in html


def test_invalid_date_filter_shows_a_message(client):
    assert "Start date is not valid" in client.get("/weather-data?start=zzz").get_data(as_text=True)


def test_upload_valid_csv_is_accepted_and_can_be_reset(client):
    good = make_weather_dataframe(30).to_csv(index=False).encode()
    assert "Upload successful. 30 records loaded." in upload_csv(client, good)
    assert "Preview of your uploaded data" in client.get("/weather-data").get_data(as_text=True)
    assert "Switched back to the sample dataset" in client.post("/reset-data", follow_redirects=True).get_data(as_text=True)


def test_upload_with_missing_columns_shows_an_error(client):
    assert "Missing required columns" in upload_csv(client, b"a,b\n1,2\n")


def test_upload_wrong_file_type_is_refused(client):
    assert "Only .csv files are allowed" in upload_csv(client, b"hello", "notes.txt")


def test_upload_with_only_bad_dates_is_refused(client):
    bad = b"date,temperature,humidity,pressure,wind_speed,rainfall\nbad,1,1,1,1,1\n"
    assert "no valid rows" in upload_csv(client, bad)


# ---------------------------------------------------------------- prediction page
def test_prediction_page_asks_to_train_when_there_is_no_model(client):
    html = client.get("/prediction").get_data(as_text=True)
    assert "has not been trained yet" in html and "Estimate rainfall" not in html


def test_train_button_then_prediction_works(client):
    html = client.post("/train-model", follow_redirects=True).get_data(as_text=True)
    assert "Model trained on sample_weather.csv" in html
    assert "Demonstration data" in html                       # warning for the sample file
    result = client.post("/prediction", data=GOOD_INPUT).get_data(as_text=True)
    assert "Estimated rainfall" in result and "not an official weather forecast" in result


def test_prediction_rejects_bad_input(client):
    client.post("/train-model")
    bad = {"temperature": "abc", "humidity": "150", "pressure": "1000", "wind_speed": ""}
    html = client.post("/prediction", data=bad).get_data(as_text=True)
    assert "must be a number" in html and "must be between 0 and 100" in html
    assert "Estimated rainfall</div>" not in html


def test_prediction_rejects_nan(client):
    client.post("/train-model")
    data = dict(GOOD_INPUT, temperature="nan")
    assert "must be a number" in client.post("/prediction", data=data).get_data(as_text=True)


# ---------------------------------------------------------------- analysis and download
def test_analysis_page_has_five_charts(client):
    html = client.get("/analysis").get_data(as_text=True)
    assert html.count("data:image/png;base64,") == 5


def test_download_analysis_csv(client):
    response = client.get("/download-analysis")
    assert response.status_code == 200
    assert "weather_analysis.csv" in response.headers["Content-Disposition"]
    lines = response.data.decode().splitlines()
    assert "rainfall_30day_avg" in lines[0] and len(lines) == ROWS + 1


# ---------------------------------------------------------------- satellite page
def test_png_upload_shows_its_dimensions(client):
    html = upload_image(client, make_image_bytes("PNG"), "goes.png")
    assert "320 x 200" in html and "goes.png" in html


def test_jpeg_upload_works(client):
    assert "640 x 480" in upload_image(client, make_image_bytes("JPEG", (640, 480)), "sat.jpg")


def test_fake_image_is_refused(client):
    assert "could not be read as an image" in upload_image(client, b"not an image", "fake.png")


def test_unsupported_image_type_is_refused(client):
    assert "Only PNG and JPEG" in upload_image(client, make_image_bytes("GIF"), "picture.gif")
    assert "Only PNG and JPEG" in upload_image(client, make_image_bytes("GIF"), "renamed.png")


def test_oversized_upload_is_refused(client):
    big = make_image_bytes("PNG") + os.urandom(6 * 1024 * 1024)
    response = client.post("/satellite", data={"image_file": (io.BytesIO(big), "big.png")},
                           content_type="multipart/form-data", headers={"Referer": "http://localhost/satellite"},
                           follow_redirects=True)
    assert "too large" in response.get_data(as_text=True)