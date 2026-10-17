"""
check_project.py
Checks that every file and folder of the project is in place.
Run from the main project folder:   python check_project.py
"""
import os

REQUIRED = [
    "app.py", "train_model.py", "data_utils.py", "analysis_utils.py", "charts.py",
    "convert_noaa.py", "make_sample_data.py", "requirements.txt", "README.md",
    "data/sample_weather.csv", "models/.gitkeep",
    "templates/base.html", "templates/index.html", "templates/weather_data.html",
    "templates/prediction.html", "templates/analysis.html", "templates/satellite.html",
    "templates/about.html",
    "static/css/style.css", "static/css/bootstrap.min.css", "static/js/bootstrap.bundle.min.js",
    "static/images", "tests/test_app.py",
]

base = os.path.dirname(os.path.abspath(__file__))
missing = [name for name in REQUIRED if not os.path.exists(os.path.join(base, name))]

if missing:
    print("Missing files or folders:")
    for name in missing:
        print("  -", name)
else:
    print("All", len(REQUIRED), "required files and folders are present.")

if os.path.exists(os.path.join(base, "models", "rainfall_model.joblib")):
    print("Model file found.")
else:
    print("Model not trained yet: run  python train_model.py")