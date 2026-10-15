"""
analysis_utils.py
Weather analysis with Pandas and NumPy, written as a small class.
Step 8 will draw charts from the numbers calculated here.
"""
import numpy as np
import pandas as pd

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


class WeatherAnalyzer:
    """Takes a CLEAN weather DataFrame (from data_utils) and analyses it."""

    NUMBER_COLUMNS = ["temperature", "humidity", "pressure", "wind_speed", "rainfall"]

    def __init__(self, df):
        self.df = df

    def summary(self):
        """Mean, median and standard deviation of every weather column (a dictionary of dictionaries)."""
        result = {}
        for column in self.NUMBER_COLUMNS:
            result[column] = {
                "mean": round(float(self.df[column].mean()), 2),
                "median": round(float(self.df[column].median()), 2),
                "std": round(float(self.df[column].std()), 2),
            }
        return result

    def monthly_average_rainfall(self):
        """
        Average daily rainfall for each calendar month (Jan to Dec),
        for example the mean of all Januaries, all Februaries and so on.
        Returns a Series with 12 values (months with no data become 0).
        """
        by_month = self.df.groupby(self.df["date"].dt.month)["rainfall"].mean()
        by_month = by_month.reindex(range(1, 13)).fillna(0).round(2)
        by_month.index = MONTH_NAMES
        return by_month

    def monthly_total_rainfall(self):
        """Total rainfall in each month of the time series (one value per month)."""
        return self.df.resample("MS", on="date")["rainfall"].sum().round(1)

    def moving_average(self, column, days=30):
        """Rolling average that smooths a noisy daily line (used for trend charts)."""
        return self.df[column].rolling(window=days, min_periods=1).mean()

    def correlation_table(self):
        """Correlation between all weather columns (values from -1 to +1)."""
        return self.df[self.NUMBER_COLUMNS].corr().round(2)

    def rainfall_correlations(self):
        """Correlation of each input column with rainfall, strongest first."""
        table = self.correlation_table()["rainfall"].drop("rainfall")
        return table.reindex(table.abs().sort_values(ascending=False).index)

    def trend_per_year(self, column):
        """
        Basic trend analysis: fit a straight line through the data with NumPy
        and return how much the column changes per year.
        Needs at least 2 days of data.
        """
        if len(self.df) < 2:
            return 0.0
        days = (self.df["date"] - self.df["date"].min()).dt.days.to_numpy()
        slope, _intercept = np.polyfit(days, self.df[column].to_numpy(), 1)
        return round(float(slope * 365), 3)

    def trend_text(self, column):
        """A sentence describing the trend, e.g. for the analysis page."""
        change = self.trend_per_year(column)
        if abs(change) < 0.01:
            return column + " shows almost no straight-line trend."
        direction = "rises" if change > 0 else "falls"
        return column + " " + direction + " by about " + str(abs(change)) + " per year (straight-line fit)."