"""
charts.py
Draws the Matplotlib charts. Each function returns the chart as a text string
(base64) that can be placed directly inside an HTML <img> tag, so no image files are saved.
"""
import base64
import io

import matplotlib
matplotlib.use("Agg")            # draw charts in memory (no pop-up windows)
import matplotlib.pyplot as plt

# Colours that match the website theme
NAVY = "#0b2545"
BLUE = "#1d4ed8"
LIGHT_BLUE = "#93c5fd"
GRID = "#dbe3ee"


def _to_base64(fig):
    """Turn a Matplotlib figure into a base64 PNG string, then free the memory."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _style(ax, title, ylabel):
    """Common look for all charts: title, light grid, no top/right border."""
    ax.set_title(title, color=NAVY, fontsize=12, fontweight="bold", loc="left")
    ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def trend_chart(analyzer, column, title, ylabel):
    """Line chart of one column over time, with a 30-day moving average on top."""
    df = analyzer.df
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(df["date"], df[column], color=LIGHT_BLUE, linewidth=1, label="Daily value")
    ax.plot(df["date"], analyzer.moving_average(column, 30), color=NAVY, linewidth=2, label="30-day average")
    ax.legend(frameon=False)
    _style(ax, title, ylabel)
    fig.autofmt_xdate()
    return _to_base64(fig)


def monthly_total_chart(analyzer):
    """Bar chart: total rainfall in every month of the data (used on the home page)."""
    totals = analyzer.monthly_total_rainfall()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(totals.index, totals.values, width=20, color=BLUE)
    _style(ax, "Total rainfall per month", "Rainfall (mm)")
    fig.autofmt_xdate()
    return _to_base64(fig)


def monthly_average_chart(analyzer):
    """Bar chart: average daily rainfall for each calendar month (Jan to Dec)."""
    averages = analyzer.monthly_average_rainfall()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(averages.index, averages.values, color=BLUE)
    _style(ax, "Average daily rainfall by month", "Rainfall (mm per day)")
    return _to_base64(fig)


def heatmap_chart(analyzer):
    """Correlation heatmap: +1 = move together, -1 = move opposite, 0 = no straight-line link."""
    table = analyzer.correlation_table().fillna(0)
    labels = list(table.columns)
    count = len(labels)

    fig, ax = plt.subplots(figsize=(6.4, 5))
    picture = ax.imshow(table.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(count))
    ax.set_yticks(range(count))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for row in range(count):                       # write the number inside every square
        for col in range(count):
            value = table.values[row][col]
            colour = "white" if abs(value) > 0.6 else NAVY
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=colour, fontsize=9)

    fig.colorbar(picture, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation between weather columns", color=NAVY, fontsize=12, fontweight="bold", loc="left")
    return _to_base64(fig)