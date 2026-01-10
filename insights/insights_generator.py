# import pandas as pd
# from utils.logger import setup_logger

# logger = setup_logger("InsightsGenerator")


# def generate_insights(forecast: pd.DataFrame):
#     logger.info("Generating insights")

#     insights = {}

#     # Trend analysis
#     trend_change = forecast["trend"].iloc[-1] - forecast["trend"].iloc[0]
#     insights["trend"] = (
#         "Increasing sales trend"
#         if trend_change > 0
#         else "Declining sales trend"
#     )

#     # Best & worst predicted day
#     best_day = forecast.loc[forecast["yhat"].idxmax(), "ds"]
#     worst_day = forecast.loc[forecast["yhat"].idxmin(), "ds"]

#     insights["best_day"] = str(best_day.date())
#     insights["worst_day"] = str(worst_day.date())

#     # Average expected sales
#     insights["average_sales"] = round(forecast["yhat"].mean(), 2)

#     # Used for exact what-if calculations
#     insights["baseline_units_per_day"] = round(
#         forecast["yhat"].tail(30).mean(), 2
#     )

#     insights["baseline_units_per_month"] = round(
#         insights["baseline_units_per_day"] * 30, 0
#     )

#     logger.info("Insights generated successfully")
#     return insights
# insights/insights_generator.py
import pandas as pd
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("InsightsGenerator")


def generate_insights(forecast: pd.DataFrame) -> dict:
    """
    Generates stable, business-safe insights from forecast output.
    """

    logger.info("Generating insights")

    insights = {}

    # ---------------- TREND (SMOOTHED) ----------------
    trend_series = forecast["trend"].rolling(7).mean()
    trend_change = trend_series.iloc[-1] - trend_series.iloc[0]

    if trend_change > 0:
        insights["trend"] = "Increasing sales trend"
        insights["trend_confidence"] = "Medium confidence"
    else:
        insights["trend"] = "Declining sales trend"
        insights["trend_confidence"] = "Medium confidence"

    # ---------------- ROBUST BEST / WORST PERIOD ----------------
    forecast["week"] = forecast["ds"].dt.to_period("W")

    weekly_avg = forecast.groupby("week")["yhat"].mean()

    insights["best_period"] = str(weekly_avg.idxmax())
    insights["worst_period"] = str(weekly_avg.idxmin())

    # ---------------- BASELINES ----------------
    last_30 = forecast["yhat"].tail(30)

    insights["average_sales"] = round(last_30.mean(), 2)
    insights["baseline_units_per_day"] = round(last_30.mean(), 2)
    insights["baseline_units_per_month"] = round(last_30.mean() * 30, 0)

    # ---------------- FORECAST STABILITY ----------------
    volatility = last_30.std() / last_30.mean()

    if volatility < 0.15:
        insights["forecast_stability"] = "Stable demand pattern"
        insights["forecast_confidence"] = "High confidence"
    elif volatility < 0.3:
        insights["forecast_stability"] = "Moderately volatile demand"
        insights["forecast_confidence"] = "Medium confidence"
    else:
        insights["forecast_stability"] = "Highly volatile demand"
        insights["forecast_confidence"] = "Low confidence"

    logger.info("Insights generated successfully")
    return insights
