import pandas as pd
from utils.logger import setup_logger

logger = setup_logger("InsightsGenerator")


def generate_insights(forecast: pd.DataFrame):
    logger.info("Generating insights")

    insights = {}

    # Trend analysis
    trend_change = forecast["trend"].iloc[-1] - forecast["trend"].iloc[0]
    insights["trend"] = (
        "Increasing sales trend"
        if trend_change > 0
        else "Declining sales trend"
    )

    # Best & worst predicted day
    best_day = forecast.loc[forecast["yhat"].idxmax(), "ds"]
    worst_day = forecast.loc[forecast["yhat"].idxmin(), "ds"]

    insights["best_day"] = str(best_day.date())
    insights["worst_day"] = str(worst_day.date())

    # Average expected sales
    insights["average_sales"] = round(forecast["yhat"].mean(), 2)

    # Used for exact what-if calculations
    insights["baseline_units_per_day"] = round(
        forecast["yhat"].tail(30).mean(), 2
    )

    insights["baseline_units_per_month"] = round(
        insights["baseline_units_per_day"] * 30, 0
    )

    logger.info("Insights generated successfully")
    return insights
