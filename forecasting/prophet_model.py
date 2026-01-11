from prophet import Prophet
import numpy as np
from utils.logger import setup_logger
from forecasting.prophet_tuning import tune_prophet
from forecasting.hybrid_model import train_residual_model, apply_residual_correction

logger = setup_logger("ProphetModel")


def train_prophet(prophet_df, forecast_days=90):
    logger.info("Training Prophet model")

    param_grid = [
        {"changepoint_prior_scale": 0.05, "seasonality_prior_scale": 8, "interval_width": 0.70},
        {"changepoint_prior_scale": 0.1, "seasonality_prior_scale": 12, "interval_width": 0.75},
        {"changepoint_prior_scale": 0.15, "seasonality_prior_scale": 15, "interval_width": 0.80},
    ]

    best_params, _ = tune_prophet(prophet_df, param_grid)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        **best_params,
    )

    model.add_seasonality("monthly", period=30.5, fourier_order=8, prior_scale=10)
    model.add_seasonality("quarterly", period=91.25, fourier_order=4, prior_scale=5)
    
    for reg in ["price", "discount", "marketing_spend"]:
        model.add_regressor(reg, prior_scale=8, standardize=True)

    df = prophet_df.copy()
    df["y"] = np.log(df["y"].clip(lower=1))

    model.fit(df)

    future = model.make_future_dataframe(periods=forecast_days)

    for col in ["price", "discount", "marketing_spend"]:
        last_30 = df[col].tail(30)
        weights = np.linspace(1, 2, len(last_30))
        future[col] = np.average(last_30, weights=weights)

    forecast = model.predict(future)

    # ---- Hybrid residual correction (log scale) ----
    residual_model = train_residual_model(df, forecast)
    forecast = apply_residual_correction(forecast, residual_model, future)

    # ---- Inverse log and reduce uncertainty bands ----
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = np.exp(forecast[col])
    
    # Tighten uncertainty bands post-transformation
    forecast_range = forecast["yhat_upper"] - forecast["yhat_lower"]
    forecast["yhat_upper"] = forecast["yhat"] + (forecast_range * 0.4)
    forecast["yhat_lower"] = forecast["yhat"] - (forecast_range * 0.4)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=1)

    return model, forecast

# hkhk
def category_wise_forecast(df, category_name, days=60):
    """
    Category-level demand forecast using Prophet with regressors.
    Log-safe, no leakage, production-ready.
    """

    category_df = df[df["product_category"] == category_name].copy()

    prophet_df = (
        category_df
        .groupby("date", as_index=False)
        .agg(
            y=("units_sold", "sum"),
            price=("price", "mean"),
            discount=("discount", "mean"),
            marketing_spend=("marketing_spend", "sum"),
        )
        .rename(columns={"date": "ds"})
        .sort_values("ds")
    )

    # ---- Log transform ----
    prophet_df["y"] = prophet_df["y"].clip(lower=1)
    prophet_df["y"] = np.log(prophet_df["y"])

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=10,
    )

    model.add_seasonality("monthly", period=30.5, fourier_order=5)

    for reg in ["price", "discount", "marketing_spend"]:
        model.add_regressor(reg)

    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=days)

    # ---- Stable future regressors ----
    for col in ["price", "discount", "marketing_spend"]:
        future[col] = prophet_df[col].tail(30).mean()

    forecast = model.predict(future)

    # ---- Inverse log ----
    forecast["yhat"] = np.exp(forecast["yhat"])
    forecast["yhat_lower"] = np.exp(forecast["yhat_lower"])
    forecast["yhat_upper"] = np.exp(forecast["yhat_upper"])

    return forecast
# # =========================
# # 3️⃣ WHAT-IF ANALYSIS
# # =========================
def what_if_forecast(
    model,
    prophet_df,
    discount_change=0.0,
    marketing_change=0.0,
    days=30
):
    logger.info("Running what-if scenario analysis")

    future = model.make_future_dataframe(periods=days)

    future["price"] = prophet_df["price"].iloc[-1]
    future["discount"] = prophet_df["discount"].iloc[-1] * (1 + discount_change)
    future["marketing_spend"] = prophet_df["marketing_spend"].iloc[-1] * (1 + marketing_change)

    forecast = model.predict(future)

    forecast["yhat"] = np.exp(forecast["yhat"])
    forecast["yhat_lower"] = np.exp(forecast["yhat_lower"])
    forecast["yhat_upper"] = np.exp(forecast["yhat_upper"])

    return forecast