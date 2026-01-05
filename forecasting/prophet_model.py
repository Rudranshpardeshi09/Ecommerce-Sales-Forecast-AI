# from prophet import Prophet
# from utils.logger import setup_logger
# import numpy as np

# logger = setup_logger("ProphetModel")

# # =========================
# # 1️⃣ OVERALL FORECAST
# # =========================
# def train_prophet(prophet_df, forecast_days=90):
#     logger.info("Training Prophet with regressors")

#     model = Prophet(
#     yearly_seasonality=True,
#     weekly_seasonality=True,
#     daily_seasonality=False,
#     changepoint_prior_scale=0.15,      # allows trend flexibility
#     seasonality_prior_scale=10.0,      # better seasonal learning
#     interval_width=0.8                 # tighter uncertainty bands
# )
    
#     model.add_seasonality(
#     name="monthly",
#     period=30.5,
#     fourier_order=5
# )


#     # Add regressors
#     model.add_regressor("price")
#     model.add_regressor("discount")
#     model.add_regressor("marketing_spend")

#     prophet_df["y"] = prophet_df["y"].apply(lambda x: max(x, 1))
#     prophet_df["y"] = np.log(prophet_df["y"])
#     model.fit(prophet_df)
#     forecast["yhat"] = np.exp(forecast["yhat"])


#     future = model.make_future_dataframe(periods=forecast_days)

#     # Use last known regressor values
#     for col in ["price", "discount", "marketing_spend"]:
#         last_n = 30

#         future["price"] = prophet_df["price"].tail(last_n).mean()
#         future["discount"] = prophet_df["discount"].tail(last_n).mean()
#         future["marketing_spend"] = prophet_df["marketing_spend"].tail(last_n).mean()


#     forecast = model.predict(future)

#     return model, forecast


# # =========================
# # 2️⃣ CATEGORY-WISE FORECAST
# # =========================
# def category_wise_forecast(df, category_name, days=60):
#     logger.info(f"Running category-wise forecast for {category_name}")

#     category_df = df[df["product_category"] == category_name]

#     prophet_df = (
#         category_df.groupby("date")
#         .agg({
#             "units_sold": "sum",
#             "price": "mean",
#             "discount": "mean",
#             "marketing_spend": "sum"
#         })
#         .reset_index()
#         .rename(columns={"date": "ds", "units_sold": "y"})
#     )

#     model = Prophet()
#     model.add_regressor("price")
#     model.add_regressor("discount")
#     model.add_regressor("marketing_spend")

#     model.fit(prophet_df)

#     future = model.make_future_dataframe(periods=days)

#     for col in ["price", "discount", "marketing_spend"]:
#         last_n = 30

#         future["price"] = prophet_df["price"].tail(last_n).mean()
#         future["discount"] = prophet_df["discount"].tail(last_n).mean()
#         future["marketing_spend"] = prophet_df["marketing_spend"].tail(last_n).mean()


#     forecast = model.predict(future)

#     return forecast


# # =========================
# # 3️⃣ WHAT-IF ANALYSIS
# # =========================
# def what_if_forecast(
#     model,
#     prophet_df,
#     discount_change=0.0,
#     marketing_change=0.0,
#     days=30
# ):
#     logger.info("Running what-if scenario analysis")

#     future = model.make_future_dataframe(periods=days)

#     future["price"] = prophet_df["price"].iloc[-1]
#     future["discount"] = prophet_df["discount"].iloc[-1] * (1 + discount_change)
#     future["marketing_spend"] = prophet_df["marketing_spend"].iloc[-1] * (1 + marketing_change)

#     forecast = model.predict(future)
#     return forecast

from prophet import Prophet
from utils.logger import setup_logger
import numpy as np
from forecasting.prophet_tuning import tune_prophet
from forecasting.hybrid_model import train_residual_model, apply_residual_correction


logger = setup_logger("ProphetModel")

# =========================
# 1️⃣ OVERALL FORECAST
# =========================
def train_prophet(prophet_df, forecast_days=90):
    logger.info("Training Prophet with regressors")

    param_grid = [
    {"changepoint_prior_scale": 0.05, "seasonality_prior_scale": 5, "interval_width": 0.8},
    {"changepoint_prior_scale": 0.15, "seasonality_prior_scale": 10, "interval_width": 0.8},
    {"changepoint_prior_scale": 0.3, "seasonality_prior_scale": 15, "interval_width": 0.9},
]

    best_params, _ = tune_prophet(prophet_df, param_grid)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=best_params["changepoint_prior_scale"],
        seasonality_prior_scale=best_params["seasonality_prior_scale"],
        interval_width=best_params["interval_width"]
    )

    # Monthly seasonality
    model.add_seasonality(
        name="monthly",
        period=30.5,
        fourier_order=5
    )

    # Regressors
    model.add_regressor("price")
    model.add_regressor("discount")
    model.add_regressor("marketing_spend")

    # ---- Log transform target (safe) ----
    prophet_df = prophet_df.copy()
    prophet_df["y"] = prophet_df["y"].clip(lower=1)
    prophet_df["y"] = np.log(prophet_df["y"])

    model.fit(prophet_df.rename(columns={"y_log": "y"}))

    # ---- Future dataframe ----
    future = model.make_future_dataframe(periods=forecast_days)

    last_n = 30
    future["price"] = prophet_df["price"].tail(last_n).mean()
    future["discount"] = prophet_df["discount"].tail(last_n).mean()
    future["marketing_spend"] = prophet_df["marketing_spend"].tail(last_n).mean()
    

    forecast = model.predict(future)
     # ---------------- RESIDUAL MODEL (LOG SCALE) ----------------
    residual_model = train_residual_model(
        prophet_df.rename(columns={"y_log": "y"}),
        forecast
    )

    forecast = apply_residual_correction(
        forecast,
        residual_model,
        future
    )


    # ---- Inverse log transform ----
    forecast["yhat"] = np.exp(forecast["yhat"])
    forecast["yhat_lower"] = np.exp(forecast["yhat_lower"])
    forecast["yhat_upper"] = np.exp(forecast["yhat_upper"])

    return model, forecast


# =========================
# 2️⃣ CATEGORY-WISE FORECAST
# =========================
def category_wise_forecast(df, category_name, days=60):
    logger.info(f"Running category-wise forecast for {category_name}")

    category_df = df[df["product_category"] == category_name]

    prophet_df = (
        category_df.groupby("date")
        .agg({
            "units_sold": "sum",
            "price": "mean",
            "discount": "mean",
            "marketing_spend": "sum"
        })
        .reset_index()
        .rename(columns={"date": "ds", "units_sold": "y"})
    )

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=10.0
    )

    model.add_seasonality(
        name="monthly",
        period=30.5,
        fourier_order=5
    )

    model.add_regressor("price")
    model.add_regressor("discount")
    model.add_regressor("marketing_spend")

    prophet_df = prophet_df.copy()
    prophet_df["y"] = prophet_df["y"].clip(lower=1)
    prophet_df["y"] = np.log(prophet_df["y"])

    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=days)

    last_n = 30
    future["price"] = prophet_df["price"].tail(last_n).mean()
    future["discount"] = prophet_df["discount"].tail(last_n).mean()
    future["marketing_spend"] = prophet_df["marketing_spend"].tail(last_n).mean()

    forecast = model.predict(future)

    forecast["yhat"] = np.exp(forecast["yhat"])
    forecast["yhat_lower"] = np.exp(forecast["yhat_lower"])
    forecast["yhat_upper"] = np.exp(forecast["yhat_upper"])

    return forecast


# =========================
# 3️⃣ WHAT-IF ANALYSIS
# =========================
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
