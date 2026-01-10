# from sklearn.ensemble import RandomForestRegressor
# import numpy as np

# def train_residual_model(prophet_df, forecast):
#     """
#     Train ML model on Prophet residuals
#     """

#     df = prophet_df.copy()

#     df["prophet_pred"] = forecast.loc[:len(df)-1, "yhat"].values
#     df["residual"] = df["y"] - df["prophet_pred"]

#     features = df[["price", "discount", "marketing_spend"]]
#     target = df["residual"]

#     model = RandomForestRegressor(
#         n_estimators=200,
#         max_depth=8,
#         random_state=42
#     )

#     model.fit(features, target)
#     return model


# def apply_residual_correction(forecast, residual_model, future_df):
#     residual_pred = residual_model.predict(
#         future_df[["price", "discount", "marketing_spend"]]
#     )

#     forecast["yhat_hybrid"] = forecast["yhat"] + residual_pred
#     return forecast
# forecasting/hybrid_model.py
from sklearn.ensemble import RandomForestRegressor


def train_residual_model(train_df, forecast_df):
    """
    Train residual ML model on Prophet log-scale residuals.
    """

    df = train_df.copy()
    df["prophet_pred"] = forecast_df.loc[: len(df) - 1, "yhat"].values
    df["residual"] = df["y"] - df["prophet_pred"]

    features = df[["price", "discount", "marketing_spend"]]
    target = df["residual"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42,
    )

    model.fit(features, target)
    return model


def apply_residual_correction(forecast, residual_model, future_df):
    """
    Apply residual correction on log scale only.
    """

    residual_pred = residual_model.predict(
        future_df[["price", "discount", "marketing_spend"]]
    )

    forecast["yhat_hybrid"] = forecast["yhat"] + residual_pred
    return forecast
