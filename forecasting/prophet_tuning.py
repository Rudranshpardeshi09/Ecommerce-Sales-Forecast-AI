from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import numpy as np


def tune_prophet(prophet_df, param_grid, horizon=30):
    """
    Time-based hyperparameter tuning for Prophet.
    """

    best_mae = float("inf")
    best_params = None

    train_df = prophet_df.iloc[:-horizon]
    valid_df = prophet_df.iloc[-horizon:]

    train_df = train_df.copy()
    valid_df = valid_df.copy()

    train_df["y"] = np.log(train_df["y"].clip(lower=1))
    valid_df["y"] = np.log(valid_df["y"].clip(lower=1))

    for params in param_grid:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=params["changepoint_prior_scale"],
            seasonality_prior_scale=params["seasonality_prior_scale"],
            seasonality_mode="multiplicative",
            interval_width=params["interval_width"],
        )

        model.add_seasonality("monthly", period=30.5, fourier_order=8, prior_scale=10)
        model.add_seasonality("quarterly", period=91.25, fourier_order=4, prior_scale=5)
        for reg in ["price", "discount", "marketing_spend"]:
            model.add_regressor(reg, prior_scale=5, standardize=True)

        model.fit(train_df)

        forecast = model.predict(valid_df)

        mae = mean_absolute_error(valid_df["y"], forecast["yhat"])

        if mae < best_mae:
            best_mae = mae
            best_params = params

    return best_params, round(best_mae, 3)
