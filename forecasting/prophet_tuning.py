from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import numpy as np

def tune_prophet(prophet_df, param_grid, horizon=30):
    """
    Simple grid search for Prophet hyperparameters
    Returns best parameter set based on MAE
    """

    best_mae = float("inf")
    best_params = None

    for params in param_grid:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=params["changepoint_prior_scale"],
            seasonality_prior_scale=params["seasonality_prior_scale"],
            interval_width=params["interval_width"]
        )

        model.add_seasonality("monthly", period=30.5, fourier_order=5)

        model.add_regressor("price")
        model.add_regressor("discount")
        model.add_regressor("marketing_spend")

        df = prophet_df.copy()
        df["y"] = np.log(df["y"].clip(lower=1))

        model.fit(df)

        future = model.make_future_dataframe(periods=horizon)
        for col in ["price", "discount", "marketing_spend"]:
            future[col] = df[col].tail(30).mean()

        forecast = model.predict(future)

        y_true = df["y"].tail(horizon)
        y_pred = forecast["yhat"].iloc[:horizon]

        mae = mean_absolute_error(y_true, y_pred)

        if mae < best_mae:
            best_mae = mae
            best_params = params

    return best_params, best_mae
