from sklearn.ensemble import RandomForestRegressor


def train_residual_model(train_df, forecast_df):
    """
    Train enhanced residual ML model on Prophet log-scale residuals.
    Improves forecast accuracy by capturing patterns Prophet misses.
    """

    df = train_df.copy()
    df["prophet_pred"] = forecast_df.loc[: len(df) - 1, "yhat"].values
    df["residual"] = df["y"] - df["prophet_pred"]

    # Add interaction features for better residual modeling
    df["price_discount_interaction"] = df["price"] * df["discount"]
    df["marketing_price_interaction"] = df["marketing_spend"] * df["price"]
    
    features = df[[
        "price", "discount", "marketing_spend",
        "price_discount_interaction", "marketing_price_interaction"
    ]]
    target = df["residual"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(features, target)
    return model


def apply_residual_correction(forecast, residual_model, future_df):
    """
    Apply residual correction on log scale with damping factor.
    Prevents over-correction that could inflate uncertainty.
    """

    # Prepare features with interactions
    future_features = future_df[[
        "price", "discount", "marketing_spend"
    ]].copy()
    future_features["price_discount_interaction"] = (
        future_features["price"] * future_features["discount"]
    )
    future_features["marketing_price_interaction"] = (
        future_features["marketing_spend"] * future_features["price"]
    )

    residual_pred = residual_model.predict(future_features)
    
    # Apply damping factor to prevent over-correction
    damping_factor = 0.7
    forecast["yhat_hybrid"] = forecast["yhat"] + (residual_pred * damping_factor)
    return forecast
