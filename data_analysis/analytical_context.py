import pandas as pd


def _corr_strength(value: float) -> str:
    abs_val = abs(value)
    if abs_val >= 0.7:
        return "strong"
    elif abs_val >= 0.4:
        return "moderate"
    elif abs_val >= 0.2:
        return "weak"
    return "negligible"


def build_analytical_context(df: pd.DataFrame) -> dict:
    """
    Builds structured, LLM-ready analytical context.
    """

    context = {}

    # ---------------- OVERALL STATS ----------------
    context["overall"] = {
        "avg_units_sold": round(df["units_sold"].mean(), 2),
        "avg_price": round(df["price"].mean(), 2),
        "avg_discount": round(df["discount"].mean(), 2),
        "avg_marketing_spend": round(df["marketing_spend"].mean(), 2),
        "sales_volatility": round(df["units_sold"].std() / df["units_sold"].mean(), 2),
    }

    # ---------------- CORRELATIONS ----------------
    corr_df = df[
        ["units_sold", "price", "discount", "marketing_spend"]
    ].corr()

    correlations = {}
    for var in ["price", "discount", "marketing_spend"]:
        val = corr_df.loc["units_sold", var]
        correlations[var] = {
            "value": round(val, 3),
            "strength": _corr_strength(val),
            "direction": "positive" if val > 0 else "negative",
            "business_note": "Correlation only; causality not implied",
        }

    context["correlations"] = correlations

    # ---------------- CATEGORY SUMMARY ----------------
    context["by_category"] = (
        df.groupby("product_category")
        .agg(
            avg_units=("units_sold", "mean"),
            avg_price=("price", "mean"),
            avg_discount=("discount", "mean"),
            avg_marketing=("marketing_spend", "mean"),
        )
        .round(2)
        .to_dict(orient="index")
    )

    # ---------------- SEASONAL SIGNAL ----------------
    df["month"] = df["date"].dt.month
    context["monthly_pattern"] = (
        df.groupby("month")["units_sold"]
        .mean()
        .round(2)
        .to_dict()
    )

    return context
