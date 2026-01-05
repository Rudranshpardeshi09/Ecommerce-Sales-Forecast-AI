import pandas as pd

def build_analytical_context(df: pd.DataFrame):
    """
    Extracts historical, statistical, and relational insights
    used by the LLM for precise answers.
    """

    context = {}

    # ---------------- BASIC STATS ----------------
    context["overall"] = {
        "avg_units_sold": round(df["units_sold"].mean(), 2),
        "avg_price": round(df["price"].mean(), 2),
        "avg_discount": round(df["discount"].mean(), 2),
        "avg_marketing_spend": round(df["marketing_spend"].mean(), 2),
    }

    # ---------------- CORRELATIONS ----------------
    corr = df[[
        "units_sold",
        "price",
        "discount",
        "marketing_spend"
    ]].corr()

    context["correlations"] = corr["units_sold"].to_dict()

    # ---------------- CATEGORY-WISE SUMMARY ----------------
    category_summary = (
        df.groupby("product_category")
        .agg(
            avg_units=("units_sold", "mean"),
            avg_price=("price", "mean"),
            avg_discount=("discount", "mean"),
            avg_marketing=("marketing_spend", "mean")
        )
        .round(2)
        .to_dict(orient="index")
    )

    context["by_category"] = category_summary

    return context
