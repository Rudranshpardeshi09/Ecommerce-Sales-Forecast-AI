# def analyze_relationship(df, x_col, y_col):
#     corr = df[[x_col, y_col]].corr().iloc[0, 1]

#     abs_corr = abs(corr)
#     if abs_corr >= 0.7:
#         strength = "strong"
#     elif abs_corr >= 0.4:
#         strength = "moderate"
#     else:
#         strength = "weak"

#     direction = "positive" if corr > 0 else "negative"

#     drivers = {"price", "discount", "marketing_spend"}
#     outcome = "units_sold"

#     if x_col in drivers and y_col == outcome:
#         causal = f"{x_col} impacts {y_col}"
#     elif y_col in drivers and x_col == outcome:
#         causal = f"{y_col} impacts {x_col}"
#     else:
#         causal = "associative (no clear causal direction)"

#     return {
#         "correlation": round(corr, 3),
#         "strength": strength,
#         "direction": direction,
#         "causal_statement": causal
#     }
# data_analysis/relationship_analysis.py
import pandas as pd


def _strength_label(corr: float) -> str:
    abs_corr = abs(corr)
    if abs_corr >= 0.7:
        return "strong"
    elif abs_corr >= 0.4:
        return "moderate"
    elif abs_corr >= 0.2:
        return "weak"
    return "negligible"


def analyze_relationship(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    """
    Analyzes statistical relationship with business-safe wording.
    """

    corr = df[[x_col, y_col]].corr().iloc[0, 1]
    strength = _strength_label(corr)

    if corr > 0:
        direction = "positive"
    elif corr < 0:
        direction = "negative"
    else:
        direction = "neutral"

    # ---- Business-safe causality framing ----
    drivers = {"price", "discount", "marketing_spend"}
    outcome = "units_sold"

    if strength in ["strong", "moderate"]:
        if x_col in drivers and y_col == outcome:
            causality = (
                f"{x_col} is a meaningful demand driver, "
                "though causality is not proven"
            )
        elif y_col in drivers and x_col == outcome:
            causality = (
                f"{y_col} is a meaningful demand driver, "
                "though causality is not proven"
            )
        else:
            causality = "Statistically related variables with no direct causal direction"
    else:
        causality = (
            "Relationship is weak and should not be treated as a primary business driver"
        )

    # ---- Confidence score (for UI & LLM) ----
    confidence = (
        "High" if strength == "strong"
        else "Medium" if strength == "moderate"
        else "Low"
    )

    return {
        "correlation": round(corr, 3),
        "strength": strength,
        "direction": direction,
        "confidence": confidence,
        "causal_statement": causality,
    }
