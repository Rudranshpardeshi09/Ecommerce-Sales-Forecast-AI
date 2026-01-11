def generate_plot_insight(x_col: str, y_col: str, corr: float) -> str:
    """
    Generates concise, executive-safe insight for charts.
    """

    abs_corr = abs(corr)

    if abs_corr >= 0.7:
        strength = "strong"
        confidence = "High confidence"
    elif abs_corr >= 0.4:
        strength = "moderate"
        confidence = "Medium confidence"
    elif abs_corr >= 0.2:
        strength = "weak"
        confidence = "Low confidence"
    else:
        strength = "negligible"
        confidence = "Low confidence"

    direction = "positive" if corr > 0 else "negative"

    return f"""
• **{strength.title()} {direction} relationship** between **{x_col}** and **{y_col}**
• Indicates association only — **not causation**
• Suitable for directional insight, not standalone decision-making
• {confidence}
"""
