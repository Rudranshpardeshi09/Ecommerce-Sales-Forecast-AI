def generate_plot_insight(x_col, y_col, corr):
    strength = abs(corr)

    if strength > 0.7:
        level = "strong"
    elif strength > 0.4:
        level = "moderate"
    else:
        level = "weak"

    direction = "positive" if corr > 0 else "negative"

    return f"""
• There is a **{level} {direction} relationship** between **{x_col}** and **{y_col}**.
• This relationship is **associative** and should **not be interpreted as causal**.
"""
