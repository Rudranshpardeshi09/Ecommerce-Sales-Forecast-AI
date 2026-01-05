def analyze_relationship(df, x_col, y_col):
    corr = df[[x_col, y_col]].corr().iloc[0, 1]

    abs_corr = abs(corr)
    if abs_corr >= 0.7:
        strength = "strong"
    elif abs_corr >= 0.4:
        strength = "moderate"
    else:
        strength = "weak"

    direction = "positive" if corr > 0 else "negative"

    drivers = {"price", "discount", "marketing_spend"}
    outcome = "units_sold"

    if x_col in drivers and y_col == outcome:
        causal = f"{x_col} impacts {y_col}"
    elif y_col in drivers and x_col == outcome:
        causal = f"{y_col} impacts {x_col}"
    else:
        causal = "associative (no clear causal direction)"

    return {
        "correlation": round(corr, 3),
        "strength": strength,
        "direction": direction,
        "causal_statement": causal
    }
