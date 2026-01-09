import streamlit as st
import pandas as pd
import plotly.express as px
from preprocessing.data_preprocessing import load_data, preprocess_data
from forecasting.prophet_model import category_wise_forecast, train_prophet, what_if_forecast
from insights.insights_generator import generate_insights
from llm.llm_qa import ask_llm, explain_chart
from data_analysis.plot_insights import generate_plot_insight
from forecasting.model_evaluation import evaluate_forecast
from data_analysis.relationship_analysis import analyze_relationship

if "show_intro" not in st.session_state:
    st.session_state.show_intro = True

# ---------------- GLOBAL UI THEME ----------------
st.markdown(
    """
<style>

/* ========== GLOBAL BACKGROUND ========== */
html, body, [class*="css"] {
    background-color: #0e1117 !important;
    color: #e6e6e6 !important;
    font-family: "Inter", sans-serif;
}

/* ========== SIDEBAR ========== */
section[data-testid="stSidebar"] {
    background-color: #111827 !important;
    border-right: 1px solid #1f2933;
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* ========== HEADERS ========== */
h1, h2, h3, h4 {
    color: #f9fafb !important;
    font-weight: 600;
}

/* ========== METRIC CARDS ========== */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #111827, #1f2937);
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.4);
    border: 1px solid #374151;
}

/* Metric label */
div[data-testid="metric-container"] label {
    color: #9ca3af !important;
}

/* Metric value */
div[data-testid="metric-container"] div {
    color: #22c55e !important;
    font-size: 26px;
    font-weight: 700;
}

/* ========== CARDS / SECTIONS ========== */
.card {
    background: linear-gradient(135deg, #111827, #1f2937);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.45);
    border: 1px solid #374151;
}

/* ========== BUTTONS ========== */
button {
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
    color: #000 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    border: none !important;
}

button:hover {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
}

/* ========== INPUTS & SELECTS ========== */
input, textarea, select {
    background-color: #020617 !important;
    color: #e5e7eb !important;
    border-radius: 8px !important;
    border: 1px solid #374151 !important;
}

/* ========== RADIO / CHECKBOX ========== */
label {
    color: #d1d5db !important;
}

/* ========== EXPANDERS ========== */
details {
    background-color: #020617 !important;
    border-radius: 10px;
    padding: 10px;
    border: 1px solid #374151;
}

/* ========== PLOTLY CONTAINER FIX ========== */
div[data-testid="stPlotlyChart"] {
    background-color: #020617;
    border-radius: 14px;
    padding: 10px;
}

/* ========== DIVIDERS ========== */
hr {
    border: 1px solid #1f2937;
}

/* ========== SCROLLBAR ========== */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #374151;
    border-radius: 10px;
}

</style>
""",
    unsafe_allow_html=True
)

def render_confidence_text(text: str):
    """
    Adds confidence badges and color coding.
    Presentation-only change.
    """
    text = text.replace(
        "High confidence",
        "🟢 <span style='color:green;font-weight:bold'>High confidence</span>"
    )
    text = text.replace(
        "Medium confidence",
        "🟠 <span style='color:orange;font-weight:bold'>Medium confidence</span>"
    )
    text = text.replace(
        "Low confidence",
        "🔴 <span style='color:red;font-weight:bold'>Low confidence</span>"
    )
    return text

 # ---------------- LOAD & PREPROCESS ----------------
df = load_data("data/commerce_Sales_Prediction_Dataset.csv")
processed_df, prophet_df, _ = preprocess_data(df)

# ---------- INTRO CARD STATE ----------
if "intro_step" not in st.session_state:
    st.session_state.intro_step = 0

st.markdown("""
    <style>
    .intro-card {
        background: linear-gradient(145deg, #020617, #020617);
        border: 1px solid #1e293b;
        border-radius: 18px;
        padding: 32px;
        margin-top: 20px;
        box-shadow: 0 0 25px rgba(34,197,94,0.15);
        animation: fadeIn 0.7s ease-in-out;
    }

    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(15px);}
        to {opacity: 1; transform: translateY(0);}
    }

    .neon-title {
        font-size: 34px;
        font-weight: 800;
        background: linear-gradient(90deg, #22c55e, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .neon-sub {
        color: #94a3b8;
        font-size: 16px;
    }

    .nav-btn {
        display: flex;
        justify-content: space-between;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

if st.session_state.intro_step == 0:
    st.markdown("""
        <div class="intro-card">
            <div class="neon-title">🚀 Sales Forecasting & AI Insight Platform</div>

            1. A decision-intelligence system that combines time-series forecasting, 
            statistical relationship analysis, and LLM-driven business explanations 
            to support data-driven decision making.

            2. Core Technologies Used

            3. Prophet for time-series forecasting

            4. Pandas, NumPy, and Plotly for data processing and visualization

            5. Gemini LLM for executive-level insights

            6. Streamlit for the interactive dashboard interface
        </div>
    """, unsafe_allow_html=True)


    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start", key="start_btn"):
            st.session_state.intro_step = 1
            st.rerun()

elif st.session_state.intro_step == 1:
    st.markdown("""
        <div class="intro-card">
            <div class="neon-title">📂 Dataset Overview</div>

                The dataset captures transactional and marketing drivers influencing 
                demand. It supports trend analysis, driver attribution, and forecasting.
                Key columns include:
                - Date: Transaction date
                - Product Category: Type of product sold
                - Customer Segment: Buyer segment
                - Units Sold: Quantity sold
                - Price: Sale price
                - Discount: Discount applied
                - Marketing Spend: Promotional expenditure

        </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        processed_df[
            ["date", "product_category", "units_sold", "price", "discount", "marketing_spend"]
        ].head(),
        width="stretch"
    )

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        if st.button("⬅ Previous"):
            st.session_state.intro_step = 0
            st.rerun() 
    with col6:
        if st.button("Next ➡"):
            st.session_state.intro_step = 2
            st.rerun() 

elif st.session_state.intro_step == 2:
    st.markdown("""
        <div class="intro-card">
            <div class="neon-title">🧠 What This Project Delivers</div>
            
            - This platform transforms raw sales data into actionable intelligence.
            Prophet models identify demand trends and seasonality, while statistical
            relationship analysis highlights key demand drivers.
        
            - An integrated LLM layer converts forecasts and metrics into executive-ready
            explanations, enabling faster, data-driven decisions across pricing,
            promotions, and inventory planning.
            
            - The system is designed for both short-term operational confidence and
            long-term strategic awareness, with uncertainty transparently communicated.
            
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("⬅ Previous"):
            st.session_state.intro_step = 1
            st.rerun() 
    with col4:
        if st.button("🚀 Enter Dashboard"):
            st.session_state.intro_step = 3
            st.rerun() 

# ⛔ STOP HERE if intro is still active
if st.session_state.intro_step < 3:
    st.stop()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    layout="wide"
)
# ---------------- DASHBOARD TITLE ----------------
st.markdown("""
<div style="display:flex; align-items:center; gap:15px;">
    <h1>📊 Sales Forecast Dashboard</h1>
</div>
<p style="color:#6b7280;">Executive overview of sales performance, trends, and forecasts</p>
""", unsafe_allow_html=True)

st.divider()
# ---------------- MAIN DASHBOARD ----------------
if st.session_state.intro_step >= 3:

        # ---------------- TRAIN MODEL ----------------
        model, forecast = train_prophet(prophet_df)

        # ---------------- INSIGHTS ----------------
        insights = generate_insights(forecast)
        eval_metrics = evaluate_forecast(
            prophet_df["y"].tail(30),
            forecast["yhat"].iloc[:30]
        )


        # ---------------- METRICS (UNCHANGED) ----------------
        st.markdown("### 🔑 Key Metrics")

        with st.container():
            col1, col2, col3 = st.columns(3)

            col1.metric("Avg Units Sold", insights["average_sales"])
            col2.metric("Best Day", insights["best_day"])
            col3.metric("Worst Day", insights["worst_day"])




        # ---------------- OVERALL FORECAST (UNCHANGED) ----------------
        st.subheader("📈 Overall Sales Forecast")
        with st.expander("ℹ️ What does this forecast show?"):
            st.write(
                """
                This chart shows the **predicted future units sold over time**.
                - The central line is the expected demand forecast.
                - Predictions are based on historical trends, seasonality,
                price, discount, and marketing spend.
                """
            )

        # st.line_chart(forecast.set_index("ds")[["yhat"]])
        fig_forecast = px.line(
            forecast,
            x="ds",
            y="yhat",
            title="Overall Sales Forecast (Units Sold)",
            labels={"ds": "Date", "yhat": "Units Sold"},
            color_discrete_sequence=["#22c55e"]
        )

        fig_forecast.update_traces(
            hovertemplate="Date: %{x}<br>Units Sold: %{y}<extra></extra>"
        )

        st.plotly_chart(fig_forecast, width="stretch")


        # -----executive summary button-----
        st.divider()
        st.header("🧠 Executive Summary")

        st.caption(
            "Click to generate an AI-driven executive summary. "
        )

        if st.button("📌 Generate Executive Summary", key="exec_summary_btn"):

            summary = ask_llm(
                """
            You are a senior business analyst preparing an executive briefing for leadership.

            TASK:
            Generate a single executive summary paragraph strictly based on data, forecasts, and model performance.

            MANDATORY DATA POINTS (ALL MUST BE USED):
            - Overall sales trend direction
            - Short-term forecast confidence based on MAE and RMSE
            - Long-term forecast uncertainty
            - Impact of price, discount, and marketing on demand
            - Business suitability of the model for planning decisions

            CONTEXT:
            - Forecasting model: Prophet with regressors
            - Model accuracy metrics: MAE and RMSE (lower is better)
            - Forecast uncertainty increases with time horizon
            - Relationships represent correlation, not causation

            STRICT OUTPUT RULES:
            - EXACTLY 70–90 words
            - ONE paragraph only
            - No bullet points
            - Executive, analytical tone
            - No generic phrases
            - No disclaimers like “data not available”
            - Do not explain methodology
            - Focus on implications and decisions

            AUDIENCE:
            C-level executives and senior management
            """,
                insights,
                df=processed_df,
                eval_metrics=eval_metrics
            )

            # st.markdown(render_confidence_text(summary), unsafe_allow_html=True)
            st.markdown(summary)

            st.markdown("</div>", unsafe_allow_html=True)


        with st.expander("📉 Understanding forecast uncertainty"):
            st.write(
                """
                Prophet forecasts include **uncertainty intervals**:

                - **Lower bound (yhat_lower)**: Conservative estimate  
                - **Upper bound (yhat_upper)**: Optimistic estimate  
                - **Actual demand is likely to fall within this range**

                Wider bands = higher uncertainty  
                Narrow bands = more stable demand pattern
                """
            )


        # ---------------- BUSINESS INSIGHTS (UNCHANGED) ----------------
        st.subheader("📌 Business Insights")
        for k, v in insights.items():
            st.write(f"**{k.replace('_',' ').title()}**: {v}")

        st.divider()
        st.header("📦 Category Growth Comparison")

        st.caption(
            "Compares recent sales momentum across product categories to identify growth leaders and laggards."
        )

        # ---------------- CATEGORY GROWTH LOGIC ----------------
        category_trend_df = (
            processed_df
            .groupby(["product_category", "date"])["units_sold"]
            .sum()
            .reset_index()
        )

        # Use recent window (last 30 days vs previous 30 days)
        latest_date = category_trend_df["date"].max()
        recent_start = latest_date - pd.Timedelta(days=30)
        previous_start = latest_date - pd.Timedelta(days=60)

        recent_sales = (
            category_trend_df[category_trend_df["date"] >= recent_start]
            .groupby("product_category")["units_sold"]
            .sum()
        )

        previous_sales = (
            category_trend_df[
                (category_trend_df["date"] >= previous_start) &
                (category_trend_df["date"] < recent_start)
            ]
            .groupby("product_category")["units_sold"]
            .sum()
        )

        growth_df = (
            pd.DataFrame({
                "Recent Sales": recent_sales,
                "Previous Sales": previous_sales
            })
            .fillna(0)
        )

        growth_df["Growth %"] = (
            (growth_df["Recent Sales"] - growth_df["Previous Sales"])
            / growth_df["Previous Sales"].replace(0, 1)
        ) * 100

        # ---------------- GROWTH STATUS LABEL ----------------
        def growth_badge(value):
            if value > 5:
                return "🟢 Growing ↑"
            elif value < -5:
                return "🔴 Declining ↓"
            else:
                return "🟠 Stable →"

        growth_df["Status"] = growth_df["Growth %"].apply(growth_badge)

        growth_df = growth_df.reset_index().sort_values("Growth %", ascending=False)

        fig_growth = px.bar(
            growth_df,
            x="product_category",
            y="Growth %",
            color="Growth %",
            color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
            title="Category-wise Sales Growth (Last 30 Days vs Previous 30 Days)"
        )

        fig_growth.update_layout(
            xaxis_title="Product Category",
            yaxis_title="Growth Percentage (%)",
            template="plotly_dark",
            height=420
        )

        st.plotly_chart(fig_growth, width="stretch")

        if st.button("📊 Explain Category Growth with AI", key="category_growth_ai"):

            category_summary = growth_df[[
                "product_category", "Recent Sales", "Previous Sales", "Growth %"
            ]].to_dict(orient="records")

            prompt = f"""
        You are a senior business analyst preparing an executive briefing.

        DATA PROVIDED:
        Category-wise sales comparison between two periods (last 30 days vs previous 30 days):
        {category_summary}

        OBJECTIVE:
        Explain which categories are growing, which are declining, and what this means for the business.

        INSTRUCTIONS:
        - Length: 60–80 words
        - Executive, decision-oriented tone
        - Highlight growth leaders and laggards
        - Avoid speculation
        - Do NOT use bullet points
        - Do NOT mention data availability issues
        - Focus on demand momentum and planning implications
        """

            explanation = ask_llm(
                prompt,
                insights,
                df=processed_df,
                eval_metrics=eval_metrics
            )

            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(180deg, #020617, #020617);
                    border: 1px solid #1e293b;
                    border-radius: 14px;
                    padding: 20px;
                ">
                {explanation}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.subheader("🤖 Explain Category Performance")

        for _, row in growth_df.iterrows():
            category = row["product_category"]
            growth_pct = row["Growth %"]
            status = row["Status"]

            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(
                    f"**{category}** — {status} ({growth_pct:.1f}%)"
                )

            with col2:
                if st.button(
                    f"Explain {category}",
                    key=f"explain_{category}"
                ):
                    if "Declining" in status:
                        prompt = f"""
                    You are a senior business analyst.

                    CATEGORY: {category}
                    STATUS: Declining ({growth_pct:.1f}%)

                    TASK:
                    Explain why this category is declining and what the forecast implies.

                    RULES:
                    - 50–60 words
                    - No optimistic framing
                    - Focus on risk, trend, and corrective action
                    - Use forecast cautiously
                    """
                    elif "Growing" in status:
                        prompt = f"""
                    You are a senior business analyst.

                    CATEGORY: {category}
                    STATUS: Growing ({growth_pct:.1f}%)

                    TASK:
                    Explain growth sustainability using forecast outlook.

                    RULES:
                    - 50–60 words
                    - Highlight opportunities and risks
                    """
                    else:
                        prompt = f"""
                    You are a senior business analyst.

                    CATEGORY: {category}
                    STATUS: Stable ({growth_pct:.1f}%)

                    TASK:
                    Explain stability and potential directional risks.

                    RULES:
                    - 50–60 words
                    """

                    explanation = ask_llm(
                        prompt,
                        insights,
                        df=processed_df,
                        eval_metrics=eval_metrics
                    )

                    st.markdown(
                        f"""
                        <div style="
                            background:#020617;
                            border:1px solid #1e293b;
                            border-radius:12px;
                            padding:16px;
                            margin-bottom:12px;
                        ">
                        {explanation}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        # ============================================================
        # 🔥 NEW FEATURES BELOW (EMBEDDED, NOT BREAKING)
        # ============================================================

        st.divider()
        st.header("📊 Data Relationships Explorer")

        numeric_cols = ["units_sold", "price", "discount", "marketing_spend"]

        selected_col = st.selectbox(
            "Select a column",
            numeric_cols,
            key="data_relationships_column_selector"
        )

        chart_type = st.radio(
            "Select visualization type",
            ["Bar (Binned Average)", "Line (Trend)", "Histogram",  "Scatter", "Density"],
            horizontal=True,
            key="data_relationships_chart_type"
        )

        # ---------- GRID LAYOUT ----------
        cols_per_row = 2
        cards = []

        for col in numeric_cols:
            if col != selected_col:
                cards.append(col)

        rows = [cards[i:i + cols_per_row] for i in range(0, len(cards), cols_per_row)]

        for row in rows:
            col_layout = st.columns(cols_per_row)

            for idx, col in enumerate(row):
                with col_layout[idx]:
                    st.markdown(
                        f"""
                        <div style="
                            background: linear-gradient(180deg, #0f172a, #020617);
                            padding: 18px;
                            border-radius: 14px;
                            border: 1px solid #1e293b;
                        ">
                        <h4 style="margin-bottom:8px;">{selected_col.replace('_',' ').title()} vs {col.replace('_',' ').title()}</h4>
                        """,
                        unsafe_allow_html=True
                    )

                    # -------- CHART --------
                    if chart_type == "Scatter":
                        fig = px.scatter(
                            processed_df,
                            x=selected_col,
                            y=col,
                            opacity=0.3
                        )
                    elif chart_type == "Density":
                        fig = px.density_heatmap(
                            processed_df,
                            x=selected_col,
                            y=col,
                            nbinsx=30,
                            nbinsy=30,
                            color_continuous_scale="Blues"
                        )
                    elif chart_type == "Line (Trend)":
                        sorted_df = processed_df.sort_values(selected_col)
                        fig = px.line(sorted_df, x=selected_col, y=col)
                    elif chart_type == "Histogram":
                        fig = px.histogram(processed_df, x=selected_col, nbins=30)
                    else:
                        temp = processed_df[[selected_col, col]].dropna().copy()
                        bins = pd.qcut(temp[selected_col], q=10, duplicates="drop")
                        temp["bin"] = bins.apply(lambda x: f"{int(x.left)}–{int(x.right)}")
                        summary = temp.groupby("bin")[col].mean().reset_index()
                        fig = px.bar(summary, x="bin", y=col)

                    fig.update_layout(
                        template="plotly_dark",
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=260
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # -------- QUICK INSIGHT (NO LLM) --------
                    corr_value = processed_df[[selected_col, col]].corr().iloc[0, 1]
                    st.markdown(generate_plot_insight(selected_col, col, corr_value))

                    # -------- AI EXPLANATION (ON DEMAND ONLY) --------
                    explain_key = f"ai_{selected_col}_{col}_{chart_type}"

                    if st.button("🤖 Explain with AI", key=explain_key):
                        analysis = analyze_relationship(processed_df, selected_col, col)

                        explanation = explain_chart(
                            x_col=selected_col,
                            y_col=col,
                            chart_type=chart_type,
                            correlation=analysis["correlation"],
                            strength=analysis["strength"],
                            direction=analysis["direction"],
                            causal_statement=analysis["causal_statement"]
                        )

                        st.markdown("**📘 Executive Explanation**")
                        st.write(explanation)

                    st.markdown("</div>", unsafe_allow_html=True)


        # ✅ LLM EXPLANATION
        st.divider()
        st.markdown("""
        <div style="background:black;padding:20px;border-radius:12px;
        box-shadow:0 4px 10px rgba(0,0,0,0.08);">
        <h2>🤖 LLM Relationship Explainer</h2>
        <p style="color:#6b7280;">
        Data-driven explanation of relationships for business decision-making
        </p>
        </div>
        """, unsafe_allow_html=True)


        st.caption(
            "Select any two variables to get a clear, data-driven explanation "
            "of their relationship."
        )

        col1, col2 = st.columns(2)

        with col1:
            explain_x = st.selectbox(
                "Select X variable",
                numeric_cols,
                key="llm_explain_x"
            )

        with col2:
            explain_y = st.selectbox(
                "Select Y variable",
                numeric_cols,
                key="llm_explain_y"
            )

        if explain_x == explain_y:
            st.warning("Please select two different variables.")
        else:
            if st.button("Explain Relationship", key="llm_explain_btn"):

                # -------- CORRELATION (DATA-DRIVEN) --------
                analysis = analyze_relationship(
            processed_df,
            explain_x,
            explain_y
        )


                st.markdown(
                    generate_plot_insight(explain_x, explain_y, analysis["correlation"])
                )

                explanation = explain_chart(
            x_col=explain_x,
            y_col=explain_y,
            chart_type="Relationship Analysis",
            correlation=analysis["correlation"],
            strength=analysis["strength"],
            direction=analysis["direction"],
            causal_statement=analysis["causal_statement"]
        )




                st.subheader("📘 Explanation (Plain English)")
                st.write(explanation)


        # ---------------- SIDEBAR CONTROLS ----------------
        st.sidebar.markdown("## 🔧 Forecast Filters")
        st.sidebar.markdown("---")


        selected_category = st.sidebar.selectbox(
            "Select Product Category",
            processed_df["product_category"].unique()
        )

        discount_change = st.sidebar.slider(
            "Increase Discount (%)",
            min_value=0,
            max_value=50,
            value=0
        ) / 100

        marketing_change = st.sidebar.slider(
            "Increase Marketing Spend (%)",
            min_value=0,
            max_value=100,
            value=0
        ) / 100

        # ---------------- CATEGORY-WISE FORECAST ----------------
        st.subheader(f"📦 Category-wise Forecast: {selected_category}")

        category_forecast = category_wise_forecast(
            processed_df,
            selected_category
        )

        with st.expander("ℹ️ How to interpret category-wise forecast"):
            st.write(
                """
                This chart isolates demand for the selected product category.
                It helps compare how different categories respond to
                pricing, discounts, and marketing activity.
                """
            )

        fig_category = px.line(
            category_forecast,
            x="ds",
            y="yhat",
            title=f"Forecast for {selected_category}",
            labels={"ds": "Date", "yhat": "Units Sold"},
        )

        fig_category.update_traces(
            hovertemplate="Date: %{x}<br>Units Sold: %{y}<extra></extra>"
        )

        st.plotly_chart(fig_category, width="stretch")


        # ---------------- GROWTH VS FORECAST OUTLOOK ----------------
        st.divider()
        st.header("📈 Growth vs Forecast Outlook")

        st.caption(
            "Combines recent growth momentum with forward-looking demand forecasts."
        )

        forecast_summary = []

        for category in growth_df["product_category"]:
            cat_forecast = category_wise_forecast(
                processed_df,
                category,
                days=30
            )

            avg_future_demand = cat_forecast["yhat"].tail(30).mean()
            recent_growth = growth_df.loc[
                growth_df["product_category"] == category,
                "Growth %"
            ].values[0]

            forecast_summary.append({
                "Category": category,
                "Recent Growth (%)": recent_growth,
                "Avg Forecast Demand (30d)": avg_future_demand
            })

        forecast_summary_df = pd.DataFrame(forecast_summary)

        fig_combo = px.scatter(
            forecast_summary_df,
            x="Recent Growth (%)",
            y="Avg Forecast Demand (30d)",
            text="Category",
            size="Avg Forecast Demand (30d)",
            color="Recent Growth (%)",
            color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
            title="Category Growth vs Forecasted Demand"
        )

        fig_combo.update_traces(textposition="top center")
        fig_combo.update_layout(template="plotly_dark", height=480)

        st.plotly_chart(fig_combo, width="stretch")


        # ---------------- WHAT-IF ANALYSIS ----------------
        st.subheader("🔮 What-If Scenario Analysis")

        what_if = what_if_forecast(
            model,
            prophet_df,
            discount_change=discount_change,
            marketing_change=marketing_change
        )

        with st.expander("ℹ️ What-if scenario explanation"):
            st.write(
                """
                This simulation shows **expected demand changes** if you modify
                discount or marketing spend.
                - Baseline: last known real values
                - Scenario: adjusted values from sliders
                """
            )

        fig_whatif = px.line(
            what_if,
            x="ds",
            y="yhat",
            title="What-If Scenario Forecast",
            labels={"ds": "Date", "yhat": "Units Sold"},
        )

        fig_whatif.update_traces(
            hovertemplate="Date: %{x}<br>Projected Units Sold: %{y}<extra></extra>"
        )

        st.plotly_chart(fig_whatif, width="stretch")

        # ---------------- LLM Q&A (UNCHANGED) ----------------
        st.divider()
        st.subheader("🤖 Ask Business Questions")
        st.caption("Click a question to get instant insights. Custom questions are optional.")

        # ---------- PRESET QUESTIONS ----------
        preset_questions = {
            "📈 Overall sales trend & confidence":
                "What is the overall sales trend and how confident is the forecast?",
            "💰 Price impact on demand":
                "How does price affect units sold in the short and long term?",
            "🏷️ Discount effectiveness":
                "Does increasing discount significantly improve sales volume?",
            "📢 Marketing ROI":
                "How effective is marketing spend in driving additional sales?",
            "🔮 Next 30-day outlook":
                "What should the business expect in the next 30 days?"
        }

        for label, question_text in preset_questions.items():
            if st.button(label, key=f"preset_{label}"):
                answer = ask_llm(
                    question_text,
                    insights,
                    df=processed_df,
                    eval_metrics=eval_metrics
                )
                st.markdown(render_confidence_text(answer), unsafe_allow_html=True)
                st.divider()

        # ---------- OPTIONAL CUSTOM QUESTION ----------
        st.markdown("### ✍️ Ask a custom question (optional)")

        custom_question = st.text_input(
            "Type your business question",
            placeholder="e.g. What happens if marketing spend increases by 20% next month?"
        )

        if st.button("Ask AI", key="btn_custom_question") and custom_question.strip():
            answer = ask_llm(
                custom_question,
                insights,
                df=processed_df,
                eval_metrics=eval_metrics
            )
            st.markdown(render_confidence_text(answer), unsafe_allow_html=True)

        # model performance    
        st.markdown("### 📐 Model Performance")

        with st.container():
            col1, col2 = st.columns(2)
            col1.metric("MAE", eval_metrics["MAE"])
            col2.metric("RMSE", eval_metrics["RMSE"])


        with st.expander("🤖 Model Performance Explained"):
            st.caption("Click a button to get an AI explanation.")

            col_a, col_b = st.columns(2)

            with col_a:
                if st.button("📊 Is this model good?", key="btn_model_quality"):
                    explanation = ask_llm(
                        "Is this forecasting model accurate and reliable for business decisions?",
                        insights,
                        df=processed_df,
                        eval_metrics=eval_metrics
                    )
                    st.markdown(render_confidence_text(explanation), unsafe_allow_html=True)

            with col_b:
                if st.button("⚠️ How confident is the forecast?", key="btn_model_confidence"):
                    explanation = ask_llm(
                        "Explain forecast confidence using MAE, RMSE, and uncertainty over time.",
                        insights,
                        df=processed_df,
                        eval_metrics=eval_metrics
                    )
                    st.markdown(render_confidence_text(explanation), unsafe_allow_html=True)


