import streamlit as st
import pandas as pd
import plotly.express as px
import re
from preprocessing.data_preprocessing import load_data, preprocess_data
from forecasting.prophet_model import category_wise_forecast, train_prophet, what_if_forecast
from insights.insights_generator import generate_insights
from llm.llm_qa import ask_llm, explain_chart
from data_analysis.plot_insights import generate_plot_insight
from forecasting.model_evaluation import evaluate_forecast
from data_analysis.relationship_analysis import analyze_relationship

if "show_intro" not in st.session_state:
    st.session_state.show_intro = True

# Initialize sidebar page state
if "active_sidebar_page" not in st.session_state:
    st.session_state.active_sidebar_page = None

# Initialize category explanation state
if "active_category_explanation" not in st.session_state:
    st.session_state.active_category_explanation = None

# Initialize business question answer state
if "active_question_answer" not in st.session_state:
    st.session_state.active_question_answer = None

# Initialize relationship explanation state
if "active_relationship_explanation" not in st.session_state:
    st.session_state.active_relationship_explanation = None

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
# df = load_data("data/commerce_Sales_Prediction_Dataset.csv")
# processed_df, prophet_df, _ = preprocess_data(df)
@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    df = load_data("data/commerce_Sales_Prediction_Dataset.csv")
    return preprocess_data(df)

processed_df, prophet_df, _ = load_and_prepare_data()

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

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.markdown("## More Options")
st.sidebar.markdown("---")

# Sidebar buttons
sidebar_pages = {
    "Data Relationships Explorer": "data_relationships",
    "Category-wise Forecast": "category_forecast",
    "What-If Scenario Analysis": "what_if",
    "LLM Relationship Explainer": "llm_relationship",
    "Executive Summary": "executive_summary",
    "Model Performance Explained": "model_performance",
    "Ask Business Questions": "ask_questions"
}

for page_name, page_key in sidebar_pages.items():
    if st.sidebar.button(page_name, key=f"sidebar_{page_key}", width='stretch'):
        st.session_state.active_sidebar_page = page_key
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Close button at bottom 
st.sidebar.markdown("""
    <style>
    .stButton > button[kind=""] {
        background: linear-gradient(135deg, #ef4444, #dc2626) !important;
        color: white !important;
    }
    .stButton > button[kind=""]:hover {
        background: linear-gradient(135deg, #dc2626, #b91c1c) !important;
    }
    </style>
""", unsafe_allow_html=True)

if st.sidebar.button("Close Options", key="close_sidebar", width='stretch'):
    st.session_state.active_sidebar_page = None
    st.rerun()

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
    @st.cache_resource(show_spinner=True)
    def get_trained_model(prophet_df):
        return train_prophet(prophet_df)

    model, forecast = get_trained_model(prophet_df)

    # ---------------- INSIGHTS ----------------
    insights = generate_insights(forecast)
    eval_metrics = evaluate_forecast(
        actual=prophet_df["y"].iloc[-30:],
        predicted=forecast["yhat"].iloc[-30:]
    )

    # ---------------- OVERALL FORECAST ----------------
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
    
    # Generate forecast insights button
    if "forecast_insights_shown" not in st.session_state:
        st.session_state.forecast_insights_shown = False
    
    if st.button("🔍 Generate Insight", key="generate_forecast_insight", width='content'):
        st.session_state.forecast_insights_shown = not st.session_state.forecast_insights_shown
        st.rerun()
    
    if st.session_state.forecast_insights_shown:
        # Generate business analyst insights from the forecast
        def generate_forecast_insights(forecast, insights, eval_metrics):
            """Generate 5 business analyst insights from the forecast graph"""
            points = []
            
            # 1. Trend direction and momentum
            trend_series = forecast["trend"].rolling(7).mean()
            trend_change = trend_series.iloc[-1] - trend_series.iloc[0]
            if trend_change > 0:
                momentum = "upward"
                direction_icon = "📈"
            else:
                momentum = "downward"
                direction_icon = "📉"
            trend_pct = abs((trend_change / abs(trend_series.iloc[0])) * 100) if trend_series.iloc[0] != 0 else 0
            points.append(f"{direction_icon} **Trend Analysis**: The forecast shows a {momentum} trajectory with approximately {trend_pct:.1f}% change over the forecast period, indicating {'sustained growth momentum' if momentum == 'upward' else 'potential demand contraction'}.")
            
            # 2. Volatility and stability
            last_30 = forecast["yhat"].tail(30)
            volatility = (last_30.std() / last_30.mean()) * 100 if last_30.mean() > 0 else 0
            if volatility < 10:
                stability = "highly stable"
                stability_icon = "✅"
            elif volatility < 20:
                stability = "moderately stable"
                stability_icon = "⚖️"
            else:
                stability = "volatile"
                stability_icon = "⚠️"
            points.append(f"{stability_icon} **Demand Stability**: Forecast volatility of {volatility:.1f}% indicates {stability} demand patterns, {'enabling reliable inventory planning' if volatility < 15 else 'requiring flexible supply chain management'}.")
            
            # 3. Peak vs trough performance
            peak = forecast["yhat"].max()
            trough = forecast["yhat"].min()
            peak_date = forecast.loc[forecast["yhat"].idxmax(), "ds"]
            trough_date = forecast.loc[forecast["yhat"].idxmin(), "ds"]
            peak_trough_gap = ((peak - trough) / trough) * 100 if trough > 0 else 0
            points.append(f"📊 **Performance Range**: Peak demand ({peak:.0f} units) occurs around {peak_date.strftime('%B %d')} while lowest demand ({trough:.0f} units) is projected around {trough_date.strftime('%B %d')}, representing a {peak_trough_gap:.1f}% variation that {'suggests strong seasonality' if peak_trough_gap > 30 else 'indicates relatively consistent demand'}.")
            
            # 4. Forecast confidence and uncertainty
            uncertainty_range = ((forecast["yhat_upper"].tail(30) - forecast["yhat_lower"].tail(30)).mean()) / last_30.mean() * 100
            mae = eval_metrics.get("MAE", 0)
            if uncertainty_range < 15 and mae < last_30.mean() * 0.1:
                confidence = "high"
                confidence_icon = "🟢"
            elif uncertainty_range < 25:
                confidence = "moderate"
                confidence_icon = "🟠"
            else:
                confidence = "low"
                confidence_icon = "🔴"
            points.append(f"{confidence_icon} **Forecast Confidence**: With a mean absolute error of {mae:.2f} units and uncertainty bands of ±{uncertainty_range:.1f}%, the forecast demonstrates {confidence} reliability for {'operational decision-making' if confidence == 'high' else 'strategic planning purposes'}.")
            
            # 5. Business implications and recommendations
            avg_forecast = last_30.mean()
            forecast_change = ((forecast["yhat"].iloc[-1] - forecast["yhat"].iloc[0]) / forecast["yhat"].iloc[0]) * 100 if forecast["yhat"].iloc[0] > 0 else 0
            if forecast_change > 5:
                implication = "positive growth outlook"
                recommendation = "consider capacity expansion and inventory buildup"
            elif forecast_change < -5:
                implication = "declining demand trend"
                recommendation = "implement demand stimulation strategies or optimize inventory levels"
            else:
                implication = "stable demand expectations"
                recommendation = "maintain current operational levels with focus on efficiency"
            points.append(f"💼 **Business Implications**: The forecast projects {forecast_change:.1f}% change by period end, indicating a {implication}. Management should {recommendation} to align with projected demand.")
            
            return points
        
        forecast_insights = generate_forecast_insights(forecast, insights, eval_metrics)
        
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #111827, #1f2937);
                border: 2px solid #22c55e;
                border-radius: 16px;
                padding: 24px;
                margin: 20px 0;
                box-shadow: 0 10px 30px rgba(34, 197, 94, 0.2);
            ">
                <h4 style="color: #22c55e; margin-bottom: 20px; font-size: 18px; font-weight: 600;">
                    📊 Business Analyst Insights: Sales Forecast Analysis
                </h4>
                <ol style="color: #e5e7eb; line-height: 1.8; font-size: 14px; padding-left: 20px; margin: 0;">
                    {''.join([f'<li style="margin-bottom: 16px;">{point}</li>' for point in forecast_insights])}
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------- BUSINESS INSIGHTS ----------------
    st.subheader("📌 Business Insights")
    
    # Generate additional insights
    def generate_additional_insights(insights, forecast, processed_df, eval_metrics):
        """Generate additional business insights from available data"""
        additional = {}
        
        # Growth trajectory
        trend_series = forecast["trend"].rolling(7).mean()
        first_value = trend_series.iloc[0]
        last_value = trend_series.iloc[-1]
        
        # Handle NaN or zero values
        if pd.isna(first_value) or first_value == 0 or pd.isna(last_value):
            # Fallback: use raw trend values
            first_value = forecast["trend"].iloc[0]
            last_value = forecast["trend"].iloc[-1]
        
        if pd.isna(first_value) or first_value == 0:
            trend_change_pct = 0
        else:
            trend_change_pct = ((last_value - first_value) / abs(first_value)) * 100
        
        if pd.isna(trend_change_pct):
            trend_change_pct = 0
        
        additional["growth_trajectory"] = f"{abs(trend_change_pct):.1f}% {'increase' if trend_change_pct > 0 else 'decrease'}"
        
        # Forecast uncertainty range
        last_30_forecast = forecast["yhat"].tail(30)
        uncertainty_range = ((forecast["yhat_upper"].tail(30) - forecast["yhat_lower"].tail(30)).mean()) / last_30_forecast.mean() * 100
        additional["forecast_uncertainty"] = f"±{uncertainty_range:.1f}% range"
        
        # Peak vs average performance
        peak_forecast = forecast["yhat"].max()
        avg_forecast = forecast["yhat"].mean()
        peak_advantage = ((peak_forecast - avg_forecast) / avg_forecast) * 100
        additional["peak_performance"] = f"{peak_advantage:.1f}% above average"
        
        # Model accuracy summary
        mae = eval_metrics.get("MAE", 0)
        rmse = eval_metrics.get("RMSE", 0)
        avg_value = last_30_forecast.mean()
        accuracy_pct = (1 - (mae / avg_value)) * 100 if avg_value > 0 else 0
        additional["model_accuracy"] = f"{accuracy_pct:.1f}% accuracy"
        
        # Price sensitivity (from processed_df)
        if "price" in processed_df.columns and "units_sold" in processed_df.columns:
            price_corr = processed_df[["price", "units_sold"]].corr().iloc[0, 1]
            if abs(price_corr) < 0.2:
                additional["price_sensitivity"] = "Low sensitivity"
            elif price_corr < 0:
                additional["price_sensitivity"] = "Negative correlation"
            else:
                additional["price_sensitivity"] = "Positive correlation"
        
        # Marketing effectiveness
        if "marketing_spend" in processed_df.columns:
            marketing_corr = processed_df[["marketing_spend", "units_sold"]].corr().iloc[0, 1]
            if abs(marketing_corr) >= 0.4:
                additional["marketing_effectiveness"] = "Strong impact"
            elif abs(marketing_corr) >= 0.2:
                additional["marketing_effectiveness"] = "Moderate impact"
            else:
                additional["marketing_effectiveness"] = "Limited impact"
        
        # Category diversity
        if "product_category" in processed_df.columns:
            num_categories = processed_df["product_category"].nunique()
            additional["category_diversity"] = f"{num_categories} categories"
        
        # Forecast horizon confidence
        short_term_std = forecast["yhat"].tail(15).std()
        long_term_std = forecast["yhat"].tail(30).std()
        if long_term_std / short_term_std > 1.2:
            additional["forecast_horizon"] = "Decreasing confidence"
        else:
            additional["forecast_horizon"] = "Stable confidence"
        
        return additional
    
    # Combine original and additional insights
    additional_insights = generate_additional_insights(insights, forecast, processed_df, eval_metrics)
    
    # Prepare all insights for display with icons and categories
    all_insights = [
        # Trend & Performance
        {"icon": "📈", "title": "Sales Trend", "value": insights.get("trend", "N/A"), "subtitle": insights.get("trend_confidence", "")},
        {"icon": "🎯", "title": "Growth Trajectory", "value": additional_insights.get("growth_trajectory", "N/A"), "subtitle": "7-day rolling trend"},
        {"icon": "⭐", "title": "Best Performance Period", "value": insights.get("best_period", "N/A"), "subtitle": "Peak weekly average"},
        {"icon": "📉", "title": "Worst Performance Period", "value": insights.get("worst_period", "N/A"), "subtitle": "Lowest weekly average"},
        
        # Forecast Metrics
        {"icon": "📊", "title": "Average Daily Sales", "value": f"{insights.get('average_sales', 0):.2f} units", "subtitle": "Last 30 days baseline"},
        {"icon": "📅", "title": "Monthly Baseline", "value": f"{insights.get('baseline_units_per_month', 0):.0f} units", "subtitle": "30-day projection"},
        {"icon": "🔮", "title": "Forecast Uncertainty", "value": additional_insights.get("forecast_uncertainty", "N/A"), "subtitle": "Confidence interval range"},
        
        # Model Performance
        {"icon": "🎯", "title": "Model Accuracy", "value": additional_insights.get("model_accuracy", "N/A"), "subtitle": "Based on MAE & RMSE"},
        {"icon": "📐", "title": "Forecast Stability", "value": insights.get("forecast_stability", "N/A"), "subtitle": insights.get("forecast_confidence", "")},
        {"icon": "📈", "title": "Forecast Horizon", "value": additional_insights.get("forecast_horizon", "N/A"), "subtitle": "Time-based confidence"},
        
        # Market Dynamics
        {"icon": "📢", "title": "Marketing Effectiveness", "value": additional_insights.get("marketing_effectiveness", "N/A"), "subtitle": "Spend impact analysis"},
        {"icon": "📦", "title": "Product Diversity", "value": additional_insights.get("category_diversity", "N/A"), "subtitle": "Category coverage"},
    ]
    
    # Display insights in a grid layout (3 columns)
    cols_per_row = 3
    num_rows = (len(all_insights) + cols_per_row - 1) // cols_per_row
    
    for row_idx in range(num_rows):
        cols = st.columns(cols_per_row)
        start_idx = row_idx * cols_per_row
        end_idx = min(start_idx + cols_per_row, len(all_insights))
        
        for col_idx, col in enumerate(cols):
            if start_idx + col_idx < end_idx:
                insight = all_insights[start_idx + col_idx]
                with col:
                    st.markdown(
                        f"""
                        <div style="
                            background: linear-gradient(135deg, #111827, #1f2937);
                            border: 1px solid #374151;
                            border-radius: 14px;
                            padding: 20px;
                            margin-bottom: 16px;
                            box-shadow: 0 6px 18px rgba(0,0,0,0.4);
                            transition: transform 0.2s, box-shadow 0.2s;
                            height: 100%;
                            display: flex;
                            flex-direction: column;
                            justify-content: space-between;
                        ">
                            <div>
                                <div style="
                                    font-size: 32px;
                                    margin-bottom: 12px;
                                    display: flex;
                                    align-items: center;
                                    gap: 10px;
                                ">
                                    {insight["icon"]}
                                    <h4 style="
                                        margin: 0;
                                        color: #f9fafb;
                                        font-size: 16px;
                                        font-weight: 600;
                                    ">{insight["title"]}</h4>
                                </div>
                                <div style="
                                    color: #22c55e;
                                    font-size: 24px;
                                    font-weight: 700;
                                    margin: 12px 0;
                                ">{insight["value"]}</div>
                            </div>
                            <div style="
                                color: #9ca3af;
                                font-size: 12px;
                                margin-top: auto;
                                padding-top: 8px;
                                border-top: 1px solid #374151;
                            ">{insight["subtitle"]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    st.divider()
    st.header("📦 Category Growth Comparison")

    st.caption(
        "Compares recent sales momentum across product categories to identify growth leaders and laggards."
    )

    # ---------------- CATEGORY GROWTH LOGIC ----------------
    category_trend_df = (
    processed_df
    .groupby(["product_category", "date"], observed=True)["units_sold"]
    .sum()
    .reset_index()
    )

    # Use recent window (last 30 days vs previous 30 days)
    latest_date = category_trend_df["date"].max()
    recent_start = latest_date - pd.Timedelta(days=30)
    previous_start = latest_date - pd.Timedelta(days=60)

    recent_sales = (
        category_trend_df[category_trend_df["date"] >= recent_start]
        .groupby("product_category",observed=True)["units_sold"]
        .sum()
    )

    previous_sales = (
        category_trend_df[
            (category_trend_df["date"] >= previous_start) &
            (category_trend_df["date"] < recent_start)
        ]
        .groupby("product_category",observed=True)["units_sold"]
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

    st.subheader("🤖 Explain Category Performance")

    # Display categories with their info and buttons side by side
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
                key=f"explain_{category}",
                use_container_width=True
            ):
                st.session_state.active_category_explanation = category
                st.rerun()

    # Display explanation box when a category is selected
    if st.session_state.active_category_explanation is not None:
        selected_cat = st.session_state.active_category_explanation
        cat_row = growth_df[growth_df["product_category"] == selected_cat].iloc[0]
        category = cat_row["product_category"]
        growth_pct = cat_row["Growth %"]
        status = cat_row["Status"]

        explanation = ask_llm(
            question=category,
            insights={
                "category_status": status,
                "category_growth": f"{growth_pct:.1f}%"
            },
            df=processed_df,
            intent="category_explanation"
        )

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #111827, #1f2937);
                border: 2px solid #22c55e;
                border-radius: 16px;
                padding: 24px;
                margin: 20px 0;
                box-shadow: 0 10px 30px rgba(34, 197, 94, 0.2);
            ">
                <h4 style="color: #22c55e; margin-bottom: 12px;">📊 Explanation for {category}</h4>
                <p style="color: #e5e7eb; line-height: 1.6; font-size: 15px;">{explanation}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ================= INSIGHTS DASHBOARD =================
    st.divider()
    st.header("📊 Sales & Marketing Insights Dashboard")
    st.caption("Complete analysis of sales trends, customer segments, marketing effectiveness, and product performance")
    
    # -------- Sales & Category Analysis (3x3 Grid) --------
    st.subheader("📈 Sales & Category Performance (Top Insights)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1. Sales by Product Category")
        category_sales = processed_df.groupby('product_category')['units_sold'].sum().sort_values(ascending=False)
        fig1 = px.bar(
            x=category_sales.index,
            y=category_sales.values,
            labels={"x": "Product Category", "y": "Units Sold"},
            color=category_sales.values,
            color_continuous_scale="Viridis"
        )
        fig1.update_layout(template="plotly_dark", height=350, showlegend=False)
        st.plotly_chart(fig1, width='stretch')
    
    with col2:
        st.markdown("### 2. Customer Segment Contribution")
        segment_sales = processed_df.groupby('customer_segment')['units_sold'].sum().sort_values(ascending=False)
        fig2 = px.pie(
            values=segment_sales.values,
            names=segment_sales.index,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig2.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig2, width='stretch')
    
    with col3:
        st.markdown("### 3. Monthly Sales Trend - 2023")
        sales_2023 = processed_df[processed_df['date'].dt.year == 2023].copy()
        monthly_sales = sales_2023.groupby(sales_2023['date'].dt.month)['units_sold'].sum()
        fig3 = px.line(
            x=monthly_sales.index,
            y=monthly_sales.values,
            labels={"x": "Month", "y": "Units Sold"},
            markers=True,
            color_discrete_sequence=["#22c55e"]
        )
        fig3.update_layout(template="plotly_dark", height=350, showlegend=False)
        st.plotly_chart(fig3, width='stretch')
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("### 4. Discount Sensitivity by Segment")
        processed_df_temp = processed_df.copy()
        processed_df_temp['Discount_Bucket'] = pd.cut(
            processed_df_temp['discount'],
            bins=[-1, 10, 30, 60],
            labels=['Low Discount', 'Medium Discount', 'High Discount']
        )
        discount_impact = processed_df_temp.groupby(['customer_segment', 'Discount_Bucket'], observed=True)['units_sold'].mean().unstack()
        fig4 = px.bar(
            discount_impact.reset_index(),
            x='customer_segment',
            y=discount_impact.columns.tolist(),
            title="Units Sold by Discount Level",
            barmode='group'
        )
        fig4.update_layout(template="plotly_dark", height=350, xaxis_title="Customer Segment", yaxis_title="Avg Units Sold")
        st.plotly_chart(fig4, width='stretch')
    
    with col5:
        st.markdown("### 5. Marketing Spend Distribution")
        category_marketing = processed_df.groupby('product_category')['marketing_spend'].sum().sort_values(ascending=False)
        fig5 = px.bar(
            x=category_marketing.index,
            y=category_marketing.values,
            labels={"x": "Product Category", "y": "Marketing Spend"},
            color=category_marketing.values,
            color_continuous_scale="Blues"
        )
        fig5.update_layout(template="plotly_dark", height=350, showlegend=False)
        st.plotly_chart(fig5, width='stretch')
    
    with col6:
        st.markdown("### 6. Category vs Segment Sales Heatmap")
        heatmap_data = processed_df.groupby(['customer_segment', 'product_category'])['units_sold'].sum().unstack()
        fig6 = px.imshow(
            heatmap_data,
            labels=dict(x="Product Category", y="Customer Segment", color="Units Sold"),
            color_continuous_scale="Greens",
            aspect="auto"
        )
        fig6.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig6, width='stretch')
    
    # -------- Marketing & ROI Analysis (5x5 Grid) --------
    st.divider()
    st.subheader("💰 Marketing Effectiveness & ROI Analysis")
    
    col7, col8, col9 = st.columns(3)
    
    with col7:
        st.markdown("### 1. Marketing Efficiency by Category")
        category_efficiency = (
            processed_df.groupby('product_category')[['marketing_spend', 'units_sold']]
            .sum()
        )
        category_efficiency['Efficiency'] = (
            category_efficiency['units_sold'] / category_efficiency['marketing_spend']
        )
        category_efficiency = category_efficiency.sort_values('Efficiency')
        fig7 = px.bar(
            x=category_efficiency['Efficiency'],
            y=category_efficiency.index,
            orientation='h',
            color=category_efficiency['Efficiency'],
            color_continuous_scale="Reds"
        )
        fig7.update_layout(template="plotly_dark", height=300, xaxis_title="Units per ₹ Spent", yaxis_title="Category", showlegend=False)
        st.plotly_chart(fig7, width='stretch')
    
    with col8:
        st.markdown("### 2. Marketing ROI by Category")
        roi_category = (
            processed_df.groupby('product_category')[['marketing_spend', 'units_sold']]
            .sum()
        )
        roi_category['ROI'] = (
            (roi_category['units_sold'] - roi_category['marketing_spend']) /
            roi_category['marketing_spend'] * 100
        )
        roi_category = roi_category.sort_values('ROI')
        fig8 = px.bar(
            x=roi_category['ROI'],
            y=roi_category.index,
            orientation='h',
            color=roi_category['ROI'],
            color_continuous_scale="RdYlGn"
        )
        fig8.update_layout(template="plotly_dark", height=300, xaxis_title="ROI (%)", yaxis_title="Category", showlegend=False)
        st.plotly_chart(fig8, width='stretch')
    
    with col9:
        st.markdown("### 3. Segment-wise Marketing ROI")
        roi_segment = (
            processed_df.groupby('customer_segment')[['marketing_spend', 'units_sold']]
            .sum()
        )
        roi_segment['ROI'] = (
            (roi_segment['units_sold'] - roi_segment['marketing_spend']) /
            roi_segment['marketing_spend'] * 100
        )
        fig9 = px.bar(
            x=roi_segment.index,
            y=roi_segment['ROI'],
            color=roi_segment['ROI'],
            color_continuous_scale="RdYlGn",
            labels={"x": "Customer Segment", "y": "ROI (%)"}
        )
        fig9.update_layout(template="plotly_dark", height=300, showlegend=False)
        st.plotly_chart(fig9, width='stretch')

    col10, col11 = st.columns(2)

    with col10:
        st.markdown("### 4. Marketing vs Sales Comparison")
        category_summary = (
            processed_df.groupby('product_category')[['marketing_spend', 'units_sold']]
            .sum()
        )
        fig10 = px.bar(
            category_summary.reset_index(),
            x='product_category',
            y=['marketing_spend', 'units_sold'],
            barmode='group',
            labels={"value": "Amount", "product_category": "Category"}
        )
        fig10.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig10, width='stretch')
    
    with col11:
        st.markdown("### 5. Monthly Marketing ROI - 2023")
        roi_month = (
            processed_df[processed_df['date'].dt.year == 2023]
            .groupby(processed_df['date'].dt.month)[['marketing_spend', 'units_sold']]
            .sum()
        )
        roi_month['ROI'] = (
            (roi_month['units_sold'] - roi_month['marketing_spend']) /
            roi_month['marketing_spend'] * 100
        )
        fig11 = px.line(
            x=roi_month.index,
            y=roi_month['ROI'],
            markers=True,
            color_discrete_sequence=["#f59e0b"],
            labels={"x": "Month", "y": "ROI (%)"}
        )
        fig11.update_layout(template="plotly_dark", height=300, showlegend=False)
        st.plotly_chart(fig11, width='stretch')

    col12, col13 = st.columns(2)

    with col12:
        st.markdown("### 6. Monthly Marketing ROI - 2024")
        roi_month = (
            processed_df[processed_df['date'].dt.year == 2024]
            .groupby(processed_df['date'].dt.month)[['marketing_spend', 'units_sold']]
            .sum()
        )
        roi_month['ROI'] = (
            (roi_month['units_sold'] - roi_month['marketing_spend']) /
            roi_month['marketing_spend'] * 100
        )
        fig11 = px.line(
            x=roi_month.index,
            y=roi_month['ROI'],
            markers=True,
            color_discrete_sequence=["#f59e0b"],
            labels={"x": "Month", "y": "ROI (%)"}
        )
        fig11.update_layout(template="plotly_dark", height=300, showlegend=False)
        st.plotly_chart(fig11, width='stretch')

    with col13:
        st.markdown("### 7. Monthly Marketing ROI - 2025")
        roi_month = (
            processed_df[processed_df['date'].dt.year == 2025]
            .groupby(processed_df['date'].dt.month)[['marketing_spend', 'units_sold']]
            .sum()
        )
        roi_month['ROI'] = (
            (roi_month['units_sold'] - roi_month['marketing_spend']) /
            roi_month['marketing_spend'] * 100
        )
        fig11 = px.line(
            x=roi_month.index,
            y=roi_month['ROI'],
            markers=True,
            color_discrete_sequence=["#f59e0b"],
            labels={"x": "Month", "y": "ROI (%)"}
        )
        fig11.update_layout(template="plotly_dark", height=300, showlegend=False)
        st.plotly_chart(fig11, width='stretch')

    # -------- GROWTH VS FORECAST OUTLOOK ----------------
    st.divider()
    st.header("📈 Growth vs Forecast Outlook")

    st.caption(
        "Combines recent growth momentum with forward-looking demand forecasts."
    )

    @st.cache_data(show_spinner=False)
    def build_growth_forecast_df(processed_df, growth_df):
        rows = []

        for category in growth_df["product_category"]:
            cat_forecast = category_wise_forecast(
                processed_df, category, days=30
            )

            rows.append({
                "Category": category,
                "Recent Growth (%)": growth_df.loc[
                    growth_df["product_category"] == category,
                    "Growth %"
                ].values[0],
                "Avg Forecast Demand (30d)": cat_forecast["yhat"].tail(30).mean()
            })

        return pd.DataFrame(rows)

    forecast_summary_df = build_growth_forecast_df(
        processed_df, growth_df
    )

    # 🔒 SAFETY CHECK (VERY IMPORTANT)
    st.write("Forecast Summary Preview")
    st.dataframe(forecast_summary_df)

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
    
    # Explain Table button
    if "table_explanation_shown" not in st.session_state:
        st.session_state.table_explanation_shown = False
    
    if st.button("🔍 Explain Table", key="explain_table_btn", width='content'):
        st.session_state.table_explanation_shown = not st.session_state.table_explanation_shown
        st.rerun()
    
    if st.session_state.table_explanation_shown:
        # Generate 7-8 points explaining the table
        def generate_table_explanation(forecast_summary_df, growth_df):
            """Generate 7-8 business analyst insights explaining the growth vs forecast table"""
            points = []
            
            # 1. Overall pattern analysis
            high_growth_categories = forecast_summary_df[forecast_summary_df["Recent Growth (%)"] > 10]
            low_growth_categories = forecast_summary_df[forecast_summary_df["Recent Growth (%)"] < -10]
            if len(high_growth_categories) > len(low_growth_categories):
                pattern = "positive momentum across most categories"
                pattern_icon = "📈"
            elif len(low_growth_categories) > len(high_growth_categories):
                pattern = "declining trends in majority of categories"
                pattern_icon = "📉"
            else:
                pattern = "mixed performance with balanced growth and decline"
                pattern_icon = "⚖️"
            points.append(f"{pattern_icon} **Overall Pattern**: The table reveals {pattern}, indicating {'strong market positioning' if len(high_growth_categories) > len(low_growth_categories) else 'potential market challenges requiring strategic intervention'}.")
            
            # 2. Growth leaders
            top_growth = forecast_summary_df.nlargest(1, "Recent Growth (%)")
            if len(top_growth) > 0:
                top_cat = top_growth.iloc[0]
                points.append(f"⭐ **Growth Leader**: {top_cat['Category']} shows the strongest recent growth at {top_cat['Recent Growth (%)']:.1f}%, with forecasted demand of {top_cat['Avg Forecast Demand (30d)']:.0f} units, suggesting {'sustained market demand' if top_cat['Recent Growth (%)'] > 20 else 'recovering market position'}.")
            
            # 3. High growth, high forecast correlation
            positive_growth = forecast_summary_df[forecast_summary_df["Recent Growth (%)"] > 0]
            if len(positive_growth) > 0:
                avg_forecast_positive = positive_growth["Avg Forecast Demand (30d)"].mean()
                avg_forecast_all = forecast_summary_df["Avg Forecast Demand (30d)"].mean()
                if avg_forecast_positive > avg_forecast_all * 1.1:
                    correlation = "strong positive correlation"
                    correlation_icon = "✅"
                else:
                    correlation = "moderate correlation"
                    correlation_icon = "⚖️"
                points.append(f"{correlation_icon} **Growth-Forecast Correlation**: Growing categories average {avg_forecast_positive:.0f} units forecasted demand, showing {correlation} between recent momentum and future expectations, {'validating growth sustainability' if correlation == 'strong positive correlation' else 'indicating potential forecast adjustment needs'}.")
            
            # 4. Declining categories analysis
            declining = forecast_summary_df[forecast_summary_df["Recent Growth (%)"] < -5]
            if len(declining) > 0:
                avg_forecast_declining = declining["Avg Forecast Demand (30d)"].mean()
                points.append(f"⚠️ **Declining Categories**: {len(declining)} category(ies) show negative growth, with average forecasted demand of {avg_forecast_declining:.0f} units, suggesting {'market repositioning may be required' if avg_forecast_declining < forecast_summary_df['Avg Forecast Demand (30d)'].mean() * 0.8 else 'forecast indicates potential recovery'}.")
            
            # 5. Forecast demand distribution
            max_forecast = forecast_summary_df["Avg Forecast Demand (30d)"].max()
            min_forecast = forecast_summary_df["Avg Forecast Demand (30d)"].min()
            forecast_range = ((max_forecast - min_forecast) / min_forecast) * 100 if min_forecast > 0 else 0
            if forecast_range > 100:
                distribution = "highly variable"
                distribution_icon = "📊"
            else:
                distribution = "relatively uniform"
                distribution_icon = "📈"
            points.append(f"{distribution_icon} **Demand Distribution**: Forecasted demand ranges from {min_forecast:.0f} to {max_forecast:.0f} units across categories ({forecast_range:.1f}% variation), indicating {distribution} category performance that {'requires category-specific strategies' if distribution == 'highly variable' else 'enables standardized planning approaches'}.")
            
            # 6. Strategic quadrant analysis
            high_growth_high_demand = forecast_summary_df[
                (forecast_summary_df["Recent Growth (%)"] > 5) & 
                (forecast_summary_df["Avg Forecast Demand (30d)"] > forecast_summary_df["Avg Forecast Demand (30d)"].median())
            ]
            if len(high_growth_high_demand) > 0:
                points.append(f"🎯 **Strategic Winners**: {len(high_growth_high_demand)} category(ies) combine strong recent growth (>5%) with above-median forecasted demand, representing priority investment opportunities for resource allocation and expansion initiatives.")
            
            # 7. Forecast vs growth alignment
            misaligned = forecast_summary_df[
                ((forecast_summary_df["Recent Growth (%)"] > 10) & (forecast_summary_df["Avg Forecast Demand (30d)"] < forecast_summary_df["Avg Forecast Demand (30d)"].median())) |
                ((forecast_summary_df["Recent Growth (%)"] < -10) & (forecast_summary_df["Avg Forecast Demand (30d)"] > forecast_summary_df["Avg Forecast Demand (30d)"].median()))
            ]
            if len(misaligned) > 0:
                points.append(f"🔍 **Forecast-Growth Misalignment**: {len(misaligned)} category(ies) show recent growth trends that {'contradict forecast expectations' if len(misaligned) > 0 else 'require model review'}, indicating potential forecasting model refinement or external factor influences that need investigation.")
            
            # 8. Overall business recommendation
            avg_growth = forecast_summary_df["Recent Growth (%)"].mean()
            total_forecast_demand = forecast_summary_df["Avg Forecast Demand (30d)"].sum()
            if avg_growth > 5:
                recommendation = "capitalize on growth momentum through inventory expansion and marketing amplification"
                rec_icon = "💡"
            elif avg_growth < -5:
                recommendation = "implement demand stimulation strategies and consider portfolio optimization"
                rec_icon = "🛠️"
            else:
                recommendation = "maintain stable operations while monitoring category-specific developments"
                rec_icon = "📊"
            points.append(f"{rec_icon} **Business Recommendation**: With average growth of {avg_growth:.1f}% and total forecasted demand of {total_forecast_demand:.0f} units across all categories, the organization should {recommendation} to align operations with projected market dynamics.")
            
            return points
        
        table_explanation_points = generate_table_explanation(forecast_summary_df, growth_df)
        
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #111827, #1f2937);
                border: 2px solid #22c55e;
                border-radius: 16px;
                padding: 24px;
                margin: 20px 0;
                box-shadow: 0 10px 30px rgba(34, 197, 94, 0.2);
            ">
                <h4 style="color: #22c55e; margin-bottom: 20px; font-size: 18px; font-weight: 600;">
                    📊 Table Analysis: Growth vs Forecast Insights
                </h4>
                <ol style="color: #e5e7eb; line-height: 1.8; font-size: 14px; padding-left: 20px; margin: 0;">
                    {''.join([f'<li style="margin-bottom: 16px;">{point}</li>' for point in table_explanation_points])}
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------- SIDEBAR CONTENT SECTIONS ----------------
    if st.session_state.active_sidebar_page is not None:
        st.divider()
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #111827, #1f2937);
            border-radius: 16px;
            padding: 24px;
            margin: 20px 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.45);
            border: 2px solid #22c55e;
        ">
        """, unsafe_allow_html=True)

        # Data Relationships Explorer
        if st.session_state.active_sidebar_page == "data_relationships":
            st.header("📊 Data Relationships Explorer")

            numeric_cols = ["units_sold", "price", "discount", "marketing_spend"]

            selected_col = st.selectbox(
                "Select a column",
                numeric_cols,
                key="data_relationships_column_selector"
            )

            chart_type = st.radio(
                "Select visualization type",
                ["Bar", "Line (Trend)", "Histogram",  "Scatter", "Density"],
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
                            summary = temp.groupby("bin",observed=True)[col].mean().reset_index()
                            fig = px.bar(summary, x="bin", y=col)

                        fig.update_layout(
                            template="plotly_dark",
                            margin=dict(l=10, r=10, t=10, b=10),
                            height=260
                        )

                        chart_key = f"plot_{selected_col}_{col}_{chart_type}"

                        st.plotly_chart(
                            fig,
                            width="stretch",
                            key=chart_key
                        )

                        # -------- EXPLAIN RELATION BUTTON --------
                        relationship_key = f"{selected_col}_{col}"
                        if st.button(
                            "🔍 Explain Relation",
                            key=f"explain_relation_{relationship_key}",
                            width='stretch'
                        ):
                            st.session_state.active_relationship_explanation = relationship_key
                            st.rerun()

                        # -------- QUICK INSIGHT (SHOWN ONLY WHEN BUTTON CLICKED) --------
                        if st.session_state.active_relationship_explanation == relationship_key:
                            corr_value = processed_df[[selected_col, col]].corr().iloc[0, 1]
                            insight_text = generate_plot_insight(selected_col, col, corr_value)
                            
                            # Convert markdown bullet points to HTML format
                            insight_lines = insight_text.strip().split('\n')
                            html_bullets = []
                            for line in insight_lines:
                                line = line.strip()
                                if line and line.startswith('•'):
                                    # Remove the bullet and convert markdown to HTML
                                    content = line[1:].strip()
                                    # Convert **text** to <strong>text</strong>
                                    content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
                                    html_bullets.append(f'<li style="margin-bottom: 10px; padding-left: 4px;">{content}</li>')
                            
                            formatted_insight = '<ul style="margin: 0; padding-left: 20px; list-style-type: none;">' + ''.join(html_bullets) + '</ul>'
                            
                            st.markdown(
                                f"""
                                <div style="
                                    background: linear-gradient(135deg, #111827, #1f2937);
                                    border: 2px solid #22c55e;
                                    border-radius: 12px;
                                    padding: 20px;
                                    margin-top: 12px;
                                    margin-bottom: 12px;
                                    box-shadow: 0 6px 20px rgba(34, 197, 94, 0.15);
                                ">
                                    <h5 style="color: #22c55e; margin-bottom: 16px; font-size: 16px; font-weight: 600;">
                                        📊 Relationship Analysis: {selected_col.replace('_',' ').title()} vs {col.replace('_',' ').title()}
                                    </h5>
                                    <div style="color: #e5e7eb; line-height: 1.8; font-size: 14px;">
                                        {formatted_insight}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        st.markdown("</div>", unsafe_allow_html=True)

        # Category-wise Forecast
        elif st.session_state.active_sidebar_page == "category_forecast":
            st.header("📦 Category-wise Forecast")
            
            st.markdown("## 🔧 Forecast Filters")
            
            selected_category = st.selectbox(
                "Select Product Category",
                processed_df["product_category"].unique(),
                key="category_forecast_selectbox"
            )
            
            st.divider()
            
            st.subheader(f"Forecast for: {selected_category}")

            @st.cache_data(show_spinner=False)
            def get_category_forecast(df, category, days=60):
                return category_wise_forecast(df, category, days)

            category_forecast = get_category_forecast(
                processed_df, selected_category
            )

            with st.expander("ℹ️ How to interpret category-wise forecast"):
                st.write(
                    """
                    **What This Graph Shows:**
                    
                    This chart displays the predicted future sales for the specific product category you've selected. 
                    Think of it as a business forecast that tells you how many units of this category you can expect 
                    to sell in the coming days.
                    
                    **Key Things to Look For:**
                    
                    1. **Direction**: Is the line going up, down, or staying flat? An upward trend means sales are 
                       expected to grow, while a downward trend suggests declining demand.
                    
                    2. **Patterns**: Look for repeating patterns - these might indicate seasonality (e.g., higher 
                       sales during certain months or weeks).
                    
                    3. **Stability**: A smooth line suggests predictable demand, while a wavy line indicates more 
                       variable sales expectations.
                    
                    4. **Comparison**: Use this to compare different categories - some may grow faster than others, 
                       helping you decide where to focus your marketing efforts or inventory.
                    
                    **Why This Matters:**
                    Understanding category-specific forecasts helps you make better decisions about inventory management, 
                    marketing budgets, and resource allocation for each product line.
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
            
            # Explain Graph button for category forecast
            if "category_graph_insight_shown" not in st.session_state:
                st.session_state.category_graph_insight_shown = False
            
            if st.button("🔍 Explain Graph", key=f"explain_category_graph_{selected_category}", width='content'):
                st.session_state.category_graph_insight_shown = not st.session_state.category_graph_insight_shown
                st.rerun()
            
            if st.session_state.category_graph_insight_shown:
                def generate_category_graph_insights(category_forecast, selected_category):
                    """Generate business analyst insights for category forecast graph"""
                    points = []
                    
                    # 1. Overall trend direction
                    first_value = category_forecast["yhat"].iloc[0]
                    last_value = category_forecast["yhat"].iloc[-1]
                    trend_change = ((last_value - first_value) / first_value) * 100 if first_value > 0 else 0
                    
                    if trend_change > 5:
                        direction = "strong upward trend"
                        direction_icon = "📈"
                        implication = "suggests growing market demand and potential for expansion"
                    elif trend_change > 0:
                        direction = "moderate upward trend"
                        direction_icon = "📊"
                        implication = "indicates steady growth requiring sustained marketing support"
                    elif trend_change < -5:
                        direction = "declining trend"
                        direction_icon = "📉"
                        implication = "signals market challenges requiring strategic intervention"
                    else:
                        direction = "stable/flat trend"
                        direction_icon = "➡️"
                        implication = "suggests consistent demand patterns suitable for stable inventory planning"
                    
                    points.append(f"{direction_icon} **Trend Direction**: The graph for {selected_category} shows a {direction} ({trend_change:.1f}% change over the forecast period), which {implication}.")
                    
                    # 2. Volatility analysis
                    volatility = (category_forecast["yhat"].std() / category_forecast["yhat"].mean()) * 100 if category_forecast["yhat"].mean() > 0 else 0
                    if volatility < 10:
                        stability = "highly stable and predictable"
                        stability_icon = "✅"
                        action = "enables confident inventory planning and consistent supply chain management"
                    elif volatility < 20:
                        stability = "moderately stable"
                        stability_icon = "⚖️"
                        action = "requires flexible inventory management with buffer stock considerations"
                    else:
                        stability = "highly volatile"
                        stability_icon = "⚠️"
                        action = "demands dynamic inventory systems and close monitoring to avoid stockouts or overstock"
                    
                    points.append(f"{stability_icon} **Demand Stability**: With {volatility:.1f}% volatility, {selected_category} exhibits {stability} demand patterns that {action}.")
                    
                    # 3. Peak performance analysis
                    peak = category_forecast["yhat"].max()
                    avg_forecast = category_forecast["yhat"].mean()
                    peak_date = category_forecast.loc[category_forecast["yhat"].idxmax(), "ds"]
                    peak_advantage = ((peak - avg_forecast) / avg_forecast) * 100 if avg_forecast > 0 else 0
                    
                    points.append(f"⭐ **Peak Performance**: Maximum forecasted demand reaches {peak:.0f} units around {peak_date.strftime('%B %d, %Y')}, which is {peak_advantage:.1f}% above the average. This peak period {'represents a strategic opportunity for promotional campaigns' if peak_advantage > 30 else 'indicates relatively consistent demand levels'}.")
                    
                    # 4. Forecast confidence
                    uncertainty_range = ((category_forecast["yhat_upper"] - category_forecast["yhat_lower"]).mean()) / avg_forecast * 100 if avg_forecast > 0 else 0
                    if uncertainty_range < 15:
                        confidence = "high confidence"
                        confidence_icon = "🟢"
                        recommendation = "suitable for operational decision-making and inventory commitments"
                    elif uncertainty_range < 30:
                        confidence = "moderate confidence"
                        confidence_icon = "🟠"
                        recommendation = "appropriate for strategic planning with contingency considerations"
                    else:
                        confidence = "lower confidence"
                        confidence_icon = "🔴"
                        recommendation = "requires caution and regular model updates for reliable planning"
                    
                    points.append(f"{confidence_icon} **Forecast Confidence**: The uncertainty bands (confidence intervals) show {confidence} levels (±{uncertainty_range:.1f}% range), making this forecast {recommendation}.")
                    
                    # 5. Business recommendation
                    if trend_change > 10:
                        rec = "increase inventory allocation and marketing investment to capitalize on strong growth momentum"
                        rec_icon = "💡"
                    elif trend_change < -10:
                        rec = "review pricing strategy, consider promotional campaigns, or reassess product positioning to address declining demand"
                        rec_icon = "🛠️"
                    else:
                        rec = "maintain current operational levels while monitoring market conditions for opportunities or risks"
                        rec_icon = "📊"
                    
                    points.append(f"{rec_icon} **Strategic Recommendation**: Based on the forecast trajectory for {selected_category}, management should {rec} to align business operations with projected demand patterns.")
                    
                    return points
                
                category_insights = generate_category_graph_insights(category_forecast, selected_category)
                
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #111827, #1f2937);
                        border: 2px solid #22c55e;
                        border-radius: 16px;
                        padding: 24px;
                        margin: 20px 0;
                        box-shadow: 0 10px 30px rgba(34, 197, 94, 0.2);
                    ">
                        <h4 style="color: #22c55e; margin-bottom: 20px; font-size: 18px; font-weight: 600;">
                            📊 Business Analyst Insights: {selected_category} Forecast Analysis
                        </h4>
                        <ol style="color: #e5e7eb; line-height: 1.8; font-size: 14px; padding-left: 20px; margin: 0;">
                            {''.join([f'<li style="margin-bottom: 16px;">{point}</li>' for point in category_insights])}
                        </ol>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # What-If Scenario Analysis
        elif st.session_state.active_sidebar_page == "what_if":
            st.header("🔮 What-If Scenario Analysis")
            
            st.markdown("## 🔧 Forecast Filters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                discount_change = st.slider(
                    "Increase Discount (%)",
                    min_value=0,
                    max_value=50,
                    value=0,
                    key="what_if_discount"
                ) / 100
            
            with col2:
                marketing_change = st.slider(
                    "Increase Marketing Spend (%)",
                    min_value=0,
                    max_value=100,
                    value=0,
                    key="what_if_marketing"
                ) / 100
            
            st.divider()
            
            @st.cache_data(show_spinner=False)
            def get_what_if_forecast(_model, prophet_df, d_change, m_change):
                return what_if_forecast(
                    _model,
                    prophet_df,
                    discount_change=d_change,
                    marketing_change=m_change
                )

            what_if = get_what_if_forecast(
                model, prophet_df, discount_change, marketing_change
            )

            with st.expander("ℹ️ What-if scenario explanation"):
                st.write(
                    """
                    **What This Graph Shows:**
                    
                    This is a simulation tool that helps you see what might happen to sales if you change your 
                    business strategy. Think of it as a "crystal ball" for your business decisions - it shows 
                    predicted outcomes before you actually make changes.
                    
                    **How It Works:**
                    
                    1. **Baseline Forecast**: Without any changes, this shows what sales would look like based on 
                       current trends and patterns.
                    
                    2. **Scenario Simulation**: When you adjust the discount or marketing spend sliders, the graph 
                       recalculates to show the potential impact on future sales.
                    
                    3. **Real-World Application**: Use this to answer questions like:
                       - "What if I increase discounts by 20%?"
                       - "How would sales change if I double my marketing budget?"
                       - "Is it worth investing more in promotions?"
                    
                    **What to Look For:**
                    - **Line goes up**: Your proposed changes could increase sales
                    - **Line goes down**: The changes might negatively impact sales
                    - **Line stays similar**: The changes have minimal effect
                    
                    **Important Note:**
                    These are predictions based on historical patterns. Real-world results may vary due to 
                    competition, market conditions, and other factors. Use this as a planning tool, not an 
                    absolute guarantee.
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
            
            # Explain Graph button for what-if scenario
            if "whatif_graph_insight_shown" not in st.session_state:
                st.session_state.whatif_graph_insight_shown = False
            
            if st.button("🔍 Explain Graph", key="explain_whatif_graph", width='content'):
                st.session_state.whatif_graph_insight_shown = not st.session_state.whatif_graph_insight_shown
                st.rerun()
            
            if st.session_state.whatif_graph_insight_shown:
                def generate_whatif_graph_insights(what_if, discount_change, marketing_change):
                    """Generate business analyst insights for what-if scenario graph"""
                    points = []
                    
                    # 1. Scenario summary
                    discount_pct = discount_change * 100
                    marketing_pct = marketing_change * 100
                    
                    if discount_pct > 0 and marketing_pct > 0:
                        scenario = f"combined strategy with {discount_pct:.0f}% discount increase and {marketing_pct:.0f}% marketing spend increase"
                        scenario_icon = "🔄"
                    elif discount_pct > 0:
                        scenario = f"{discount_pct:.0f}% discount increase strategy"
                        scenario_icon = "💰"
                    elif marketing_pct > 0:
                        scenario = f"{marketing_pct:.0f}% marketing spend increase strategy"
                        scenario_icon = "📢"
                    else:
                        scenario = "baseline scenario (no changes)"
                        scenario_icon = "📊"
                    
                    points.append(f"{scenario_icon} **Scenario Overview**: This simulation models the {scenario} compared to baseline operations. The graph shows projected demand changes over the forecast horizon.")
                    
                    # 2. Impact magnitude
                    avg_forecast = what_if["yhat"].mean()
                    # We need baseline for comparison - approximate it
                    baseline_approx = what_if["yhat"].iloc[0]  # First value as baseline proxy
                    impact = ((avg_forecast - baseline_approx) / baseline_approx) * 100 if baseline_approx > 0 else 0
                    
                    if impact > 10:
                        impact_level = "significant positive impact"
                        impact_icon = "📈"
                        interpretation = "strongly suggests the proposed changes would substantially boost sales"
                    elif impact > 3:
                        impact_level = "moderate positive impact"
                        impact_icon = "📊"
                        interpretation = "indicates the strategy could yield meaningful sales improvements"
                    elif impact > -3:
                        impact_level = "minimal impact"
                        impact_icon = "➡️"
                        interpretation = "suggests the changes may not significantly alter demand patterns"
                    else:
                        impact_level = "negative impact"
                        impact_icon = "📉"
                        interpretation = "warns that the proposed changes could reduce sales volumes"
                    
                    points.append(f"{impact_icon} **Expected Impact**: The simulation projects {impact:.1f}% average change in demand, indicating {impact_level} that {interpretation}.")
                    
                    # 3. Trend direction
                    first_val = what_if["yhat"].iloc[0]
                    last_val = what_if["yhat"].iloc[-1]
                    trend = ((last_val - first_val) / first_val) * 100 if first_val > 0 else 0
                    
                    if trend > 5:
                        trend_desc = "accelerating growth trajectory"
                        trend_icon = "🚀"
                    elif trend > 0:
                        trend_desc = "steady upward trajectory"
                        trend_icon = "📈"
                    elif trend > -5:
                        trend_desc = "relatively stable pattern"
                        trend_icon = "➡️"
                    else:
                        trend_desc = "declining trajectory"
                        trend_icon = "📉"
                    
                    points.append(f"{trend_icon} **Trajectory Analysis**: Over the forecast period, demand shows a {trend_desc} ({trend:.1f}% end-to-end change), {'suggesting sustained positive momentum' if trend > 3 else 'indicating potential stability concerns' if trend < -3 else 'reflecting relatively consistent demand expectations'}.")
                    
                    # 4. ROI considerations
                    if discount_pct > 0 and marketing_pct > 0:
                        points.append(f"💼 **Investment Analysis**: The scenario combines price reduction (discount) and marketing investment. While discounting may improve volume, it reduces profit margins. Marketing investment increases costs but may build long-term brand value. The net impact depends on margin structures and customer acquisition costs.")
                    elif discount_pct > 0:
                        points.append(f"💼 **Profitability Considerations**: A {discount_pct:.0f}% discount increase improves volume but reduces per-unit profit margins. The strategy is viable if volume growth compensates for margin compression, typically requiring {abs(discount_pct * 1.5):.0f}%+ volume increase to maintain profitability.")
                    elif marketing_pct > 0:
                        points.append(f"💼 **Marketing ROI**: A {marketing_pct:.0f}% marketing spend increase requires generating sufficient incremental sales to exceed the investment. Break-even analysis suggests the forecasted demand increase should justify the marketing cost increase for positive ROI.")
                    else:
                        points.append(f"💼 **Baseline Comparison**: This represents current operations without changes. Use this as a reference point to evaluate the incremental value of proposed strategic adjustments.")
                    
                    # 5. Risk and uncertainty
                    volatility = (what_if["yhat"].std() / avg_forecast) * 100 if avg_forecast > 0 else 0
                    if volatility > 20:
                        risk = "higher volatility"
                        risk_icon = "⚠️"
                        caution = "demands careful monitoring and flexible execution to adapt to changing market responses"
                    else:
                        risk = "moderate volatility"
                        risk_icon = "✅"
                        caution = "enables more predictable planning and resource allocation"
                    
                    points.append(f"{risk_icon} **Execution Risk**: The forecast exhibits {risk} ({volatility:.1f}% standard deviation), which {caution}. Actual results may vary from projections due to competitive responses, market conditions, and external factors.")
                    
                    # 6. Recommendation
                    if impact > 10 and trend > 3:
                        recommendation = "proceed with implementation, monitor results closely, and scale up if early indicators are positive"
                        rec_icon = "✅"
                    elif impact > 3:
                        recommendation = "consider pilot testing in a limited market or time period before full rollout to validate assumptions"
                        rec_icon = "🔄"
                    elif impact < -3:
                        recommendation = "reconsider the proposed changes or explore alternative strategies with potentially better outcomes"
                        rec_icon = "🛑"
                    else:
                        recommendation = "evaluate the cost-benefit trade-off carefully, as minimal impact may not justify the investment or margin sacrifice"
                        rec_icon = "⚖️"
                    
                    points.append(f"{rec_icon} **Strategic Recommendation**: Given the projected impact of {impact:.1f}% and trajectory of {trend:.1f}%, management should {recommendation} to optimize decision-making and resource allocation.")
                    
                    return points
                
                whatif_insights = generate_whatif_graph_insights(what_if, discount_change, marketing_change)
                
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #111827, #1f2937);
                        border: 2px solid #22c55e;
                        border-radius: 16px;
                        padding: 24px;
                        margin: 20px 0;
                        box-shadow: 0 10px 30px rgba(34, 197, 94, 0.2);
                    ">
                        <h4 style="color: #22c55e; margin-bottom: 20px; font-size: 18px; font-weight: 600;">
                            📊 Business Analyst Insights: What-If Scenario Analysis
                        </h4>
                        <ol style="color: #e5e7eb; line-height: 1.8; font-size: 14px; padding-left: 20px; margin: 0;">
                            {''.join([f'<li style="margin-bottom: 16px;">{point}</li>' for point in whatif_insights])}
                        </ol>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # LLM Relationship Explainer
        elif st.session_state.active_sidebar_page == "llm_relationship":
            st.markdown("""
            <div style="background:black;padding:20px;border-radius:12px;
            box-shadow:0 4px 10px rgba(0,0,0,0.08);">
            <h2>🤖 AI Relationship Explainer</h2>
            <p style="color:#6b7280;">
            Data-driven explanation of relationships for business decision-making
            </p>
            </div>
            """, unsafe_allow_html=True)

            st.caption(
                "Select any two variables to get a clear, data-driven explanation "
                "of their relationship."
            )

            numeric_cols = ["units_sold", "price", "discount", "marketing_spend"]
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
                        correlation=analysis["correlation"],
                        strength=analysis["strength"],
                        direction=analysis["direction"],
                        causal_statement=analysis["causal_statement"]
                    )

                    st.subheader("📘 Explanation (Plain English)")
                    st.write(explanation)

        # Executive Summary
        elif st.session_state.active_sidebar_page == "executive_summary":
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
                - No disclaimers like "data not available"
                - Do not explain methodology
                - Focus on implications and decisions

                AUDIENCE:
                C-level executives and senior management
                """,
                    insights,
                    df=processed_df,
                    eval_metrics=eval_metrics
                )

                st.markdown(summary)

            with st.expander("📉 Understanding forecast uncertainty"):
                st.write(
                    """
                    **What Are Uncertainty Intervals?**
                    
                    Think of forecast uncertainty like a weather forecast. When a meteorologist says "high of 75°F 
                    with a range of 70-80°F," they're acknowledging that the actual temperature could vary. Similarly, 
                    our sales forecast shows a predicted value plus a range of possible outcomes.
                    
                    **The Three Lines Explained:**
                    
                    1. **Middle Line (Forecast)**: This is our "best guess" - the most likely sales number based on 
                       historical patterns, trends, and seasonality. It's what we'd bet on if we had to pick one number.
                    
                    2. **Lower Bound (Conservative Estimate)**: This represents the pessimistic scenario - the lowest 
                       reasonable sales number we might see. Use this for worst-case planning (minimum inventory needs, 
                       cash flow planning).
                    
                    3. **Upper Bound (Optimistic Estimate)**: This represents the best-case scenario - the highest 
                       reasonable sales number. Use this for opportunity planning (potential upside, capacity needs).
                    
                    **What the Width Tells You:**
                    
                    - **Narrow Bands (Close Together)**: High confidence. The model is relatively certain about future 
                      demand. Good for making firm commitments, ordering inventory, or setting budgets.
                    
                    - **Wide Bands (Far Apart)**: Lower confidence. More variability is expected. Better for flexible 
                      planning - keep options open, use rolling forecasts, maintain buffer inventory.
                    
                    **Why Uncertainty Exists:**
                    
                    Forecasting gets harder the further you look ahead. Think of it like throwing a ball - you can 
                    predict where it'll land if you throw it a few feet, but it's much harder if you throw it far away. 
                    Similarly:
                    - Short-term forecasts (1-4 weeks): Usually more accurate, narrower bands
                    - Long-term forecasts (2-3 months): Less accurate, wider bands
                    
                    **How to Use This Information:**
                    
                    Don't just look at the middle line - consider the entire range when making decisions. If the bands 
                    are wide, be more flexible in your planning. If they're narrow, you can be more confident in your 
                    decisions. Always have a backup plan for outcomes outside the expected range.
                    """
                )

        # Model Performance Explained
        elif st.session_state.active_sidebar_page == "model_performance":
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

        # Ask Business Questions
        elif st.session_state.active_sidebar_page == "ask_questions":
            st.header("🤖 Ask Business Questions")
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

            # Display all preset questions in one row
            question_cols = st.columns(len(preset_questions))
            for idx, (label, question_text) in enumerate(preset_questions.items()):
                with question_cols[idx]:
                    if st.button(label, key=f"preset_{label}", width='stretch'):
                        st.session_state.active_question_answer = {
                            "question": question_text,
                            "label": label
                        }
                        st.rerun()

            # Display answer box below questions (only when a question is clicked)
            if st.session_state.active_question_answer is not None:
                st.markdown("<br>", unsafe_allow_html=True)
                question_data = st.session_state.active_question_answer
                
                # Check if answer is already cached
                answer_key = f"answer_{question_data['label']}"
                if answer_key not in st.session_state:
                    with st.spinner("Generating answer..."):
                        answer = ask_llm(
                            question_data["question"],
                            insights,
                            df=processed_df,
                            eval_metrics=eval_metrics
                        )
                        st.session_state[answer_key] = answer
                else:
                    answer = st.session_state[answer_key]
                
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #111827, #1f2937);
                        border: 2px solid #22c55e;
                        border-radius: 16px;
                        padding: 24px;
                        margin: 20px 0;
                        box-shadow: 0 10px 30px rgba(34, 197, 94, 0.2);
                    ">
                        <h4 style="color: #22c55e; margin-bottom: 12px;">💡 {question_data["label"]}</h4>
                        <p style="color: #e5e7eb; line-height: 1.6; font-size: 15px;">{render_confidence_text(answer)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.divider()

            # ---------- OPTIONAL CUSTOM QUESTION ----------

            custom_question = st.text_input(
                "Type your business question",
                placeholder="e.g. What happens if marketing spend increases by 20% next month?",
                key="custom_question_input"
            )
            if len(custom_question.strip()) < 5:
                st.warning("Please ask a meaningful business question.")
            else:
                if st.button("Ask AI", key="btn_custom_question") and custom_question.strip():
                    answer = ask_llm(
                        custom_question,
                        insights,
                        df=processed_df,
                        eval_metrics=eval_metrics
                    )
                    st.markdown(
                        f"""
                        <div style="
                            background: linear-gradient(135deg, #111827, #1f2937);
                            border: 2px solid #22c55e;
                            border-radius: 16px;
                            padding: 24px;
                            margin: 20px 0;
                            box-shadow: 0 10px 30px rgba(34, 197, 94, 0.2);
                        ">
                            <h4 style="color: #22c55e; margin-bottom: 12px;">💡 Custom Question</h4>
                            <p style="color: #e5e7eb; line-height: 1.6; font-size: 15px;">{render_confidence_text(answer)}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        st.markdown("</div>", unsafe_allow_html=True)
