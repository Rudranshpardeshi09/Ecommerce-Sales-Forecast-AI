# import os
# # from click import prompt
# from dotenv import load_dotenv
# import google.generativeai as genai
# # from streamlit import text
# from utils.logger import setup_logger
# from functools import lru_cache
# import re
# from data_analysis.analytical_context import build_analytical_context

# load_dotenv()
# logger = setup_logger("Gemini_LLM")

# # ---------------- CONFIG ----------------
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     raise EnvironmentError("GEMINI_API_KEY not set")

# genai.configure(api_key=GEMINI_API_KEY)

# def get_working_model():
#     """
#     Automatically finds a Gemini model that supports text generation.
#     This prevents 404 errors when Google changes model names.
#     """
#     for model in genai.list_models():
#         if "generateContent" in model.supported_generation_methods:
#             return genai.GenerativeModel(model.name)

#     raise RuntimeError("No compatible Gemini model found")

# # 🔥 AUTO-SELECTED MODEL
# model = get_working_model()

# # ---------------- CACHE LLM RESPONSES ----------------
# @lru_cache(maxsize=128)
# def _cached_llm_call(prompt: str) -> str:
#     try:
#         response = model.generate_content(
#             prompt,
#             generation_config={
#                 "temperature": 0.3,
#                 "max_output_tokens": 1200
#             }
#         )
#         text = response.text.strip()
#         # Do NOT cache incomplete answers
#         if len(text) < 100 or text.endswith(("The overall", "1.", "•")):
#             logger.warning("Incomplete LLM response detected; skipping cache")
#             return text

#         return text
#     except Exception as e:
#         logger.error(f"LLM error: {e}")
#         return (
#             "⚠️ LLM explanation temporarily unavailable due to API limits.\n\n"
#             "The chart and numeric insights above remain valid."
#         )



# # ---------------- MAIN FUNCTION (UNCHANGED SIGNATURE) ---------------

# def ask_llm(question: str, insights: dict, df=None, eval_metrics=None) -> str:

#     logger.info("LLM question received")

#     q = question.lower()

#     # -------- SAFE DEFAULTS (PREVENT UNBOUND ERRORS) --------
#     corr_text = ""
#     relation_text = ""


#     # -------- EXECUTIVE SUMMARY and category explaination  --------
#     goto_fallback = any(
#     k in q
#     for k in [
#         # Executive summaries
#         "executive summary",
#         "executive briefing",
#         "executive overview",
#         "management summary",
#         "board summary",

#         # Category explanation
#         "explain electronics",
#         "explain category",
#         "category performance",
#         "category outlook",
#         "category analysis",
#         "why is",
#         "declining",
#         "growing",
#         "growth",
#         "decline"
#     ]
# )




#     mae = eval_metrics.get("MAE") if eval_metrics else None
#     rmse = eval_metrics.get("RMSE") if eval_metrics else None


#     # Build analytical context ONLY if df is available
#     analytical_context = build_analytical_context(df) if df is not None else {}

#     baseline_day = insights.get("baseline_units_per_day", 0)
#     # baseline_month = insights.get("baseline_units_per_month", 0)

#     # -------- FUTURE HORIZON --------
#     horizon_days = 30
#     match = re.search(r"(\d+)\s*days", q)
#     if match:
#         horizon_days = int(match.group(1))
#     elif "month" in q:
#         horizon_days = 30

#     # -------- PRICE QUESTIONS --------
#     if "price" in q and not goto_fallback:
#         correlations = analytical_context.get("correlations", {})
#         corr = correlations.get("price", None)

#         if corr is not None:
#             if abs(corr) < 0.1:
#                 corr_text = (
#                     f"• Historical correlation between price and units sold: "
#                     f"**{corr:.2f}**, indicating a negligible relationship."
#                 )
#                 relation_text = "negligible"
#             elif corr < 0:
#                 corr_text = (
#                     f"• Historical correlation between price and units sold: "
#                     f"**{corr:.2f}**, indicating a negative relationship."
#                 )
#                 relation_text = "negative"
#             else:
#                 corr_text = (
#                     f"• Historical correlation between price and units sold: "
#                     f"**{corr:.2f}**, indicating a positive relationship."
#                 )
#                 relation_text = "positive"
#         else:
#             corr_text = (
#                 "• Price–demand relationship inferred directionally from model structure."
#             )
#             relation_text = "directional"


#         system_answer = f"""
# 📊 **Price Impact Outlook (Next {horizon_days} Days)**

# {corr_text}
# • This indicates a **{relation_text} relationship**.

# 📈 **Forecast Outlook**
# • Expected baseline demand: ~{baseline_day * horizon_days:.0f} units
# • Price stability → forecast trend continues
# • Price increase → may exert mild downward pressure
# • Price reduction → may offer limited demand uplift


# 🟠 **Medium confidence** — derived from model structure and forecast trend.
# """

#         return system_answer

#     # -------- PERCENTAGE SALES CHANGE --------
#     percent_match = re.search(r"(\d+)%", q)
#     # -------- BLOCK SCENARIOS FOR DECLINING CATEGORIES --------
#     if "declining" in q and percent_match:
#         goto_fallback = True

#     if percent_match and not goto_fallback and "scenario" in q:

#         pct = float(percent_match.group(1)) / 100
#         new_daily = baseline_day * (1 + pct)
#         delta = (new_daily - baseline_day) * horizon_days

#         return f"""
#         📊 **Sales Increase Scenario ({horizon_days} Days)**

#         • Current avg demand: {baseline_day:.2f} units/day
#         • Increase applied: {int(pct*100)}%
#         • New expected demand: {new_daily:.2f} units/day
#         • Additional units over {horizon_days} days: **~{delta:.0f} units**

#         🟢 **High confidence** — computed directly from forecast baseline
#         """

#         # -------- SAFE CONTEXT EXTRACTION --------
#     trend = insights.get("trend", "Not specified")

#     # Relationship defaults (used only if applicable)
#     x_col = analytical_context.get("x_col", "N/A")
#     y_col = analytical_context.get("y_col", "N/A")
#     correlation = analytical_context.get("correlation", "N/A")
#     strength = analytical_context.get("strength", "N/A")
#     direction = analytical_context.get("direction", "N/A")
#     causal_statement = analytical_context.get(
#         "causal_statement",
#         "Associative relationship; causality not established"
#     )

#     # -------- FALLBACK --------
#     prompt = f"""
#     You are a senior business analyst answering based strictly on data.

# AVAILABLE DATA CONTEXT:
# - Dataset contains: units_sold, price, discount, marketing_spend
# - Relationships are quantified using correlation analysis
# - Sales forecasts are generated using Prophet with regressors
# - Forecast uncertainty increases with time horizon
# - Model accuracy is evaluated using MAE and RMSE

# RELATIONSHIP FACTS (IF APPLICABLE):
# - X variable: {x_col}
# - Y variable: {y_col}
# - Correlation value: {correlation}
# - Relationship strength: {strength}
# - Direction: {direction}
# - Business causality: {causal_statement}

# FORECAST CONTEXT:
# - Trend direction: {trend}
# - Short-term forecast confidence: moderate to high
# - Long-term forecast confidence: lower due to uncertainty

# MODEL PERFORMANCE:
# - MAE: {mae}
# - RMSE: {rmse}

# QUESTION:
# {question}

# STRICT INSTRUCTIONS:
# - Answer in **50–60 words**
# - Use professional business-analyst tone
# - Be data-driven and trend-oriented
# - Do NOT say “data not available” if proxies exist
# - Distinguish correlation vs causation
# - Avoid speculation and generic statements

#     """


#     answer = _cached_llm_call(prompt)

#     # 🔒 Retry once if response is suspiciously short or incomplete
#     if len(answer.split()) < 25:
#         logger.warning("LLM response too short; retrying once")
#         answer = _cached_llm_call(prompt)

#     return answer
# # explain_chart
# @lru_cache(maxsize=128)
# def explain_chart(
#     x_col: str,
#     y_col: str,
#     #chart_type: str,
#     correlation: float,
#     strength: str,
#     direction: str,
#     causal_statement: str
# ) -> str:

#     prompt = f"""
# You are a senior business analyst writing for executives.

# DATA FACTS (DO NOT MODIFY):
# - X variable: {x_col}
# - Y variable: {y_col}
# - Correlation value: {correlation}
# - Relationship strength: {strength}
# - Direction: {direction}
# - Business causality: {causal_statement}

# STRICT OUTPUT FORMAT (FOLLOW EXACTLY):

# SECTION 1 — Executive Summary  
# • Write **ONE paragraph of EXACTLY 30 WORDS**
# • No greetings
# • No filler
# • Business-focused
# • Trend-oriented
# • If relationship is weak, clearly state it

# SECTION 2 — Key Takeaway  
# • Write **TWO concise sentence**
# • Must start with: **"Key takeaway:"**

# SECTION 3 — Executive Bullet Summary  
# • Write **EXACTLY 3 bullet points**
# • Each bullet ≤ 40 words
# • Focus on impact, risk, and decision relevance

# RULES:
# - Do NOT reverse causality
# - If relationship is weak, say it is not a strong driver
# - Avoid generic language
# - Use only the data facts above
# - Can only use professional business-analyst tone
# - Should be suitable for executive presentation
# - data-driven and trend-oriented
# - Avoid speculation
# - Maintain correct business causality
# - Use clear, executive-ready language

# """


#     return _cached_llm_call(prompt)

# llm/llm_qa.py
import os
import re
from functools import lru_cache
from dotenv import load_dotenv
import google.generativeai as genai
from utils.logger import setup_logger
from data_analysis.analytical_context import build_analytical_context

load_dotenv()
logger = setup_logger("Gemini_LLM")

# ---------------- CONFIG ----------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def get_working_model():
    for model in genai.list_models():
        if "generateContent" in model.supported_generation_methods:
            return genai.GenerativeModel(model.name)
    raise RuntimeError("No compatible Gemini model found")


model = get_working_model()


@lru_cache(maxsize=128)
def _cached_llm_call(prompt: str) -> str:
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.25,
                "max_output_tokens": 1400
            }
        )

        text = response.text.strip()

        # 🚨 VALIDATION: reject incomplete answers
        if (
            len(text.split()) < 60 or
            text.endswith(("with", "and", "of", "RMSE", "MAE", ","))
        ):
            logger.warning("Incomplete LLM response detected — retrying once")
            raise ValueError("Incomplete response")

        return text

    except Exception:
        # Retry ONCE without cache
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.25,
                "max_output_tokens": 1600
            }
        )
        return response.text.strip()



# ---------------- PUBLIC API ----------------

def ask_llm(question: str, insights: dict, df=None, eval_metrics=None, intent: str = "executive_summary") -> str:
    """
    Business-safe LLM interface.
    """

    analytical_context = build_analytical_context(df) if df is not None else {}

    if intent == "category_explanation":
        prompt = f"""
    You are a senior business analyst.

    TASK:
    Explain the performance of the following product category using data.

    CATEGORY CONTEXT:
    - Category name: {question}
    - Growth status: {insights.get("category_status")}
    - Growth percentage: {insights.get("category_growth")}

    RULES:
    - 50–60 words
    - Category-specific (DO NOT discuss overall company trend)
    - If growing, explain drivers and sustainability
    - If declining, explain risks and corrective actions
    - Executive tone
    - No generic company-wide commentary
    """

    else:
        prompt = f"""
        You are a senior business analyst preparing an executive briefing for leadership.

        TASK:
        Generate ONE complete executive summary paragraph.

        MANDATORY REQUIREMENTS (ALL MUST BE MET):
        - EXACTLY 80–90 words
        - ONE paragraph only
        - Complete sentences (no truncation)
        - Must end with a clear business conclusion

        MANDATORY CONTENT:
        - Overall sales trend direction
        - Demand stability or volatility
        - Short-term forecast confidence using MAE ({eval_metrics.get("MAE") if eval_metrics else "N/A"})
        and RMSE ({eval_metrics.get("RMSE") if eval_metrics else "N/A"})
        - Long-term forecast uncertainty
        - Impact of price, discount, and marketing
        - Business suitability of the forecast for planning

        AVAILABLE DATA CONTEXT:
        - Trend: {insights.get("trend")}
        - Forecast confidence: {insights.get("forecast_confidence")}
        - Demand stability: {insights.get("forecast_stability")}
        - Average daily demand: {insights.get("baseline_units_per_day")}
        - Correlations: {analytical_context.get("correlations")}

        CONSTRAINTS:
        - Executive tone
        - Data-driven
        - No bullet points
        - No methodology explanation
        - No filler
        - No generic phrases
        - Do NOT stop mid-sentence

        AUDIENCE:
        C-level executives
        """

    try:
        answer = _cached_llm_call(prompt)

        # 🔒 VALIDATION: reject incomplete or truncated responses
        if (
            len(answer.split()) < 75 or
            answer.endswith(("with", "and", "of", "RMSE", "MAE", ","))
        ):
            logger.warning("Incomplete LLM response detected — retrying once")
            raise ValueError("Incomplete response")

        return answer

    except Exception as e:
        logger.error(f"LLM error or retry triggered: {e}")

        # 🔁 Retry ONCE without cache pollution
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.25,
                    "max_output_tokens": 1600
                }
            )
            return response.text.strip()
        except Exception:
            return (
                "AI explanation temporarily unavailable. "
                "Please rely on charts and quantitative insights."
            )


from functools import lru_cache


@lru_cache(maxsize=128)
def explain_chart(
    x_col: str,
    y_col: str,
    correlation: float,
    strength: str,
    direction: str,
    causal_statement: str,
) -> str:
    """
    Generates an executive-ready explanation of a data relationship.
    Used ONLY when user explicitly requests AI explanation.
    """

    prompt = f"""
    You are a senior business analyst writing for executives.

    DATA FACTS (DO NOT MODIFY):
    - X variable: {x_col}
    - Y variable: {y_col}
    - Correlation value: {correlation}
    - Relationship strength: {strength}
    - Direction: {direction}
    - Business interpretation: {causal_statement}

    STRICT OUTPUT FORMAT:

    SECTION 1 — Executive Summary
    • ONE paragraph
    • EXACTLY 30–35 words
    • No filler or generic language
    • If relationship is weak, clearly say it is not a strong driver

    SECTION 2 — Key Takeaway
    • EXACTLY one sentence
    • Must start with ":"

    SECTION 3 — Executive Bullet Summary
    • EXACTLY 3 bullet points
    • Each bullet ≤ 30 words
    • Focus on impact, risk, and decision relevance

    RULES:
    - Never claim causation
    - Avoid speculation
    - Use only the data above
    - Business-focused, executive tone
    """

    try:
        response = _cached_llm_call(prompt)
        return response
    except Exception:
        return (
            "AI explanation temporarily unavailable. "
            "Please rely on the correlation statistics and visual trends."
        )
