import os
from functools import lru_cache

from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types as genai_types

    GEMINI_BACKEND = "google-genai"
except ImportError:
    genai = None
    genai_types = None
    try:
        import google.generativeai as legacy_genai

        GEMINI_BACKEND = "google-generativeai"
    except ImportError:
        legacy_genai = None
        GEMINI_BACKEND = "unavailable"

from data_analysis.analytical_context import build_analytical_context
from utils.logger import setup_logger

load_dotenv()
logger = setup_logger("Gemini_LLM")

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
PRIMARY_MAX_OUTPUT_TOKENS = 700
RETRY_MAX_OUTPUT_TOKENS = 900


def _get_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key

    try:
        import streamlit as st

        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_model():
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    if GEMINI_BACKEND == "google-genai":
        return genai.Client(api_key=api_key)

    if GEMINI_BACKEND == "google-generativeai":
        legacy_genai.configure(api_key=api_key)
        return legacy_genai.GenerativeModel(DEFAULT_GEMINI_MODEL)

    raise RuntimeError("No Gemini SDK is installed")


def _generate_text(prompt: str, max_output_tokens: int) -> str:
    model_or_client = get_model()

    if GEMINI_BACKEND == "google-genai":
        response = model_or_client.models.generate_content(
            model=DEFAULT_GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.25,
                max_output_tokens=max_output_tokens,
            ),
        )
        return (response.text or "").strip()

    response = model_or_client.generate_content(
        prompt,
        generation_config={
            "temperature": 0.25,
            "max_output_tokens": max_output_tokens,
        },
    )
    return response.text.strip()


def _looks_incomplete(text: str) -> bool:
    text = text.strip()
    if not text:
        return True

    truncated_endings = (
        "with",
        "and",
        "of",
        "to",
        "for",
        "in",
        "on",
        "at",
        "by",
        "from",
        "RMSE",
        "MAE",
        ",",
        ":",
        ";",
        "(",
        "[",
        "{",
        "-",
    )
    if text.endswith(truncated_endings):
        return True

    if text.count("(") > text.count(")"):
        return True
    if text.count("[") > text.count("]"):
        return True
    if text.count("{") > text.count("}"):
        return True
    if text.count('"') % 2 == 1:
        return True

    last_line = text.splitlines()[-1].strip()
    if last_line in {"-", "*", "•"}:
        return True

    if len(text.split()) >= 20 and text[-1] not in ".!?)]\"":
        return True

    return False


def _generate_complete_text(prompt: str) -> str:
    text = _generate_text(prompt, max_output_tokens=PRIMARY_MAX_OUTPUT_TOKENS)
    if not _looks_incomplete(text):
        return text

    logger.warning("Incomplete LLM response detected - retrying once")
    completion_prompt = (
        f"{prompt}\n\n"
        "Return the answer again as a fully completed response. "
        "Do not stop mid-sentence. End cleanly."
    )
    retry_text = _generate_text(completion_prompt, max_output_tokens=RETRY_MAX_OUTPUT_TOKENS)
    if not _looks_incomplete(retry_text):
        return retry_text

    logger.warning("Retry response still appears incomplete; returning best available response")
    return retry_text if len(retry_text) >= len(text) else text


def _friendly_llm_unavailable_message(feature_name: str, reason: Exception | None = None) -> str:
    reason_text = str(reason).lower() if reason else ""

    message = (
        f"Gemini-generated {feature_name} is unavailable right now. "
        "The forecast, charts, and model metrics are still available."
    )

    if "api_key" in reason_text or "configured" in reason_text or "credential" in reason_text:
        return (
            f"{message} If you're the app owner, add `GEMINI_API_KEY` "
            "to Streamlit Cloud secrets and redeploy."
        )

    if "deadline" in reason_text or "timeout" in reason_text or "timed out" in reason_text:
        return f"{message} The Gemini request timed out, so please try again in a minute."

    if "quota" in reason_text or "429" in reason_text or "rate limit" in reason_text:
        return f"{message} The Gemini API is rate-limited at the moment, so please try again shortly."

    return f"{message} Please try again shortly."


@lru_cache(maxsize=128)
def _cached_llm_call(prompt: str) -> str:
    try:
        return _generate_complete_text(prompt)

    except Exception as e:
        logger.warning(f"Primary Gemini call failed, retrying once: {e}")
        return _generate_text(prompt, max_output_tokens=RETRY_MAX_OUTPUT_TOKENS)


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
    - 50-60 words
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
        - EXACTLY 80-90 words
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
        return _cached_llm_call(prompt)

    except Exception as e:
        logger.error(f"LLM error or retry triggered: {e}")

        try:
            return _generate_complete_text(prompt)
        except Exception as retry_error:
            return _friendly_llm_unavailable_message("business insights", retry_error)


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
    Used only when the user explicitly requests an AI explanation.
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

    SECTION 1 - Executive Summary
    - ONE paragraph
    - EXACTLY 30-35 words
    - No filler or generic language
    - If relationship is weak, clearly say it is not a strong driver

    SECTION 2 - Key Takeaway
    - EXACTLY one sentence
    - Must start with: "Key takeaway:"

    SECTION 3 - Executive Bullet Summary
    - EXACTLY 3 bullet points
    - Each bullet <= 30 words
    - Focus on impact, risk, and decision relevance

    RULES:
    - Never claim causation
    - Avoid speculation
    - Use only the data above
    - Business-focused, executive tone
    """

    try:
        return _cached_llm_call(prompt)
    except Exception as e:
        return _friendly_llm_unavailable_message("relationship explanations", e)
