# Sales Forecast AI

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)](README.md)

**Advanced AI-Powered Sales Forecasting & Analytics Platform**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Data Flow](#-data-flow)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Performance Metrics](#-performance-metrics)
- [Contributing](#-contributing)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## 🎯 Overview

**Sales Forecast AI** is an enterprise-grade, full-stack machine learning platform designed for time-series sales forecasting and business intelligence. It combines advanced forecasting algorithms with natural language processing to provide actionable insights for data-driven decision-making.

### Problem Statement

Sales forecasting is critical for inventory management, resource allocation, and strategic planning. This platform addresses the challenges of:
- Complex temporal patterns and seasonality in sales data
- Understanding relationships between marketing drivers and sales performance
- Generating human-readable business insights from statistical models
- Providing interactive exploration of forecast scenarios

### Solution

Sales Forecast AI integrates:
- **Prophet-based time-series forecasting** for accurate trend and seasonality capture
- **Hybrid ML ensemble** combining Prophet with Random Forest residual correction
- **Causal relationship analysis** between pricing, discounts, marketing spend, and sales
- **AI-powered insights generation** using Gemini LLM for business context
- **Interactive dashboard** built with Streamlit for real-time exploration

---

## ✨ Key Features

### 1. **Advanced Forecasting**
- Multi-method forecasting using Facebook Prophet
- Hybrid ensemble models with residual correction
- Category-wise forecasting for product segments
- Uncertainty quantification with confidence intervals

### 2. **Driver Analysis**
- Statistical relationship analysis between business variables
- Price elasticity and discount effectiveness evaluation
- Marketing spend ROI measurement
- Feature importance scoring

### 3. **What-If Scenarios**
- Interactive scenario modeling
- Real-time forecast adjustments based on parameter changes
- Impact simulation for pricing and promotional strategies

### 4. **AI-Powered Insights**
- Natural language explanations of forecast patterns
- Chart interpretations powered by Gemini AI
- Business context generation with analytical depth
- Automated business question answering

### 5. **Interactive Dashboard**
- Real-time forecast visualizations with Plotly
- Dark theme UI with professional design
- Session-based state management
- Responsive and intuitive navigation

### 6. **Model Evaluation**
- Comprehensive metrics (MAE, RMSE, MAPE)
- Residual analysis and error distribution
- Cross-validation performance tracking

---

## 🛠 Tech Stack

### **Backend & Data Processing**
| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.8+ |
| Data Processing | Pandas, NumPy | Latest |
| Data Storage | CSV | Native |

### **Machine Learning & Forecasting**
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Time-Series Forecasting | Facebook Prophet | Primary forecast model |
| Residual Modeling | Scikit-learn (Random Forest) | Ensemble correction |
| Statistical Analysis | Statsmodels | Relationship analysis |
| ML Utilities | Scikit-learn | Preprocessing & metrics |

### **Frontend & Visualization**
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Dashboard Framework | Streamlit | Interactive UI |
| Plotting Library | Plotly Express | Interactive visualizations |
| Charting | Matplotlib, Seaborn | Static visualizations |

### **AI & LLM Integration**
| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM Provider | Google Generative AI (Gemini) | Insight generation |
| LLM Framework | LangChain | Chain-of-thought processing |
| API Framework | FastAPI | REST API (optional) |

### **Utilities & DevOps**
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Environment Management | python-dotenv | Configuration management |
| Progress Tracking | tqdm | CLI progress bars |
| Web Server | Uvicorn | ASGI server (optional) |

---

## 🏗 Architecture

### **High-Level System Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                   FRONTEND LAYER                             │
│            Streamlit Interactive Dashboard                   │
│  (Visualizations, User Inputs, Real-time Interactions)      │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                 API/SERVICE LAYER                            │
│  FastAPI Endpoints | Session Management | Caching           │
└────────────┬──────────────┬────────────────┬────────────────┘
             │              │                │
    ┌────────▼────┐ ┌──────▼──────┐ ┌─────▼─────┐
    │ Forecasting │ │  Analytics  │ │ LLM/AI    │
    │  Pipeline   │ │  Engine     │ │ Service   │
    └────────┬────┘ └──────┬──────┘ └─────┬─────┘
             │              │              │
    ┌────────▼──────────────▼──────────────▼─────┐
    │        DATA PROCESSING LAYER               │
    │  - Data Preprocessing                      │
    │  - Feature Engineering                     │
    │  - Data Validation & Cleaning              │
    └────────┬──────────────────────────────────┘
             │
    ┌────────▼──────────────────────────────────┐
    │      DATA LAYER                            │
    │  - CSV Data Storage                        │
    │  - Time-Series Database (Optional)         │
    └────────────────────────────────────────────┘
```

### **Module Architecture**

```
salse-forecast-ai/
│
├── preprocessing/          # Data pipeline
│   └── data_preprocessing.py
│
├── forecasting/            # ML models
│   ├── prophet_model.py     # Primary forecasting
│   ├── hybrid_model.py      # Ensemble correction
│   ├── prophet_tuning.py    # Hyperparameter optimization
│   └── model_evaluation.py  # Performance metrics
│
├── data_analysis/          # Analytics & insights
│   ├── analytical_context.py      # Business context
│   ├── plot_insights.py            # Visualization generation
│   └── relationship_analysis.py    # Causal analysis
│
├── insights/               # AI insight generation
│   └── insights_generator.py
│
├── llm/                    # LLM integration
│   └── llm_qa.py           # Gemini API wrapper
│
├── utils/                  # Utilities
│   └── logger.py           # Logging configuration
│
├── app.py                  # Streamlit main app
├── requirements.txt        # Dependencies
├── .env                    # API keys (create this)
└── data/                   # Dataset storage
    └── commerce_Sales_Prediction_Dataset.csv
```

---

## 🔄 Data Flow

### **End-to-End Data Pipeline**

```
INPUT DATA (CSV)
      │
      ▼
┌──────────────────────────┐
│  Data Loading            │
│  - Read CSV file         │
│  - Initial validation    │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Data Preprocessing      │
│  - Format conversion     │
│  - Missing value handling│
│  - Outlier management    │
│  - Date standardization  │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Feature Engineering     │
│  - Interaction features  │
│  - Temporal features     │
│  - Categorical encoding  │
└──────────┬───────────────┘
           │
           ├─────────────────────────────┐
           │                             │
           ▼                             ▼
    ┌─────────────────┐      ┌──────────────────┐
    │ Forecasting     │      │ Relationship     │
    │ Pipeline        │      │ Analysis         │
    │                 │      │                  │
    │ 1. Prophet      │      │ 1. Correlation  │
    │ 2. Hybrid ML    │      │ 2. Elasticity   │
    │ 3. Evaluation   │      │ 3. Causality    │
    └────────┬────────┘      └────────┬─────────┘
             │                        │
             ▼                        ▼
    ┌──────────────────────────────────────┐
    │ Analytics & Insights                 │
    │ - Statistical summaries              │
    │ - Relationship interpretations       │
    │ - Business implications              │
    └────────┬─────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────┐
    │ LLM Processing (Gemini)              │
    │ - Natural language generation        │
    │ - Chart explanations                 │
    │ - Business context                   │
    └────────┬─────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────┐
    │ Streamlit Dashboard                  │
    │ - Real-time visualizations           │
    │ - User interactions                  │
    │ - What-if scenarios                  │
    └──────────────────────────────────────┘
```

### **Key Data Transformations**

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| Loading | CSV file | Parse & validate | DataFrame |
| Cleaning | Raw data | Handle nulls, outliers | Clean data |
| Aggregation | Detailed records | Group by date | Daily sales |
| Forecasting | Historical data | Prophet + ML | Forecast + CI |
| Analysis | Full dataset | Correlation, elasticity | Metrics |
| Insights | Results | LLM processing | Natural language |

---

## 📂 Project Structure

```
salse-forecast-ai/
│
├── 📄 app.py
│   └─ Main Streamlit application entry point
│      • Session state management
│      • UI theme and styling
│      • Page routing logic
│
├── 📄 requirements.txt
│   └─ Python package dependencies
│
├── 📄 .env
│   └─ Environment variables (create this)
│      • GEMINI_API_KEY
│      • Other API keys
│
├── 📄 question.txt
│   └─ Pre-defined business questions
│      • Sales trend questions
│      • Driver analysis questions
│      • Relationship exploration
│
├── 📁 preprocessing/
│   └── data_preprocessing.py
│       • load_data(): Load CSV files
│       • preprocess_data(): Clean & transform data
│       • Feature engineering
│       • Categorical encoding
│
├── 📁 forecasting/
│   ├── prophet_model.py
│   │   • train_prophet(): Train Prophet models
│   │   • category_wise_forecast(): Segment forecasting
│   │   • what_if_forecast(): Scenario modeling
│   │
│   ├── hybrid_model.py
│   │   • train_residual_model(): ML ensemble
│   │   • apply_residual_correction(): Accuracy boost
│   │
│   ├── prophet_tuning.py
│   │   • Hyperparameter optimization
│   │   • Model selection
│   │
│   └── model_evaluation.py
│       • evaluate_forecast(): Performance metrics
│       • MAE, RMSE, MAPE calculation
│
├── 📁 data_analysis/
│   ├── analytical_context.py
│   │   • build_analytical_context(): Generate context
│   │   • Statistical summaries
│   │
│   ├── relationship_analysis.py
│   │   • analyze_relationship(): Correlation analysis
│   │   • Elasticity computation
│   │   • Causality assessment
│   │
│   └── plot_insights.py
│       • generate_plot_insight(): Chart generation
│       • Visualization utilities
│
├── 📁 insights/
│   └── insights_generator.py
│       • generate_insights(): Business insight synthesis
│       • Multi-model analysis
│
├── 📁 llm/
│   └── llm_qa.py
│       • ask_llm(): Gemini API wrapper
│       • explain_chart(): Chart explanation
│       • Context-aware responses
│
├── 📁 utils/
│   └── logger.py
│       • setup_logger(): Logging configuration
│       • Error tracking
│       • Debug information
│
├── 📁 data/
│   └── commerce_Sales_Prediction_Dataset.csv
│       • Historical sales data
│       • Product categories
│       • Customer segments
│       • Marketing metrics
│
└── 📁 logs/
    └─ Application logs (auto-generated)
```

---

## 🚀 Installation & Setup

### **Prerequisites**

- **Python 3.8+** installed on your system
- **pip** package manager
- **Git** (optional, for version control)
- **Gemini API Key** (get from [Google AI Studio](https://makersuite.google.com/app/apikey))

### **Step 1: Clone or Download the Repository**

```bash
# Clone using Git
git clone https://github.com/yourusername/salse-forecast-ai.git
cd salse-forecast-ai

# OR download and extract the ZIP file manually
```

### **Step 2: Create Virtual Environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### **Step 3: Install Dependencies**

```bash
# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

### **Step 4: Configure Environment Variables**

Create a `.env` file in the root directory:

```env
# Google Generative AI
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Other API keys
# OPENAI_API_KEY=your_openai_key
# LOG_LEVEL=INFO
```

**How to get Gemini API Key:**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create new secret key"
3. Copy and paste the key into `.env`

### **Step 5: Verify Installation**

```bash
# Test Python packages
python -c "import streamlit, pandas, prophet; print('All imports successful!')"

# Check directory structure
tree  # Windows: tree or use dir
ls -la  # macOS/Linux
```

### **Step 6: Run the Application**

```bash
# Start Streamlit dashboard
streamlit run app.py

# Application will open at: http://localhost:8501
```

---

## 📖 Usage Guide

### **Dashboard Navigation**

1. **Home Page**
   - Overview of data statistics
   - Key metrics display
   - Interactive exploration start point

2. **Forecasting**
   - View time-series forecasts
   - Explore forecast uncertainty
   - Category-wise breakdowns
   - Download forecast data

3. **Driver Analysis**
   - Understand relationships between variables
   - Explore price elasticity
   - Discount effectiveness
   - Marketing ROI

4. **What-If Scenarios**
   - Adjust parameters interactively
   - See real-time forecast updates
   - Scenario comparison
   - Impact simulation

5. **AI Insights**
   - Ask business questions
   - Get LLM-powered explanations
   - Chart interpretations
   - Analytical context

6. **Model Evaluation**
   - Performance metrics
   - Error distribution
   - Model comparison
   - Statistical tests

### **Common Workflows**

#### **Workflow 1: Generate Monthly Forecast**
```
1. Upload/Load data → Data loads automatically
2. Navigate to "Forecasting" page
3. Select forecast period (e.g., next 90 days)
4. View predictions with confidence intervals
5. Download results as CSV
```

#### **Workflow 2: Analyze Sales Drivers**
```
1. Go to "Driver Analysis" page
2. Select two variables from dropdowns
3. View relationship visualization
4. Read AI explanation
5. Ask follow-up questions via chat
```

#### **Workflow 3: Model What-If Scenario**
```
1. Navigate to "What-If Analysis"
2. Adjust parameters (price, discount, marketing)
3. Observe real-time forecast changes
4. Compare scenarios side-by-side
5. Export results
```

---

## ⚙️ Configuration

### **Model Configuration**

Edit parameters in respective modules:

**Prophet Configuration** (`forecasting/prophet_model.py`):
```python
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative'
)
```

**Hybrid Model Configuration** (`forecasting/hybrid_model.py`):
```python
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=5,
    min_samples_split=5,
    random_state=42
)
```

**LLM Configuration** (`llm/llm_qa.py`):
```python
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
```

### **Streamlit Configuration**

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#22c55e"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#1f2937"
textColor = "#e6e6e6"

[server]
headless = true
port = 8501
```

---

## 📊 API Documentation

### **Main Functions Reference**

#### **Preprocessing Module**
```python
from preprocessing.data_preprocessing import load_data, preprocess_data

# Load dataset
df = load_data("data/commerce_Sales_Prediction_Dataset.csv")

# Preprocess and prepare for forecasting
prophet_df, ml_features = preprocess_data(df)
```

#### **Forecasting Module**
```python
from forecasting.prophet_model import train_prophet, category_wise_forecast
from forecasting.model_evaluation import evaluate_forecast

# Train Prophet model
model, forecast = train_prophet(prophet_df, periods=90)

# Category-wise forecasts
category_forecasts = category_wise_forecast(df, periods=90)

# Evaluate accuracy
metrics = evaluate_forecast(forecast, actual_data)
```

#### **Analysis Module**
```python
from data_analysis.relationship_analysis import analyze_relationship
from data_analysis.plot_insights import generate_plot_insight

# Analyze relationships
correlation, elasticity = analyze_relationship(df, var1, var2)

# Generate insights
plot_data = generate_plot_insight(df, x_col, y_col)
```

#### **LLM Module**
```python
from llm.llm_qa import ask_llm, explain_chart

# Ask business questions
answer = ask_llm(question, context)

# Explain charts
explanation = explain_chart(chart_data, chart_type)
```

#### **Insights Module**
```python
from insights.insights_generator import generate_insights

# Generate business insights
insights = generate_insights(forecast, analysis_results)
```

---

## 📈 Performance Metrics

### **Model Evaluation Metrics**

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Average absolute error (same units as data) |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | Penalizes larger errors more heavily |
| **MAPE** | $\frac{100}{n}\sum\|\frac{y_i - \hat{y}_i}{y_i}\|$ | Percentage error (scale-independent) |

### **Typical Performance**

- **Forecast Accuracy**: MAPE 10-15% on test data
- **Confidence Intervals**: 80% coverage of actual values
- **Residual Distribution**: ~Normal, centered at 0

### **Optimization Tips**

1. **Improve Accuracy**:
   - Increase historical data timespan
   - Fine-tune Prophet seasonality parameters
   - Adjust residual model hyperparameters

2. **Reduce Latency**:
   - Cache predictions
   - Pre-compute common queries
   - Optimize data loading

3. **Enhance Insights**:
   - Provide more context to LLM
   - Fine-tune system prompts
   - Include domain expertise

---

## 🤝 Contributing

### **How to Contribute**

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/salse-forecast-ai.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make changes**
   - Follow PEP 8 style guide
   - Add docstrings to functions
   - Include unit tests

4. **Commit and push**
   ```bash
   git add .
   git commit -m "Add feature: description"
   git push origin feature/your-feature-name
   ```

5. **Submit a Pull Request**
   - Describe changes clearly
   - Reference related issues
   - Request review

### **Code Style Guidelines**

```python
# Functions should have docstrings
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw data for forecasting.
    
    Args:
        df: Input DataFrame with raw data
        
    Returns:
        Processed DataFrame ready for forecasting
    """
    # implementation
    pass

# Use type hints
def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calculate performance metrics."""
    pass
```

---

## 🔧 Troubleshooting

### **Common Issues & Solutions**

#### **Issue 1: "GEMINI_API_KEY not set"**
```bash
# Solution:
# 1. Check .env file exists in root directory
# 2. Verify GEMINI_API_KEY is set correctly
# 3. Reload terminal/restart application
# 4. Check for typos in key
```

#### **Issue 2: "ModuleNotFoundError: No module named 'prophet'"**
```bash
# Solution:
pip install --upgrade pip
pip install -r requirements.txt
# If still fails:
pip install pystan==2.19.1.1
pip install fbprophet
```

#### **Issue 3: "CSV file not found"**
```python
# Solution: Verify file path
# Correct: data/commerce_Sales_Prediction_Dataset.csv
# Check working directory: os.getcwd()
```

#### **Issue 4: Streamlit not opening at localhost:8501**
```bash
# Solution:
# 1. Check port is not in use: netstat -an | find ":8501"
# 2. Specify custom port: streamlit run app.py --server.port 8502
# 3. Check firewall settings
```

#### **Issue 5: LLM responses are slow**
```python
# Solution:
# 1. Check internet connection
# 2. Verify API key quota not exceeded
# 3. Use caching: @lru_cache(maxsize=128)
# 4. Batch requests
```

#### **Issue 6: Out of Memory Error**
```bash
# Solution:
# 1. Reduce dataset size
# 2. Use data sampling
# 3. Increase system RAM
# 4. Check for memory leaks in loops
```

### **Debug Mode**

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set in .env
LOG_LEVEL=DEBUG
```

### **Performance Optimization**

```python
# Cache expensive operations
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(data_hash):
    # Only runs once per unique input
    pass

# Use pandas query for fast filtering
fast_result = df.query('category == "Electronics"')
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Sales Forecast AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

---

## 📧 Support & Contact

- **Issues**: Open an issue on [GitHub Issues](https://github.com/yourusername/salse-forecast-ai/issues)
- **Discussions**: Start a discussion on [GitHub Discussions](https://github.com/yourusername/salse-forecast-ai/discussions)
- **Email**: your-email@example.com
- **Documentation**: [Full Documentation](https://docs.example.com)

---

## 🎓 Learning Resources

- [Prophet Documentation](https://facebook.github.io/prophet/)
- [Streamlit Tutorial](https://docs.streamlit.io/)
- [Scikit-learn Guide](https://scikit-learn.org/stable/)
- [Plotly Express](https://plotly.com/python/plotly-express/)
- [LangChain Documentation](https://python.langchain.com/)

---

## 🙏 Acknowledgments

- Facebook Prophet team for the time-series forecasting library
- Streamlit for the dashboard framework
- Google for Generative AI API
- Community contributors and feedback

---

<div align="center">


[⬆ Back to Top](#sales-forecast-ai)

</div>
