import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ARIMA Forecasting Masterclass",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 50%, #4a9bd9 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(30, 58, 95, 0.3);
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 1.05rem;
        opacity: 0.9;
        margin: 0;
        font-weight: 300;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #f0f7ff, #ffffff);
        border-left: 5px solid #2d6a9f;
        padding: 1rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 2rem 0 1.5rem 0;
    }
    .section-header h2 {
        color: #1e3a5f;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0;
    }

    /* Concept cards */
    .concept-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s;
    }
    .concept-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }
    .concept-card h4 {
        color: #2d6a9f;
        font-weight: 600;
        margin: 0 0 0.75rem 0;
        font-size: 1.1rem;
    }
    .concept-card p {
        color: #4a5568;
        line-height: 1.7;
        margin: 0;
    }

    /* Formula box */
    .formula-box {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.25rem;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.05rem;
    }

    /* Insight box */
    .insight-box {
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        border: 1px solid #93c5fd;
        border-radius: 10px;
        padding: 1.25rem;
        margin: 1rem 0;
    }
    .insight-box strong {
        color: #1e40af;
    }

    /* Warning box */
    .warning-box {
        background: linear-gradient(135deg, #fefce8, #fef9c3);
        border: 1px solid #fcd34d;
        border-radius: 10px;
        padding: 1.25rem;
        margin: 1rem 0;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        flex: 1;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .metric-card .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3a5f;
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.25rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #f8fafc;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.25rem;
        font-weight: 500;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1e3a5f;
    }

    /* Result badge */
    .result-pass {
        background: #dcfce7;
        color: #166534;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .result-fail {
        background: #fee2e2;
        color: #991b1b;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Step indicator */
    .step-indicator {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        background: #2d6a9f;
        color: white;
        border-radius: 50%;
        font-weight: 700;
        font-size: 0.9rem;
        margin-right: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load the classic airline passengers dataset."""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    try:
        df = pd.read_csv(url)
    except Exception:
        # Fallback: generate synthetic airline data
        dates = pd.date_range('1949-01', periods=144, freq='MS')
        np.random.seed(42)
        trend = np.linspace(100, 500, 144)
        seasonal = 40 * np.sin(np.linspace(0, 12 * 2 * np.pi, 144))
        noise = np.random.normal(0, 10, 144)
        passengers = (trend + seasonal + noise).astype(int)
        df = pd.DataFrame({'Month': dates.strftime('%Y-%m'), 'Passengers': passengers})

    df.columns = ['Month', 'Passengers']
    df['Month'] = pd.to_datetime(df['Month'])
    df = df.set_index('Month')
    return df

df = load_data()


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def run_adf(series, name):
    result = adfuller(series.dropna(), autolag='AIC')
    return {
        'Series': name,
        'Test Statistic': round(result[0], 4),
        'p-value': round(result[1], 6),
        'Lags Used': result[2],
        'Critical 1%': round(result[4]['1%'], 4),
        'Critical 5%': round(result[4]['5%'], 4),
        'Critical 10%': round(result[4]['10%'], 4),
        'Stationary?': '✅ Yes' if result[1] < 0.05 else '❌ No'
    }

def forecast_metrics(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = mean_absolute_percentage_error(actual, forecast) * 100
    return mae, rmse, mape


# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Go to section:",
        [
            "🏠 Home & Overview",
            "📖 ARIMA Theory",
            "📊 Dataset Exploration",
            "🔬 Stationarity Testing",
            "📉 ACF & PACF Analysis",
            "⚙️ Model Building",
            "🔍 Diagnostics",
            "🚀 Forecasting & Results",
            "🌊 SARIMA — Seasonal Upgrade",
            "📋 Summary & Cheat Sheet"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 📐 Quick Reference")
    st.markdown("""
    **ARIMA(p, d, q)**
    - **p** → AR order (PACF)
    - **d** → Differencing order
    - **q** → MA order (ACF)
    """)
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#94a3b8; font-size:0.8rem;'>"
        "SP Jain School of Global Management<br>Global MBA Program<br>Dr. Anshul Gupta</div>",
        unsafe_allow_html=True
    )


# ═════════════════════════════════════════════════════════════
# PAGE: HOME & OVERVIEW
# ═════════════════════════════════════════════════════════════
if page == "🏠 Home & Overview":
    st.markdown("""
    <div class="main-header">
        <h1>📈 ARIMA Forecasting Masterclass</h1>
        <p>A comprehensive, interactive guide to understanding and implementing ARIMA models — from theory to forecasting results.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header"><h2>What You'll Learn</h2></div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="concept-card">
            <h4>📖 Theory & Concepts</h4>
            <p>Understand what ARIMA stands for, the intuition behind AR, I, and MA components, and how they combine to model real-world time series data.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="concept-card">
            <h4>📊 Exploratory Analysis</h4>
            <p>Visualize trend, seasonality, and noise in data. Learn to test for stationarity and read ACF/PACF plots to select model parameters.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="concept-card">
            <h4>🚀 Model & Forecast</h4>
            <p>Build ARIMA models interactively, diagnose residuals, evaluate accuracy metrics, and produce future forecasts with confidence intervals.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header"><h2>The ARIMA Workflow</h2></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="concept-card">
        <p>
        <span class="step-indicator">1</span> <strong>Visualize</strong> the raw time series — identify trend, seasonality, outliers<br><br>
        <span class="step-indicator">2</span> <strong>Test stationarity</strong> using the Augmented Dickey-Fuller (ADF) test<br><br>
        <span class="step-indicator">3</span> <strong>Difference</strong> the series (if needed) to achieve stationarity<br><br>
        <span class="step-indicator">4</span> <strong>Analyze ACF & PACF</strong> plots to determine p and q parameters<br><br>
        <span class="step-indicator">5</span> <strong>Fit the ARIMA model</strong> with chosen (p, d, q) parameters<br><br>
        <span class="step-indicator">6</span> <strong>Diagnose</strong> residuals — they should look like white noise<br><br>
        <span class="step-indicator">7</span> <strong>Forecast</strong> future values and evaluate accuracy
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header"><h2>Dataset: International Airline Passengers (1949–1960)</h2></div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.line(
            df, y='Passengers',
            title='Monthly Airline Passenger Count',
            labels={'Month': '', 'Passengers': 'Passengers (thousands)'},
            template='plotly_white'
        )
        fig.update_traces(line=dict(color='#2d6a9f', width=2.5))
        fig.update_layout(
            font=dict(family='Inter'),
            title_font_size=16,
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"""
        <div class="concept-card">
            <h4>Dataset at a Glance</h4>
            <p>
            <strong>Period:</strong> {df.index.min().strftime('%b %Y')} – {df.index.max().strftime('%b %Y')}<br><br>
            <strong>Frequency:</strong> Monthly<br><br>
            <strong>Observations:</strong> {len(df)}<br><br>
            <strong>Min:</strong> {df['Passengers'].min():,}<br><br>
            <strong>Max:</strong> {df['Passengers'].max():,}<br><br>
            <strong>Mean:</strong> {df['Passengers'].mean():,.0f}<br><br>
            <strong>Std Dev:</strong> {df['Passengers'].std():,.1f}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
            <strong>💡 Why this dataset?</strong><br>
            It exhibits clear upward <strong>trend</strong> and strong <strong>seasonal</strong> pattern (summer peaks) — perfect for demonstrating ARIMA concepts.
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE: ARIMA THEORY
# ═════════════════════════════════════════════════════════════
elif page == "📖 ARIMA Theory":
    st.markdown("""
    <div class="main-header">
        <h1>📖 ARIMA Theory</h1>
        <p>Understanding the building blocks: AutoRegressive, Integrated, and Moving Average components.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- What is ARIMA ---
    st.markdown('<div class="section-header"><h2>What is ARIMA?</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="concept-card">
        <p>
        <strong>ARIMA</strong> stands for <strong>A</strong>uto<strong>R</strong>egressive <strong>I</strong>ntegrated <strong>M</strong>oving <strong>A</strong>verage.
        It is one of the most widely used models for time series forecasting. ARIMA combines three ideas:
        </p>
        <p>
        1. <strong>AR (AutoRegressive):</strong> The current value depends on its own past values.<br>
        2. <strong>I (Integrated):</strong> Differencing the data to make it stationary.<br>
        3. <strong>MA (Moving Average):</strong> The current value depends on past forecast errors.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
        <strong>ARIMA(p, d, q)</strong><br><br>
        p = number of autoregressive terms &nbsp;|&nbsp; d = number of differences &nbsp;|&nbsp; q = number of moving average terms
    </div>
    """, unsafe_allow_html=True)

    # --- AR Component ---
    st.markdown('<div class="section-header"><h2>1️⃣ AutoRegressive (AR) Component</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class="concept-card">
            <h4>Intuition</h4>
            <p>
            Think of it like predicting tomorrow's temperature based on today's and yesterday's temperature.
            The current value is a <strong>weighted sum of its own previous values</strong>.
            </p>
            <p>
            <strong>Business analogy:</strong> This month's sales are likely influenced by last month's and the month before — there's inertia in business metrics.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="formula-box">
            <strong>AR(p) Model:</strong><br><br>
            Y<sub>t</sub> = c + φ₁·Y<sub>t-1</sub> + φ₂·Y<sub>t-2</sub> + ... + φ<sub>p</sub>·Y<sub>t-p</sub> + ε<sub>t</sub><br><br>
            <em>Where φ are coefficients and ε is white noise error</em>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # AR demo
        np.random.seed(42)
        ar_data = [0, 0]
        for i in range(2, 100):
            ar_data.append(0.7 * ar_data[i-1] + np.random.normal(0, 1))
        fig_ar = px.line(
            y=ar_data, title='Simulated AR(1) Process (φ=0.7)',
            labels={'x': 'Time', 'y': 'Value'}, template='plotly_white'
        )
        fig_ar.update_traces(line=dict(color='#2d6a9f', width=2))
        fig_ar.update_layout(height=300, font=dict(family='Inter'), showlegend=False, title_font_size=14)
        st.plotly_chart(fig_ar, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            <strong>🔑 Key Insight:</strong> Higher φ means stronger dependence on past values — the series looks smoother and more persistent.
        </div>
        """, unsafe_allow_html=True)

    # --- I Component ---
    st.markdown('<div class="section-header"><h2>2️⃣ Integrated (I) Component — Differencing</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class="concept-card">
            <h4>Intuition</h4>
            <p>
            Many real-world time series have trends — they go up or down over time. ARIMA requires
            <strong>stationary</strong> data (constant mean and variance over time).
            </p>
            <p>
            <strong>Differencing</strong> removes the trend by computing the change between consecutive observations:
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="formula-box">
            <strong>First Difference:</strong><br><br>
            Y'<sub>t</sub> = Y<sub>t</sub> − Y<sub>t-1</sub><br><br>
            <em>d=1 means we difference once; d=2 means we difference the differenced series</em>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Rule of Thumb:</strong> Most business time series need d=1 (rarely d=2). Over-differencing can introduce artificial patterns.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        df_diff = df['Passengers'].diff().dropna()
        fig_diff = make_subplots(rows=2, cols=1, subplot_titles=('Original Series', 'After First Differencing (d=1)'))
        fig_diff.add_trace(go.Scatter(y=df['Passengers'].values, mode='lines', line=dict(color='#2d6a9f', width=2), name='Original'), row=1, col=1)
        fig_diff.add_trace(go.Scatter(y=df_diff.values, mode='lines', line=dict(color='#e67e22', width=2), name='Differenced'), row=2, col=1)
        fig_diff.update_layout(height=420, template='plotly_white', font=dict(family='Inter'), showlegend=False, title_font_size=13)
        st.plotly_chart(fig_diff, use_container_width=True)

    # --- MA Component ---
    st.markdown('<div class="section-header"><h2>3️⃣ Moving Average (MA) Component</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class="concept-card">
            <h4>Intuition</h4>
            <p>
            Instead of depending on past <em>values</em>, the MA component depends on past <em>forecast errors</em>.
            Think of it as a correction mechanism: "I was off by this much yesterday, so let me adjust today's prediction."
            </p>
            <p>
            <strong>Business analogy:</strong> If your demand forecast was too high last month, you'd naturally revise this month's forecast downward.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="formula-box">
            <strong>MA(q) Model:</strong><br><br>
            Y<sub>t</sub> = μ + ε<sub>t</sub> + θ₁·ε<sub>t-1</sub> + θ₂·ε<sub>t-2</sub> + ... + θ<sub>q</sub>·ε<sub>t-q</sub><br><br>
            <em>Where θ are coefficients and ε are past errors</em>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        np.random.seed(42)
        errors = np.random.normal(0, 1, 100)
        ma_data = [errors[0]]
        for i in range(1, 100):
            ma_data.append(errors[i] + 0.7 * errors[i-1])
        fig_ma = px.line(
            y=ma_data, title='Simulated MA(1) Process (θ=0.7)',
            labels={'x': 'Time', 'y': 'Value'}, template='plotly_white'
        )
        fig_ma.update_traces(line=dict(color='#27ae60', width=2))
        fig_ma.update_layout(height=300, font=dict(family='Inter'), showlegend=False, title_font_size=14)
        st.plotly_chart(fig_ma, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            <strong>🔑 Key Insight:</strong> MA processes have "short memory" — shocks die out after q periods. Compare this with AR where effects persist longer.
        </div>
        """, unsafe_allow_html=True)

    # --- Stationarity ---
    st.markdown('<div class="section-header"><h2>🎯 The Stationarity Requirement</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="concept-card">
        <h4>Why Does Stationarity Matter?</h4>
        <p>
        A <strong>stationary</strong> time series has statistical properties (mean, variance, autocorrelation) that are constant over time.
        ARIMA's AR and MA components assume stationarity — they model patterns that repeat consistently.
        </p>
        <p>
        <strong>Non-stationary signals:</strong> Upward/downward trends, changing variance, seasonal patterns that grow over time.<br>
        <strong>How we fix it:</strong> Differencing (the "I" in ARIMA) and sometimes log transformations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="formula-box">
            <strong>Augmented Dickey-Fuller (ADF) Test</strong><br><br>
            H₀: The series has a unit root (non-stationary)<br>
            H₁: The series is stationary<br><br>
            <em>If p-value < 0.05 → Reject H₀ → Series is stationary ✅</em>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="concept-card">
            <h4>Identifying p, d, q from plots</h4>
            <p>
            <strong>d:</strong> How many times you difference until ADF says "stationary"<br><br>
            <strong>p (AR order):</strong> Count significant lags in <strong>PACF</strong> plot<br><br>
            <strong>q (MA order):</strong> Count significant lags in <strong>ACF</strong> plot
            </p>
        </div>
        """, unsafe_allow_html=True)

    # --- ACF / PACF ---
    st.markdown('<div class="section-header"><h2>📊 ACF & PACF — Your Parameter Selection Tools</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="concept-card">
            <h4>ACF (AutoCorrelation Function)</h4>
            <p>
            Measures the correlation between a time series and its lagged versions. Includes both direct and
            indirect effects.
            </p>
            <p>
            <strong>Used for:</strong> Determining <strong>q</strong> (MA order).<br>
            <strong>Look for:</strong> Sharp cutoff after lag q → suggests MA(q).
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="concept-card">
            <h4>PACF (Partial AutoCorrelation Function)</h4>
            <p>
            Measures the <em>direct</em> correlation between a time series and a specific lag, removing the effect
            of all intermediate lags.
            </p>
            <p>
            <strong>Used for:</strong> Determining <strong>p</strong> (AR order).<br>
            <strong>Look for:</strong> Sharp cutoff after lag p → suggests AR(p).
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <strong>📋 Quick Decision Guide:</strong><br><br>
        • ACF tails off gradually + PACF cuts off sharply → <strong>AR model</strong> (use PACF to find p)<br>
        • ACF cuts off sharply + PACF tails off gradually → <strong>MA model</strong> (use ACF to find q)<br>
        • Both tail off gradually → <strong>ARMA model</strong> (need both p and q)
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE: DATASET EXPLORATION
# ═════════════════════════════════════════════════════════════
elif page == "📊 Dataset Exploration":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Dataset Exploration</h1>
        <p>Visual exploration of the International Airline Passengers dataset — uncovering trend, seasonality, and patterns.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Raw Data ---
    st.markdown('<div class="section-header"><h2>Raw Time Series</h2></div>', unsafe_allow_html=True)

    fig1 = px.line(
        df, y='Passengers',
        labels={'Month': '', 'Passengers': 'Passengers (thousands)'},
        template='plotly_white'
    )
    fig1.update_traces(line=dict(color='#2d6a9f', width=2.5))
    fig1.update_layout(font=dict(family='Inter'), height=400, hovermode='x unified')
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <strong>👀 Observations:</strong> Clear upward <strong>trend</strong> (passengers growing year over year),
        <strong>seasonal peaks</strong> every summer (Jun–Aug), and the seasonal amplitude is <em>increasing</em> with the level —
        this suggests a multiplicative seasonal pattern.
    </div>
    """, unsafe_allow_html=True)

    # --- Descriptive Stats ---
    st.markdown('<div class="section-header"><h2>Descriptive Statistics</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        stats = df['Passengers'].describe()
        st.dataframe(
            pd.DataFrame(stats).T.style.format("{:.1f}"),
            use_container_width=True
        )
    with col2:
        fig_hist = px.histogram(
            df, x='Passengers', nbins=25,
            title='Distribution of Passenger Counts',
            template='plotly_white', color_discrete_sequence=['#2d6a9f']
        )
        fig_hist.update_layout(font=dict(family='Inter'), height=300, title_font_size=14)
        st.plotly_chart(fig_hist, use_container_width=True)

    # --- Seasonal Decomposition ---
    st.markdown('<div class="section-header"><h2>Seasonal Decomposition</h2></div>', unsafe_allow_html=True)

    decomp_type = st.radio("Decomposition type:", ["Multiplicative (recommended)", "Additive"], horizontal=True)
    model_type = 'multiplicative' if 'Multiplicative' in decomp_type else 'additive'

    decomposition = seasonal_decompose(df['Passengers'], model=model_type, period=12)

    fig_decomp = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'),
        vertical_spacing=0.08
    )
    fig_decomp.add_trace(go.Scatter(x=df.index, y=decomposition.observed, mode='lines', line=dict(color='#2d6a9f', width=2), name='Observed'), row=1, col=1)
    fig_decomp.add_trace(go.Scatter(x=df.index, y=decomposition.trend, mode='lines', line=dict(color='#e67e22', width=2.5), name='Trend'), row=2, col=1)
    fig_decomp.add_trace(go.Scatter(x=df.index, y=decomposition.seasonal, mode='lines', line=dict(color='#27ae60', width=2), name='Seasonal'), row=3, col=1)
    fig_decomp.add_trace(go.Scatter(x=df.index, y=decomposition.resid, mode='lines', line=dict(color='#e74c3c', width=1.5), name='Residual'), row=4, col=1)
    fig_decomp.update_layout(height=700, template='plotly_white', font=dict(family='Inter'), showlegend=False)
    st.plotly_chart(fig_decomp, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box">
        <strong>💡 Decomposition Insights ({model_type.title()}):</strong><br><br>
        • <strong>Trend:</strong> Steady upward growth from ~100 to ~400+ passengers over the 12-year period.<br>
        • <strong>Seasonal:</strong> Repeating 12-month pattern with clear summer peaks (Jul–Aug) and winter troughs (Nov–Feb).<br>
        • <strong>Residual:</strong> What's left after removing trend and seasonality — ideally random noise.
    </div>
    """, unsafe_allow_html=True)

    # --- Monthly Boxplot ---
    st.markdown('<div class="section-header"><h2>Monthly Seasonality Pattern</h2></div>', unsafe_allow_html=True)

    df_month = df.copy()
    df_month['MonthName'] = df_month.index.strftime('%b')
    df_month['MonthNum'] = df_month.index.month
    df_month = df_month.sort_values('MonthNum')

    fig_box = px.box(
        df_month, x='MonthName', y='Passengers',
        title='Passenger Distribution by Month',
        template='plotly_white', color_discrete_sequence=['#2d6a9f'],
        category_orders={'MonthName': ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']}
    )
    fig_box.update_layout(font=dict(family='Inter'), height=400, title_font_size=14, xaxis_title='', yaxis_title='Passengers (thousands)')
    st.plotly_chart(fig_box, use_container_width=True)

    # --- Yearly Overlay ---
    st.markdown('<div class="section-header"><h2>Year-over-Year Comparison</h2></div>', unsafe_allow_html=True)

    df_yearly = df.copy()
    df_yearly['Year'] = df_yearly.index.year
    df_yearly['MonthNum'] = df_yearly.index.month

    fig_yearly = px.line(
        df_yearly, x='MonthNum', y='Passengers', color='Year',
        title='Passenger Traffic by Year',
        labels={'MonthNum': 'Month', 'Passengers': 'Passengers (thousands)'},
        template='plotly_white'
    )
    fig_yearly.update_layout(
        font=dict(family='Inter'), height=400, title_font_size=14,
        xaxis=dict(tickmode='array', tickvals=list(range(1,13)), ticktext=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    )
    st.plotly_chart(fig_yearly, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <strong>💡 Key Takeaway:</strong> Each year follows the same seasonal shape, but at a higher level — confirming both
        trend and seasonality. The gap between years also widens, suggesting the seasonal effect is proportional to the level (multiplicative).
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE: STATIONARITY TESTING
# ═════════════════════════════════════════════════════════════
elif page == "🔬 Stationarity Testing":
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Stationarity Testing</h1>
        <p>Testing and transforming the data to meet ARIMA's stationarity requirement.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Log Transform ---
    st.markdown('<div class="section-header"><h2>Step 1: Log Transformation</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="concept-card">
        <p>Since the airline data shows <strong>increasing variance</strong> (seasonal swings get bigger as the level rises),
        we first apply a <strong>log transformation</strong> to stabilize the variance. This converts the multiplicative pattern to additive.</p>
    </div>
    """, unsafe_allow_html=True)

    df_log = np.log(df['Passengers'])

    fig_log = make_subplots(rows=1, cols=2, subplot_titles=('Original Scale', 'Log-Transformed'))
    fig_log.add_trace(go.Scatter(x=df.index, y=df['Passengers'], mode='lines', line=dict(color='#2d6a9f', width=2), name='Original'), row=1, col=1)
    fig_log.add_trace(go.Scatter(x=df.index, y=df_log, mode='lines', line=dict(color='#27ae60', width=2), name='Log'), row=1, col=2)
    fig_log.update_layout(height=350, template='plotly_white', font=dict(family='Inter'), showlegend=False)
    st.plotly_chart(fig_log, use_container_width=True)

    # --- ADF Test: Original & Log ---
    st.markdown('<div class="section-header"><h2>Step 2: ADF Test on Original & Log Series</h2></div>', unsafe_allow_html=True)

    def run_adf(series, name):
        result = adfuller(series.dropna(), autolag='AIC')
        return {
            'Series': name,
            'Test Statistic': round(result[0], 4),
            'p-value': round(result[1], 6),
            'Lags Used': result[2],
            'Critical 1%': round(result[4]['1%'], 4),
            'Critical 5%': round(result[4]['5%'], 4),
            'Critical 10%': round(result[4]['10%'], 4),
            'Stationary?': '✅ Yes' if result[1] < 0.05 else '❌ No'
        }

    adf_results = [
        run_adf(df['Passengers'], 'Original'),
        run_adf(df_log, 'Log-Transformed'),
    ]

    st.dataframe(pd.DataFrame(adf_results).set_index('Series'), use_container_width=True)

    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Result:</strong> Both original and log-transformed series are <strong>non-stationary</strong> (p-value > 0.05).
        The trend is still present. We need to difference the data.
    </div>
    """, unsafe_allow_html=True)

    # --- Differencing ---
    st.markdown('<div class="section-header"><h2>Step 3: Differencing</h2></div>', unsafe_allow_html=True)

    df_log_diff1 = df_log.diff().dropna()
    df_log_diff2 = df_log.diff().diff().dropna()

    adf_diff_results = [
        run_adf(df_log_diff1, 'Log + 1st Difference (d=1)'),
        run_adf(df_log_diff2, 'Log + 2nd Difference (d=2)'),
    ]
    st.dataframe(pd.DataFrame(adf_diff_results).set_index('Series'), use_container_width=True)

    fig_diffs = make_subplots(rows=2, cols=1, subplot_titles=('Log + First Difference (d=1)', 'Log + Second Difference (d=2)'))
    fig_diffs.add_trace(go.Scatter(y=df_log_diff1.values, mode='lines', line=dict(color='#e67e22', width=1.5), name='d=1'), row=1, col=1)
    fig_diffs.add_trace(go.Scatter(y=df_log_diff2.values, mode='lines', line=dict(color='#e74c3c', width=1.5), name='d=2'), row=2, col=1)
    fig_diffs.update_layout(height=450, template='plotly_white', font=dict(family='Inter'), showlegend=False)
    st.plotly_chart(fig_diffs, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <strong>✅ Conclusion:</strong> The log + first-differenced series (d=1) passes the ADF test.
        We'll proceed with <strong>d=1</strong>. The series now fluctuates around zero with relatively constant variance — it's stationary!
    </div>
    """, unsafe_allow_html=True)

    # --- Rolling Stats ---
    st.markdown('<div class="section-header"><h2>Visual Confirmation: Rolling Statistics</h2></div>', unsafe_allow_html=True)

    window = st.slider("Rolling window size:", 6, 24, 12)
    rolling_mean = df_log_diff1.rolling(window=window).mean()
    rolling_std = df_log_diff1.rolling(window=window).std()

    fig_rolling = go.Figure()
    fig_rolling.add_trace(go.Scatter(x=df_log_diff1.index, y=df_log_diff1.values, mode='lines', name='Differenced Log Series', line=dict(color='#bdc3c7', width=1)))
    fig_rolling.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean.values, mode='lines', name=f'Rolling Mean ({window}m)', line=dict(color='#e74c3c', width=2.5)))
    fig_rolling.add_trace(go.Scatter(x=rolling_std.index, y=rolling_std.values, mode='lines', name=f'Rolling Std ({window}m)', line=dict(color='#2d6a9f', width=2.5)))
    fig_rolling.update_layout(
        height=400, template='plotly_white', font=dict(family='Inter'),
        title='Rolling Mean & Std of Differenced Log Series', title_font_size=14,
        hovermode='x unified'
    )
    st.plotly_chart(fig_rolling, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <strong>💡 Interpretation:</strong> For a stationary series, the rolling mean should hover around zero and the rolling standard deviation
        should remain roughly constant. If you see drift or widening/narrowing, the series may not be fully stationary.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE: ACF & PACF
# ═════════════════════════════════════════════════════════════
elif page == "📉 ACF & PACF Analysis":
    st.markdown("""
    <div class="main-header">
        <h1>📉 ACF & PACF Analysis</h1>
        <p>Using autocorrelation plots to determine the optimal p and q parameters for our ARIMA model.</p>
    </div>
    """, unsafe_allow_html=True)

    df_log = np.log(df['Passengers'])
    df_log_diff1 = df_log.diff().dropna()

    n_lags = st.slider("Number of lags to display:", 10, 40, 25)

    # --- ACF ---
    st.markdown('<div class="section-header"><h2>ACF (AutoCorrelation Function)</h2></div>', unsafe_allow_html=True)

    acf_values = acf(df_log_diff1, nlags=n_lags, alpha=0.05)
    acf_vals = acf_values[0]
    acf_ci = acf_values[1]

    fig_acf = go.Figure()
    for i in range(len(acf_vals)):
        color = '#e74c3c' if abs(acf_vals[i]) > 1.96/np.sqrt(len(df_log_diff1)) and i > 0 else '#2d6a9f'
        fig_acf.add_trace(go.Bar(x=[i], y=[acf_vals[i]], marker_color=color, showlegend=False, width=0.3))
    # Confidence band
    ci_val = 1.96 / np.sqrt(len(df_log_diff1))
    fig_acf.add_hline(y=ci_val, line_dash="dash", line_color="#95a5a6", annotation_text="95% CI")
    fig_acf.add_hline(y=-ci_val, line_dash="dash", line_color="#95a5a6")
    fig_acf.add_hline(y=0, line_color="black", line_width=0.5)
    fig_acf.update_layout(
        title='ACF of Differenced Log Series → Helps determine q',
        xaxis_title='Lag', yaxis_title='Autocorrelation',
        height=380, template='plotly_white', font=dict(family='Inter'), title_font_size=14
    )
    st.plotly_chart(fig_acf, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <strong>📖 How to read this:</strong> Bars in <span style="color:#e74c3c; font-weight:bold;">red</span> are statistically significant
        (beyond the 95% confidence band). For the <strong>MA order (q)</strong>, count how many initial lags are significant before the ACF "cuts off."
        Also note the spike at lag 12 — that's the seasonal autocorrelation (12-month cycle).
    </div>
    """, unsafe_allow_html=True)

    # --- PACF ---
    st.markdown('<div class="section-header"><h2>PACF (Partial AutoCorrelation Function)</h2></div>', unsafe_allow_html=True)

    pacf_vals = pacf(df_log_diff1, nlags=n_lags, method='ywm')

    fig_pacf = go.Figure()
    for i in range(len(pacf_vals)):
        color = '#e74c3c' if abs(pacf_vals[i]) > 1.96/np.sqrt(len(df_log_diff1)) and i > 0 else '#27ae60'
        fig_pacf.add_trace(go.Bar(x=[i], y=[pacf_vals[i]], marker_color=color, showlegend=False, width=0.3))
    fig_pacf.add_hline(y=ci_val, line_dash="dash", line_color="#95a5a6", annotation_text="95% CI")
    fig_pacf.add_hline(y=-ci_val, line_dash="dash", line_color="#95a5a6")
    fig_pacf.add_hline(y=0, line_color="black", line_width=0.5)
    fig_pacf.update_layout(
        title='PACF of Differenced Log Series → Helps determine p',
        xaxis_title='Lag', yaxis_title='Partial Autocorrelation',
        height=380, template='plotly_white', font=dict(family='Inter'), title_font_size=14
    )
    st.plotly_chart(fig_pacf, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <strong>📖 How to read this:</strong> For the <strong>AR order (p)</strong>, count how many initial lags are significant before the PACF "cuts off."
        Spikes at seasonal lags (12, 24) indicate seasonal AR components.
    </div>
    """, unsafe_allow_html=True)

    # --- Parameter Suggestion ---
    st.markdown('<div class="section-header"><h2>🎯 Suggested Parameters</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="concept-card">
        <h4>Based on ACF & PACF Analysis</h4>
        <p>
        From the plots above, we can observe:<br><br>
        • <strong>PACF</strong> shows a significant spike at lag 1, suggesting <strong>p = 1</strong> (or p = 2 if lag 2 is also significant)<br><br>
        • <strong>ACF</strong> shows a significant spike at lag 1, suggesting <strong>q = 1</strong> (or q = 2)<br><br>
        • We've established <strong>d = 1</strong> from the stationarity tests<br><br>
        • Strong spikes at lag 12 in both plots suggest seasonal components → SARIMA may be even better
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
        <strong>Recommended starting point: ARIMA(2, 1, 2)</strong><br><br>
        <em>We'll test variations in the Model Building section and compare using AIC/BIC.</em>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE: MODEL BUILDING
# ═════════════════════════════════════════════════════════════
elif page == "⚙️ Model Building":
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ Model Building</h1>
        <p>Fit ARIMA models with different parameters and compare their performance using AIC and BIC.</p>
    </div>
    """, unsafe_allow_html=True)

    df_log = np.log(df['Passengers'])

    # Train-Test Split
    st.markdown('<div class="section-header"><h2>Train-Test Split</h2></div>', unsafe_allow_html=True)

    test_months = st.slider("Number of months for test set:", 6, 36, 24)
    train = df_log[:-test_months]
    test = df_log[-test_months:]

    fig_split = go.Figure()
    fig_split.add_trace(go.Scatter(x=train.index, y=train.values, mode='lines', name=f'Training ({len(train)} months)', line=dict(color='#2d6a9f', width=2.5)))
    fig_split.add_trace(go.Scatter(x=test.index, y=test.values, mode='lines', name=f'Test ({len(test)} months)', line=dict(color='#e74c3c', width=2.5)))
    fig_split.add_vline(x=test.index[0], line_dash="dash", line_color="#95a5a6")
    fig_split.update_layout(
        title='Train-Test Split (Log Scale)', height=350,
        template='plotly_white', font=dict(family='Inter'), title_font_size=14,
        hovermode='x unified'
    )
    st.plotly_chart(fig_split, use_container_width=True)

    # --- Interactive Model Fitting ---
    st.markdown('<div class="section-header"><h2>Interactive Parameter Selection</h2></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        p = st.selectbox("**p** (AR order — from PACF):", [0, 1, 2, 3, 4], index=2)
    with col2:
        d = st.selectbox("**d** (Differencing order):", [0, 1, 2], index=1)
    with col3:
        q = st.selectbox("**q** (MA order — from ACF):", [0, 1, 2, 3, 4], index=2)

    if st.button("🔧 Fit ARIMA Model", use_container_width=True, type="primary"):
        with st.spinner(f"Fitting ARIMA({p},{d},{q})..."):
            try:
                model = ARIMA(train, order=(p, d, q))
                model_fit = model.fit()

                st.session_state['model_fit'] = model_fit
                st.session_state['train'] = train
                st.session_state['test'] = test
                st.session_state['order'] = (p, d, q)

                # Model Summary
                st.markdown(f'<div class="section-header"><h2>Model Summary: ARIMA({p},{d},{q})</h2></div>', unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{model_fit.aic:.1f}</div>
                        <div class="metric-label">AIC</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{model_fit.bic:.1f}</div>
                        <div class="metric-label">BIC</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{model_fit.llf:.1f}</div>
                        <div class="metric-label">Log Likelihood</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    n_params = p + q + (1 if d > 0 else 0)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{n_params}</div>
                        <div class="metric-label">Parameters</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div class="insight-box">
                    <strong>📋 AIC & BIC Explained:</strong><br><br>
                    • <strong>AIC (Akaike Information Criterion):</strong> Balances model fit vs. complexity. Lower = better.<br>
                    • <strong>BIC (Bayesian Information Criterion):</strong> Like AIC but penalizes complexity more. Lower = better.<br>
                    • Use these to compare different (p,d,q) combinations — the model with the lowest AIC/BIC wins.
                </div>
                """, unsafe_allow_html=True)

                with st.expander("📄 Full Model Summary (statsmodels output)"):
                    st.text(str(model_fit.summary()))

            except Exception as e:
                st.error(f"Model fitting failed: {str(e)}. Try different p, d, q values.")

    # --- Model Comparison ---
    st.markdown('<div class="section-header"><h2>Model Comparison Grid</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="concept-card">
        <p>Let's systematically compare common ARIMA configurations to find the best fit.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔍 Run Comparison (may take a moment)", use_container_width=True):
        comparison_results = []
        configs = [(1,1,0), (0,1,1), (1,1,1), (2,1,1), (1,1,2), (2,1,2), (3,1,1), (1,1,3), (2,1,0), (0,1,2)]

        progress = st.progress(0)
        for idx, (pi, di, qi) in enumerate(configs):
            try:
                m = ARIMA(train, order=(pi, di, qi)).fit()
                # Forecast on test set
                forecast = m.forecast(steps=len(test))
                mae = mean_absolute_error(test, forecast)
                comparison_results.append({
                    'Model': f'ARIMA({pi},{di},{qi})',
                    'AIC': round(m.aic, 2),
                    'BIC': round(m.bic, 2),
                    'Log Likelihood': round(m.llf, 2),
                    'Test MAE (log)': round(mae, 4)
                })
            except:
                comparison_results.append({
                    'Model': f'ARIMA({pi},{di},{qi})',
                    'AIC': 'Failed', 'BIC': 'Failed',
                    'Log Likelihood': 'Failed', 'Test MAE (log)': 'Failed'
                })
            progress.progress((idx + 1) / len(configs))

        comp_df = pd.DataFrame(comparison_results)
        # Highlight best
        valid = comp_df[comp_df['AIC'] != 'Failed'].copy()
        if len(valid) > 0:
            valid['AIC'] = valid['AIC'].astype(float)
            best_model = valid.loc[valid['AIC'].idxmin(), 'Model']
            st.dataframe(comp_df.set_index('Model'), use_container_width=True)
            st.markdown(f"""
            <div class="insight-box">
                <strong>🏆 Best Model (by AIC):</strong> <span style="font-size:1.2rem; font-weight:700;">{best_model}</span><br>
                Try this configuration in the interactive selector above!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.dataframe(comp_df.set_index('Model'), use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE: DIAGNOSTICS
# ═════════════════════════════════════════════════════════════
elif page == "🔍 Diagnostics":
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Model Diagnostics</h1>
        <p>Checking if the model residuals behave like white noise — the hallmark of a well-specified model.</p>
    </div>
    """, unsafe_allow_html=True)

    if 'model_fit' not in st.session_state:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ No model fitted yet!</strong> Please go to the <strong>⚙️ Model Building</strong> section first and fit an ARIMA model.
        </div>
        """, unsafe_allow_html=True)
    else:
        model_fit = st.session_state['model_fit']
        order = st.session_state['order']
        residuals = model_fit.resid

        st.markdown(f'<div class="section-header"><h2>Residual Analysis: ARIMA{order}</h2></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="concept-card">
            <h4>What are we checking?</h4>
            <p>
            If our model is good, the residuals (errors) should be <strong>white noise</strong>:
            </p>
            <p>
            ✅ <strong>No pattern</strong> in residuals over time<br>
            ✅ <strong>Normally distributed</strong> (bell curve shape)<br>
            ✅ <strong>No significant autocorrelation</strong> (ACF should be within bounds)<br>
            ✅ <strong>Constant variance</strong> (no fanning out or clustering)
            </p>
        </div>
        """, unsafe_allow_html=True)

        # 4 diagnostic plots
        fig_diag = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Residuals Over Time',
                'Histogram of Residuals',
                'ACF of Residuals',
                'Q-Q Plot (Normal)'
            ),
            vertical_spacing=0.15, horizontal_spacing=0.1
        )

        # 1. Residuals over time
        fig_diag.add_trace(
            go.Scatter(x=residuals.index, y=residuals.values, mode='lines',
                       line=dict(color='#2d6a9f', width=1.5), name='Residuals'),
            row=1, col=1
        )
        fig_diag.add_hline(y=0, line_color='red', line_dash='dash', row=1, col=1)

        # 2. Histogram
        fig_diag.add_trace(
            go.Histogram(x=residuals.values, nbinsx=30, marker_color='#2d6a9f',
                         opacity=0.7, name='Distribution'),
            row=1, col=2
        )

        # 3. ACF of residuals
        resid_acf = acf(residuals.dropna(), nlags=25)
        ci_r = 1.96 / np.sqrt(len(residuals))
        for i in range(len(resid_acf)):
            color = '#e74c3c' if abs(resid_acf[i]) > ci_r and i > 0 else '#2d6a9f'
            fig_diag.add_trace(
                go.Bar(x=[i], y=[resid_acf[i]], marker_color=color, showlegend=False, width=0.3),
                row=2, col=1
            )
        fig_diag.add_hline(y=ci_r, line_dash='dash', line_color='#95a5a6', row=2, col=1)
        fig_diag.add_hline(y=-ci_r, line_dash='dash', line_color='#95a5a6', row=2, col=1)

        # 4. QQ plot
        from scipy import stats
        sorted_resid = np.sort(residuals.dropna().values)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_resid)))
        fig_diag.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sorted_resid, mode='markers',
                       marker=dict(color='#2d6a9f', size=4), name='Q-Q'),
            row=2, col=2
        )
        # Reference line
        min_val = min(theoretical_quantiles.min(), sorted_resid.min())
        max_val = max(theoretical_quantiles.max(), sorted_resid.max())
        fig_diag.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                       line=dict(color='red', dash='dash', width=1.5), showlegend=False),
            row=2, col=2
        )

        fig_diag.update_layout(height=700, template='plotly_white', font=dict(family='Inter'), showlegend=False)
        st.plotly_chart(fig_diag, use_container_width=True)

        # --- Ljung-Box Test ---
        st.markdown('<div class="section-header"><h2>Ljung-Box Test for Autocorrelation</h2></div>', unsafe_allow_html=True)

        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_result = acorr_ljungbox(residuals.dropna(), lags=[10, 15, 20], return_df=True)

        st.dataframe(lb_result.round(4), use_container_width=True)

        all_pass = all(lb_result['lb_pvalue'] > 0.05)
        if all_pass:
            st.markdown("""
            <div class="insight-box">
                <strong>✅ Ljung-Box Test: PASS</strong><br>
                All p-values > 0.05, meaning we <em>cannot reject</em> the null hypothesis that residuals are independently distributed.
                The residuals show no significant autocorrelation — our model has captured the time series patterns well!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Ljung-Box Test: PARTIAL FAIL</strong><br>
                Some lags show significant autocorrelation (p < 0.05). This suggests the model may not have fully captured
                all patterns. Consider: adjusting p/q values, adding seasonal terms (SARIMA), or log/Box-Cox transformation.
            </div>
            """, unsafe_allow_html=True)

        # --- Normality Test ---
        st.markdown('<div class="section-header"><h2>Shapiro-Wilk Normality Test</h2></div>', unsafe_allow_html=True)

        shapiro_stat, shapiro_p = stats.shapiro(residuals.dropna())
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{shapiro_stat:.4f}</div>
                <div class="metric-label">Test Statistic</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            badge = "result-pass" if shapiro_p > 0.05 else "result-fail"
            label = "Normal ✅" if shapiro_p > 0.05 else "Not Normal ⚠️"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{shapiro_p:.4f}</div>
                <div class="metric-label">p-value — <span class="{badge}">{label}</span></div>
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE: FORECASTING & RESULTS
# ═════════════════════════════════════════════════════════════
elif page == "🚀 Forecasting & Results":
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Forecasting & Results</h1>
        <p>Evaluating model accuracy on the test set and generating future forecasts with confidence intervals.</p>
    </div>
    """, unsafe_allow_html=True)

    if 'model_fit' not in st.session_state:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ No model fitted yet!</strong> Please go to the <strong>⚙️ Model Building</strong> section first and fit an ARIMA model.
        </div>
        """, unsafe_allow_html=True)
    else:
        model_fit = st.session_state['model_fit']
        train = st.session_state['train']
        test = st.session_state['test']
        order = st.session_state['order']

        # --- Forecast on Test Set ---
        st.markdown('<div class="section-header"><h2>Test Set Forecast</h2></div>', unsafe_allow_html=True)

        forecast_result = model_fit.get_forecast(steps=len(test))
        forecast_log = forecast_result.predicted_mean
        forecast_ci_log = forecast_result.conf_int()

        # Convert back from log scale
        forecast_actual = np.exp(forecast_log)
        test_actual = np.exp(test)
        train_actual = np.exp(train)
        ci_lower = np.exp(forecast_ci_log.iloc[:, 0])
        ci_upper = np.exp(forecast_ci_log.iloc[:, 1])

        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=train_actual.index, y=train_actual.values,
            mode='lines', name='Training Data', line=dict(color='#2d6a9f', width=2)
        ))
        fig_forecast.add_trace(go.Scatter(
            x=test_actual.index, y=test_actual.values,
            mode='lines', name='Actual (Test)', line=dict(color='#2c3e50', width=2.5)
        ))
        fig_forecast.add_trace(go.Scatter(
            x=test.index, y=forecast_actual.values,
            mode='lines', name='Forecast', line=dict(color='#e74c3c', width=2.5, dash='dash')
        ))
        # Confidence interval
        fig_forecast.add_trace(go.Scatter(
            x=list(test.index) + list(test.index[::-1]),
            y=list(ci_upper.values) + list(ci_lower.values[::-1]),
            fill='toself', fillcolor='rgba(231, 76, 60, 0.15)',
            line=dict(color='rgba(231, 76, 60, 0)'), name='95% Confidence Interval'
        ))
        fig_forecast.update_layout(
            title=f'ARIMA{order} — Forecast vs Actual (Original Scale)',
            height=450, template='plotly_white', font=dict(family='Inter'),
            title_font_size=16, hovermode='x unified',
            yaxis_title='Passengers (thousands)'
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        # --- Accuracy Metrics ---
        st.markdown('<div class="section-header"><h2>Forecast Accuracy Metrics</h2></div>', unsafe_allow_html=True)

        mae = mean_absolute_error(test_actual, forecast_actual)
        rmse = np.sqrt(mean_squared_error(test_actual, forecast_actual))
        mape = mean_absolute_percentage_error(test_actual, forecast_actual) * 100

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{mae:.1f}</div>
                <div class="metric-label">MAE (Mean Abs Error)</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{rmse:.1f}</div>
                <div class="metric-label">RMSE (Root Mean Sq Error)</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            mape_color = '#166534' if mape < 10 else '#b45309' if mape < 20 else '#991b1b'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{mape_color}">{mape:.1f}%</div>
                <div class="metric-label">MAPE</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="concept-card">
            <h4>Understanding the Metrics</h4>
            <p>
            <strong>MAE (Mean Absolute Error):</strong> Average absolute difference between forecast and actual. Easy to interpret — "on average, we're off by X passengers."<br><br>
            <strong>RMSE (Root Mean Squared Error):</strong> Similar to MAE but penalizes large errors more heavily. Useful when big misses are particularly costly.<br><br>
            <strong>MAPE (Mean Absolute Percentage Error):</strong> Error as a percentage. Industry benchmarks: < 10% = Excellent, 10–20% = Good, 20–30% = Acceptable, > 30% = Poor.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # --- Forecast vs Actual Table ---
        with st.expander("📋 Detailed Forecast vs Actual Table"):
            detail_df = pd.DataFrame({
                'Actual': test_actual.values,
                'Forecast': forecast_actual.values.round(1),
                'Error': (test_actual.values - forecast_actual.values).round(1),
                'Abs % Error': ((np.abs(test_actual.values - forecast_actual.values) / test_actual.values) * 100).round(1)
            }, index=test.index.strftime('%b %Y'))
            st.dataframe(detail_df, use_container_width=True)

        # --- Future Forecast ---
        st.markdown('<div class="section-header"><h2>Future Forecast</h2></div>', unsafe_allow_html=True)

        future_months = st.slider("Months to forecast into the future:", 6, 36, 12)

        # Refit on full data
        full_model = ARIMA(np.log(df['Passengers']), order=order).fit()
        future_result = full_model.get_forecast(steps=future_months)
        future_log = future_result.predicted_mean
        future_ci_log = future_result.conf_int()

        future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=future_months, freq='MS')
        future_actual = np.exp(future_log)
        future_ci_lower = np.exp(future_ci_log.iloc[:, 0])
        future_ci_upper = np.exp(future_ci_log.iloc[:, 1])

        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(
            x=df.index, y=df['Passengers'],
            mode='lines', name='Historical', line=dict(color='#2d6a9f', width=2)
        ))
        fig_future.add_trace(go.Scatter(
            x=future_dates, y=future_actual.values,
            mode='lines', name='Forecast', line=dict(color='#e74c3c', width=2.5)
        ))
        fig_future.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(future_ci_upper.values) + list(future_ci_lower.values[::-1]),
            fill='toself', fillcolor='rgba(231, 76, 60, 0.15)',
            line=dict(color='rgba(231, 76, 60, 0)'), name='95% CI'
        ))
        fig_future.update_layout(
            title=f'ARIMA{order} — {future_months}-Month Future Forecast',
            height=450, template='plotly_white', font=dict(family='Inter'),
            title_font_size=16, hovermode='x unified',
            yaxis_title='Passengers (thousands)'
        )
        st.plotly_chart(fig_future, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            <strong>💡 Note on Confidence Intervals:</strong> The confidence bands widen as we forecast further into the future —
            this is expected and honest. It reflects increasing uncertainty. In practice, forecasts beyond 2–3 seasonal cycles
            should be treated with caution.
        </div>
        """, unsafe_allow_html=True)

        # --- Future forecast table ---
        with st.expander("📋 Future Forecast Table"):
            future_df = pd.DataFrame({
                'Forecast': future_actual.values.round(0),
                'Lower 95% CI': future_ci_lower.values.round(0),
                'Upper 95% CI': future_ci_upper.values.round(0)
            }, index=future_dates.strftime('%b %Y'))
            st.dataframe(future_df, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE: SARIMA — SEASONAL UPGRADE
# ═════════════════════════════════════════════════════════════
elif page == "🌊 SARIMA — Seasonal Upgrade":
    st.markdown("""
    <div class="main-header">
        <h1>🌊 SARIMA — Adding Seasonality to ARIMA</h1>
        <p>ARIMA captures trend but produces flat forecasts on seasonal data. SARIMA fixes this by adding seasonal components.</p>
    </div>
    """, unsafe_allow_html=True)

    df_log = np.log(df['Passengers'])

    # ── Why ARIMA Falls Short ──────────────────────────────────
    st.markdown('<div class="section-header"><h2>The Problem: ARIMA\'s Flat Forecast</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="concept-card">
        <p>
        In the previous section, the ARIMA forecast on the test set looked something like a flat line — completely missing
        the seasonal peaks and troughs. This isn't a bug. Standard ARIMA has <strong>no mechanism to model repeating
        seasonal patterns</strong>. It can handle trend (via differencing) and short-term autocorrelation (via AR and MA),
        but the 12-month summer-peak/winter-trough cycle is invisible to it.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Quick ARIMA demo forecast to show the flat line
    test_months_s = st.slider("Test set size (months):", 6, 36, 24, key="sarima_test")
    train_s = df_log[:-test_months_s]
    test_s = df_log[-test_months_s:]

    with st.spinner("Fitting ARIMA for comparison..."):
        arima_order = st.session_state.get('order', (2, 1, 2))
        try:
            arima_model = ARIMA(train_s, order=arima_order).fit()
            arima_fc = arima_model.get_forecast(steps=len(test_s))
            arima_pred = np.exp(arima_fc.predicted_mean)
        except:
            arima_order = (2, 1, 1)
            arima_model = ARIMA(train_s, order=arima_order).fit()
            arima_fc = arima_model.get_forecast(steps=len(test_s))
            arima_pred = np.exp(arima_fc.predicted_mean)

    test_actual = np.exp(test_s)
    train_actual = np.exp(train_s)

    fig_problem = go.Figure()
    fig_problem.add_trace(go.Scatter(x=train_actual.index, y=train_actual.values, mode='lines', name='Training', line=dict(color='#2d6a9f', width=2)))
    fig_problem.add_trace(go.Scatter(x=test_actual.index, y=test_actual.values, mode='lines', name='Actual (Test)', line=dict(color='#2c3e50', width=2.5)))
    fig_problem.add_trace(go.Scatter(x=test_s.index, y=arima_pred.values, mode='lines', name=f'ARIMA{arima_order} Forecast', line=dict(color='#e74c3c', width=2.5, dash='dash')))
    fig_problem.update_layout(
        title=f'ARIMA{arima_order} — Flat forecast, no seasonal pattern captured',
        height=400, template='plotly_white', font=dict(family='Inter'),
        title_font_size=15, hovermode='x unified', yaxis_title='Passengers (thousands)'
    )
    st.plotly_chart(fig_problem, use_container_width=True)

    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ The flat red line tells the whole story.</strong> ARIMA sees trend but has no concept of the
        12-month seasonal cycle. For any data with repeating seasonal patterns, we need SARIMA.
    </div>
    """, unsafe_allow_html=True)

    # ── What is SARIMA ─────────────────────────────────────────
    st.markdown('<div class="section-header"><h2>What is SARIMA?</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class="concept-card">
            <h4>SARIMA = ARIMA + Seasonal Components</h4>
            <p>
            SARIMA adds a second set of parameters that operate at the <strong>seasonal lag</strong> (e.g., every 12 months
            for monthly data):
            </p>
            <p>
            <strong>P</strong> — Seasonal AR order (how many past seasonal values to use)<br><br>
            <strong>D</strong> — Seasonal differencing (removes seasonal trend; usually 1)<br><br>
            <strong>Q</strong> — Seasonal MA order (how many past seasonal errors to use)<br><br>
            <strong>s</strong> — Seasonal period (12 for monthly data with yearly cycle)
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="formula-box">
            <strong>SARIMA(p, d, q)(P, D, Q, s)</strong><br><br>
            <em>Non-seasonal part:</em><br>
            (p, d, q) — same as ARIMA<br><br>
            <em>Seasonal part:</em><br>
            (P, D, Q, s) — operates at lag <strong>s</strong><br><br>
            For our airline data:<br>
            <strong>s = 12</strong> (monthly data, yearly cycle)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
            <strong>💡 Intuition:</strong> If July is always a peak month, SARIMA learns this by looking at what happened
            12, 24, 36 months ago — not just 1 or 2 months ago like ARIMA does.
        </div>
        """, unsafe_allow_html=True)

    # ── Seasonal Differencing Visual ───────────────────────────
    st.markdown('<div class="section-header"><h2>Seasonal Differencing (D=1, s=12)</h2></div>', unsafe_allow_html=True)

    seasonal_diff = df_log.diff(12).dropna()

    fig_sdiff = make_subplots(rows=2, cols=1, subplot_titles=(
        'Log Series (non-stationary — trend + seasonality)',
        'After Seasonal Differencing (D=1, s=12) — seasonality removed'
    ))
    fig_sdiff.add_trace(go.Scatter(y=df_log.values, x=df_log.index, mode='lines', line=dict(color='#2d6a9f', width=2), name='Log'), row=1, col=1)
    fig_sdiff.add_trace(go.Scatter(y=seasonal_diff.values, x=seasonal_diff.index, mode='lines', line=dict(color='#27ae60', width=2), name='Seasonal Diff'), row=2, col=1)
    fig_sdiff.update_layout(height=450, template='plotly_white', font=dict(family='Inter'), showlegend=False)
    st.plotly_chart(fig_sdiff, use_container_width=True)

    adf_seasonal = run_adf(seasonal_diff, 'Seasonal Differenced (D=1, s=12)')
    adf_both = run_adf(seasonal_diff.diff().dropna(), 'Seasonal + First Diff (d=1, D=1)')
    st.dataframe(pd.DataFrame([adf_seasonal, adf_both]).set_index('Series'), use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <strong>💡 Key point:</strong> Regular differencing (d=1) removes <em>trend</em>. Seasonal differencing (D=1, s=12) removes the
        <em>seasonal pattern</em>. For this dataset we typically need both: <strong>d=1, D=1</strong>.
    </div>
    """, unsafe_allow_html=True)

    # ── Fit SARIMA ─────────────────────────────────────────────
    st.markdown('<div class="section-header"><h2>Fit SARIMA Model</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="concept-card">
        <p>Select the non-seasonal (p, d, q) and seasonal (P, D, Q) parameters below. The seasonal period <strong>s</strong> is fixed at 12 for this monthly dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Non-seasonal (p, d, q)**")
        c1, c2, c3 = st.columns(3)
        sp = c1.selectbox("p", [0,1,2,3], index=1, key="sp")
        sd = c2.selectbox("d", [0,1,2], index=1, key="sd")
        sq = c3.selectbox("q", [0,1,2,3], index=1, key="sq")
    with col2:
        st.markdown("**Seasonal (P, D, Q) with s=12**")
        c4, c5, c6 = st.columns(3)
        sP = c4.selectbox("P", [0,1,2], index=1, key="sP")
        sD = c5.selectbox("D", [0,1,2], index=1, key="sD")
        sQ = c6.selectbox("Q", [0,1,2], index=1, key="sQ")

    if st.button("🔧 Fit SARIMA Model", use_container_width=True, type="primary"):
        with st.spinner(f"Fitting SARIMA({sp},{sd},{sq})({sP},{sD},{sQ},12) — this may take a moment..."):
            try:
                sarima_model = SARIMAX(
                    train_s, order=(sp, sd, sq),
                    seasonal_order=(sP, sD, sQ, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)

                st.session_state['sarima_fit'] = sarima_model
                st.session_state['sarima_order'] = (sp, sd, sq)
                st.session_state['sarima_seasonal'] = (sP, sD, sQ, 12)
                st.session_state['sarima_train'] = train_s
                st.session_state['sarima_test'] = test_s

                st.success(f"SARIMA({sp},{sd},{sq})({sP},{sD},{sQ},12) fitted successfully!")

                # ── Model Summary Metrics ──
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""<div class="metric-card"><div class="metric-value">{sarima_model.aic:.1f}</div><div class="metric-label">AIC</div></div>""", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""<div class="metric-card"><div class="metric-value">{sarima_model.bic:.1f}</div><div class="metric-label">BIC</div></div>""", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""<div class="metric-card"><div class="metric-value">{sarima_model.llf:.1f}</div><div class="metric-label">Log Likelihood</div></div>""", unsafe_allow_html=True)

                with st.expander("📄 Full Model Summary"):
                    st.text(str(sarima_model.summary()))

            except Exception as e:
                st.error(f"SARIMA fitting failed: {str(e)}. Try different parameters — reducing P or Q often helps.")

    # ── Head-to-Head Comparison ────────────────────────────────
    if 'sarima_fit' in st.session_state:
        sarima_model = st.session_state['sarima_fit']
        s_order = st.session_state['sarima_order']
        s_seasonal = st.session_state['sarima_seasonal']

        st.markdown('<div class="section-header"><h2>Head-to-Head: ARIMA vs SARIMA</h2></div>', unsafe_allow_html=True)

        # SARIMA forecast
        sarima_fc = sarima_model.get_forecast(steps=len(test_s))
        sarima_pred = np.exp(sarima_fc.predicted_mean)
        sarima_ci = sarima_fc.conf_int()
        sarima_ci_lower = np.exp(sarima_ci.iloc[:, 0])
        sarima_ci_upper = np.exp(sarima_ci.iloc[:, 1])

        # ── Comparison Plot ──
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(
            x=train_actual.index, y=train_actual.values,
            mode='lines', name='Training', line=dict(color='#2d6a9f', width=2)
        ))
        fig_compare.add_trace(go.Scatter(
            x=test_actual.index, y=test_actual.values,
            mode='lines', name='Actual (Test)', line=dict(color='#2c3e50', width=3)
        ))
        fig_compare.add_trace(go.Scatter(
            x=test_s.index, y=arima_pred.values,
            mode='lines', name=f'ARIMA{arima_order}', line=dict(color='#e74c3c', width=2.5, dash='dash')
        ))
        fig_compare.add_trace(go.Scatter(
            x=test_s.index, y=sarima_pred.values,
            mode='lines', name=f'SARIMA{s_order}{s_seasonal}', line=dict(color='#27ae60', width=2.5, dash='dash')
        ))
        # SARIMA confidence interval
        fig_compare.add_trace(go.Scatter(
            x=list(test_s.index) + list(test_s.index[::-1]),
            y=list(sarima_ci_upper.values) + list(sarima_ci_lower.values[::-1]),
            fill='toself', fillcolor='rgba(39, 174, 96, 0.12)',
            line=dict(color='rgba(39, 174, 96, 0)'), name='SARIMA 95% CI'
        ))
        fig_compare.update_layout(
            title='ARIMA vs SARIMA — Forecast Comparison',
            height=500, template='plotly_white', font=dict(family='Inter'),
            title_font_size=16, hovermode='x unified', yaxis_title='Passengers (thousands)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        # ── Accuracy Metrics Comparison ──
        arima_mae, arima_rmse, arima_mape = forecast_metrics(test_actual, arima_pred)
        sarima_mae, sarima_rmse, sarima_mape = forecast_metrics(test_actual, sarima_pred)

        st.markdown("""
        <div style="text-align:center; margin:1rem 0;">
            <span style="background:linear-gradient(135deg,#fef3c7,#fde68a); border:2px solid #f59e0b;
            border-radius:12px; padding:0.5rem 1.5rem; font-weight:700; color:#92400e; font-size:1.1rem;">
            📊 Accuracy Comparison</span>
        </div>
        """, unsafe_allow_html=True)

        comp_df = pd.DataFrame({
            'Metric': ['MAE', 'RMSE', 'MAPE (%)'],
            f'ARIMA{arima_order}': [f'{arima_mae:.1f}', f'{arima_rmse:.1f}', f'{arima_mape:.1f}%'],
            f'SARIMA{s_order}{s_seasonal}': [f'{sarima_mae:.1f}', f'{sarima_rmse:.1f}', f'{sarima_mape:.1f}%'],
            'Improvement': [
                f'{((arima_mae - sarima_mae) / arima_mae * 100):.0f}%',
                f'{((arima_rmse - sarima_rmse) / arima_rmse * 100):.0f}%',
                f'{((arima_mape - sarima_mape) / arima_mape * 100):.0f}%'
            ]
        }).set_index('Metric')
        st.dataframe(comp_df, use_container_width=True)

        mape_color = '#166534' if sarima_mape < 10 else '#b45309' if sarima_mape < 20 else '#991b1b'
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{sarima_mae:.1f}</div><div class="metric-label">SARIMA MAE</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{sarima_rmse:.1f}</div><div class="metric-label">SARIMA RMSE</div></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:{mape_color}">{sarima_mape:.1f}%</div><div class="metric-label">SARIMA MAPE</div></div>""", unsafe_allow_html=True)
        with col4:
            improv = ((arima_mape - sarima_mape) / arima_mape * 100)
            st.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:#166534">↓ {improv:.0f}%</div><div class="metric-label">MAPE Improvement</div></div>""", unsafe_allow_html=True)

        # ── SARIMA Residual Diagnostics ──
        st.markdown('<div class="section-header"><h2>SARIMA Residual Diagnostics</h2></div>', unsafe_allow_html=True)

        residuals = sarima_model.resid

        fig_diag = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Residuals Over Time', 'Histogram of Residuals', 'ACF of Residuals', 'Q-Q Plot'),
            vertical_spacing=0.15, horizontal_spacing=0.1
        )

        fig_diag.add_trace(go.Scatter(x=residuals.index, y=residuals.values, mode='lines', line=dict(color='#27ae60', width=1.5), name='Residuals'), row=1, col=1)
        fig_diag.add_hline(y=0, line_color='red', line_dash='dash', row=1, col=1)

        fig_diag.add_trace(go.Histogram(x=residuals.values, nbinsx=30, marker_color='#27ae60', opacity=0.7, name='Distribution'), row=1, col=2)

        resid_acf = acf(residuals.dropna(), nlags=25)
        ci_r = 1.96 / np.sqrt(len(residuals))
        for i in range(len(resid_acf)):
            color = '#e74c3c' if abs(resid_acf[i]) > ci_r and i > 0 else '#27ae60'
            fig_diag.add_trace(go.Bar(x=[i], y=[resid_acf[i]], marker_color=color, showlegend=False, width=0.3), row=2, col=1)
        fig_diag.add_hline(y=ci_r, line_dash='dash', line_color='#95a5a6', row=2, col=1)
        fig_diag.add_hline(y=-ci_r, line_dash='dash', line_color='#95a5a6', row=2, col=1)

        sorted_resid = np.sort(residuals.dropna().values)
        theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_resid)))
        fig_diag.add_trace(go.Scatter(x=theoretical_q, y=sorted_resid, mode='markers', marker=dict(color='#27ae60', size=4), name='Q-Q'), row=2, col=2)
        min_v, max_v = min(theoretical_q.min(), sorted_resid.min()), max(theoretical_q.max(), sorted_resid.max())
        fig_diag.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode='lines', line=dict(color='red', dash='dash', width=1.5), showlegend=False), row=2, col=2)

        fig_diag.update_layout(height=650, template='plotly_white', font=dict(family='Inter'), showlegend=False)
        st.plotly_chart(fig_diag, use_container_width=True)

        # Ljung-Box
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_result = acorr_ljungbox(residuals.dropna(), lags=[10, 15, 20], return_df=True)
        st.dataframe(lb_result.round(4), use_container_width=True)

        all_pass = all(lb_result['lb_pvalue'] > 0.05)
        if all_pass:
            st.markdown("""
            <div style="background:linear-gradient(135deg,#ecfdf5,#d1fae5); border:1px solid #6ee7b7; border-radius:10px; padding:1.25rem; margin:1rem 0;">
                <strong style="color:#065f46;">✅ Ljung-Box: PASS</strong> — Residuals show no significant autocorrelation. The SARIMA model has captured both trend and seasonal patterns effectively.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Ljung-Box: Some lags significant.</strong> The model captures most structure but minor autocorrelation remains. Consider tuning P, Q, or the non-seasonal parameters.
            </div>
            """, unsafe_allow_html=True)

        # ── Future Forecast ────────────────────────────────────
        st.markdown('<div class="section-header"><h2>SARIMA Future Forecast</h2></div>', unsafe_allow_html=True)

        future_months = st.slider("Months to forecast:", 6, 48, 24, key="sarima_future")

        with st.spinner("Fitting SARIMA on full data and forecasting..."):
            full_sarima = SARIMAX(
                df_log, order=s_order, seasonal_order=s_seasonal,
                enforce_stationarity=False, enforce_invertibility=False
            ).fit(disp=False)

            future_fc = full_sarima.get_forecast(steps=future_months)
            future_pred = np.exp(future_fc.predicted_mean)
            future_ci = future_fc.conf_int()
            future_ci_lower = np.exp(future_ci.iloc[:, 0])
            future_ci_upper = np.exp(future_ci.iloc[:, 1])
            future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=future_months, freq='MS')

        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=df.index, y=df['Passengers'], mode='lines', name='Historical', line=dict(color='#2d6a9f', width=2)))
        fig_future.add_trace(go.Scatter(x=future_dates, y=future_pred.values, mode='lines', name='SARIMA Forecast', line=dict(color='#27ae60', width=2.5)))
        fig_future.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(future_ci_upper.values) + list(future_ci_lower.values[::-1]),
            fill='toself', fillcolor='rgba(39, 174, 96, 0.12)',
            line=dict(color='rgba(39, 174, 96, 0)'), name='95% CI'
        ))
        fig_future.update_layout(
            title=f'SARIMA{s_order}{s_seasonal} — {future_months}-Month Future Forecast',
            height=450, template='plotly_white', font=dict(family='Inter'),
            title_font_size=16, hovermode='x unified', yaxis_title='Passengers (thousands)'
        )
        st.plotly_chart(fig_future, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            <strong>💡 Notice the difference:</strong> Unlike ARIMA's flat line, the SARIMA forecast shows
            the seasonal peaks and troughs continuing into the future — exactly what we'd expect for airline traffic.
            The confidence intervals widen over time, reflecting increasing uncertainty.
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📋 Future Forecast Table"):
            future_df = pd.DataFrame({
                'Forecast': future_pred.values.round(0),
                'Lower 95% CI': future_ci_lower.values.round(0),
                'Upper 95% CI': future_ci_upper.values.round(0)
            }, index=future_dates.strftime('%b %Y'))
            st.dataframe(future_df, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE: SUMMARY & CHEAT SHEET
# ═════════════════════════════════════════════════════════════
elif page == "📋 Summary & Cheat Sheet":
    st.markdown("""
    <div class="main-header">
        <h1>📋 Summary & Cheat Sheet</h1>
        <p>A one-page reference you can use for assignments, exams, and real-world projects.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- ARIMA Cheat Sheet ---
    st.markdown('<div class="section-header"><h2>ARIMA Quick Reference</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="concept-card">
            <h4>The ARIMA Equation</h4>
            <p>
            <strong>ARIMA(p, d, q)</strong> combines:<br><br>
            <strong>AR(p):</strong> Y depends on p past values<br>
            <strong>I(d):</strong> Difference d times for stationarity<br>
            <strong>MA(q):</strong> Y depends on q past errors<br><br>
            <em>Full form:</em><br>
            Y'<sub>t</sub> = c + φ₁Y'<sub>t-1</sub> + ... + φ<sub>p</sub>Y'<sub>t-p</sub> + θ₁ε<sub>t-1</sub> + ... + θ<sub>q</sub>ε<sub>t-q</sub> + ε<sub>t</sub>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="concept-card">
            <h4>Parameter Selection Rules</h4>
            <p>
            <strong>d:</strong> Use ADF test. Difference until stationary (usually d=1).<br><br>
            <strong>p (AR):</strong> Count significant PACF lags before cutoff.<br><br>
            <strong>q (MA):</strong> Count significant ACF lags before cutoff.<br><br>
            <strong>Validation:</strong> Compare AIC/BIC across candidates. Lower = better.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="concept-card">
            <h4>The 7-Step ARIMA Workflow</h4>
            <p>
            <span class="step-indicator">1</span> <strong>Plot</strong> the series — identify trend & seasonality<br><br>
            <span class="step-indicator">2</span> <strong>Transform</strong> — log if variance increases with level<br><br>
            <span class="step-indicator">3</span> <strong>Test stationarity</strong> — ADF test (p < 0.05 = stationary)<br><br>
            <span class="step-indicator">4</span> <strong>Difference</strong> — until stationary (usually d=1)<br><br>
            <span class="step-indicator">5</span> <strong>ACF/PACF</strong> — determine p and q<br><br>
            <span class="step-indicator">6</span> <strong>Fit & Compare</strong> — try several (p,d,q), pick lowest AIC<br><br>
            <span class="step-indicator">7</span> <strong>Diagnose</strong> — residuals should be white noise
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="concept-card">
            <h4>Diagnostic Checklist</h4>
            <p>
            ✅ Residuals show no pattern over time<br>
            ✅ Residuals are approximately normal (Q-Q plot)<br>
            ✅ No significant autocorrelation in residuals (ACF)<br>
            ✅ Ljung-Box test p-values > 0.05<br>
            ✅ MAPE < 20% for acceptable forecast accuracy
            </p>
        </div>
        """, unsafe_allow_html=True)

    # --- Common Patterns ---
    st.markdown('<div class="section-header"><h2>ACF/PACF Pattern Recognition</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="concept-card">
        <p>
        <table style="width:100%; border-collapse:collapse; font-size:0.95rem;">
        <tr style="background:#f0f7ff;">
            <th style="padding:10px; text-align:left; border-bottom:2px solid #2d6a9f;">ACF Pattern</th>
            <th style="padding:10px; text-align:left; border-bottom:2px solid #2d6a9f;">PACF Pattern</th>
            <th style="padding:10px; text-align:left; border-bottom:2px solid #2d6a9f;">Model Suggested</th>
        </tr>
        <tr>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0;">Tails off (decays gradually)</td>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0;">Cuts off after lag p</td>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0;"><strong>AR(p)</strong></td>
        </tr>
        <tr>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0;">Cuts off after lag q</td>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0;">Tails off (decays gradually)</td>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0;"><strong>MA(q)</strong></td>
        </tr>
        <tr>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0;">Tails off</td>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0;">Tails off</td>
            <td style="padding:10px; border-bottom:1px solid #e2e8f0;"><strong>ARMA(p, q)</strong></td>
        </tr>
        <tr>
            <td style="padding:10px;">Significant spike at lag s (e.g., 12)</td>
            <td style="padding:10px;">Significant spike at lag s</td>
            <td style="padding:10px;"><strong>Seasonal → consider SARIMA</strong></td>
        </tr>
        </table>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Key Formulas ---
    st.markdown('<div class="section-header"><h2>Key Formulas & Metrics</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="formula-box">
            <strong>ADF Test</strong><br>
            H₀: Unit root exists (non-stationary)<br>
            Reject if p-value < 0.05
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="formula-box">
            <strong>AIC</strong> = 2k − 2ln(L̂)<br>
            <em>k = parameters, L̂ = max likelihood</em>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="formula-box">
            <strong>MAE</strong> = (1/n) Σ |yᵢ − ŷᵢ|<br>
            <em>Average absolute error</em>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="formula-box">
            <strong>MAPE</strong> = (100/n) Σ |yᵢ − ŷᵢ| / |yᵢ|<br>
            <em>Percentage error — < 10% excellent</em>
        </div>
        """, unsafe_allow_html=True)

    # --- When NOT to use ARIMA ---
    st.markdown('<div class="section-header"><h2>Limitations & When NOT to Use ARIMA</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ ARIMA Limitations:</strong><br><br>
        • Assumes <strong>linear relationships</strong> — won't capture complex non-linear patterns<br>
        • Requires <strong>regularly spaced</strong> time intervals<br>
        • Works best with <strong>univariate</strong> data (single variable)<br>
        • <strong>Long seasonal periods</strong> (e.g., daily data with yearly seasonality = 365) → use SARIMA or Prophet<br>
        • <strong>Multiple external drivers</strong> (price, marketing spend, etc.) → use ARIMAX or ML methods<br>
        • <strong>Very noisy or volatile</strong> data (e.g., stock prices) → ARIMA forecasts may not outperform a random walk
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <strong>📚 What to Explore Next:</strong><br><br>
        • <strong>SARIMA</strong> — ARIMA with seasonal components (p,d,q)(P,D,Q,s)<br>
        • <strong>Auto-ARIMA</strong> — Automated parameter selection (pmdarima library)<br>
        • <strong>Facebook Prophet</strong> — Modern alternative, handles holidays and multiple seasonalities<br>
        • <strong>LSTM / Neural Networks</strong> — Deep learning for complex non-linear time series
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#64748b; padding:1rem;'>"
        "📈 <strong>ARIMA Forecasting Masterclass</strong> | SP Jain School of Global Management | Global MBA Program<br>"
        "Prepared by Dr. Anshul Gupta"
        "</div>",
        unsafe_allow_html=True
    )
