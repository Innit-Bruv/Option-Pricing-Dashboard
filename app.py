"""
Option Pricing Dashboard
Compares Monte Carlo and Black-Scholes option pricing methods
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os

import yfinance as yf
from datetime import datetime
import pytz
from scipy.stats import norm

# Configure Streamlit page
st.set_page_config(
    page_title="Option Pricing Dashboard",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'market_data' not in st.session_state:
    st.session_state.market_data = None
if 'ticker_loaded' not in st.session_state:
    st.session_state.ticker_loaded = None

# ============================================================================
# Notebook Functions
# ============================================================================

def extract_code_cells(notebook_path):
    """Reads a Jupyter notebook and extracts all code cell sources."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    code_cells = []
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                code = ''.join(source)
            else:
                code = source
            code_cells.append(code)
    
    return code_cells

def remove_main_block(code):
    """Removes 'if __name__ == \"__main__\":' blocks from code."""
    lines = code.split('\n')
    result_lines = []
    skip_indent = None
    
    for line in lines:
        if 'if __name__' in line and '==' in line and '__main__' in line:
            skip_indent = len(line) - len(line.lstrip())
            continue
        
        if skip_indent is not None:
            if line.strip() == '':
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent > skip_indent:
                continue
            else:
                skip_indent = None
        
        if skip_indent is None:
            result_lines.append(line)
    
    return '\n'.join(result_lines)

@st.cache_resource
def load_notebook_functions():
    """Loads functions from both Jupyter notebooks into separate namespaces."""
    notebook_dir = os.path.dirname(os.path.abspath(__file__))
    
    common_imports = {
        'np': np, 'numpy': np, 'pd': pd, 'pandas': pd,
        'yf': yf, 'datetime': datetime, 'pytz': pytz, 'norm': norm,
    }
    
    monte_carlo_ns = common_imports.copy()
    monte_carlo_ns['__builtins__'] = __builtins__
    
    mc_path = os.path.join(notebook_dir, "Option_Pricing_MonteCarlo.ipynb")
    for code in extract_code_cells(mc_path):
        clean_code = remove_main_block(code)
        if clean_code.strip():
            try:
                exec(clean_code, monte_carlo_ns)
            except:
                pass
    
    black_scholes_ns = common_imports.copy()
    black_scholes_ns['__builtins__'] = __builtins__
    
    bs_path = os.path.join(notebook_dir, "Option_Pricing_Scholes.ipynb")
    for code in extract_code_cells(bs_path):
        clean_code = remove_main_block(code)
        if clean_code.strip():
            try:
                exec(clean_code, black_scholes_ns)
            except:
                pass
    
    return monte_carlo_ns, black_scholes_ns

# ============================================================================
# Design System CSS
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0');
    
    :root {
        --bg-primary: #0a0a0a;
        --bg-secondary: #141414;
        --bg-tertiary: #1c1c1c;
        --bg-hover: #252525;
        --text-primary: #fafafa;
        --text-secondary: #a1a1a1;
        --text-tertiary: #525252;
        --border: #262626;
        --border-light: #333333;
        --accent-green: #22c55e;
        --accent-red: #ef4444;
        --accent-blue: #3b82f6;
        --accent-amber: #f59e0b;
    }
    
    /* Base */
    .stApp {
        background-color: var(--bg-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }
    
    p, span, div {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header */
    .header-container {
        margin-bottom: 2.5rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid var(--border);
    }
    
    .header-title {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .header-subtitle {
        font-size: 0.95rem;
        color: var(--text-secondary);
        margin-top: 0.375rem;
        font-weight: 400;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border);
    }
    
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-tertiary);
        margin-bottom: 1rem;
    }
    
    /* Inputs */
    [data-testid="stTextInput"] input {
        background-color: var(--bg-tertiary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.95rem !important;
        padding: 0.625rem 0.875rem !important;
    }
    
    [data-testid="stTextInput"] input:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.15) !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background-color: var(--border) !important;
    }
    
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: var(--text-primary) !important;
        border: none !important;
    }
    
    /* Button */
    .stButton > button {
        background-color: var(--text-primary) !important;
        color: var(--bg-primary) !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        padding: 0.625rem 1.25rem !important;
        border-radius: 6px !important;
        border: none !important;
        transition: all 0.15s ease !important;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: var(--text-secondary) !important;
        transform: none !important;
        box-shadow: none !important;
    }
    
    /* Metrics Card */
    .metric-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.25rem;
    }
    
    .metric-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border);
    }
    
    /* Method Labels */
    .method-label {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        padding: 0.375rem 0.75rem;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 1rem;
    }
    
    .method-mc {
        background: rgba(59, 130, 246, 0.12);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.25);
    }
    
    .method-bs {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.25);
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }
    
    [data-testid="stDataFrame"] th {
        background: var(--bg-tertiary) !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    
    [data-testid="stDataFrame"] td {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.875rem !important;
    }
    
    /* Selectbox */
    [data-testid="stSelectbox"] > div > div {
        background-color: var(--bg-tertiary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
    }
    
    /* Welcome Card */
    .welcome-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 2.5rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    .welcome-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .welcome-text {
        font-size: 0.95rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }
    
    /* Ticker chips */
    .ticker-grid {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        justify-content: center;
        margin-top: 1.5rem;
    }
    
    .ticker-chip {
        background: var(--bg-tertiary);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 0.75rem 1.25rem;
        text-align: center;
        transition: all 0.15s ease;
    }
    
    .ticker-chip:hover {
        border-color: var(--border-light);
        background: var(--bg-hover);
    }
    
    .ticker-symbol {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 0.95rem;
        color: var(--text-primary);
    }
    
    .ticker-name {
        font-size: 0.75rem;
        color: var(--text-tertiary);
        margin-top: 0.25rem;
    }
    
    /* Stats row */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .stat-label {
        font-size: 0.7rem;
        font-weight: 500;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    
    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-top: 0.375rem;
    }
    
    .stat-positive { color: var(--accent-green) !important; }
    .stat-negative { color: var(--accent-red) !important; }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--border);
        text-align: center;
    }
    
    .footer-text {
        font-size: 0.8rem;
        color: var(--text-tertiary);
    }
    
    /* Expander */
    [data-testid="stExpander"] {
        background: var(--bg-tertiary);
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }
    
    [data-testid="stExpander"] summary {
        padding: 0.875rem 1rem;
        font-weight: 500;
        color: var(--text-primary);
    }
    
    [data-testid="stExpander"] summary span {
        font-family: 'Material Symbols Rounded' !important;
        font-size: 20px !important;
    }
    
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] {
        padding: 0 1rem 1rem 1rem;
    }
    
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
        color: var(--text-secondary) !important;
        font-size: 0.875rem;
        line-height: 1.6;
    }
    
    /* Alert styling */
    .stSuccess {
        background-color: rgba(34, 197, 94, 0.1) !important;
        border: 1px solid rgba(34, 197, 94, 0.3) !important;
        color: #4ade80 !important;
    }
    
    .stError {
        background-color: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
    }
    
    /* Divider */
    hr {
        border-color: var(--border) !important;
        margin: 2rem 0 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Metric styling override */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.5rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        color: var(--text-tertiary) !important;
    }
    
    /* Comparison info */
    .comparison-note {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Header
# ============================================================================

st.markdown("""
<div class="header-container">
    <h1 class="header-title">Option Pricing</h1>
    <p class="header-subtitle">Compare Monte Carlo simulation against Black-Scholes analytical pricing</p>
</div>
""", unsafe_allow_html=True)

# Load notebook functions
try:
    monte_carlo_ns, black_scholes_ns = load_notebook_functions()
    notebooks_loaded = True
except Exception as e:
    st.error(f"Error loading notebooks: {str(e)}")
    notebooks_loaded = False

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.markdown("### Configuration")
    
    ticker = st.text_input(
        "Stock Ticker",
        value="AAPL",
        placeholder="AAPL, MSFT, NVDA...",
        label_visibility="collapsed"
    ).upper()
    
    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
    
    mc_simulations = st.slider(
        "Simulations",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000,
        help="Number of Monte Carlo simulation paths"
    )
    
    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
    
    run_analysis = st.button("Fetch Market Data", use_container_width=True)
    
    if st.session_state.ticker_loaded and st.session_state.ticker_loaded != ticker:
        st.session_state.data_loaded = False
        st.session_state.market_data = None
    
    st.markdown("<hr style='margin: 1.5rem 0; border-color: #262626;'>", unsafe_allow_html=True)
    
    st.markdown("### Reference")
    
    with st.expander("Monte Carlo Method"):
        st.markdown("""
        Simulates thousands of random price paths using geometric Brownian motion.
        
        The option value is the average discounted payoff across all paths.
        """)
    
    with st.expander("Black-Scholes Model"):
        st.markdown("""
        Analytical closed-form solution for European options.
        
        Assumes log-normal price distribution, constant volatility, and no dividends.
        """)

# ============================================================================
# Main Content
# ============================================================================

if notebooks_loaded:
    if run_analysis:
        with st.spinner(f"Loading {ticker}..."):
            try:
                get_market_data = monte_carlo_ns['get_market_data']
                calculate_historical_volatility = monte_carlo_ns['calculate_historical_volatility']
                get_risk_free_rate = monte_carlo_ns['get_risk_free_rate']
                calculate_time_to_expiry = monte_carlo_ns['calculate_time_to_expiry']
                
                data = get_market_data(ticker)
                
                if data['calls_df'].empty:
                    st.error(f"No option data available for {ticker}")
                else:
                    sigma = calculate_historical_volatility(data['historical_df'].copy())
                    r = get_risk_free_rate()
                    
                    data['calls_df']['T'] = data['calls_df']['expirationDate'].apply(calculate_time_to_expiry)
                    
                    st.session_state.market_data = {
                        'S': data['spot_price'],
                        'calls_df': data['calls_df'],
                        'sigma': sigma,
                        'r': r
                    }
                    st.session_state.data_loaded = True
                    st.session_state.ticker_loaded = ticker
                    st.success(f"Loaded {ticker} market data")
                    
            except Exception as e:
                st.error(f"Failed to fetch data: {str(e)}")
    
    # Display results
    if st.session_state.data_loaded and st.session_state.market_data:
        md = st.session_state.market_data
        S, calls_df, sigma, r = md['S'], md['calls_df'], md['sigma'], md['r']
        
        monte_carlo_pricer = monte_carlo_ns['monte_carlo_pricer']
        black_scholes_calculator = black_scholes_ns['black_scholes_calculator']
        
        # Market data metrics
        st.markdown('<div class="section-header">Market Data</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Spot Price", f"${S:.2f}")
        with col2:
            st.metric("Volatility (σ)", f"{sigma:.1%}")
        with col3:
            st.metric("Risk-Free Rate", f"{r:.2%}")
        with col4:
            st.metric("Simulations", f"{mc_simulations:,}")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Expiration selector
        expiry_dates = sorted(calls_df['expirationDate'].unique())
        selected_expiry = st.selectbox(
            "Expiration Date",
            expiry_dates,
            index=0,
            label_visibility="visible"
        )
        
        # Filter and calculate
        filtered_df = calls_df[calls_df['expirationDate'] == selected_expiry].copy()
        filtered_df['dist'] = abs(filtered_df['strike'] - S)
        near_money = filtered_df.nsmallest(15, 'dist').sort_values('strike').copy()
        
        with st.spinner("Calculating prices..."):
            mc_prices = []
            bs_prices = []
            
            for idx, row in near_money.iterrows():
                K, T = row['strike'], row['T']
                
                mc_price = monte_carlo_pricer(S, K, T, r, sigma, mc_simulations)
                mc_prices.append(mc_price)
                
                bs_price = black_scholes_calculator(S, K, T, r, sigma)
                bs_prices.append(bs_price)
            
            near_money['MC_Price'] = mc_prices
            near_money['BS_Price'] = bs_prices
        
        # Results section
        st.markdown('<div class="section-header">Pricing Results</div>', unsafe_allow_html=True)
        st.markdown(f'<p class="comparison-note">Comparing {len(near_money)} near-the-money options expiring {selected_expiry}</p>', unsafe_allow_html=True)
        
        # Two-column comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<span class="method-label method-mc">Monte Carlo</span>', unsafe_allow_html=True)
            mc_df = near_money[['strike', 'lastPrice', 'MC_Price']].copy()
            mc_df['Δ'] = mc_df['MC_Price'] - mc_df['lastPrice']
            mc_df.columns = ['Strike', 'Market', 'Model', 'Δ']
            mc_df['Strike'] = mc_df['Strike'].apply(lambda x: f"${x:.0f}")
            mc_df['Market'] = mc_df['Market'].apply(lambda x: f"${x:.2f}")
            mc_df['Model'] = mc_df['Model'].apply(lambda x: f"${x:.2f}")
            mc_df['Δ'] = mc_df['Δ'].apply(lambda x: f"{x:+.2f}")
            st.dataframe(mc_df.reset_index(drop=True), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown('<span class="method-label method-bs">Black-Scholes</span>', unsafe_allow_html=True)
            bs_df = near_money[['strike', 'lastPrice', 'BS_Price']].copy()
            bs_df['Δ'] = bs_df['BS_Price'] - bs_df['lastPrice']
            bs_df.columns = ['Strike', 'Market', 'Model', 'Δ']
            bs_df['Strike'] = bs_df['Strike'].apply(lambda x: f"${x:.0f}")
            bs_df['Market'] = bs_df['Market'].apply(lambda x: f"${x:.2f}")
            bs_df['Model'] = bs_df['Model'].apply(lambda x: f"${x:.2f}")
            bs_df['Δ'] = bs_df['Δ'].apply(lambda x: f"{x:+.2f}")
            st.dataframe(bs_df.reset_index(drop=True), use_container_width=True, hide_index=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Model comparison stats
        st.markdown('<div class="section-header">Model Accuracy</div>', unsafe_allow_html=True)
        
        mc_diff = near_money['MC_Price'] - near_money['lastPrice']
        bs_diff = near_money['BS_Price'] - near_money['lastPrice']
        mc_bs_diff = near_money['MC_Price'] - near_money['BS_Price']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("MC Mean Error", f"${mc_diff.mean():.3f}")
        with col2:
            st.metric("MC RMSE", f"${np.sqrt((mc_diff**2).mean()):.3f}")
        with col3:
            st.metric("BS Mean Error", f"${bs_diff.mean():.3f}")
        with col4:
            st.metric("BS RMSE", f"${np.sqrt((bs_diff**2).mean()):.3f}")
        with col5:
            st.metric("MC vs BS Δ", f"${mc_bs_diff.mean():.3f}")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Chart
        st.markdown('<div class="section-header">Price Comparison</div>', unsafe_allow_html=True)
        
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # Market price line
        fig.add_trace(go.Scatter(
            x=near_money['strike'],
            y=near_money['lastPrice'],
            mode='lines+markers',
            name='Market',
            line=dict(color='#fafafa', width=2),
            marker=dict(size=6)
        ))
        
        # Monte Carlo
        fig.add_trace(go.Scatter(
            x=near_money['strike'],
            y=near_money['MC_Price'],
            mode='lines+markers',
            name='Monte Carlo',
            line=dict(color='#3b82f6', width=2, dash='dot'),
            marker=dict(size=5)
        ))
        
        # Black-Scholes
        fig.add_trace(go.Scatter(
            x=near_money['strike'],
            y=near_money['BS_Price'],
            mode='lines+markers',
            name='Black-Scholes',
            line=dict(color='#f59e0b', width=2, dash='dash'),
            marker=dict(size=5)
        ))
        
        # Spot price marker
        fig.add_vline(
            x=S,
            line_dash="dash",
            line_color="#525252",
            annotation_text=f"Spot ${S:.0f}",
            annotation_position="top",
            annotation_font_color="#a1a1a1"
        )
        
        fig.update_layout(
            plot_bgcolor='#0a0a0a',
            paper_bgcolor='#0a0a0a',
            font=dict(family='Inter', color='#a1a1a1'),
            xaxis=dict(
                title="Strike Price",
                gridcolor='#1c1c1c',
                linecolor='#262626',
                tickformat='$,.0f'
            ),
            yaxis=dict(
                title="Option Price",
                gridcolor='#1c1c1c',
                linecolor='#262626',
                tickformat='$,.2f'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                bgcolor='rgba(0,0,0,0)'
            ),
            height=450,
            margin=dict(l=0, r=0, t=40, b=0),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif not st.session_state.data_loaded:
        # Welcome state
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-title">Get Started</div>
            <div class="welcome-text">
                Enter a stock ticker in the sidebar and click <strong>Fetch Market Data</strong> to load option chain data.
            </div>
            <div class="ticker-grid">
                <div class="ticker-chip">
                    <div class="ticker-symbol">AAPL</div>
                    <div class="ticker-name">Apple</div>
                </div>
                <div class="ticker-chip">
                    <div class="ticker-symbol">MSFT</div>
                    <div class="ticker-name">Microsoft</div>
                </div>
                <div class="ticker-chip">
                    <div class="ticker-symbol">NVDA</div>
                    <div class="ticker-name">NVIDIA</div>
                </div>
                <div class="ticker-chip">
                    <div class="ticker-symbol">TSLA</div>
                    <div class="ticker-name">Tesla</div>
                </div>
                <div class="ticker-chip">
                    <div class="ticker-symbol">GOOGL</div>
                    <div class="ticker-name">Alphabet</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p class="footer-text">Option Pricing Dashboard · Data via Yahoo Finance · For educational purposes only</p>
</div>
""", unsafe_allow_html=True)
