"""
Option Pricing Dashboard - Streamlit App
Compares Monte Carlo and Black-Scholes option pricing methods
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os

# Required imports for the notebook functions
import yfinance as yf
from datetime import datetime
import pytz
from scipy.stats import norm

# Configure Streamlit page
st.set_page_config(
    page_title="Option Pricing Dashboard",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'market_data' not in st.session_state:
    st.session_state.market_data = None
if 'ticker_loaded' not in st.session_state:
    st.session_state.ticker_loaded = None

# ============================================================================
# Extract and execute functions from Jupyter notebooks
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
    """Removes 'if __name__ == "__main__":' blocks from code."""
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
    
    # Monte Carlo namespace
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
    
    # Black-Scholes namespace
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
# Custom CSS
# ============================================================================

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 16px; margin-bottom: 2rem;
        text-align: center; box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 { color: white; font-size: 2.5rem; margin-bottom: 0.5rem; font-weight: 700; }
    .main-header p { color: rgba(255, 255, 255, 0.85); font-size: 1.1rem; }
    
    .method-badge-mc {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 0.5rem 1rem; border-radius: 20px; color: white;
        font-weight: 600; display: inline-block; margin-bottom: 1rem;
    }
    .method-badge-bs {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 0.5rem 1rem; border-radius: 20px; color: white;
        font-weight: 600; display: inline-block; margin-bottom: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; font-weight: 600; padding: 0.75rem 2rem;
        border-radius: 8px; border: none; width: 100%;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4); }
    
    .info-section {
        background: rgba(102, 126, 234, 0.1); border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px; padding: 1.5rem; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📈 Option Pricing Dashboard</h1>
    <p>Compare Monte Carlo Simulation vs Black-Scholes Model</p>
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
    st.markdown("### 🔧 Configuration")
    st.markdown("---")
    
    ticker = st.text_input(
        "Enter Stock Ticker", value="AAPL",
        placeholder="e.g., AAPL, MSFT, NVDA"
    ).upper()
    
    st.markdown("---")
    
    mc_simulations = st.slider(
        "Monte Carlo Simulations",
        min_value=1000, max_value=50000, value=10000, step=1000,
        help="More simulations = higher accuracy but slower"
    )
    
    st.markdown("---")
    
    run_analysis = st.button("🚀 Fetch Data", use_container_width=True)
    
    # Clear data if ticker changed
    if st.session_state.ticker_loaded and st.session_state.ticker_loaded != ticker:
        st.session_state.data_loaded = False
        st.session_state.market_data = None
    
    st.markdown("---")
    st.markdown("### 📊 Methods Explained")
    
    with st.expander("Monte Carlo Simulation"):
        st.markdown("""
        Uses random sampling to simulate thousands of possible 
        stock price paths. Option value = average discounted payoff.
        
        $$S_T = S_0 e^{(r - \\frac{\\sigma^2}{2})T + \\sigma\\sqrt{T}Z}$$
        """)
    
    with st.expander("Black-Scholes Model"):
        st.markdown("""
        Analytical closed-form solution for European options.
        
        $$C = S \\cdot N(d_1) - Ke^{-rT} \\cdot N(d_2)$$
        """)

# ============================================================================
# Main Content
# ============================================================================

if notebooks_loaded:
    # Fetch data when button is clicked
    if run_analysis:
        with st.spinner(f"Fetching data for {ticker}..."):
            try:
                get_market_data = monte_carlo_ns['get_market_data']
                calculate_historical_volatility = monte_carlo_ns['calculate_historical_volatility']
                get_risk_free_rate = monte_carlo_ns['get_risk_free_rate']
                calculate_time_to_expiry = monte_carlo_ns['calculate_time_to_expiry']
                
                data = get_market_data(ticker)
                
                if data['calls_df'].empty:
                    st.error(f"No option data found for {ticker}")
                else:
                    sigma = calculate_historical_volatility(data['historical_df'].copy())
                    r = get_risk_free_rate()
                    
                    # Calculate T for all options
                    data['calls_df']['T'] = data['calls_df']['expirationDate'].apply(calculate_time_to_expiry)
                    
                    # Store in session state
                    st.session_state.market_data = {
                        'S': data['spot_price'],
                        'calls_df': data['calls_df'],
                        'sigma': sigma,
                        'r': r
                    }
                    st.session_state.data_loaded = True
                    st.session_state.ticker_loaded = ticker
                    st.success(f"✅ Data loaded for {ticker}!")
                    
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                st.exception(e)
    
    # Display results if data is loaded
    if st.session_state.data_loaded and st.session_state.market_data:
        md = st.session_state.market_data
        S, calls_df, sigma, r = md['S'], md['calls_df'], md['sigma'], md['r']
        
        # Get pricing functions
        monte_carlo_pricer = monte_carlo_ns['monte_carlo_pricer']
        black_scholes_calculator = black_scholes_ns['black_scholes_calculator']
        
        # Display metrics
        st.markdown("### 📊 Market Data")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Spot Price", f"${S:.2f}")
        with col2:
            st.metric("Historical Volatility", f"{sigma:.2%}")
        with col3:
            st.metric("Risk-Free Rate", f"{r:.2%}")
        with col4:
            st.metric("MC Simulations", f"{mc_simulations:,}")
        
        st.markdown("---")
        
        # Date selector
        expiry_dates = sorted(calls_df['expirationDate'].unique())
        selected_expiry = st.selectbox("Select Expiration Date", expiry_dates, index=0)
        
        # Filter and calculate prices
        filtered_df = calls_df[calls_df['expirationDate'] == selected_expiry].copy()
        filtered_df['dist'] = abs(filtered_df['strike'] - S)
        near_money = filtered_df.nsmallest(15, 'dist').sort_values('strike').copy()
        
        # Calculate prices for selected expiry
        with st.spinner(f"Calculating prices for {selected_expiry}..."):
            mc_prices = []
            bs_prices = []
            
            for idx, row in near_money.iterrows():
                K, T = row['strike'], row['T']
                
                # Monte Carlo - uses the slider value
                mc_price = monte_carlo_pricer(S, K, T, r, sigma, mc_simulations)
                mc_prices.append(mc_price)
                
                # Black-Scholes
                bs_price = black_scholes_calculator(S, K, T, r, sigma)
                bs_prices.append(bs_price)
            
            near_money['MC_Price'] = mc_prices
            near_money['BS_Price'] = bs_prices
        
        st.markdown("### 💰 Option Pricing Results")
        st.markdown(f"*Expiry: {selected_expiry} | Spot: ${S:.2f}*")
        
        # Display comparison tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<span class="method-badge-mc">🎲 Monte Carlo Method</span>', unsafe_allow_html=True)
            mc_df = near_money[['strike', 'lastPrice', 'MC_Price']].copy()
            mc_df['MC vs Market'] = mc_df['MC_Price'] - mc_df['lastPrice']
            mc_df.columns = ['Strike', 'Market', 'Monte Carlo', 'Diff']
            for col in mc_df.columns:
                mc_df[col] = mc_df[col].apply(lambda x: f"${x:.2f}")
            st.dataframe(mc_df.reset_index(drop=True), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown('<span class="method-badge-bs">📐 Black-Scholes Method</span>', unsafe_allow_html=True)
            bs_df = near_money[['strike', 'lastPrice', 'BS_Price']].copy()
            bs_df['BS vs Market'] = bs_df['BS_Price'] - bs_df['lastPrice']
            bs_df.columns = ['Strike', 'Market', 'Black-Scholes', 'Diff']
            for col in bs_df.columns:
                bs_df[col] = bs_df[col].apply(lambda x: f"${x:.2f}")
            st.dataframe(bs_df.reset_index(drop=True), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Model comparison stats
        st.markdown("### 📈 Model Comparison")
        
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
            st.metric("Avg MC-BS Diff", f"${mc_bs_diff.mean():.3f}")
        
        # Chart
        st.markdown("### 📊 Price Visualization")
        
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=near_money['strike'], y=near_money['lastPrice'],
            mode='lines+markers', name='Market Price', line=dict(color='#ffffff', width=3)))
        fig.add_trace(go.Scatter(x=near_money['strike'], y=near_money['MC_Price'],
            mode='lines+markers', name='Monte Carlo', line=dict(color='#f5576c', width=2, dash='dash')))
        fig.add_trace(go.Scatter(x=near_money['strike'], y=near_money['BS_Price'],
            mode='lines+markers', name='Black-Scholes', line=dict(color='#00f2fe', width=2, dash='dot')))
        fig.add_vline(x=S, line_dash="dash", line_color="yellow", annotation_text=f"Spot: ${S:.2f}")
        
        fig.update_layout(
            title=f"Option Prices - {st.session_state.ticker_loaded} (Expiry: {selected_expiry})",
            xaxis_title="Strike Price ($)", yaxis_title="Option Price ($)",
            template="plotly_dark", height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif not st.session_state.data_loaded:
        st.markdown("""
        <div class="info-section">
            <h3>👋 Welcome!</h3>
            <p>Enter a stock ticker and click <strong>Fetch Data</strong> to get started.</p>
            <p>Then select different expiration dates and adjust simulations to compare pricing methods.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🏷️ Popular Tickers")
        cols = st.columns(5)
        for i, (t, name) in enumerate([("AAPL", "Apple"), ("MSFT", "Microsoft"), 
                                        ("NVDA", "NVIDIA"), ("TSLA", "Tesla"), ("GOOGL", "Alphabet")]):
            with cols[i]:
                st.info(f"**{t}**\n{name}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.85rem;">
    <p>📊 Option Pricing Dashboard | Monte Carlo & Black-Scholes</p>
</div>
""", unsafe_allow_html=True)
