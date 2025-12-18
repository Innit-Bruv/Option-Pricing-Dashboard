# 📈 Option Pricing Dashboard

A Streamlit web application that compares two popular option pricing methods: **Monte Carlo Simulation** and **Black-Scholes Model**.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Features

- **Real-time market data** fetched via Yahoo Finance
- **Monte Carlo Simulation** with adjustable number of simulations (1,000 - 50,000)
- **Black-Scholes Model** analytical pricing
- **Interactive visualizations** with Plotly charts
- Compare theoretical prices against actual market prices
- Support for multiple expiration dates

## 📊 Pricing Methods

### Monte Carlo Simulation
Uses random sampling to simulate stock price paths under Geometric Brownian Motion:

$$S_T = S_0 \exp\left[\left(r - \frac{\sigma^2}{2}\right)T + \sigma\sqrt{T}Z\right]$$

### Black-Scholes Model
Analytical closed-form solution for European call options:

$$C = S \cdot N(d_1) - Ke^{-rT} \cdot N(d_2)$$

## 🚀 Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/Innit-Bruv/Option-Pricing-Dashboard.git
cd Option-Pricing-Dashboard

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Usage

1. Enter a stock ticker (e.g., AAPL, MSFT, NVDA)
2. Click **Fetch Data** to load option chain data
3. Select an expiration date from the dropdown
4. Adjust Monte Carlo simulations using the slider
5. Compare pricing results between methods

## 📁 Project Structure

```
Option-Pricing-Dashboard/
├── app.py                          # Streamlit dashboard
├── Option_Pricing_MonteCarlo.ipynb # Monte Carlo implementation
├── Option_Pricing_Scholes.ipynb    # Black-Scholes implementation
├── requirements.txt                # Python dependencies
└── README.md
```

## 📦 Dependencies

- `streamlit` - Web application framework
- `pandas` & `numpy` - Data manipulation
- `yfinance` - Yahoo Finance API
- `scipy` - Scientific computing (normal distribution)
- `plotly` - Interactive charts
- `pytz` - Timezone handling

## 📈 Sample Output

| Strike | Market Price | Monte Carlo | Black-Scholes |
|--------|--------------|-------------|---------------|
| $250   | $20.50       | $19.85      | $19.82        |
| $255   | $17.00       | $16.45      | $16.42        |
| $260   | $14.25       | $13.62      | $13.58        |

## 🔬 Technical Details

- **Historical Volatility**: Calculated from 1-year daily log returns, annualized
- **Risk-Free Rate**: Fetched from 13-week Treasury Bill (^IRX)
- **Time to Expiry**: Calculated in trading days, converted to years

## 📝 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Pull requests welcome! For major changes, please open an issue first.

---

*Data sourced from Yahoo Finance. For educational purposes only - not financial advice.*
