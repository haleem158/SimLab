import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
import json

st.set_page_config(page_title="SimLab - Token Price Impact Model", layout="wide")

st.title("🧮 SimLab — Token Price Impact Model")

st.markdown("""
This app computes an **indicative token price proxy** by constructing a demand *level* from baseline volume and growth,
and comparing it to an **effective circulating supply**. Outputs are tuned to a market anchor or controlled
by a sensitivity parameter so results are interpretable.
""")

# --------------------------
# Upload or sample CSV
# --------------------------
st.subheader("Input data")
st.markdown("Upload a CSV with these columns (monthly rows): `Month`, `Unlocked Tokens`, `Emission`, `Transaction Volume Growth (%)`, `Staking Participation Growth (%)`. Optional columns: `Locked`, `Staked`.")
sample_btn = st.button("Load sample CSV")
if sample_btn:
    sample_csv = """Month,Unlocked Tokens,Emission,Transaction Volume Growth (%),Staking Participation Growth (%),Locked,Staked
2025-01-01,100000,50000,10,5,20000,30000
2025-02-01,120000,60000,12,6,21000,35000
2025-03-01,110000,55000,8,4,22000,31000
"""
    uploaded_file = StringIO(sample_csv)
else:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], help="Upload your monthly token emission and growth data in CSV format.")

if uploaded_file is None:
    st.info("Upload a CSV or click 'Load sample CSV' to proceed.")
    st.stop()

# --------------------------
# Read CSV & validation
# --------------------------
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

required = ["Month", "Unlocked Tokens", "Emission", "Transaction Volume Growth (%)", "Staking Participation Growth (%)"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"CSV missing required columns: {missing}")
    st.stop()

df['Month_dt'] = pd.to_datetime(df['Month'], errors='coerce')
if df['Month_dt'].isnull().any():
    st.error("Some 'Month' values could not be parsed into dates. Use ISO dates (YYYY-MM-DD).")
    st.stop()
df = df.sort_values('Month_dt').reset_index(drop=True)
n_months = len(df)

numeric_cols = ["Unlocked Tokens", "Emission", "Transaction Volume Growth (%)", "Staking Participation Growth (%)"]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
if df[numeric_cols].isnull().any().any():
    st.error("Numeric columns contain non-numeric or missing values. Clean the CSV and retry.")
    st.stop()

df['Locked'] = pd.to_numeric(df.get('Locked', 0), errors='coerce').fillna(0)
df['Staked'] = pd.to_numeric(df.get('Staked', 0), errors='coerce').fillna(0)

st.success("✅ CSV loaded and validated.")

# --------------------------
# Data semantics
# --------------------------
st.subheader("Data semantics & baseline assumptions")
flow_choice = st.radio(
    "Are `Unlocked Tokens` and `Emission` provided as monthly flows or cumulative totals?",
    ("monthly_flows", "cumulative_totals"),
    help="Choose whether the provided data are monthly new unlocks/emissions or total cumulative figures."
)

if flow_choice == "monthly_flows":
    monthly_unlocked = df['Unlocked Tokens'].values.astype(float)
    monthly_emission = df['Emission'].values.astype(float)
    cumulative_unlocked = np.cumsum(monthly_unlocked)
    cumulative_emission = np.cumsum(monthly_emission)
else:
    cumulative_unlocked = df['Unlocked Tokens'].values.astype(float)
    cumulative_emission = df['Emission'].values.astype(float)
    monthly_unlocked = np.concatenate([[cumulative_unlocked[0]], np.diff(cumulative_unlocked)])
    monthly_emission = np.concatenate([[cumulative_emission[0]], np.diff(cumulative_emission)])

initial_circ_supply = st.number_input(
    "Optional: known initial circulating supply (tokens)",
    value=0.0, min_value=0.0, step=1.0,
    help="If known, enter the circulating supply before the first data month. Otherwise leave at 0."
)
total_supply_estimate = st.number_input(
    "Optional: total token supply (for sanity checks)",
    value=0.0, min_value=0.0, step=1.0,
    help="Used only to cross-check the realism of effective supply results."
)

raw_circulating = (cumulative_unlocked + cumulative_emission) + (initial_circ_supply if initial_circ_supply>0 else 0.0)

# --------------------------
# Effective supply
# --------------------------
st.markdown("**Effective supply adjustments**")
activation_mode = st.radio(
    "Activation (fraction of unlocked+emission that becomes liquid):",
    ("global_ratio", "column_based"),
    help="Select whether to use a global activation ratio or column-based values if provided."
)
if activation_mode == "global_ratio":
    global_activation = st.slider("Global activation ratio (0-1)", 0.0, 1.0, 0.85, step=0.01,
                                  help="Fraction of total unlocked+emitted tokens that becomes tradable.")
    activation_array = np.full(n_months, global_activation)
else:
    if 'Activation' in df.columns:
        df['Activation'] = pd.to_numeric(df['Activation'], errors='coerce').fillna(1.0)
        activation_array = df['Activation'].values.astype(float)
    else:
        st.info("No 'Activation' column found; defaulting to 0.85 activation.")
        activation_array = np.full(n_months, 0.85)

locked = df['Locked'].values.astype(float)
staked = df['Staked'].values.astype(float)
effective_supply = (raw_circulating - locked - staked) * activation_array
eps = 1e-9
effective_supply = np.maximum(effective_supply, eps)

# --------------------------
# Demand index with tooltips
# --------------------------
st.subheader("Demand index construction")
st.markdown("We convert the two growth rates into demand *levels* using baseline volumes and cumulative growth.")

V0_tx = st.number_input("Baseline transaction volume index (V0_tx, arbitrary units)",
                        value=1.0, min_value=0.0, step=0.1,
                        help="Starting level for transaction activity — scales total demand index.")
V0_stake = st.number_input("Baseline staking participation index (V0_stake, arbitrary units)",
                           value=1.0, min_value=0.0, step=0.1,
                           help="Starting level for staking participation — higher value raises base demand.")

tx_growth_dec = df['Transaction Volume Growth (%)'].values.astype(float) / 100.0
stake_growth_dec = df['Staking Participation Growth (%)'].values.astype(float) / 100.0
V_tx = V0_tx * np.cumprod(1 + tx_growth_dec)
V_stake = V0_stake * np.cumprod(1 + stake_growth_dec)

alpha = st.slider("Weight for transaction volume (α)", 0.0, 1.0, 0.6, step=0.05,
                  help="Determines how much transaction growth vs staking growth drives demand.")
beta = 1.0 - alpha
demand_index = alpha * V_tx + beta * V_stake

# --------------------------
# Price tuning & formula
# --------------------------
st.subheader("Price tuning & formula")
st.markdown("Price formula: `Price_t = k * DemandIndex_t / (EffectiveSupply_t ** ε)`.")

calibrate = st.checkbox("Tune sensitivity (k) to an anchor price or market cap", value=True,
                        help="Tick this to automatically tune k using a known price or market cap.")
k = None
if calibrate:
    anchor_mode = st.radio("Anchor by:", ("current_price", "current_market_cap", "none"),
                           help="Choose the reference point used to tune k.")
    if anchor_mode == "current_price":
        anchor_price = st.number_input("Anchor current price (token units)", min_value=0.0, value=0.0, step=0.0001,
                                       help="Enter a known token price to tune the sensitivity constant.")
        anchor_supply_idx = st.number_input("Month index for anchor (0 = first row)?", min_value=0, max_value=n_months-1, value=0,
                                            help="Index corresponding to the calibration month.")
        if anchor_price > 0:
            k = anchor_price * (effective_supply[anchor_supply_idx] ** st.number_input("Enter epsilon for tuning (ε)", value=1.0)) / demand_index[anchor_supply_idx]
            st.write(f"Tuned k = {k:.6e}")
    elif anchor_mode == "current_market_cap":
        anchor_market_cap = st.number_input("Anchor market cap (token units * price)", min_value=0.0, value=0.0, step=1.0,
                                            help="Enter total market cap to tune k.")
        anchor_supply_idx = st.number_input("Month index for anchor (0 = first row)?", min_value=0, max_value=n_months-1, value=0)
        if anchor_market_cap > 0:
            implied_price = anchor_market_cap / effective_supply[anchor_supply_idx]
            k = implied_price * (effective_supply[anchor_supply_idx] ** st.number_input("Enter epsilon for tuning (ε)", value=1.0)) / demand_index[anchor_supply_idx]
            st.write(f"Tuned k = {k:.6e}")
    else:
        st.info("No anchor selected — you will set k manually below.")

if k is None:
    k = st.number_input("Sensitivity parameter k (positive scalar)", value=1e-6, format="%.8e",
                        help="Manual tuning constant linking demand and supply to price.")

epsilon = st.slider("Supply elasticity (ε)", 0.1, 2.0, 1.0, step=0.05,
                    help="Controls how strongly price responds to changes in supply.")

indicative_price = k * demand_index / (effective_supply ** epsilon + eps)

# --------------------------
# Smoothing & uncertainty
# --------------------------
st.subheader("Smoothing & uncertainty")
smoothing_window = st.slider("Smoothing window (months)", min_value=1, max_value=12, value=1,
                             help="Applies a moving average to smooth price fluctuations.")
do_mc = st.checkbox("Run Monte Carlo sensitivity", value=False,
                    help="Simulate random demand shocks to test price stability.")

mc_n = 200
mc_p25 = mc_p50 = mc_p75 = None
if do_mc:
    st.info("Running Monte Carlo — simulating small random shocks to demand.")
    np.random.seed(123)
    sims = np.zeros((mc_n, n_months))
    jitter_std = st.number_input("MC jitter stdev (fraction of demand, e.g., 0.05)",
                                 min_value=0.0, max_value=1.0, value=0.05, step=0.01,
                                 help="Controls randomness magnitude for each simulation run.")
    for i in range(mc_n):
        noise = np.random.normal(1.0, jitter_std, size=n_months)
        demand_j = demand_index * noise
        price_j = k * demand_j / (effective_supply ** epsilon + eps)
        sims[i, :] = price_j
    mc_p25 = np.percentile(sims, 25, axis=0)
    mc_p50 = np.percentile(sims, 50, axis=0)
    mc_p75 = np.percentile(sims, 75, axis=0)

# --------------------------
# Results
# --------------------------
st.subheader("Results")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Final Effective Supply ℹ️", f"{int(effective_supply[-1]):,}",
               help="Total circulating tokens after removing locked and staked balances.")
with col2:
    st.metric("Final Demand Index ℹ️", f"{demand_index[-1]:,.2f}",
               help="Combined transaction and staking index capturing demand growth.")
with col3:
    st.metric("Final Indicative Price (smoothed) ℹ️", f"{indicative_price[-1]:.6e}",
               help="Model-estimated token price after smoothing — not a market price.")

# --------------------------
# Plot & export
# --------------------------
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(df['Month_dt'].dt.strftime("%Y-%m"), effective_supply, label='Effective Supply', alpha=0.8)
ax1.set_ylabel('Effective Supply (tokens)', color='tab:blue')
ax2 = ax1.twinx()
ax2.plot(df['Month_dt'].dt.strftime("%Y-%m"), demand_index, label='Demand Index', linestyle='--', color='tab:orange')
ax2.plot(df['Month_dt'].dt.strftime("%Y-%m"), indicative_price, label='Indicative Price', color='tab:red')
ax2.set_ylabel('Demand Index / Price', color='tab:red')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')
st.pyplot(fig)

if do_mc:
    fig2, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df['Month_dt'].dt.strftime("%Y-%m"), mc_p50, label='Median price', color='tab:red')
    ax.fill_between(df['Month_dt'].dt.strftime("%Y-%m"), mc_p25, mc_p75, color='tab:red', alpha=0.2, label='25–75th pct')
    ax.legend()
    ax.set_title("Monte Carlo Price Sensitivity Band")
    st.pyplot(fig2)

# --------------------------
# Explanation footer
# --------------------------
st.subheader("Assumptions & notes (read before acting on results)")
st.write("""
1. **Demand Index** is built as a level from baseline values and growth rates. Growth rates alone are not sufficient — levels matter.  
2. **Effective supply** removes locked and staked tokens and scales by an activation ratio.  
3. **Price formula** uses two hyperparameters: sensitivity `k` (tuning constant) and elasticity `ε` (price vs supply relationship).  
4. **Tuning** to an anchor (current price or market cap) is recommended to give realistic price scale.  
5. **Monte Carlo** runs are for exploring uncertainty — not market risk modeling.  
6. This model is a **simple indicative tool**, not a trading or risk engine.  
""")
