# simlab_price_impact.py
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
and comparing it to an **effective circulating supply**. Outputs are calibrated to a market anchor or controlled
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
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

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

# parse Month into datetime and sort
df['Month_dt'] = pd.to_datetime(df['Month'], errors='coerce')
if df['Month_dt'].isnull().any():
    st.error("Some 'Month' values could not be parsed into dates. Use ISO dates (YYYY-MM-DD) or common month formats.")
    st.stop()
df = df.sort_values('Month_dt').reset_index(drop=True)
n_months = len(df)

# coerce numeric columns
numeric_cols = ["Unlocked Tokens", "Emission", "Transaction Volume Growth (%)", "Staking Participation Growth (%)"]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
if df[numeric_cols].isnull().any().any():
    st.error("Numeric columns contain non-numeric or missing values. Clean the CSV and retry.")
    st.stop()

# optional columns
df['Locked'] = pd.to_numeric(df.get('Locked', 0), errors='coerce').fillna(0)
df['Staked'] = pd.to_numeric(df.get('Staked', 0), errors='coerce').fillna(0)

st.success("CSV loaded and validated.")

# --------------------------
# User choices: flows vs cumulative
# --------------------------
st.subheader("Data semantics & baseline assumptions")
flow_choice = st.radio("Are `Unlocked Tokens` and `Emission` provided as monthly flows or cumulative totals?",
                       ("monthly_flows", "cumulative_totals"))

# Interpret columns accordingly
if flow_choice == "monthly_flows":
    monthly_unlocked = df['Unlocked Tokens'].values.astype(float)
    monthly_emission = df['Emission'].values.astype(float)
    cumulative_unlocked = np.cumsum(monthly_unlocked)
    cumulative_emission = np.cumsum(monthly_emission)
else:
    cumulative_unlocked = df['Unlocked Tokens'].values.astype(float)
    cumulative_emission = df['Emission'].values.astype(float)
    # compute monthly flows from cumulative if needed (first difference)
    monthly_unlocked = np.concatenate([[cumulative_unlocked[0]], np.diff(cumulative_unlocked)])
    monthly_emission = np.concatenate([[cumulative_emission[0]], np.diff(cumulative_emission)])

# baseline and effective supply handling
initial_circ_supply = st.number_input("Optional: known initial circulating supply (tokens). Enter 0 if unknown.", value=0.0, min_value=0.0, step=1.0)
total_supply_estimate = st.number_input("Optional: total token supply (for sanity checks). Enter 0 if unknown.", value=0.0, min_value=0.0, step=1.0)

# Compute raw circulating supply (stock)
raw_circulating = (cumulative_unlocked + cumulative_emission) + (initial_circ_supply if initial_circ_supply>0 else 0.0)

# Allow user to provide activation ratio (fraction of unlocked that becomes liquid) globally or per-row
st.markdown("**Effective supply adjustments**")
activation_mode = st.radio("Activation (which fraction of unlocked+emission enters liquid supply):", ("global_ratio", "column_based"))
if activation_mode == "global_ratio":
    global_activation = st.slider("Global activation ratio (0-1)", 0.0, 1.0, 0.85, step=0.01)
    activation_array = np.full(n_months, global_activation)
else:
    # let user provide a column or a constant
    if 'Activation' in df.columns:
        df['Activation'] = pd.to_numeric(df['Activation'], errors='coerce').fillna(1.0)
        activation_array = df['Activation'].values.astype(float)
    else:
        st.info("No 'Activation' column found; defaulting to 0.85 activation.")
        activation_array = np.full(n_months, 0.85)

# locked & staked reduce effective liquid supply
locked = df['Locked'].values.astype(float)
staked = df['Staked'].values.astype(float)
effective_supply = (raw_circulating - locked - staked) * activation_array
# clamp to minimum small positive value to avoid zeros
eps = 1e-9
effective_supply = np.maximum(effective_supply, eps)

# --------------------------
# Demand index construction (levels, not %)
# --------------------------
st.subheader("Demand index construction")
st.markdown("We convert the two growth rates into demand *levels* using baseline volumes and cumulative growth.")
V0_tx = st.number_input("Baseline transaction volume index (V0_tx, arbitrary units)", value=1.0, min_value=0.0, step=0.1)
V0_stake = st.number_input("Baseline staking participation index (V0_stake, arbitrary units)", value=1.0, min_value=0.0, step=0.1)

tx_growth_dec = df['Transaction Volume Growth (%)'].values.astype(float) / 100.0
stake_growth_dec = df['Staking Participation Growth (%)'].values.astype(float) / 100.0

V_tx = V0_tx * np.cumprod(1 + tx_growth_dec)
V_stake = V0_stake * np.cumprod(1 + stake_growth_dec)

alpha = st.slider("Weight for transaction volume (α)", 0.0, 1.0, 0.6, step=0.05)
beta = 1.0 - alpha
demand_index = alpha * V_tx + beta * V_stake

# --------------------------
# Calibration & price formula
# --------------------------
st.subheader("Price calibration & formula")
st.markdown("Price formula: `Price_t = k * DemandIndex_t / (EffectiveSupply_t ** ε)`. Calibrate `k` with an anchor price/market cap or set `k` manually.")

calibrate = st.checkbox("Calibrate sensitivity (k) to an anchor price or market cap", value=True)
k = None
if calibrate:
    anchor_mode = st.radio("Anchor by:", ("current_price", "current_market_cap", "none"))
    if anchor_mode == "current_price":
        anchor_price = st.number_input("Anchor current price (token units)", min_value=0.0, value=0.0, step=0.0001)
        anchor_supply_idx = st.number_input("Which month index corresponds to anchor (0 = first row)?", min_value=0, max_value=n_months-1, value=0)
        if anchor_price > 0:
            # compute k so Price_anchor = anchor_price
            k = anchor_price * (effective_supply[anchor_supply_idx] ** st.number_input("Enter epsilon for calibration (ε)", value=1.0)) / demand_index[anchor_supply_idx]
            st.write(f"Calibrated k = {k:.6e}")
    elif anchor_mode == "current_market_cap":
        anchor_market_cap = st.number_input("Anchor market cap (token units * price)", min_value=0.0, value=0.0, step=1.0)
        anchor_supply_idx = st.number_input("Which month index corresponds to anchor (0 = first row)?", min_value=0, max_value=n_months-1, value=0)
        if anchor_market_cap > 0:
            # price = market_cap / supply => compute k
            implied_price = anchor_market_cap / effective_supply[anchor_supply_idx]
            k = implied_price * (effective_supply[anchor_supply_idx] ** st.number_input("Enter epsilon for calibration (ε)", value=1.0)) / demand_index[anchor_supply_idx]
            st.write(f"Calibrated k = {k:.6e}")
    else:
        st.info("No anchor selected — you will set k manually below.")

if k is None:
    k = st.number_input("Sensitivity parameter k (positive scalar). If you calibrated above this field is ignored.", value=1e-6, format="%.8e")

epsilon = st.slider("Supply elasticity (ε). Higher ε => stronger price sensitivity to supply", 0.1, 2.0, 1.0, step=0.05)

# compute indicative price
indicative_price = k * demand_index / (effective_supply ** epsilon + eps)

# --------------------------
# Smoothing & Monte Carlo sensitivity
# --------------------------
st.subheader("Smoothing & uncertainty")
smoothing_window = st.slider("Smoothing window (months) for moving average, 1 = no smoothing", min_value=1, max_value=12, value=1)
if smoothing_window > 1:
    import pandas as _pd
    indicative_price_smoothed = _pd.Series(indicative_price).rolling(window=smoothing_window, min_periods=1).mean().values
else:
    indicative_price_smoothed = indicative_price.copy()

do_mc = st.checkbox("Run Monte Carlo sensitivity (small simulations)", value=False)
mc_n = 200
mc_p25 = mc_p50 = mc_p75 = None
if do_mc:
    st.info("Running Monte Carlo. This will jitter Demand and compute price percentiles.")
    np.random.seed(123)
    sims = np.zeros((mc_n, n_months))
    # jitter demand_index multiplicatively with 5% noise by default (user could parametrize)
    jitter_std = st.number_input("MC jitter stdev (fraction of demand, e.g., 0.05)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    for i in range(mc_n):
        noise = np.random.normal(1.0, jitter_std, size=n_months)
        demand_j = demand_index * noise
        price_j = k * demand_j / (effective_supply ** epsilon + eps)
        sims[i, :] = price_j
    mc_p25 = np.percentile(sims, 25, axis=0)
    mc_p50 = np.percentile(sims, 50, axis=0)
    mc_p75 = np.percentile(sims, 75, axis=0)

# --------------------------
# Display results & plots
# --------------------------
st.subheader("Results")

# assemble results dataframe
out = pd.DataFrame({
    "Month": df['Month_dt'],
    "Monthly Unlocked": monthly_unlocked,
    "Monthly Emission": monthly_emission,
    "Cumulative Unlocked": cumulative_unlocked,
    "Cumulative Emission": cumulative_emission,
    "Raw Circulating": raw_circulating,
    "Locked": locked,
    "Staked": staked,
    "Effective Supply": effective_supply,
    "Demand Index": demand_index,
    "Indicative Price": indicative_price,
    "Indicative Price (smoothed)": indicative_price_smoothed
})
out['Month_str'] = out['Month'].dt.strftime("%Y-%m")

# Top-line metrics
col1, col2, col3 = st.columns(3)
col1.metric("Final Effective Supply", f"{int(effective_supply[-1]):,}")
col2.metric("Final Demand Index", f"{demand_index[-1]:,.2f}")
col3.metric("Final Indicative Price (smoothed)", f"{indicative_price_smoothed[-1]:.6e}")

# Plot time series: effective supply, demand index, price
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(out['Month_str'], out['Effective Supply'], label='Effective Supply', alpha=0.8)
ax1.set_ylabel('Effective Supply (tokens)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xticks(out['Month_str'][::max(1, n_months // 12)])
for label in ax1.get_xticklabels():
    label.set_rotation(45)
ax2 = ax1.twinx()
ax2.plot(out['Month_str'], out['Demand Index'], label='Demand Index', linestyle='--', alpha=0.8, color='tab:orange')
ax2.plot(out['Month_str'], out['Indicative Price (smoothed)'], label='Price (smoothed)', linestyle='-', alpha=0.9, color='tab:red')
ax2.set_ylabel('Demand Index / Indicative Price', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

st.pyplot(fig)

# If MC computed, show percentile band
if do_mc:
    fig2, ax = plt.subplots(figsize=(12, 3))
    ax.plot(out['Month_str'], mc_p50, label='Median price', color='tab:red')
    ax.fill_between(out['Month_str'], mc_p25, mc_p75, color='tab:red', alpha=0.2, label='25-75th pct')
    ax.set_xticks(out['Month_str'][::max(1, n_months // 12)])
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    ax.set_title("Monte Carlo price uncertainty (25-75 percentile band)")
    ax.legend()
    st.pyplot(fig2)

# Show processed data
st.subheader("Processed table")
st.dataframe(out.drop(columns=['Month']))

# Download processed CSV
csv_bytes = out.to_csv(index=False).encode('utf-8')
st.download_button("Download processed CSV", data=csv_bytes, file_name="simlab_price_processed.csv", mime="text/csv")

# Export full scenario JSON (parameters + arrays)
export = {
    "params": {
        "flow_choice": flow_choice,
        "initial_circ_supply": float(initial_circ_supply),
        "total_supply_estimate": float(total_supply_estimate),
        "activation_mode": activation_mode,
        "global_activation": float(global_activation) if activation_mode == "global_ratio" else None,
        "V0_tx": float(V0_tx),
        "V0_stake": float(V0_stake),
        "alpha": float(alpha),
        "k": float(k),
        "epsilon": float(epsilon),
        "smoothing_window": int(smoothing_window),
        "do_mc": bool(do_mc)
    },
    "results": {
        "months": out['Month_str'].tolist(),
        "effective_supply": out['Effective Supply'].tolist(),
        "demand_index": out['Demand Index'].tolist(),
        "indicative_price": out['Indicative Price (smoothed)'].tolist()
    }
}
st.download_button("Download scenario JSON", data=json.dumps(export, indent=2), file_name="simlab_price_export.json", mime="application/json")

# --------------------------
# Assumptions & explanation
# --------------------------
st.subheader("Assumptions & notes (read before acting on results)")
st.write("""
1. **Demand Index** is built as a level from baseline values and growth rates. Growth rates alone are not sufficient — levels matter.
2. **Effective supply** removes locked and staked tokens and scales by an activation ratio (fraction that actually becomes liquid).
3. **Price formula** uses two hyperparameters: sensitivity `k` (units calibration) and elasticity `ε` (price vs supply relationship).
4. **Calibration** to an anchor (current price or market cap) is recommended to give price outputs real units.
5. **Monte Carlo** runs are a simple way to show sensitivity to demand shocks; they do not replace rigorous stress testing.
6. This model is a **simple indicative tool** — not a market-making or risk model. Do not use it for trading decisions without further validation and calibration.
""")


