# simlab_tokenomics_v2.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ----------------------
# Helper / Simulation
# ----------------------

def validate_inputs(params):
    assert 0 <= params["staking_rate"] <= 1, "staking_rate must be [0,1]"
    assert 0 <= params["burn_rate"] <= 1, "burn_rate must be [0,1]"
    assert 0 <= params["annual_inflation_rate"] <= 10, "inflation rate unrealistic >1000%"
    assert params["initial_supply"] <= params["total_supply"], "initial_supply cannot exceed total_supply"
    assert params["years"] >= 1 and params["years"] <= 200, "years must be between 1 and 200"

def vesting_unlocked_fraction(t_month, vesting_months, vesting_curve="linear"):
    # returns fraction of vested tokens unlocked at month t_month
    if vesting_months <= 0:
        return 1.0
    if vesting_curve == "linear":
        return min(1.0, t_month / vesting_months)
    if vesting_curve == "exponential":
        # faster early unlock, then asymptote
        lam = 5.0 / vesting_months  # shape param
        return min(1.0, 1 - np.exp(-lam * t_month))
    return min(1.0, t_month / vesting_months)

def simulate_supply(params, rng=None):
    """
    params: dict of parameters
    returns DataFrame with monthly steps and final summary values.
    This simulation:
      - works in monthly steps
      - issues new_tokens based on either circulating_supply or remaining_supply
      - staking rewards are paid from new_tokens according to staking_reward_share
      - burn applies to non-staked circulating supply and ties optionally to tx_activity
      - respects total_supply cap and vesting unlock schedule
      - supports time-varying inflation decay and staking participation adoption
    """
    if rng is None:
        rng = np.random.default_rng(params.get("seed", None))

    months = params["years"] * 12
    # initial state
    total_supply = float(params["total_supply"])
    circulating = float(params["initial_supply"])
    burned = 0.0
    minted_cumulative = max(0.0, circulating - params.get("initial_locked", 0.0))
    locked = max(0.0, params.get("initial_locked", 0.0))  # tokens minted but locked/vested
    # track monthly history
    rows = []
    for m in range(1, months + 1):
        year = (m - 1) // 12 + 1
        # time-varying parameters
        # inflation decay: exponential decay from initial_inflation to min_inflation
        decay_rate = params["inflation_decay_rate"]
        inf0 = params["annual_inflation_rate"]
        inf_min = params["min_inflation_rate"]
        # continuous decay per month
        inflation_t = inf_min + (inf0 - inf_min) * np.exp(-decay_rate * (m - 1) / 12.0)

        # staking adoption curve (can increase/decrease)
        stake_base = params["staking_rate"]
        stake_growth = params["staking_adoption_slope"]
        staking_rate_t = min(1.0, max(0.0, stake_base + stake_growth * np.log1p(m/12.0)))

        # compute unlocked fraction from vesting schedule
        unlocked_fraction = vesting_unlocked_fraction(m, params["vesting_months"], params["vesting_curve"])
        unlocked_tokens = total_supply * unlocked_fraction
        # Ensure circulating does not exceed unlocked tokens
        circulating = min(circulating, unlocked_tokens)

        # remaining cap (tokens that can still be minted)
        remaining_cap = max(0.0, total_supply - (circulating + locked + burned))

        # Determine new tokens (monthly)
        if params["inflation_applies_to"] == "circulating":
            base_for_inflation = circulating
        else:  # remaining
            base_for_inflation = remaining_cap

        # stochastic perturbation (optional)
        if params["stochastic"]:
            infl_noise = rng.normal(0.0, params["inflation_volatility_monthly"])
            inflation_t_eff = max(0.0, inflation_t * (1.0 + infl_noise))
        else:
            inflation_t_eff = inflation_t

        # monthly issuance (simple annual rate converted to monthly)
        new_tokens = base_for_inflation * ( (1 + inflation_t_eff) ** (1/12.0) - 1.0 )

        # force cap: cannot mint more than remaining cap
        new_tokens = min(new_tokens, remaining_cap)

        # staking rewards: a fraction of new_tokens allocated to stakers
        staking_reward_share = params["staking_reward_share"]  # portion of new_tokens paid to stakers
        staking_rewards = new_tokens * staking_reward_share

        # other issuance (treasury, team) = new_tokens - staking_rewards
        treasury_issuance = new_tokens - staking_rewards

        # burn: applies mainly to non-staked circulating tokens, optionally tied to activity
        staked_supply = circulating * staking_rate_t
        non_staked = max(0.0, circulating - staked_supply)
        # compute burn from non-staked based on burn_rate_annual converted monthly,
        # and optionally scaled by tx_activity_index (a proxy for on-chain activity, default 1)
        burn_rate_monthly = 1 - (1 - params["burn_rate"]) ** (1/12.0)
        burn_from_activity = non_staked * burn_rate_monthly * params["tx_activity_index"]
        # clip burn cannot exceed non_staked
        burned_tokens = min(non_staked, burn_from_activity)

        # update supplies
        circulating += new_tokens + staking_rewards - burned_tokens
        burned += burned_tokens
        minted_cumulative += new_tokens

        # locked tokens may gradually unlock: we model locked -> circulating via vesting only when unlocked_fraction grows;
        # ensure locked reflects difference between total minted (including team allocations) and circulating + burned
        # For simplicity, track locked as min(remaining locked schedule)
        locked = max(0.0, total_supply - (circulating + burned + (remaining_cap - new_tokens)))  # rough estimator

        # price model (simple demand_index / circulating supply) — illustrative only
        demand_index_base = params["demand_index_base"]
        demand_growth = params["demand_growth_rate"]
        demand_index = demand_index_base * (1 + demand_growth) ** ( (m - 1) / 12.0 )
        # price per token = price_scale * demand_index / circulating (avoid div by zero)
        price = params["price_scale"] * demand_index / max(1.0, circulating)

        rows.append({
            "month": m,
            "year": year,
            "new_tokens": new_tokens,
            "staking_rewards": staking_rewards,
            "treasury_issuance": treasury_issuance,
            "burned_tokens": burned_tokens,
            "circulating": circulating,
            "staked_supply": staked_supply,
            "burned_cumulative": burned,
            "minted_cumulative": minted_cumulative,
            "locked": locked,
            "price": price,
            "demand_index": demand_index,
            "inflation_annual_equiv": inflation_t_eff
        })

        # early stop if cap reached and no more dynamics (optional)
        if params["stop_if_cap_reached"] and (abs(remaining_cap - new_tokens) < 1e-9) and new_tokens == 0:
            # fill remaining months with steady state snapshots
            for mm in range(m+1, months+1):
                rows.append({**rows[-1], "month": mm, "year": (mm-1)//12+1})
            break

    df = pd.DataFrame(rows)
    return df

def monte_carlo_simulation(params, runs=50):
    rng_master = np.random.default_rng(params.get("seed", None))
    outcomes = []
    for i in range(runs):
        seed = int(rng_master.integers(0, 2**31 - 1))
        df = simulate_supply({**params, "seed": seed})
        # take yearly snapshots by picking month=12,24,...
        final = df.iloc[-1].to_dict()
        outcomes.append(df)
    # stack into a combined panel: we will compute mean & CI per month
    panel = pd.concat(outcomes, keys=range(len(outcomes)), names=["run", "row"]).reset_index(level=0)
    return panel

# ----------------------
# Streamlit UI
# ----------------------

st.set_page_config(page_title="SimLab — Tokenomics Simulator", layout="wide")
st.title("SimLab — Token Supply Simulator")

with st.sidebar:
    st.header("Base parameters")
    total_supply = st.number_input("Total Token Supply (cap)", value=1_000_000_000, step=1_000_000)
    initial_supply = st.number_input("Initial Circulating Supply", value=100_000_000, step=1_000_000)
    initial_locked = st.number_input("Initial Locked/Vested (not circulating)", value=0, step=1_000_000)
    years = st.number_input("Years to Simulate", min_value=1, max_value=200, value=20)

    st.header("Issuance & rewards")
    annual_inflation_rate = st.slider("Initial Annual Inflation Rate (%)", 0.0, 100.0, 5.0) / 100.0
    min_inflation_rate = st.slider("Minimum Annual Inflation Rate (%) (floor)", 0.0, 100.0, 1.0) / 100.0
    inflation_decay_rate = st.slider("Inflation Decay Rate (per year, higher = faster decay)", 0.0, 10.0, 0.5)
    inflation_applies_to = st.selectbox("Inflation applies to", options=["circulating", "remaining"],
                                        help="Apply inflation to current circulating supply, or remaining unminted supply")
    staking_rate = st.slider("Base Staking Participation Rate (%)", 0.0, 100.0, 40.0) / 100.0
    staking_adoption_slope = st.number_input("Staking adoption slope (small positive moves staking up over time)", value=0.0, format="%.4f")
    staking_reward_share = st.slider("Share of new issuance paid to stakers (%)", 0.0, 100.0, 75.0) / 100.0

    st.header("Burn & activity")
    burn_rate = st.slider("Annual burn rate on non-staked supply (%)", 0.0, 100.0, 2.0) / 100.0
    tx_activity_index = st.slider("Transaction activity index (multiplier)", 0.0, 5.0, 1.0)

    st.header("Vesting & unlocking")
    vesting_months = st.number_input("Vesting duration (months)", min_value=0, max_value=1200, value=0)
    vesting_curve = st.selectbox("Vesting curve", ["linear", "exponential"])

    st.header("Price & demand (illustrative)")
    demand_index_base = st.number_input("Demand index base", value=1.0, step=0.1, format="%.2f")
    demand_growth_rate = st.number_input("Demand growth rate (annual)", value=0.02, format="%.4f")
    price_scale = st.number_input("Price scale factor", value=1.0, format="%.4f")

    st.header("Simulation controls")
    stochastic = st.checkbox("Enable stochastic perturbations (Monte Carlo noise)", value=True)
    inflation_volatility_monthly = st.number_input("Inflation monthly volatility (stdev, e.g. 0.01)", value=0.01, format="%.4f")
    runs = st.number_input("Monte Carlo runs (if stochastic, 1 = deterministic)", min_value=1, max_value=500, value=50)
    seed = st.number_input("Random seed (0 for random)", value=0)
    stop_if_cap_reached = st.checkbox("Stop early if cap reached & no more issuance", value=True)

    st.markdown("---")
    st.button("Run Simulation", key="run_button")

# gather params dict
params = {
    "total_supply": float(total_supply),
    "initial_supply": float(initial_supply),
    "initial_locked": float(initial_locked),
    "years": int(years),
    "annual_inflation_rate": float(annual_inflation_rate),
    "min_inflation_rate": float(min_inflation_rate),
    "inflation_decay_rate": float(inflation_decay_rate),
    "inflation_applies_to": inflation_applies_to,
    "staking_rate": float(staking_rate),
    "staking_adoption_slope": float(staking_adoption_slope),
    "staking_reward_share": float(staking_reward_share),
    "burn_rate": float(burn_rate),
    "tx_activity_index": float(tx_activity_index),
    "vesting_months": int(vesting_months),
    "vesting_curve": vesting_curve,
    "demand_index_base": float(demand_index_base),
    "demand_growth_rate": float(demand_growth_rate),
    "price_scale": float(price_scale),
    "stochastic": bool(stochastic),
    "inflation_volatility_monthly": float(inflation_volatility_monthly),
    "seed": None if seed == 0 else int(seed),
    "stop_if_cap_reached": bool(stop_if_cap_reached),
}

# validate
try:
    validate_inputs(params)
except AssertionError as e:
    st.error(f"Input validation error: {e}")
    st.stop()

# Run simulation(s)
if st.session_state.get("run_button", False) or st.button("Run Simulation (main)"):
    if params["stochastic"] and runs > 1:
        # Monte Carlo panel - returns panel with 'run' index
        panel = monte_carlo_simulation(params, runs=int(runs))
        # compute stats per month
        agg = panel.groupby("month").agg({
            "circulating": ["mean", "std"],
            "burned_cumulative": ["mean"],
            "price": ["mean"]
        }).reset_index()
        agg.columns = ["month", "circulating_mean", "circulating_std", "burned_cum_mean", "price_mean"]
        agg["upper"] = agg["circulating_mean"] + 1.96 * agg["circulating_std"]
        agg["lower"] = np.maximum(0.0, agg["circulating_mean"] - 1.96 * agg["circulating_std"])
        # plot mean with 95% CI
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=agg["month"]/12.0, y=agg["circulating_mean"],
                                 mode="lines", name="Circulating (mean)"))
        fig.add_trace(go.Scatter(x=agg["month"]/12.0, y=agg["upper"],
                                 mode="lines", name="Upper 95%"))
        fig.add_trace(go.Scatter(x=agg["month"]/12.0, y=agg["lower"],
                                 mode="lines", name="Lower 95%"))
        fig.update_layout(title="Circulating Supply (mean ± 95% CI)", xaxis_title="Years", yaxis_title="Tokens")
        st.plotly_chart(fig, use_container_width=True)

        st.write("Final (mean) circulating:", f"{agg.iloc[-1]['circulating_mean']:,.0f}")
        st.write("Final (mean) burned cumulative:", f"{agg.iloc[-1]['burned_cum_mean']:,.0f}")
        st.write("Final (mean) price:", f"{agg.iloc[-1]['price_mean']:.6f}")

        # allow CSV download of aggregated panel
        csv = panel.to_csv(index=False).encode("utf-8")
        st.download_button("Download Monte Carlo panel CSV", csv, "mc_panel.csv", "text/csv")
    else:
        # single deterministic run
        df = simulate_supply(params, rng=np.random.default_rng(params.get("seed", None)))
        # convert month -> year for display
        df["year_decimal"] = df["month"] / 12.0

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["year_decimal"], y=df["circulating"], mode="lines+markers", name="Circulating"))
        fig.add_trace(go.Scatter(x=df["year_decimal"], y=df["burned_cumulative"], mode="lines+markers", name="Burned (cum)"))
        fig.add_trace(go.Scatter(x=df["year_decimal"], y=df["staked_supply"], mode="lines+markers", name="Staked"))
        fig.update_layout(title="Token Supply Dynamics", xaxis_title="Years", yaxis_title="Tokens")
        st.plotly_chart(fig, use_container_width=True)

        # price plot
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["year_decimal"], y=df["price"], mode="lines", name="Price (illustrative)"))
        fig2.update_layout(title="Illustrative Price Path", xaxis_title="Years", yaxis_title="Price")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Summary (final month)")
        final = df.iloc[-1]
        st.write(f"Circulating: {final['circulating']:,.0f}")
        st.write(f"Burned (cumulative): {final['burned_cumulative']:,.0f}")
        st.write(f"Staked (last month): {final['staked_supply']:,.0f}")
        st.write(f"Price (illustrative): {final['price']:.6f}")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download simulation CSV", csv, "simulation.csv", "text/csv")

    st.success("Simulation complete.")



