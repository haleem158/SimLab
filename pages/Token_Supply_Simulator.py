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
    assert 1 <= params["years"] <= 200, "years must be between 1 and 200"

def vesting_unlocked_fraction(t_month, vesting_months, vesting_curve="linear"):
    if vesting_months <= 0:
        return 1.0
    if vesting_curve == "linear":
        return min(1.0, t_month / vesting_months)
    if vesting_curve == "exponential":
        lam = 5.0 / vesting_months
        return min(1.0, 1 - np.exp(-lam * t_month))
    return min(1.0, t_month / vesting_months)

def simulate_supply(params, rng=None):
    if rng is None:
        rng = np.random.default_rng(params.get("seed", None))
    months = int(params["years"] * 12)
    total_supply = float(params["total_supply"])
    circulating = float(params["initial_supply"])
    burned = 0.0
    minted_cumulative = max(0.0, circulating - float(params.get("initial_locked", 0.0)))
    locked = max(0.0, float(params.get("initial_locked", 0.0)))
    rows = []

    for m in range(1, months + 1):
        year = (m - 1) // 12 + 1

        # Inflation decay
        decay_rate = params["inflation_decay_rate"]
        inf0 = params["annual_inflation_rate"]
        inf_min = params["min_inflation_rate"]
        inflation_t = inf_min + (inf0 - inf_min) * np.exp(-decay_rate * (m - 1) / 12.0)

        # Staking adoption
        stake_base = params["staking_rate"]
        stake_growth = params["staking_adoption_slope"]
        staking_rate_t = min(1.0, max(0.0, stake_base + stake_growth * np.log1p(m / 12.0)))

        # Vesting unlock
        unlocked_fraction = vesting_unlocked_fraction(m, params["vesting_months"], params["vesting_curve"])
        unlocked_tokens = total_supply * unlocked_fraction
        circulating = min(circulating, unlocked_tokens)

        remaining_cap = max(0.0, total_supply - (circulating + locked + burned))
        base_for_inflation = circulating if params["inflation_applies_to"] == "circulating" else remaining_cap

        # Stochastic inflation noise
        if params["stochastic"]:
            infl_noise = rng.normal(0.0, params["inflation_volatility_monthly"])
            inflation_t_eff = max(0.0, inflation_t * (1.0 + infl_noise))
        else:
            inflation_t_eff = inflation_t

        # Monthly issuance
        new_tokens = base_for_inflation * ((1 + inflation_t_eff) ** (1 / 12.0) - 1.0)
        new_tokens = min(new_tokens, remaining_cap)

        staking_rewards = new_tokens * params["staking_reward_share"]
        treasury_issuance = new_tokens - staking_rewards

        staked_supply = circulating * staking_rate_t
        non_staked = max(0.0, circulating - staked_supply)
        burn_rate_monthly = 1 - (1 - params["burn_rate"]) ** (1 / 12.0)
        burn_from_activity = non_staked * burn_rate_monthly * params["tx_activity_index"]
        burned_tokens = min(non_staked, burn_from_activity)

        circulating += new_tokens + staking_rewards - burned_tokens
        burned += burned_tokens
        minted_cumulative += new_tokens

        locked = max(0.0, total_supply - (circulating + burned + (remaining_cap - new_tokens)))

        demand_index = params["demand_index_base"] * (1 + params["demand_growth_rate"]) ** ((m - 1) / 12.0)
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

        if params["stop_if_cap_reached"] and remaining_cap <= 1e-9:
            for mm in range(m + 1, months + 1):
                rows.append({**rows[-1], "month": mm, "year": (mm - 1) // 12 + 1})
            break

    return pd.DataFrame(rows)

def monte_carlo_simulation(params, runs=50):
    rng_master = np.random.default_rng(params.get("seed", None))
    outcomes = []
    for i in range(runs):
        seed = int(rng_master.integers(0, 2**31 - 1))
        df = simulate_supply({**params, "seed": seed})
        outcomes.append(df)
    panel = pd.concat(outcomes, keys=range(len(outcomes)), names=["run", "row"]).reset_index(level=0)
    return panel

# ----------------------
# Streamlit UI
# ----------------------

st.set_page_config(page_title="SimLab — Token Supply Simulator", layout="wide")
st.title("SimLab — Token Supply Simulator")

with st.sidebar:
    st.title("⚙️ Simulation Controls")

    # Bitcoin Example
    if st.button("📘 Load Bitcoin-Style Example Scenario"):
        st.session_state.update({
            "total_supply": 21_000_000.0,
            "initial_supply": 19_000_000.0,
            "initial_locked": 0.0,
            "years": 100,
            "annual_inflation_rate": 0.015,
            "min_inflation_rate": 0.0,
            "inflation_decay_rate": 0.25,
            "inflation_applies_to": "remaining",
            "staking_rate": 0.0,
            "staking_adoption_slope": 0.0,
            "staking_reward_share": 0.0,
            "burn_rate": 0.0,
            "tx_activity_index": 1.0,
            "vesting_months": 0,
            "vesting_curve": "linear",
            "demand_index_base": 1.0,
            "demand_growth_rate": 0.02,
            "price_scale": 1.0,
            "stochastic": False,
            "inflation_volatility_monthly": 0.0,
            "runs": 1,
            "seed": 0,
            "stop_if_cap_reached": True,
        })
        st.success("✅ Bitcoin-style parameters loaded! Scroll down or click *Run Simulation*.")

    st.markdown("---")
    st.header("Base Parameters")

    total_supply = st.number_input(
        "Total Token Supply (cap)", min_value=1e5, max_value=1e12,
        value=float(st.session_state.get("total_supply", 1_000_000_000.0)),
        help="Total maximum token cap."
    )
    initial_supply = st.number_input(
        "Initial Circulating Supply", min_value=0.0, max_value=float(total_supply),
        value=float(st.session_state.get("initial_supply", 100_000_000.0))
    )
    initial_locked = st.number_input(
        "Initial Locked Supply", min_value=0.0, max_value=float(total_supply),
        value=float(st.session_state.get("initial_locked", 0.0))
    )
    years = st.number_input(
        "Years to Simulate", min_value=1, max_value=200,
        value=int(st.session_state.get("years", 20))
    )

    annual_inflation_rate = st.slider(
        "Initial Annual Inflation Rate (%)", 0.0, 100.0,
        float(st.session_state.get("annual_inflation_rate", 5.0))
    ) / 100.0
    min_inflation_rate = st.slider(
        "Minimum Inflation Rate (%)", 0.0, 100.0,
        float(st.session_state.get("min_inflation_rate", 1.0))
    ) / 100.0
    inflation_decay_rate = st.slider(
        "Inflation Decay Rate", 0.0, 10.0,
        float(st.session_state.get("inflation_decay_rate", 0.5))
    )

    inflation_applies_to = st.selectbox(
        "Inflation applies to", ["circulating", "remaining"],
        index=["circulating", "remaining"].index(st.session_state.get("inflation_applies_to", "circulating"))
    )

    staking_rate = st.slider(
        "Staking Rate (%)", 0.0, 100.0,
        float(st.session_state.get("staking_rate", 40.0))
    ) / 100.0
    staking_adoption_slope = st.number_input(
        "Staking Adoption Slope", -0.05, 0.05,
        float(st.session_state.get("staking_adoption_slope", 0.0))
    )
    staking_reward_share = st.slider(
        "Staking Reward Share (%)", 0.0, 100.0,
        float(st.session_state.get("staking_reward_share", 75.0))
    ) / 100.0

    burn_rate = st.slider(
        "Annual Burn Rate (%)", 0.0, 100.0,
        float(st.session_state.get("burn_rate", 2.0))
    ) / 100.0
    tx_activity_index = st.slider(
        "Transaction Activity Index", 0.0, 5.0,
        float(st.session_state.get("tx_activity_index", 1.0))
    )

    vesting_months = st.number_input(
        "Vesting Duration (months)", 0, 1200,
        int(st.session_state.get("vesting_months", 0))
    )
    vesting_curve = st.selectbox(
        "Vesting Curve", ["linear", "exponential"],
        index=["linear", "exponential"].index(st.session_state.get("vesting_curve", "linear"))
    )

    demand_index_base = st.number_input(
        "Demand Index Base", 0.1, 10.0,
        float(st.session_state.get("demand_index_base", 1.0))
    )
    demand_growth_rate = st.number_input(
        "Demand Growth Rate (annual)", 0.0, 0.5,
        float(st.session_state.get("demand_growth_rate", 0.02))
    )
    price_scale = st.number_input(
        "Price Scale Factor", 0.1, 10.0,
        float(st.session_state.get("price_scale", 1.0))
    )

    stochastic = st.checkbox(
        "Enable Stochastic Perturbations",
        value=bool(st.session_state.get("stochastic", True))
    )
    inflation_volatility_monthly = st.number_input(
        "Inflation Monthly Volatility", 0.0, 0.1,
        float(st.session_state.get("inflation_volatility_monthly", 0.01))
    )
    runs = st.number_input(
        "Monte Carlo Runs", 1, 500,
        int(st.session_state.get("runs", 50))
    )
    seed = st.number_input("Random Seed (0=random)", 0, 9999, int(st.session_state.get("seed", 0)))
    stop_if_cap_reached = st.checkbox(
        "Stop if cap reached", value=bool(st.session_state.get("stop_if_cap_reached", True))
    )

    st.markdown("---")
    run_clicked = st.button("▶️ Run Simulation")

# ----------------------
# Collect Params & Run
# ----------------------

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

try:
    validate_inputs(params)
except AssertionError as e:
    st.error(f"Input validation error: {e}")
    st.stop()

# Run simulation logic
if run_clicked:
    if params["stochastic"] and runs > 1:
        panel = monte_carlo_simulation(params, runs=int(runs))
        agg = panel.groupby("month").agg({
            "circulating": ["mean", "std"],
            "burned_cumulative": ["mean"],
            "price": ["mean"]
        }).reset_index()
        agg.columns = ["month", "circulating_mean", "circulating_std", "burned_cum_mean", "price_mean"]
        agg["upper"] = agg["circulating_mean"] + 1.96 * agg["circulating_std"]
        agg["lower"] = np.maximum(0.0, agg["circulating_mean"] - 1.96 * agg["circulating_std"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=agg["month"]/12.0, y=agg["circulating_mean"], mode="lines", name="Mean Circulating"))
        fig.add_trace(go.Scatter(x=agg["month"]/12.0, y=agg["upper"], mode="lines", name="Upper 95%"))
        fig.add_trace(go.Scatter(x=agg["month"]/12.0, y=agg["lower"], mode="lines", name="Lower 95%"))
        fig.update_layout(title="Circulating Supply (Mean ± 95% CI)", xaxis_title="Years", yaxis_title="Tokens")
        st.plotly_chart(fig, use_container_width=True)
    else:
        df = simulate_supply(params)
        df["year_decimal"] = df["month"] / 12.0
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["year_decimal"], y=df["circulating"], mode="lines", name="Circulating"))
        fig.add_trace(go.Scatter(x=df["year_decimal"], y=df["burned_cumulative"], mode="lines", name="Burned"))
        fig.update_layout(title="Token Supply Dynamics", xaxis_title="Years", yaxis_title="Tokens")
        st.plotly_chart(fig, use_container_width=True)

    st.success("Simulation complete ✅")
