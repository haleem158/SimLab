# simlab_tokenomics_v2.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ----------------------
# Page config (only once)
# ----------------------
st.set_page_config(page_title="SimLab — Token Supply Simulator", layout="wide")
st.title("SimLab — Token Supply Simulator")

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
    # returns fraction of vested tokens unlocked at month t_month
    if vesting_months <= 0:
        return 1.0
    if vesting_curve == "linear":
        return min(1.0, t_month / vesting_months)
    if vesting_curve == "exponential":
        lam = 5.0 / vesting_months # shape param
        return min(1.0, 1 - np.exp(-lam * t_month))
    return min(1.0, t_month / vesting_months)

def simulate_supply(params, rng=None):
    """
    params: dict of parameters
    returns DataFrame with monthly steps and final summary values.
    """
    if rng is None:
        rng = np.random.default_rng(params.get("seed", None))

    months = int(params["years"]) * 12
    total_supply = float(params["total_supply"])
    circulating = float(params["initial_supply"])
    burned = 0.0
    minted_cumulative = max(0.0, circulating - float(params.get("initial_locked", 0.0)))
    locked = max(0.0, float(params.get("initial_locked", 0.0)))

    rows = []
    for m in range(1, months + 1):
        year = (m - 1) // 12 + 1

        # time-varying parameters
        decay_rate = float(params["inflation_decay_rate"])
        inf0 = float(params["annual_inflation_rate"])
        inf_min = float(params["min_inflation_rate"])
        inflation_t = inf_min + (inf0 - inf_min) * np.exp(-decay_rate * (m - 1) / 12.0)

        # staking adoption curve
        stake_base = float(params["staking_rate"])
        stake_growth = float(params["staking_adoption_slope"])
        staking_rate_t = min(1.0, max(0.0, stake_base + stake_growth * np.log1p(m / 12.0)))

        # vesting unlock
        unlocked_fraction = vesting_unlocked_fraction(m, int(params["vesting_months"]), params["vesting_curve"])
        unlocked_tokens = total_supply * unlocked_fraction
        # ensure circulating not exceeding unlocked tokens
        circulating = min(circulating, unlocked_tokens)

        # remaining cap
        remaining_cap = max(0.0, total_supply - (circulating + locked + burned))

        # base for inflation
        base_for_inflation = circulating if params["inflation_applies_to"] == "circulating" else remaining_cap

        # stochastic perturbation
        if params["stochastic"]:
            infl_noise = rng.normal(0.0, float(params["inflation_volatility_monthly"]))
            inflation_t_eff = max(0.0, inflation_t * (1.0 + infl_noise))
        else:
            inflation_t_eff = inflation_t

        # monthly issuance
        new_tokens = base_for_inflation * (((1 + inflation_t_eff) ** (1/12.0)) - 1.0)
        new_tokens = min(new_tokens, remaining_cap)

        # staking rewards and treasury issuance
        staking_rewards = new_tokens * float(params["staking_reward_share"])
        treasury_issuance = new_tokens - staking_rewards

        # burn calculation (applies to non-staked circulating)
        staked_supply = circulating * staking_rate_t
        non_staked = max(0.0, circulating - staked_supply)
        burn_rate_monthly = 1 - (1 - float(params["burn_rate"])) ** (1/12.0)
        burn_from_activity = non_staked * burn_rate_monthly * float(params["tx_activity_index"])
        burned_tokens = min(non_staked, burn_from_activity)

        # update supplies
        circulating += new_tokens + staking_rewards - burned_tokens
        burned += burned_tokens
        minted_cumulative += new_tokens

        # rough locked estimator
        locked = max(0.0, total_supply - (circulating + burned + (remaining_cap - new_tokens)))

        # price model (illustrative)
        demand_index_base = float(params["demand_index_base"])
        demand_growth = float(params["demand_growth_rate"])
        demand_index = demand_index_base * (1 + demand_growth) ** ((m - 1) / 12.0)
        price = float(params["price_scale"]) * demand_index / max(1.0, circulating)

        rows.append({
            "month": int(m),
            "year": int(year),
            "new_tokens": float(new_tokens),
            "staking_rewards": float(staking_rewards),
            "treasury_issuance": float(treasury_issuance),
            "burned_tokens": float(burned_tokens),
            "circulating": float(circulating),
            "staked_supply": float(staked_supply),
            "burned_cumulative": float(burned),
            "minted_cumulative": float(minted_cumulative),
            "locked": float(locked),
            "price": float(price),
            "demand_index": float(demand_index),
            "inflation_annual_equiv": float(inflation_t_eff)
        })

        # early stop if cap reached and no new issuance
        if params["stop_if_cap_reached"] and (remaining_cap <= 1e-9):
            # copy last row forward
            for mm in range(m + 1, months + 1):
                r = rows[-1].copy()
                r["month"] = int(mm)
                r["year"] = int((mm - 1) // 12 + 1)
                rows.append(r)
            break

    return pd.DataFrame(rows)

def monte_carlo_simulation(params, runs=50):
    rng_master = np.random.default_rng(params.get("seed", None))
    outcomes = []
    for i in range(int(runs)):
        seed = int(rng_master.integers(0, 2**31 - 1))
        df = simulate_supply({**params, "seed": seed})
        outcomes.append(df)
    panel = pd.concat(outcomes, keys=range(len(outcomes)), names=["run", "row"]).reset_index(level=0)
    return panel

# ----------------------
# Sidebar UI with Loaders + Tooltips
# ----------------------

with st.sidebar:
    st.title("⚙️ Simulation Controls")

    # Bitcoin example loader
    if st.button("📘 Load Bitcoin-Style Example Scenario"):
        # use floats where appropriate
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
        st.success("✅ Bitcoin-style parameters loaded! Inspect inputs then click Run Simulation.")

    st.markdown("---")
    st.header("Base parameters")

    # Total supply: use floats consistently
    total_supply = st.number_input(
        label="Total Token Supply (cap)",
        min_value=float(1e5),
        max_value=float(1e12),
        value=float(st.session_state.get("total_supply", 1_000_000_000.0)),
        step=float(1e5),
        help="Hard cap on total tokens ever (typical range: 10M–10B)."
    )

    initial_supply = st.number_input(
        label="Initial Circulating Supply",
        min_value=0.0,
        max_value=float(total_supply),
        value=float(st.session_state.get("initial_supply", 100_000_000.0)),
        step=float(1e5),
        help="Tokens initially unlocked / in circulation (typical: 5–20% of total)."
    )

    initial_locked = st.number_input(
        label="Initial Locked / Vested (not circulating)",
        min_value=0.0,
        max_value=float(total_supply),
        value=float(st.session_state.get("initial_locked", 0.0)),
        step=float(1e5),
        help="Tokens minted but locked (team, advisors, investors)."
    )

    years = st.number_input(
        label="Years to Simulate",
        min_value=int(1),
        max_value=int(200),
        value=int(st.session_state.get("years", 20)),
        step=int(1),
        help="Simulation horizon (1–200 years)."
    )

    st.markdown("---")
    st.header("Issuance & rewards (educational hints shown above each control)")

    # explanation + slider for inflation
    st.markdown("""
**Inflation — industry ranges & rationale**
- Typical launch inflation: **3–10%** to fund validators and ecosystem.
- Long-term floors: **0.5–2%** to sustain incentives.
- Extremely high inflation (>15%) usually unsustainable.
""")
    annual_inflation_rate = st.slider(
        label="Initial Annual Inflation Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state.get("annual_inflation_rate", 0.05)) * 100.0,
        help="Annual issuance as % (displayed in %). Typical: 3–10%."
    ) / 100.0

    st.markdown("""
**Minimum Inflation (floor)**
- A floor like 0–2% prevents issuance dropping to zero prematurely if protocol wants ongoing incentives.
""")
    min_inflation_rate = st.slider(
        label="Minimum Annual Inflation Rate (%) (floor)",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state.get("min_inflation_rate", 0.01)) * 100.0,
        help="Long-term min inflation as %."
    ) / 100.0

    st.markdown("""
**Inflation decay rate**
- Controls speed of decay from initial to min (typical: 0.1–1.0 per year).
- Higher => faster approach to floor.
""")
    inflation_decay_rate = st.slider(
        label="Inflation Decay Rate (per year)",
        min_value=0.0,
        max_value=10.0,
        value=float(st.session_state.get("inflation_decay_rate", 0.5)),
        help="Higher value -> faster decay of inflation toward minimum."
    )

    inflation_applies_to = st.selectbox(
        label="Inflation applies to",
        options=["circulating", "remaining"],
        index=["circulating", "remaining"].index(str(st.session_state.get("inflation_applies_to", "circulating"))),
        help="Apply inflation to current circulating supply (PoS style) or remaining unminted supply (capped)."
    )

    st.markdown("""
**Staking participation**
- Mature PoS networks commonly see **30–70%** of supply staked.
""")
    staking_rate = st.slider(
        label="Base Staking Participation Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state.get("staking_rate", 0.40)) * 100.0,
        help="Fraction of supply staked (displayed %)."
    ) / 100.0

    staking_adoption_slope = st.number_input(
        label="Staking adoption slope",
        min_value=-0.05,
        max_value=0.05,
        value=float(st.session_state.get("staking_adoption_slope", 0.0)),
        step=0.001,
        help="Small annual change in staking participation (e.g., 0.01 ~ +1%/yr)."
    )

    st.markdown("""
**Share of new issuance paid to stakers**
- Common splits: **60–90%** to stakers, the rest to treasury/ecosystem.
""")
    staking_reward_share = st.slider(
        label="Share of issuance paid to stakers (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state.get("staking_reward_share", 0.75)) * 100.0,
        help="Percentage of new issuance given to stakers."
    ) / 100.0

    st.markdown("---")
    st.header("Burn & activity")

    st.markdown("""
**Burn rate on non-staked supply**
- Examples: typical protocol burns often in range **0.1–3%** annually depending on fee model.
""")
    burn_rate = st.slider(
        label="Annual burn rate on non-staked supply (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state.get("burn_rate", 0.02)) * 100.0,
        help="Annual burn rate applied to non-staked portion (displayed %)."
    ) / 100.0

    tx_activity_index = st.slider(
        label="Transaction activity index (multiplier)",
        min_value=0.0,
        max_value=5.0,
        value=float(st.session_state.get("tx_activity_index", 1.0)),
        step=0.1,
        help="Proxy for on-chain activity; scales burn."
    )

    st.markdown("---")
    st.header("Vesting & unlocking")

    vesting_months = st.number_input(
        label="Vesting duration (months)",
        min_value=int(0),
        max_value=int(1200),
        value=int(st.session_state.get("vesting_months", 0)),
        step=int(1),
        help="Typical team vesting: 12–48 months."
    )
    vesting_curve = st.selectbox(
        label="Vesting curve",
        options=["linear", "exponential"],
        index=["linear", "exponential"].index(str(st.session_state.get("vesting_curve", "linear"))),
        help="Linear = steady unlock; exponential = faster early unlock."
    )

    st.markdown("---")
    st.header("Price & demand (illustrative)")

    demand_index_base = st.number_input(
        label="Demand index base",
        min_value=0.001,
        max_value=100.0,
        value=float(st.session_state.get("demand_index_base", 1.0)),
        step=0.1,
        help="Arbitrary base demand index (1.0 = neutral)."
    )
    demand_growth_rate = st.number_input(
        label="Demand growth rate (annual)",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("demand_growth_rate", 0.02)),
        step=0.001,
        help="Assumed annual demand growth (typical 0.01–0.05)."
    )
    price_scale = st.number_input(
        label="Price scale factor",
        min_value=0.001,
        max_value=100.0,
        value=float(st.session_state.get("price_scale", 1.0)),
        step=0.1,
        help="Scaling factor for the illustrative price model."
    )

    st.markdown("---")
    st.header("Simulation controls")

    stochastic = st.checkbox(
        label="Enable stochastic perturbations (Monte Carlo noise)",
        value=bool(st.session_state.get("stochastic", True)),
        help="Adds monthly random noise to inflation path when enabled."
    )

    inflation_volatility_monthly = st.number_input(
        label="Inflation monthly volatility (stdev)",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("inflation_volatility_monthly", 0.01)),
        step=0.001,
        help="Monthly volatility used in stochastic runs (typical 0.005–0.02)."
    )

    runs = st.number_input(
        label="Monte Carlo runs (1 = deterministic)",
        min_value=int(1),
        max_value=int(500),
        value=int(st.session_state.get("runs", 50)),
        step=int(1),
        help="Number of independent stochastic simulations to average over."
    )

    seed = st.number_input(
        label="Random seed (0 = random)",
        min_value=int(0),
        max_value=int(2**31-1),
        value=int(st.session_state.get("seed", 0)),
        step=int(1),
        help="Set a fixed seed for reproducible Monte Carlo runs (0 => random)."
    )

    stop_if_cap_reached = st.checkbox(
        label="Stop early if cap reached & no further issuance",
        value=bool(st.session_state.get("stop_if_cap_reached", True)),
        help="If True, stops minting when cap effectively exhausted."
    )

    st.markdown("---")
    # Use a single clear Run button
    run_clicked = st.button("▶️ Run Simulation")

# ----------------------
# Collect params and validate
# ----------------------
params = {
    "total_supply": float(total_supply),
    "initial_supply": float(initial_supply),
    "initial_locked": float(initial_locked),
    "years": int(years),
    "annual_inflation_rate": float(annual_inflation_rate),
    "min_inflation_rate": float(min_inflation_rate),
    "inflation_decay_rate": float(inflation_decay_rate),
    "inflation_applies_to": str(inflation_applies_to),
    "staking_rate": float(staking_rate),
    "staking_adoption_slope": float(staking_adoption_slope),
    "staking_reward_share": float(staking_reward_share),
    "burn_rate": float(burn_rate),
    "tx_activity_index": float(tx_activity_index),
    "vesting_months": int(vesting_months),
    "vesting_curve": str(vesting_curve),
    "demand_index_base": float(demand_index_base),
    "demand_growth_rate": float(demand_growth_rate),
    "price_scale": float(price_scale),
    "stochastic": bool(stochastic),
    "inflation_volatility_monthly": float(inflation_volatility_monthly),
    "seed": None if int(seed) == 0 else int(seed),
    "runs": int(runs),
    "stop_if_cap_reached": bool(stop_if_cap_reached),
}

# Validate and stop if invalid
try:
    validate_inputs(params)
except AssertionError as e:
    st.error(f"Input validation error: {e}")
    st.stop()

# ----------------------
# Run simulation(s) and show results
# ----------------------
if run_clicked:
    if params["stochastic"] and params["runs"] > 1:
        # Monte Carlo
        panel = monte_carlo_simulation(params, runs=params["runs"])
        # aggregate per month
        agg = panel.groupby("month").agg({
            "circulating": ["mean", "std"],
            "burned_cumulative": ["mean"],
            "price": ["mean"]
        }).reset_index()
        agg.columns = ["month", "circulating_mean", "circulating_std", "burned_cum_mean", "price_mean"]
        agg["upper"] = agg["circulating_mean"] + 1.96 * agg["circulating_std"]
        agg["lower"] = np.maximum(0.0, agg["circulating_mean"] - 1.96 * agg["circulating_std"])

        # Plot mean and CI
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=agg["month"] / 12.0, y=agg["circulating_mean"], mode="lines", name="Circulating (mean)"))
        fig.add_trace(go.Scatter(x=agg["month"] / 12.0, y=agg["upper"], mode="lines", name="Upper 95%"))
        fig.add_trace(go.Scatter(x=agg["month"] / 12.0, y=agg["lower"], mode="lines", name="Lower 95%"))
        fig.update_layout(title="Circulating Supply (mean ± 95% CI)", xaxis_title="Years", yaxis_title="Tokens")
        st.plotly_chart(fig, use_container_width=True)

        st.write("Final (mean) circulating:", f"{agg.iloc[-1]['circulating_mean']:,.0f}")
        st.write("Final (mean) burned cumulative:", f"{agg.iloc[-1]['burned_cum_mean']:,.0f}")
        st.write("Final (mean) price:", f"{agg.iloc[-1]['price_mean']:.6f}")

        csv = panel.to_csv(index=False).encode("utf-8")
        st.download_button("Download Monte Carlo panel CSV", csv, "mc_panel.csv", "text/csv")

    else:
        # Deterministic single run
        df = simulate_supply(params, rng=np.random.default_rng(params.get("seed", None)))
        df["year_decimal"] = df["month"] / 12.0

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["year_decimal"], y=df["circulating"], mode="lines+markers", name="Circulating"))
        fig.add_trace(go.Scatter(x=df["year_decimal"], y=df["burned_cumulative"], mode="lines+markers", name="Burned (cum)"))
        fig.add_trace(go.Scatter(x=df["year_decimal"], y=df["staked_supply"], mode="lines+markers", name="Staked"))
        fig.update_layout(title="Token Supply Dynamics", xaxis_title="Years", yaxis_title="Tokens")
        st.plotly_chart(fig, use_container_width=True)

        # Price plot
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

    st.success("Simulation complete ✅")

# ----------------------
# End of file
# ----------------------
