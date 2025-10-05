"""
SimLab - Vesting Schedule Simulator (Improved & Economically Sound)

Features:
- Correct cumulative supply aggregation (no double-counting).
- Allocation validation (sum <= 100% of total supply).
- Cliff truncation when simulation months < cliff + vesting.
- Optionally add stochastic noise to monthly unlocks (realism).
- Behavioral "activation" ratios: fraction of unlocked tokens that actually enter circulation.
- Governance lock fraction: portion of tokens considered non-economic but may carry voting power.
- Inflation (monthly) computed as change in circulating supply / total supply.
- Discounted present value (monthly discount rate).
- Session-state storage saves parameters + numeric data for export/inspection.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

st.set_page_config(layout="wide", page_title="SimLab - Vesting Schedule Simulator (Sound)")

st.title("SimLab — Vesting Schedule Simulator (Mathematically & Economically Sound)")

# -------------------------
# Global inputs
# -------------------------
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    total_supply = st.number_input("Total Token Supply (tokens)", value=1_000_000_000, step=1_000_000, min_value=1)
with col2:
    months = st.slider("Simulation horizon (months)", min_value=12, max_value=240, value=120, step=12)
with col3:
    profile_name = st.text_input("Profile name", value="default_profile")

st.markdown("---")

# -------------------------
# Role inputs (up to 10)
# -------------------------
st.subheader("Role allocations (max 10 roles)")
num_roles = st.number_input("Number of roles", min_value=1, max_value=10, value=4, step=1)

default_roles = ["Team", "Investors", "Foundation", "Community", "Advisors", "Ecosystem", "Liquidity", "Partnerships", "Treasury", "Rewards"]

roles = []
allocations = {}
cliffs = {}
vesting_periods = {}
behavior_activation = {}
governance_fraction = {}

for i in range(int(num_roles)):
    role_default = default_roles[i] if i < len(default_roles) else f"Role_{i+1}"
    st.markdown(f"**Role {i+1}**")
    col_a, col_b, col_c, col_d = st.columns([3, 2, 2, 2])
    with col_a:
        role_name = st.text_input(f"Name (role {i+1})", value=role_default, key=f"name_{i}")
    with col_b:
        alloc_pct = st.number_input(
            f"{role_name} allocation (% of total supply)",
            min_value=0.0, max_value=100.0, value=10.0, step=0.1, key=f"alloc_{i}"
        )
    with col_c:
        cliff_months = st.number_input(f"{role_name} cliff (months)", min_value=0, max_value=months, value=12, step=1, key=f"cliff_{i}")
    with col_d:
        vest_months = st.number_input(f"{role_name} vesting period (months)", min_value=1, max_value=240, value=48, step=1, key=f"vest_{i}")
    # behavioral inputs
    col_e, col_f = st.columns([2, 2])
    with col_e:
        activation = st.slider(
            f"{role_name} activation ratio (fraction of unlocked tokens that enter circulation)",
            min_value=0.0, max_value=1.0, value=0.8, step=0.01, key=f"act_{i}"
        )
    with col_f:
        gov_frac = st.slider(
            f"{role_name} governance fraction (portion of unlocked tokens that remain governance-locked)",
            min_value=0.0, max_value=1.0, value=0.0, step=0.01, key=f"gov_{i}"
        )

    # record
    roles.append(role_name)
    allocations[role_name] = alloc_pct / 100.0
    cliffs[role_name] = int(cliff_months)
    vesting_periods[role_name] = int(vest_months)
    behavior_activation[role_name] = float(activation)
    governance_fraction[role_name] = float(gov_frac)

st.markdown("---")

# -------------------------
# Simulation options
# -------------------------
st.subheader("Simulation options & assumptions")
col_a, col_b, col_c = st.columns([3, 2, 2])
with col_a:
    stochastic = st.checkbox("Add stochastic variation to monthly unlocks (simulate delays / noise)", value=False)
    if stochastic:
        noise_std_pct = st.number_input("Monthly unlock noise stdev (fraction of monthly amount)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        seed = st.number_input("Random seed (integer)", min_value=0, value=42, step=1)
with col_b:
    annual_discount_rate = st.number_input("Annual discount rate for PV calculations (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1) / 100.0
with col_c:
    clip_to_total = st.checkbox("Enforce aggregate cap <= total supply (clamp) if rounding issues occur", value=True)

# -------------------------
# Validation
# -------------------------
total_alloc = sum(allocations.values())
if total_alloc > 1.0 + 1e-12:
    st.error(f"Total allocations exceed 100% of total supply ({total_alloc*100:.2f}%). Please reduce allocations.")
    st.stop()

# -------------------------
# Run simulation
# -------------------------
if st.button("Run simulation"):
    np.random.seed(int(seed) if stochastic else None)

    # Prepare per-role monthly unlock arrays
    vesting_unlocked = {role: np.zeros(months) for role in roles}  # tokens unlocked per month (not cumulative)
    governance_locked = {role: np.zeros(months) for role in roles}  # portion of unlocked tokens that remain governance-locked
    effective_circulation_by_role = {role: np.zeros(months) for role in roles}  # active circulating tokens due to behavior_activation

    # Compute monthly unlock schedule per role with cliff truncation and optional stochasticity
    for role in roles:
        alloc_tokens = allocations[role] * total_supply
        cliff = cliffs[role]
        vesting = vesting_periods[role]

        # truncate vesting if cliff + vesting exceeds simulation horizon
        if cliff >= months:
            # entire vesting is beyond the simulation horizon -> nothing unlocks within horizon
            effective_vesting = 0
        else:
            effective_vesting = min(vesting, months - cliff)

        if effective_vesting <= 0:
            monthly_nominal = 0.0
        else:
            monthly_nominal = alloc_tokens / vesting  # monthly amount based on full vesting period (financially sensible)
            # but only effective_vesting months will actually occur inside the horizon

        for m in range(months):
            if m < cliff:
                unlock = 0.0
            elif cliff <= m < cliff + effective_vesting:
                # apply nominal monthly amount; if stochastic, add multiplicative noise but ensure non-negative
                if stochastic and monthly_nominal > 0:
                    noise = np.random.normal(loc=1.0, scale=noise_std_pct)
                    unlock = max(0.0, monthly_nominal * noise)
                else:
                    unlock = monthly_nominal
            else:
                unlock = 0.0

            vesting_unlocked[role][m] = unlock

        # Governance locked portion of the unlocked tokens (portion of each unlock that remains governance-locked)
        gov_frac = governance_fraction.get(role, 0.0)
        governance_locked[role] = vesting_unlocked[role] * gov_frac

        # Effective circulating tokens from this role (activation ratio * unlocked * (1 - governance_locked_fraction))
        activation = behavior_activation.get(role, 1.0)
        effective_circulation_by_role[role] = vesting_unlocked[role] * (1.0 - gov_frac) * activation

    # Aggregate metrics
    cumulative_unlocked_per_role = {role: np.cumsum(vesting_unlocked[role]) for role in roles}   # cumulative unlocked tokens
    cumulative_governance_locked = {role: np.cumsum(governance_locked[role]) for role in roles}
    cumulative_effective_circ = {role: np.cumsum(effective_circulation_by_role[role]) for role in roles}

    # Correct aggregation: sum cumulative unlocked once across roles
    total_cumulative_unlocked = np.sum(np.vstack([cumulative_unlocked_per_role[r] for r in roles]), axis=0)
    total_cumulative_effective_circ = np.sum(np.vstack([cumulative_effective_circ[r] for r in roles]), axis=0)
    total_cumulative_gov_locked = np.sum(np.vstack([cumulative_governance_locked[r] for r in roles]), axis=0)

    # Enforce supply cap if requested
    if clip_to_total:
        total_cumulative_unlocked = np.minimum(total_cumulative_unlocked, float(total_supply))
        total_cumulative_effective_circ = np.minimum(total_cumulative_effective_circ, float(total_supply))
        # governance locked is a subset of unlocked; clamp consistent with unlocked
        total_cumulative_gov_locked = np.minimum(total_cumulative_gov_locked, total_cumulative_unlocked)

    # inflation rate (monthly) = change in effective circulating supply / total supply
    monthly_inflation = np.zeros(months)
    monthly_inflation[0] = total_cumulative_effective_circ[0] / total_supply
    monthly_inflation[1:] = (total_cumulative_effective_circ[1:] - total_cumulative_effective_circ[:-1]) / total_supply

    # present value calculation (discounting each month's new unlocked tokens)
    monthly_discount_rate = (1 + annual_discount_rate) ** (1/12) - 1 if annual_discount_rate > 0 else 0.0
    monthly_new_unlocked = np.zeros(months)
    monthly_new_unlocked[0] = total_cumulative_unlocked[0]
    monthly_new_unlocked[1:] = total_cumulative_unlocked[1:] - total_cumulative_unlocked[:-1]
    discount_factors = 1 / ((1 + monthly_discount_rate) ** np.arange(months))
    pv_unlocked = np.sum(monthly_new_unlocked * discount_factors)

    # Save full dataset to session_state for later inspection/export
    profile_data = {
        "params": {
            "total_supply": total_supply,
            "months": months,
            "annual_discount_rate": annual_discount_rate,
            "stochastic": stochastic,
            "noise_std_pct": noise_std_pct if stochastic else 0.0,
            "seed": int(seed) if stochastic else None,
            "clip_to_total": clip_to_total
        },
        "roles": deepcopy(roles),
        "allocations": deepcopy(allocations),
        "cliffs": deepcopy(cliffs),
        "vesting_periods": deepcopy(vesting_periods),
        "behavior_activation": deepcopy(behavior_activation),
        "governance_fraction": deepcopy(governance_fraction),
        "vesting_unlocked": {r: vesting_unlocked[r].tolist() for r in roles},
        "cumulative_unlocked": {r: cumulative_unlocked_per_role[r].tolist() for r in roles},
        "effective_circulation_by_role": {r: cumulative_effective_circ[r].tolist() for r in roles},
        "total_cumulative_unlocked": total_cumulative_unlocked.tolist(),
        "total_effective_circulating": total_cumulative_effective_circ.tolist(),
        "total_governance_locked": total_cumulative_gov_locked.tolist(),
        "monthly_inflation": monthly_inflation.tolist(),
        "pv_unlocked": float(pv_unlocked)
    }

    if "profiles" not in st.session_state:
        st.session_state["profiles"] = {}
    st.session_state["profiles"][profile_name] = profile_data

    # -------------------------
    # Plotting
    # -------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax = axes[0]
    # plot cumulative unlocked by role (stacked)
    cum_matrix = np.vstack([cumulative_unlocked_per_role[r] for r in roles])
    ax.stackplot(np.arange(months), cum_matrix, labels=roles)
    ax.set_ylabel("Cumulative unlocked tokens")
    ax.set_title(f"Vesting — cumulative unlocked (profile: {profile_name})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    ax2 = axes[1]
    ax2.plot(np.arange(months), total_cumulative_effective_circ, label="Effective circulating supply (cum)")
    ax2.plot(np.arange(months), total_cumulative_gov_locked, label="Governance locked (cum)")
    ax2.set_xlabel("Months")
    ax2.set_ylabel("Tokens")
    ax2.set_title("Effective circulating vs governance-locked")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    st.pyplot(fig)

    # inflation plot
    fig2, ax3 = plt.subplots(figsize=(10, 3))
    ax3.bar(np.arange(months), monthly_inflation)
    ax3.set_xlabel("Months")
    ax3.set_ylabel("Monthly inflation (Δ circ / total supply)")
    ax3.set_title("Monthly inflation rate from vesting (effective circulation)")
    st.pyplot(fig2)

    # summary metrics
    st.subheader("Summary metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final cumulative unlocked", f"{int(total_cumulative_unlocked[-1]):,}")
    with col2:
        st.metric("Final effective circulating", f"{int(total_cumulative_effective_circ[-1]):,}")
    with col3:
        st.metric("PV of unlocked (tokens)", f"{int(pv_unlocked):,}")

    st.markdown("**Notes / assumptions:**")
    st.write("""
    - Monthly nominal unlock = allocation_tokens / vesting_period (but only occurs inside the simulation horizon).
    - 'Activation ratio' models the fraction of unlocked tokens that enter economic circulation (e.g., immediate sell/transfer behavior).
    - 'Governance fraction' models portion of unlocked tokens that remain non-economic but may retain voting power.
    - If stochastic variation is enabled, monthly unlocks receive multiplicative noise (sampled from a normal distribution centred at 1.0). Noise is truncated to avoid negative unlocks.
    - Present value uses an annual discount rate supplied above, converted to monthly.
    """)

    st.success("Simulation complete and saved to session state.")

# -------------------------
# Saved profiles display & export
# -------------------------
if "profiles" in st.session_state and st.session_state["profiles"]:
    st.subheader("Saved Profiles")
    for pname, pdata in st.session_state["profiles"].items():
        col1, col2, col3 = st.columns([3, 1, 1])
        col1.write(f"**{pname}** — final effective circ: {int(pdata['total_effective_circulating'][-1]):,}, PV unlocked: {int(pdata['pv_unlocked']):,}")
        if col2.button("Show data (table)", key=f"show_{pname}"):
            df = pd.DataFrame({
                "month": np.arange(pdata["params"]["months"]),
                "total_cumulative_unlocked": pdata["total_cumulative_unlocked"],
                "total_effective_circulating": pdata["total_effective_circulating"],
                "total_governance_locked": pdata["total_governance_locked"],
                "monthly_inflation": pdata["monthly_inflation"]
            })
            st.dataframe(df)
        if col3.download_button("Download JSON", data=pd.io.json.dumps(pdata, indent=2), file_name=f"{pname}_vesting_profile.json"):
            pass



