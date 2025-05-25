import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("SimLab - Vesting Schedule Simulator")

# Total supply input
total_supply = st.number_input("Total Token Supply", value=1_000_000_000, step=1_000_000)

# Number of months
months = st.slider("Number of Months to Simulate", min_value=12, max_value=120, value=120, step=12)

# Profile name
profile_name = st.text_input("Name this simulation profile")

roles = []
allocations = {}
cliffs = {}
vesting_periods = {}

# Max 10 roles
for i in range(1, 11):
    role = st.text_input(f"Role {i} Name", value=f"Role {i}")
    if role:
        roles.append(role)
        allocations[role] = st.slider(f"{role} Allocation (as % of total supply)", 0.0, 1.0, 0.1, 0.01)
        cliffs[role] = st.slider(f"{role} Cliff (months)", 0, 50, 12)
        vesting_periods[role] = st.slider(f"{role} Vesting Period (months)", 1, 120, 48)

if st.button("Run Simulation"):

    vesting_unlocked = {role: np.zeros(months) for role in roles}
    circulating_supply = np.zeros(months)

    for role in roles:
        allocation_tokens = allocations[role] * total_supply
        cliff = cliffs[role]
        vesting = vesting_periods[role]

        if vesting > 0:
            monthly_unlock = allocation_tokens / vesting

        for month in range(cliff, months):
            if month < cliff + vesting:
                vesting_unlocked[role][month] = monthly_unlock

        circulating_supply += np.cumsum(vesting_unlocked[role])

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 5))
    for role in roles:
        ax.plot(np.cumsum(vesting_unlocked[role]), label=role)
    ax.set_xlabel("Months")
    ax.set_ylabel("Unlocked Tokens")
    ax.set_title(f"Token Vesting Schedule - {profile_name}")
    ax.legend()

    st.pyplot(fig)

    # Save to session state (simulate profiles)
    if "profiles" not in st.session_state:
        st.session_state["profiles"] = {}

    st.session_state["profiles"][profile_name] = fig

# Option to c
if "profiles" in st.session_state and st.session_state["profiles"]:
    st.subheader("Saved Profiles")
    for prof_name, chart in st.session_state["profiles"].items():
        st.write(f"**{prof_name}**")
        st.pyplot(chart)