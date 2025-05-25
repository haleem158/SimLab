import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Streamlit page config
st.set_page_config(page_title="SimLab — Tokenomics Simulation Hub", layout="wide")

# Module Title + Intro
st.title("Token Supply Simulator")
st.write("""
Simulate how your protocol’s circulating and total token supply evolves over time 
based on emissions schedules, inflation rates, burn mechanisms, and staking assumptions.
""")

# Sidebar Input Panel
st.sidebar.header("Simulation Parameters")

total_supply = st.sidebar.number_input("Total Token Supply", value=1_000_000_000, step=10_000_000)
initial_supply = st.sidebar.number_input("Initial Circulating Supply", value=100_000_000, step=10_000_000)
years = st.sidebar.number_input("Years to Simulate", min_value=1, max_value=100, value=20)
annual_inflation_rate = st.sidebar.slider("Annual Inflation Rate (%)", 0.0, 50.0, 5.0) / 100
burn_rate = st.sidebar.slider("Annual Burn Rate (%)", 0.0, 50.0, 2.0) / 100
staking_rate = st.sidebar.slider("Staking Participation Rate (%)", 0.0, 100.0, 40.0) / 100
staking_reward_rate = st.sidebar.slider("Staking Reward Rate (%)", 0.0, 50.0, 10.0) / 100

run_simulation = st.sidebar.button("Run Simulation")

# Simulation Logic
if run_simulation:
    circulating_supply = initial_supply
    burned_supply = 0
    data = {"Year": [], "Circulating Supply": [], "Burned Supply": [], "Staked Supply": []}

    for year in range(1, years + 1):
        new_tokens = total_supply * annual_inflation_rate
        burned_tokens = circulating_supply * burn_rate
        staking_rewards = (circulating_supply * staking_rate) * staking_reward_rate

        circulating_supply += new_tokens + staking_rewards - burned_tokens
        burned_supply += burned_tokens
        staked_supply = circulating_supply * staking_rate

        data["Year"].append(year)
        data["Circulating Supply"].append(circulating_supply)
        data["Burned Supply"].append(burned_supply)
        data["Staked Supply"].append(staked_supply)

    df = pd.DataFrame(data)

    # Charts
    st.header("Token Supply Dynamics Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Year"], y=df["Circulating Supply"], mode="lines+markers", name="Circulating Supply"))
    fig.add_trace(go.Scatter(x=df["Year"], y=df["Burned Supply"], mode="lines+markers", name="Burned Supply"))
    fig.add_trace(go.Scatter(x=df["Year"], y=df["Staked Supply"], mode="lines+markers", name="Staked Supply"))

    fig.update_layout(
        title="Token Supply Simulation",
        xaxis_title="Year",
        yaxis_title="Tokens",
        template="plotly_white",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary
    st.subheader("Simulation Summary")
    st.write(f"*Final Circulating Supply:* {circulating_supply:,.0f} tokens")
    st.write(f"*Final Burned Supply:* {burned_supply:,.0f} tokens")
    st.write(f"*Final Staked Supply:* {staked_supply:,.0f} tokens")

    # Download option
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Simulation Results as CSV", csv, "token_supply_simulation.csv", "text/csv")