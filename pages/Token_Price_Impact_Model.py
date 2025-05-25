import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Token Price Impact Model")

# CSV Upload
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Calculate cumulative circulating supply
    df['Circulating Supply'] = df['Unlocked Tokens'].cumsum() + df['Emission'].cumsum()

    # Calculate Demand Index as simple average (can adjust weights)
    df['Demand Index'] = (df['Transaction Volume Growth (%)'] + df['Staking Participation Growth (%)']) / 2

    # Calculate Indicative Price
    df['Indicative Price'] = df['Demand Index'] / df['Circulating Supply']

    # Plot results
    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(df['Month'], df['Circulating Supply'], label='Circulating Supply')
    ax.plot(df['Month'], df['Demand Index'], label='Demand Index')
    ax.plot(df['Month'], df['Indicative Price'], label='Indicative Price')

    ax.set_xlabel('Month')
    ax.set_title('Token Supply, Demand & Indicative Price Over Time')
    ax.legend()
    st.pyplot(fig)

    st.write("Raw Data", df)

else:
    st.write("Please upload a CSV file to begin.")