import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("🧮 Token Price Impact Model")

st.markdown("""
Welcome to the **Token Price Impact Model**.  
To begin, please upload a CSV file containing the following columns:

- `Month` – e.g. "Jan 2025", "Feb 2025", etc.
- `Unlocked Tokens` – Number of tokens unlocked that month.
- `Emission` – Additional tokens emitted (e.g., from rewards or incentives).
- `Transaction Volume Growth (%)` – Monthly growth rate of transaction volume.
- `Staking Participation Growth (%)` – Monthly growth in staking activity.

We will compute:
- **Circulating Supply** = cumulative `Unlocked Tokens` + `Emission`
- **Demand Index** = average of the two growth rates
- **Indicative Price** = Demand Index ÷ Circulating Supply

👉 You’ll see a line chart with supply, demand, and price over time.
""")

# Sample table for user guidance
st.subheader("📄 Sample CSV Format:")
sample_data = pd.DataFrame({
    "Month": ["Jan 2025", "Feb 2025", "Mar 2025"],
    "Unlocked Tokens": [100000, 150000, 130000],
    "Emission": [50000, 60000, 55000],
    "Transaction Volume Growth (%)": [10, 15, 12],
    "Staking Participation Growth (%)": [5, 8, 10],
})
st.dataframe(sample_data)

# File upload
st.subheader("📤 Upload Your CSV:")
uploaded_file = st.file_uploader("Upload a CSV file with the columns above", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Computations
    df['Circulating Supply'] = df['Unlocked Tokens'].cumsum() + df['Emission'].cumsum()
    df['Demand Index'] = (df['Transaction Volume Growth (%)'] + df['Staking Participation Growth (%)']) / 2
    df['Indicative Price'] = df['Demand Index'] / df['Circulating Supply']

    # Plotting
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df['Month'], df['Circulating Supply'], label='Circulating Supply')
    ax.plot(df['Month'], df['Demand Index'], label='Demand Index')
    ax.plot(df['Month'], df['Indicative Price'], label='Indicative Price')

    ax.set_xlabel('Month')
    ax.set_title('Token Supply, Demand & Indicative Price Over Time')
    ax.legend()
    st.pyplot(fig)

    st.subheader("📊 Processed Data:")
    st.dataframe(df)

else:
    st.info("Waiting for a CSV file to be uploaded...")
