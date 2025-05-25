import streamlit as st
from st_pages import _get_pages_from_config

# Automatically load pages from .streamlit/pages.toml
_get_pages_from_config()

# Set page config (optional but nice)
st.set_page_config(page_title="SimLab: Tokenomics Simulation Playground", page_icon=":bar_chart:", layout="wide")

# Optional: Add logo image if you have one
st.image("simlab_logo.jpg", width=200)

# App title
st.title("Welcome to SimLab")

# App description text
st.write("""
*SimLab* is a tokenomics simulation playground for Web3 builders, analysts, and researchers.

Explore how token supply, vesting schedules, and *price-impact models* interact with your protocol’s growth assumptions.

Use our simulation modules to *stress test your token model, incentive programs, and market dynamics — all in one place*.

---

*Select a module from the sidebar to get started.*
""")

st.markdown("### Navigate to a module from the sidebar:")
st.markdown("- *Token Supply Simulator*")
st.markdown("- *Vesting Schedule Simulator*")
st.markdown("- *Token Price Impact Model*")