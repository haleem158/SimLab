import streamlit as st
from PIL import Image

# Set page config
st.set_page_config(
    page_title="SimLab — Tokenomics Simulation Suite",
    page_icon="🧪",
    layout="wide"
)

# Header layout
col1, col2 = st.columns([1, 3])
with col1:
    st.image("simlab_logo.jpg", width=120)
with col2:
    st.title("SimLab — Tokenomics Simulation Suite")
    st.markdown("### Understand, Design, and Simulate Token Economies with Ease ⚙️")

st.markdown("---")

# Intro
st.markdown("""
Welcome to **SimLab**, your interactive lab for exploring **tokenomics** —  
the economic mechanisms that govern blockchain tokens.

This app lets you simulate **token supply, vesting, and price dynamics**  
using transparent, data-driven models.

---
### What You Can Do in SimLab
| Simulation Type | Description |
|------------------|-------------|
| **Token Supply Simulator** | Explore how inflation, staking, and burning affect token supply over time. |
|  **Vesting Schedule Simulator** | Visualize how team/investor tokens unlock across different schedules. |
| **Token Price Impact Model** | Estimate how demand and staking adoption might influence price. |

---

###  How to Get Started
1. Use the **sidebar navigation** to choose a simulator.
2. Adjust the parameters in the sidebar to match your assumptions.
3. Click **Run Simulation** to generate dynamic charts and insights.
4. Download results as CSV for further analysis.

---

###  Tips for Beginners
- Hover over each sidebar label — many include tooltips that explain what they mean.
- Start with smaller **years** or **lower supply** to run simulations faster.
- Each simulation is independent, so feel free to experiment.

---

### Recommended Learning Path
1. **Start with the Token Supply Simulator** — understand inflation, staking, and burn.
2. Then go to **Vesting Schedule Simulator** — see how locked tokens release over time.
3. Finally, explore **Price Impact Model** — link activity and demand to market effects.

---

###  About SimLab
SimLab is designed for **token economists, analysts, and builders** who want to:
- Validate token models before launch  
- Understand long-term emission patterns  
- Simulate staking, burning, and demand dynamics  

---

### 🔗 Continue
Use the sidebar or click below to open a simulator:
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.page_link("pages/Token_Supply_Simulator.py", label="Token Supply Simulator", icon="🔢")
with col2:
    st.page_link("pages/Vesting_Schedule_Simulator.py", label="Vesting Schedule Simulator", icon="📆")
with col3:
    st.page_link("pages/Token_Price_Impact_Model.py", label="Token Price Impact Model", icon="📊")

st.markdown("---")
st.caption("Developed by SimLab • Empowering data-driven token design.")