import os
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

# --- Data loading functions -----------------------------------------------

@st.cache_data
def load_chain_data():
    data_dir = Path(__file__).parent / "data"
    # find the first CSV matching our master chain metrics
    csvs = sorted(data_dir.glob("master_chain_metrics*.csv"))
    if not csvs:
        st.error(f"No master_chain_metrics CSV found in {data_dir}")
        return pd.DataFrame()
    df = pd.read_csv(csvs[0], parse_dates=["Date"])
    # standardize column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    return df

@st.cache_data
def load_regulatory_data():
    data_dir = Path(__file__).parent / "data"
    path = data_dir / "regulatory_milestones.csv"
    if not path.exists():
        st.error(f"No regulatory_milestones.csv found in {data_dir}")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["Date"])
    return df

# --- App configuration -----------------------------------------------------

st.set_page_config(
    page_title="Blockchain Insights Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar navigation ----------------------------------------------------

st.sidebar.title("Navigation")
pages = [
    "Research Objectives",
    "Regulatory Decisions",
    "Data Overview",
    "Insights",
    "Comparison",
]
page = st.sidebar.radio("Go to", pages)

# Load data
chain_df = load_chain_data()
reg_df = load_regulatory_data()

# --- Page: Research Objectives --------------------------------------------

if page == "Research Objectives":
    st.title("📋 Research Objectives")
    st.markdown(
        """
1. **Core Concepts:** Explain blockchain fundamentals (decentralization, consensus, smart contracts).  
2. **Applications:** Document current & future use-cases in finance (DeFi, trade finance, CBDC).  
3. **Benefits:** Quantify cost, efficiency, transparency gains vs. legacy.  
4. **Challenges:** Examine regulatory, security, scalability obstacles.  
5. **Adoption Trends:** Analyze on-chain adoption & institutional uptake (2016–2025).
        """
    )

# --- Page: Regulatory Decisions -------------------------------------------

elif page == "Regulatory Decisions":
    st.title("⚖️ Regulatory & Institutional Milestones")
    st.dataframe(
        reg_df.sort_values("Date").reset_index(drop=True),
        use_container_width=True
    )

# --- Page: Data Overview ---------------------------------------------------

elif page == "Data Overview":
    st.title("🔍 Data Overview")
    st.write(f"Data covers {chain_df['date'].nunique()} days × {chain_df['chain'].nunique()} chains.")
    st.dataframe(chain_df.head(10), use_container_width=True)
    with st.expander("Show all columns"):
        st.write(list(chain_df.columns))

# --- Page: Insights --------------------------------------------------------

elif page == "Insights":
    st.title("💡 Dynamic Insights")
    # Select chain and metric
    chains = sorted(chain_df["chain"].unique())
    chain_sel = st.selectbox("Select Chain", chains, index=chains.index("bitcoin"))
    metric_cols = [c for c in chain_df.columns if c not in ["date", "chain"]]
    metric_sel = st.selectbox("Select Metric", metric_cols)

    df_chain = chain_df[chain_df["chain"] == chain_sel]
    latest = df_chain.iloc[-1][metric_sel]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(
            label=f"Latest {metric_sel.replace('_', ' ').title()} ({chain_sel.title()})",
            value=f"{latest:.4f}"
        )
    with col2:
        fig = px.line(
            df_chain, x="date", y=metric_sel,
            title=f"{metric_sel.replace('_', ' ').title()} Over Time — {chain_sel.title()}",
            labels={"date": "Date", metric_sel: metric_sel.replace("_", " ").title()}
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Page: Comparison ------------------------------------------------------

elif page == "Comparison":
    st.title("⚔️ Chain Comparison")
    metric_cols = [c for c in chain_df.columns if c not in ["date", "chain"]]
    metric_sel = st.selectbox("Metric to Compare", metric_cols)

    pivot = chain_df.pivot(index="date", columns="chain", values=metric_sel)
    fig = px.line(
        pivot,
        labels={"value": metric_sel.replace("_", " ").title(), "date": "Date"},
        title=f"Comparison of {metric_sel.replace('_', ' ').title()}"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- End of app ------------------------------------------------------------

