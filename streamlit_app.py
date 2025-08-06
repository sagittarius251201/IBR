from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# --- Data loading functions -----------------------------------------------

@st.cache_data
def load_chain_data():
    # Try data/ folder first, then repo root
    base = Path(__file__).parent
    for sub in ["data", "."]:
        data_dir = base / sub
        csvs = sorted(data_dir.glob("master_chain_metrics*.csv"))
        if csvs:
            df = pd.read_csv(csvs[0], parse_dates=["Date"])
            df.columns = [c.lower() for c in df.columns]
            return df
    st.error("No `master_chain_metrics*.csv` found in data/ or repo root.")
    return pd.DataFrame()

@st.cache_data
def load_regulatory_data():
    base = Path(__file__).parent
    for sub in ["data", "."]:
        path = base / sub / "regulatory_milestones.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["Date"])
            return df
    st.error("No `regulatory_milestones.csv` found in data/ or repo root.")
    return pd.DataFrame()

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
    if not chain_df.empty:
        days = chain_df['date'].nunique() if 'date' in chain_df else len(chain_df)
        chains = chain_df['chain'].nunique() if 'chain' in chain_df else "?"
        st.write(f"Data covers {days} days × {chains} chains.")
        st.dataframe(chain_df.head(10), use_container_width=True)
        with st.expander("Show all columns"):
            st.write(list(chain_df.columns))
    else:
        st.warning("No chain data loaded.")

# --- Page: Insights --------------------------------------------------------

elif page == "Insights":
    st.title("💡 Dynamic Insights")
    if chain_df.empty:
        st.warning("No chain data to show.")
    else:
        chains = sorted(chain_df["chain"].unique())
        chain_sel = st.selectbox("Select Chain", chains, index=0)
        metrics = [c for c in chain_df.columns if c not in ["date", "chain"]]
        metric_sel = st.selectbox("Select Metric", metrics)
        dfc = chain_df[chain_df["chain"] == chain_sel]
        if dfc.empty or metric_sel not in dfc:
            st.warning("No data for this selection.")
        else:
            latest = dfc.iloc[-1][metric_sel]
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(
                    label=f"{metric_sel.replace('_',' ').title()} ({chain_sel.title()})",
                    value=f"{latest:.4f}"
                )
            with col2:
                fig = px.line(
                    dfc, x="date", y=metric_sel,
                    title=f"{metric_sel.replace('_',' ').title()} over Time — {chain_sel.title()}",
                    labels={"date": "Date", metric_sel: metric_sel.replace("_", " ").title()}
                )
                st.plotly_chart(fig, use_container_width=True)

# --- Page: Comparison ------------------------------------------------------

elif page == "Comparison":
    st.title("⚔️ Chain Comparison")
    if chain_df.empty:
        st.warning("No chain data to compare.")
    else:
        metrics = [c for c in chain_df.columns if c not in ["date","chain"]]
        metric_sel = st.selectbox("Metric to Compare", metrics, index=0)
        pivot = chain_df.pivot(index="date", columns="chain", values=metric_sel)
        fig = px.line(
            pivot,
            labels={"value": metric_sel.replace("_"," ").title(), "date": "Date"},
            title=f"Comparison of {metric_sel.replace('_',' ').title()}"
        )
        st.plotly_chart(fig, use_container_width=True)
