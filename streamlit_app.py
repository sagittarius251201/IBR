from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# --- Data loading functions -----------------------------------------------

@st.cache_data
def load_chain_data():
    base = Path(__file__).parent
    # locate CSV
    for sub in ("data", "."):
        data_dir = base / sub
        files = sorted(data_dir.glob("master_chain_metrics*.csv"))
        if files:
            df = pd.read_csv(files[0], parse_dates=[col for col in ("Date","date") if col in pd.read_csv(files[0], nrows=0)])
            df.columns = [c.lower() for c in df.columns]
            return df
    st.error("No master_chain_metrics CSV found")
    return pd.DataFrame()

@st.cache_data
def load_regulatory_data():
    base = Path(__file__).parent
    for sub in ("data", "."):
        path = base / sub / "regulatory_milestones.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["Date"])
            return df
    st.error("No regulatory_milestones.csv found")
    return pd.DataFrame()

# --- App config ------------------------------------------------------------
st.set_page_config(page_title="Blockchain Insights", layout="wide")

# Sidebar
st.sidebar.title("Navigation")
pages = ["Research Objectives","Regulatory Decisions","Data Overview","Insights","Comparison"]
page = st.sidebar.radio("Go to", pages)

# Load data
df_wide = load_chain_data()     # wide format: one row per date, many metric_chain columns
reg_df = load_regulatory_data()

# Build long-format chain_df for interactive pages
if not df_wide.empty:
    # melt to metric_chain,value
    long = df_wide.melt(id_vars=["date"], var_name="metric_chain", value_name="value")
    # split metric_chain into metric and chain
    split = long["metric_chain"].str.rsplit("_", n=1, expand=True)
    long["metric"] = split[0]
    long["chain"] = split[1]
    chain_df = long.pivot_table(index=["date","chain"], columns="metric", values="value").reset_index()
else:
    chain_df = pd.DataFrame()

# --- Pages ---------------------------------------------------------------

if page=="Research Objectives":
    st.title("📋 Research Objectives")
    st.markdown("""
1. **Core Concepts:** Explain blockchain fundamentals (decentralization, consensus, smart contracts).  
2. **Applications:** Document current & future use-cases in finance (DeFi, trade finance, CBDC).  
3. **Benefits:** Quantify cost, efficiency, transparency gains vs. legacy.  
4. **Challenges:** Examine regulatory, security, scalability obstacles.  
5. **Adoption Trends:** Analyze on-chain adoption & institutional uptake (2016–2025).  
""")

elif page=="Regulatory Decisions":
    st.title("⚖️ Regulatory & Institutional Milestones")
    if not reg_df.empty:
        st.dataframe(reg_df.sort_values("Date").reset_index(drop=True), use_container_width=True)
    else:
        st.warning("No regulatory data loaded.")

elif page=="Data Overview":
    st.title("🔍 Data Overview")
    if df_wide.empty:
        st.warning("No chain data loaded.")
    else:
        st.write(f"Data shape: {df_wide.shape[0]} days × {df_wide.shape[1]-1} metrics")
        st.dataframe(df_wide.head(10), use_container_width=True)
        with st.expander("Show all columns"):
            st.write(list(df_wide.columns))

elif page=="Insights":
    st.title("💡 Dynamic Insights")
    if chain_df.empty:
        st.warning("No data to display.")
    else:
        chains = sorted(chain_df["chain"].unique())
        chain = st.selectbox("Select Chain", chains)
        metrics = [c for c in chain_df.columns if c not in ["date","chain"]]
        metric = st.selectbox("Select Metric", metrics)
        dfc = chain_df[chain_df["chain"]==chain]
        if dfc.empty or metric not in dfc:
            st.warning("No data for selection.")
        else:
            latest = dfc.iloc[-1][metric]
            col1,col2 = st.columns([1,2])
            with col1:
                st.metric(label=f"{metric.replace('_',' ').title()} ({chain.title()})", value=f"{latest:.4f}")
            with col2:
                fig = px.line(dfc, x="date", y=metric,
                              title=f"{metric.replace('_',' ').title()} — {chain.title()}",
                              labels={"date":"Date", metric:metric.replace('_',' ').title()})
                st.plotly_chart(fig, use_container_width=True)

elif page=="Comparison":
    st.title("⚔️ Chain Comparison")
    if chain_df.empty:
        st.warning("No data to compare.")
    else:
        metrics = [c for c in chain_df.columns if c not in ["date","chain"]]
        metric = st.selectbox("Metric to Compare", metrics)
        pivot = chain_df.pivot(index="date", columns="chain", values=metric)
        fig = px.line(pivot, labels={"value":metric.replace('_',' ').title(),"date":"Date"},
                      title=f"Comparison: {metric.replace('_',' ').title()}")
        st.plotly_chart(fig, use_container_width=True)

