from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Data loading with caching for performance ----------------------------

@st.cache_data
def load_chain_wide():
    base = Path(__file__).parent
    for sub in ("data", "."):
        files = sorted((base / sub).glob("master_chain_metrics*.csv"))
        if files:
            df = pd.read_csv(files[0], parse_dates=[c for c in ("Date","date") if c in pd.read_csv(files[0], nrows=0)])
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={"date":"date"})
            return df
    return pd.DataFrame()

@st.cache_data
def load_regulatory():
    base = Path(__file__).parent
    for sub in ("data", "."):
        p = base / sub / "regulatory_milestones.csv"
        if p.exists():
            return pd.read_csv(p, parse_dates=["Date"])
    return pd.DataFrame()

@st.cache_data
def load_benchmarks():
    # Hardcoded benchmarks for Comparison tab
    data = {
        "benchmark": ["visa_avg_tps","visa_peak_tps","mc_avg_tps","mc_peak_tps",
                      "swift_settlement_days","dtcc_tplus1_adoption_pct","t2s_settlement_days"],
        "value": [1700,65000,5000,59000,1.25,95,0.10]
    }
    df = pd.DataFrame(data)
    df["label"] = df["benchmark"].map({
        "visa_avg_tps":"Visa Avg TPS","visa_peak_tps":"Visa Peak TPS",
        "mc_avg_tps":"Mastercard Avg TPS","mc_peak_tps":"Mastercard Peak TPS",
        "swift_settlement_days":"SWIFT gpi Avg Settlement (days)",
        "dtcc_tplus1_adoption_pct":"DTCC T+1 Adoption (%)",
        "t2s_settlement_days":"ECB T2S Avg Settlement (days)"
    })
    return df

# --- App config ------------------------------------------------------------
st.set_page_config(page_title="Blockchain & Finance Dashboard", layout="wide")

# --- Sidebar -------------------------------------------------------------
st.sidebar.title("Navigation")
tabs = ["Research Objectives","Regulatory Decisions","Data Overview","Insights","Comparison"]
page = st.sidebar.radio("Go to", tabs)

# Load data
df_wide = load_chain_wide()
reg_df = load_regulatory()
bench_df = load_benchmarks()

# Prepare long form for Insights & Comparison
if not df_wide.empty:
    long = df_wide.melt(id_vars=["date"], var_name="metric_chain", value_name="value")
    split = long["metric_chain"].str.rsplit("_",1,expand=True)
    long["metric"] = split[0]
    long["chain"]  = split[1]
    chain_df = long.dropna(subset=["value"])
else:
    chain_df = pd.DataFrame()

# --- Pages ---------------------------------------------------------------

if page=="Research Objectives":
    st.title("📋 Research Objectives")
    st.markdown("""
1. **Core Concepts:** Decentralization, consensus, smart contracts.  
2. **Applications:** DeFi, trade finance, CBDCs, tokenization.  
3. **Benefits:** Cost savings, efficiency, transparency vs legacy.  
4. **Challenges:** Regulatory, security, scalability, interoperability.  
5. **Adoption Trends:** On-chain usage, institutional uptake (2016–2025).
    """)

elif page=="Regulatory Decisions":
    st.title("⚖️ Regulatory & Institutional Milestones")
    if not reg_df.empty:
        st.dataframe(reg_df.sort_values("Date"), use_container_width=True)
    else:
        st.warning("No regulatory data available.")

elif page=="Data Overview":
    st.title("🔍 Data Overview")
    if df_wide.empty:
        st.warning("No time-series data loaded.")
    else:
        st.write(f"Dataset covers **{df_wide['date'].nunique()} days** × **{len(df_wide.columns)-1} metrics**")
        st.dataframe(df_wide.head(10), use_container_width=True)
        with st.expander("All Columns"):
            st.write(df_wide.columns.tolist())

elif page=="Insights":
    st.title("💡 Dynamic Insights")
    if chain_df.empty:
        st.warning("No data to show.")
    else:
        col1, col2 = st.columns(2)
        chains = sorted(chain_df["chain"].unique())
        with col1:
            chain_sel = st.selectbox("Chain", chains, index=0)
        metrics = sorted(chain_df["metric"].unique())
        with col2:
            metric_sel = st.selectbox("Metric", metrics, index=metrics.index(metrics[0]))
        dfc = chain_df[(chain_df.chain==chain_sel)&(chain_df.metric==metric_sel)]
        dfc = dfc.set_index("date").sort_index()
        # KPI
        latest = dfc["value"].iat[-1]
        st.metric(f"Latest {metric_sel.replace('_',' ').title()} ({chain_sel.title()})", f"{latest:.4f}")
        # Chart
        fig = px.line(dfc, x=dfc.index, y="value", title=f"{metric_sel.replace('_',' ').title()} over Time")
        fig.update_layout(margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)
        # Basic insight
        pct_change = (dfc["value"].iloc[-1]/dfc["value"].iloc[0]-1)*100
        st.markdown(f"**Insight:** `{chain_sel.title()}`’s **{metric_sel.replace('_',' ').title()}** changed **{pct_change:.1f}%** since {dfc.index[0].date()}.")

elif page=="Comparison":
    st.title("⚔️ Chain & Legacy Comparison")
    if chain_df.empty:
        st.warning("No data to compare.")
    else:
        st.markdown("**Select a metric and date range. Toggle benchmarks.**")
        metrics = sorted(chain_df["metric"].unique())
        metric_sel = st.selectbox("Metric", metrics, index=0)
        # date slider
        dmin, dmax = chain_df.date.min(), chain_df.date.max()
        start, end = st.slider("Date Range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
        # chart type
        chart_type = st.selectbox("Chart Type", ["Line","Area","Bar"])
        # include benchmarks?
        include_bench = st.checkbox("Show Traditional Finance Benchmarks", value=True)
        # prepare data
        dfc = chain_df[(chain_df.metric==metric_sel)&(chain_df.date.between(start,end))]
        pivot = dfc.pivot(index="date", columns="chain", values="value")
        # build figure
        if chart_type=="Line":
            fig = go.Figure()
            for chain in pivot.columns:
                fig.add_trace(go.Scatter(x=pivot.index, y=pivot[chain], mode="lines", name=chain.title()))
            if include_bench:
                bench = bench_df[bench_df.benchmark==metric_sel]
                for _,row in bench.iterrows():
                    fig.add_hline(y=row.value, line_dash="dash", annotation_text=row.label, annotation_position="top left")
        elif chart_type=="Area":
            fig = px.area(pivot, x=pivot.index, y=pivot.columns, labels={"value":metric_sel,"date":"Date"})
        else:  # Bar
            monthly = pivot.resample("M").mean().reset_index().melt(id_vars="date", var_name="chain", value_name="value")
            fig = px.bar(monthly, x="date", y="value", color="chain", barmode="group")
        fig.update_layout(title=f"{metric_sel.replace('_',' ').title()} Comparison", margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)
        # dynamic insights
        st.markdown("### Insights from Comparison")
        summary = pivot.loc[start:end].agg(["first","last"]).T
        summary["pct_change"] = (summary["last"]/summary["first"]-1)*100
        st.dataframe(summary[["first","last","pct_change"]].rename(columns={
            "first":"Start","last":"End","pct_change":"% Change"
        }), use_container_width=True)
