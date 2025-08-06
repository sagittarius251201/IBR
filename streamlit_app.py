from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Data loading with caching --------------------------------------------

@st.cache_data
def load_chain_wide():
    base = Path(__file__).parent
    for sub in ("data", "."):
        data_dir = base / sub
        files = sorted(data_dir.glob("master_chain_metrics*.csv"))
        if files:
            sample = pd.read_csv(files[0], nrows=0)
            date_cols = [c for c in ("Date","date") if c in sample.columns]
            df = pd.read_csv(files[0], parse_dates=date_cols)
            df.columns = [c.lower() for c in df.columns]
            if "date" not in df.columns and "Date" in df.columns:
                df = df.rename(columns={"Date":"date"})
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

# --- Sidebar --------------------------------------------------------------
st.sidebar.title("Navigation")
tabs = ["Research Objectives","Regulatory Decisions","Data Overview","Insights","Comparison"]
page = st.sidebar.radio("Go to", tabs)

# --- Load data ------------------------------------------------------------
df_wide = load_chain_wide()
reg_df = load_regulatory()
bench_df = load_benchmarks()

# --- Prepare long-form for interactive pages -----------------------------
if not df_wide.empty and "date" in df_wide.columns:
    long = df_wide.melt(id_vars=["date"], var_name="metric_chain", value_name="value")
    long["metric_chain"] = long["metric_chain"].astype(str)
    split = long["metric_chain"].str.rsplit("_", n=1, expand=True)
    long["metric"] = split[0]
    long["chain"]  = split[1]
    chain_df = long.dropna(subset=["value"])
else:
    chain_df = pd.DataFrame()

# --- Pages ---------------------------------------------------------------

if page == "Research Objectives":
    st.title("📋 Research Objectives")
    st.markdown("""
1. **Core Concepts:** Decentralization, consensus, smart contracts.  
2. **Applications:** DeFi, trade finance, CBDCs, tokenization.  
3. **Benefits:** Cost savings, efficiency, transparency vs legacy.  
4. **Challenges:** Regulatory, security, scalability, interoperability.  
5. **Adoption Trends:** On-chain usage & institutional uptake (2016–2025).
""")

elif page == "Regulatory Decisions":
    st.title("⚖️ Regulatory & Institutional Milestones")
    if not reg_df.empty:
        st.dataframe(reg_df.sort_values("Date"), use_container_width=True)
    else:
        st.warning("No regulatory data available.")

elif page == "Data Overview":
    st.title("🔍 Data Overview")
    if df_wide.empty or "date" not in df_wide.columns:
        st.warning("No chain data loaded.")
    else:
        days = df_wide["date"].nunique()
        metrics = len(df_wide.columns) - 1
        st.write(f"Data covers **{days} days** × **{metrics} metrics**")
        st.dataframe(df_wide.head(10), use_container_width=True)
        with st.expander("All Columns"):
            st.write(df_wide.columns.tolist())

elif page == "Insights":
    st.title("💡 Dynamic Insights")
    if chain_df.empty or "chain" not in chain_df.columns:
        st.warning("No data to show.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            chains = sorted(chain_df["chain"].unique())
            chain_sel = st.selectbox("Chain", chains)
        with col2:
            metrics = sorted(chain_df["metric"].unique())
            metric_sel = st.selectbox("Metric", metrics)
        dfc = chain_df[(chain_df.chain==chain_sel)&(chain_df.metric==metric_sel)]
        if dfc.empty:
            st.warning("No data for this selection.")
        else:
            dfc = dfc.set_index("date").sort_index()
            latest = dfc["value"].iat[-1]
            st.metric(f"{metric_sel.replace('_',' ').title()} ({chain_sel.title()})", f"{latest:.4f}")
            fig = px.line(
                dfc, y="value",
                title=f"{metric_sel.replace('_',' ').title()} — {chain_sel.title()}",
                labels={"value":metric_sel.replace("_"," ").title(), "date":"Date"}
            )
            st.plotly_chart(fig, use_container_width=True)
            pct = (dfc["value"].iloc[-1]/dfc["value"].iloc[0] - 1)*100
            st.markdown(f"**Insight:** {chain_sel.title()}'s **{metric_sel.replace('_',' ').title()}** changed **{pct:.1f}%** since {dfc.index[0].date()}.")

elif page == "Comparison":
    st.title("⚔️ Chain & Legacy Comparison")
    if chain_df.empty or "chain" not in chain_df.columns:
        st.warning("No data to compare.")
    else:
        st.markdown("**Select metric, date range, chart type, and toggle benchmarks.**")
        metrics = sorted(chain_df["metric"].unique())
        metric_sel = st.selectbox("Metric", metrics)
        dmin, dmax = chain_df["date"].min(), chain_df["date"].max()
        start, end = st.slider("Date Range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
        chart_type = st.selectbox("Chart Type", ["Line","Area","Bar"])
        show_bench = st.checkbox("Show Traditional Benchmarks", value=True)

        dfc = chain_df[(chain_df.metric==metric_sel)&(chain_df.date.between(start, end))]
        pivot = dfc.pivot(index="date", columns="chain", values="value")

        # Build figure
        if chart_type == "Line":
            fig = go.Figure()
            for c in pivot.columns:
                fig.add_trace(go.Scatter(x=pivot.index, y=pivot[c], mode="lines", name=c.title()))
            if show_bench and metric_sel in bench_df["benchmark"].values:
                for _, row in bench_df[bench_df["benchmark"]==metric_sel].iterrows():
                    fig.add_hline(y=row.value, line_dash="dash", annotation_text=row.label, annotation_position="top left")
        elif chart_type == "Area":
            fig = px.area(pivot, x=pivot.index, y=pivot.columns)
        else:  # Bar chart aggregated monthly
            monthly = pivot.resample("M").mean().reset_index().melt(id_vars="date", var_name="chain", value_name="value")
            fig = px.bar(monthly, x="date", y="value", color="chain", barmode="group")

        fig.update_layout(title=metric_sel.replace("_"," ").title(), margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Insights table
        summary = pivot.loc[start:end].agg(["first","last"]).T
        summary["% Change"] = (summary["last"]/summary["first"] - 1)*100
        st.dataframe(
            summary.rename(columns={"first":"Start","last":"End"})[["Start","End","% Change"]],
            use_container_width=True
        )
