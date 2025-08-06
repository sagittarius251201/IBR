from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Optional imports
try:
    from statsmodels.tsa.seasonal import STL
    has_stl = True
except ModuleNotFoundError:
    has_stl = False

try:
    from prophet import Prophet
    has_prophet = True
except ModuleNotFoundError:
    has_prophet = False

# --- Caching data loads for performance -----------------------------------

@st.cache_data
def load_chain_wide():
    base = Path(__file__).parent
    for sub in ("data", "."):
        files = sorted((base / sub).glob("master_chain_metrics*.csv"))
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
        "benchmark": [
            "visa_avg_tps","visa_peak_tps","mc_avg_tps","mc_peak_tps",
            "swift_settlement_days","dtcc_tplus1_adoption_pct","t2s_settlement_days"
        ],
        "value": [1700,65000,5000,59000,1.25,95,0.10]
    }
    df = pd.DataFrame(data)
    df["label"] = df["benchmark"].map({
        "visa_avg_tps":"Visa Avg TPS",
        "visa_peak_tps":"Visa Peak TPS",
        "mc_avg_tps":"Mastercard Avg TPS",
        "mc_peak_tps":"Mastercard Peak TPS",
        "swift_settlement_days":"SWIFT gpi Avg Settlement (days)",
        "dtcc_tplus1_adoption_pct":"DTCC T+1 Adoption (%)",
        "t2s_settlement_days":"ECB T2S Avg Settlement (days)"
    })
    return df

# --- App config ------------------------------------------------------------
st.set_page_config(page_title="Blockchain & Finance Dashboard", layout="wide")
st.sidebar.title("Navigation")
tabs = [
    "Research Objectives",
    "Regulatory Timeline",
    "Data Overview",
    "Insights",
    "Risk & Correlation",
]
if has_prophet:
    tabs.append("Forecast")
tabs.append("Comparison")
page = st.sidebar.radio("Go to", tabs)

# --- Load data ------------------------------------------------------------
df_wide  = load_chain_wide()
reg_df   = load_regulatory()
bench_df = load_benchmarks()

# --- Build long chain_df ---------------------------------------------
chains = ["bitcoin","ethereum","solana"]
records = []
if not df_wide.empty and "date" in df_wide.columns:
    for chain in chains:
        for col in df_wide.columns:
            if col.endswith(f"_{chain}"):
                metric = col[: -(len(chain)+1)]
                sub = df_wide[["date",col]].dropna().rename(columns={col:"value"})
                sub["chain"], sub["metric"] = chain, metric
                records.append(sub)
    chain_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    if not chain_df.empty:
        chain_df["date"] = pd.to_datetime(chain_df["date"]).dt.date
else:
    chain_df = pd.DataFrame()

# --- Pages ---------------------------------------------------------------
if page=="Research Objectives":
    st.title("📋 Research Objectives")
    st.markdown(
        """
1. **Core Concepts:** decentralization, consensus, smart contracts  
2. **Applications:** DeFi, trade finance, CBDCs, tokenization  
3. **Benefits:** cost savings, efficiency, transparency vs legacy  
4. **Challenges:** regulatory, security, scalability, interoperability  
5. **Adoption Trends:** on-chain usage & institutional uptake (2016–2025)  
        """
    )

elif page=="Regulatory Timeline":
    st.title("🗓️ Regulatory & Institutional Timeline")
    if reg_df.empty:
        st.warning("No regulatory data.")
    else:
        fig = px.timeline(
            reg_df, x_start="Date", x_end="Date", y="Milestone",
            title="Key Regulatory Milestones"
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

elif page=="Data Overview":
    st.title("🔍 Data Overview")
    if df_wide.empty:
        st.warning("No data loaded.")
    else:
        st.write(f"**{df_wide['date'].nunique()} days × {len(df_wide.columns)-1} metrics**")
        st.dataframe(df_wide.head(10), use_container_width=True)
        with st.expander("All Columns"):
            st.write(list(df_wide.columns))

elif page=="Insights":
    st.title("💡 Dynamic Insights")
    if chain_df.empty:
        st.warning("No data.")
        st.stop()
    col1, col2 = st.columns(2)
    with col1:
        chain_sel = st.selectbox("Chain", chains)
    with col2:
        metric_sel = st.selectbox("Metric", sorted(chain_df["metric"].unique()))
    dfc = (
        chain_df[(chain_df.chain==chain_sel)&(chain_df.metric==metric_sel)]
        .set_index("date").sort_index()
    )
    if dfc.empty:
        st.warning("No data for this selection.")
        st.stop()
    st.metric(f"{metric_sel.title()} ({chain_sel.title()})", f"{dfc['value'].iat[-1]:.4f}")
    fig = px.line(
        dfc, y="value", title=f"{metric_sel.title()} over time",
        labels={"value": metric_sel.title(), "date":"Date"}
    )
    st.plotly_chart(fig, use_container_width=True)
    pct = (dfc["value"].iloc[-1]/dfc["value"].iloc[0]-1)*100
    st.markdown(
        f"**Insight:** {chain_sel.title()}'s **{metric_sel.title()}** changed **{pct:.1f}%** "
        f"since {dfc.index[0]}."
    )

elif page=="Risk & Correlation":
    st.title("📈 Volatility & Correlation")
    if "price_bitcoin" in df_wide.columns:
        price = df_wide.set_index("date")["price_bitcoin"].pct_change().dropna()
        vol30 = price.rolling(30).std() * np.sqrt(365)
        st.line_chart(vol30, height=200, use_container_width=True, caption="30-day Volatility (BTC)")
    if chain_df.empty:
        st.warning("No on-chain data for correlation.")
    else:
        metric_corr = st.selectbox("Metric to Correlate", sorted(chain_df["metric"].unique()))
        pivot = (
            chain_df[chain_df.metric==metric_corr]
            .pivot(index="date", columns="chain", values="value")
        )
        corr = pivot.corr()
        fig = px.imshow(corr, text_auto=True, title=f"Correlation of {metric_corr} across Chains")
        st.plotly_chart(fig, use_container_width=True)

elif page=="Forecast":
    st.title("🔮 Price Forecast (Bitcoin)")
    if not has_prophet:
        st.warning("Prophet not installed.")
    elif "price_bitcoin" not in df_wide.columns:
        st.warning("No BTC price data.")
    else:
        dfp = (
            df_wide[["date","price_bitcoin"]]
            .rename(columns={"date":"ds","price_bitcoin":"y"})
            .dropna()
        )
        m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        m.fit(dfp)
        future = m.make_future_dataframe(periods=30)
        fc = m.predict(future)
        fig = px.line(
            fc, x="ds", y=["y","yhat","yhat_upper","yhat_lower"],
            labels={"value":"Price","ds":"Date"},
            title="BTC Price Forecast (30d)"
        )
        st.plotly_chart(fig, use_container_width=True)

elif page=="Comparison":
    st.title("⚔️ Chain & Legacy Comparison")
    if chain_df.empty:
        st.warning("No data to compare.")
        st.stop()
    metric_sel = st.selectbox("Metric", sorted(chain_df["metric"].unique()))
    dmin,dmax = chain_df.date.min(), chain_df.date.max()
    start,end = st.slider("Date Range", value=(dmin,dmax), min_value=dmin, max_value=dmax)
    chart_type = st.selectbox("Chart Type", ["Line","Area","Bar"])
    show_bench = st.checkbox("Show Benchmarks", True)

    dfc = chain_df[
        (chain_df.metric==metric_sel) &
        (chain_df.date.between(start, end))
    ]
    pivot = dfc.pivot(index="date", columns="chain", values="value")

    if chart_type=="Line":
        fig = go.Figure()
        for c in pivot.columns:
            fig.add_trace(go.Scatter(
                x=pivot.index, y=pivot[c], mode="lines", name=c.title()
            ))
        if show_bench and metric_sel in bench_df.benchmark.values:
            for _,r in bench_df[bench_df.benchmark==metric_sel].iterrows():
                fig.add_hline(
                    y=r.value, line_dash="dash",
                    annotation_text=r.label, annotation_position="top left"
                )
    elif chart_type=="Area":
        fig = px.area(pivot, x=pivot.index, y=pivot.columns)
    else:
        monthly = (
            pivot.resample("M")
            .mean()
            .reset_index()
            .melt(
                id_vars="date",
                var_name="chain",
                value_name="value"
            )
        )
        fig = px.bar(monthly, x="date", y="value", color="chain", barmode="group")

    fig.update_layout(margin=dict(l=20,r=20,t=40,b=20), title=metric_sel.title())
    st.plotly_chart(fig, use_container_width=True)

    summary = pivot.loc[start:end].agg(["first","last"]).T
    summary["% Change"] = (summary["last"]/summary["first"]-1)*100
    st.dataframe(
        summary.rename(columns={"first":"Start","last":"End"})[["Start","End","% Change"]],
        use_container_width=True
    )
