from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Optional advanced imports
try:
    from statsmodels.tsa.seasonal import STL
    has_stl = True
except ImportError:
    has_stl = False

try:
    from prophet import Prophet
    has_prophet = True
except ImportError:
    has_prophet = False

# --- Load & cache the master chain metrics CSV ---------------------------
@st.cache_data
def load_master():
    # search both data/ and repo root
    base = Path(__file__).parent
    for sub in ("data", "."):
        folder = base / sub
        if folder.exists():
            files = list(folder.glob("master_chain_metrics_updated*.csv"))
            if files:
                df = pd.read_csv(files[0], parse_dates=["Date"], low_memory=False)
                df.columns = [c.lower() for c in df.columns]
                df = df.rename(columns={"date":"date"})
                df["date"] = pd.to_datetime(df["date"]).dt.date
                return df
    st.error("⚠️ master_chain_metrics_updated.csv not found in data/ or repo root")
    return pd.DataFrame()

@st.cache_data
def load_regulatory():
    for sub in ("data", "."):
        p = Path(__file__).parent / sub / "regulatory_milestones.csv"
        if p.exists():
            return pd.read_csv(p, parse_dates=["Date"])
    return pd.DataFrame()

@st.cache_data
def load_sp500():
    for sub in ("data", "."):
        folder = Path(__file__).parent / sub
        if folder.exists():
            paths = list(folder.glob("*S&P*500*Historical*.csv"))
            if paths:
                sp = pd.read_csv(paths[0], parse_dates=["Date"])
                sp = sp.rename(columns={"Date":"date","Close":"value"})
                sp = sp[["date","value"]].dropna()
                sp["chain"], sp["metric"] = "sp500","price"
                sp["date"] = sp["date"].dt.date
                return sp
    return pd.DataFrame()

# --- App setup ------------------------------------------------------------
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
df_wide  = load_master()
reg_df   = load_regulatory()
sp500_df = load_sp500()

# --- Build long-form chain_df ---------------------------------------------
chains = ["bitcoin","ethereum","solana"] + (["sp500"] if not sp500_df.empty else [])
records = []

if not df_wide.empty:
    for col in df_wide.columns:
        parts = col.rsplit("_",1)
        if len(parts)==2 and parts[1] in chains:
            metric, chain = parts
            tmp = df_wide[["date",col]].dropna().rename(columns={col:"value"})
            tmp["chain"], tmp["metric"] = chain, metric
            records.append(tmp)

# include S&P 500 as chain if present
if not sp500_df.empty:
    records.append(sp500_df)

chain_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
if not chain_df.empty:
    chain_df["date"] = pd.to_datetime(chain_df["date"])

# --- Pages ---------------------------------------------------------------
if page=="Research Objectives":
    st.title("📋 Research Objectives")
    st.markdown("""
1. **Core Concepts:** decentralization, consensus, smart contracts  
2. **Applications:** DeFi, trade finance, CBDCs, tokenization  
3. **Benefits:** cost, efficiency, transparency vs legacy  
4. **Challenges:** regulatory, security, scalability, interoperability  
5. **Adoption Trends:** on-chain usage & institutional uptake (2000–2025)  
""")

elif page=="Regulatory Timeline":
    st.title("🗓️ Regulatory & Institutional Timeline")
    if reg_df.empty:
        st.warning("No regulatory data.")
    else:
        fig = px.timeline(reg_df, x_start="Date", x_end="Date", y="Milestone",
                          title="Key Regulatory Milestones")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

elif page=="Data Overview":
    st.title("🔍 Data Overview")
    if df_wide.empty:
        st.warning("No master metrics loaded.")
    else:
        st.write(f"**{df_wide['date'].nunique()} days × {len(df_wide.columns)-1} metrics**")
        st.dataframe(df_wide.head(10), use_container_width=True)
        with st.expander("All Columns"):
            st.write(df_wide.columns.tolist())

elif page=="Insights":
    st.title("💡 Dynamic Insights")
    if chain_df.empty:
        st.warning("No data to show.")
        st.stop()
    c1, c2 = st.columns(2)
    with c1:
        chain_sel = st.selectbox("Chain", sorted(chain_df["chain"].unique()))
    with c2:
        metric_sel = st.selectbox("Metric", sorted(chain_df["metric"].unique()))
    dfc = chain_df.query("chain==@chain_sel and metric==@metric_sel").set_index("date").sort_index()
    if dfc.empty:
        st.warning("No data for that selection.")
        st.stop()
    st.metric(f"{metric_sel.replace('_',' ').title()} ({chain_sel.upper()})", f"{dfc['value'].iat[-1]:.4f}")
    fig = px.line(dfc, y="value", title=f"{metric_sel.replace('_',' ').title()} over time",
                  labels={"value":metric_sel.replace('_',' ').title(),"date":"Date"})
    st.plotly_chart(fig, use_container_width=True)
    pct = (dfc['value'].iloc[-1] / dfc['value'].iloc[0] - 1) * 100
    st.markdown(f"**Insight:** {chain_sel.upper()}'s **{metric_sel.replace('_',' ').title()}** changed **{pct:.1f}%** since {dfc.index[0]}.")

elif page=="Risk & Correlation":
    st.title("📈 Volatility & Correlation")
    if "price_bitcoin" in df_wide.columns:
        price = df_wide.set_index("date")["price_bitcoin"].pct_change().dropna()
        vol30 = price.rolling(30).std() * np.sqrt(365)
        st.line_chart(vol30, height=200, caption="30-day Volatility (BTC)")
    if chain_df.empty:
        st.warning("No on-chain data for correlation.")
    else:
        metric_corr = st.selectbox("Metric to Correlate", sorted(chain_df["metric"].unique()))
        pivot = (chain_df.query("metric==@metric_corr")
                 .pivot(index="date", columns="chain", values="value"))
        corr = pivot.corr()
        fig = px.imshow(corr, text_auto=True, title=f"Correlation of {metric_corr} across Chains")
        st.plotly_chart(fig, use_container_width=True)

elif page=="Forecast":
    st.title("🔮 Price Forecast (Bitcoin)")
    if not has_prophet:
        st.warning("Install `prophet` for forecasting.")
    elif "price_bitcoin" not in df_wide.columns:
        st.warning("No BTC price data.")
    else:
        dfp = df_wide[["date","price_bitcoin"]].rename(columns={"date":"ds","price_bitcoin":"y"}).dropna()
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        m.fit(dfp)
        future = m.make_future_dataframe(periods=30)
        fc = m.predict(future)
        fig = px.line(fc, x="ds", y=["y","yhat","yhat_upper","yhat_lower"],
                      labels={"value":"Price","ds":"Date"}, title="BTC Price Forecast (30d)")
        st.plotly_chart(fig, use_container_width=True)

elif page=="Comparison":
    st.title("⚔️ Chain & Legacy Comparison")
    if chain_df.empty:
        st.warning("No data to compare.")
        st.stop()
    metric_sel = st.selectbox("Metric", sorted(chain_df["metric"].unique()))
    dmin, dmax = chain_df.date.min(), chain_df.date.max()
    start, end = st.slider("Date Range", (dmin, dmax), value=(dmin, dmax))
    chart_type = st.selectbox("Chart Type", ["Line","Area","Bar"])
    show_bench = st.checkbox("Show Benchmarks", True)

    dfc = chain_df.query("metric==@metric_sel and date>=@start and date<=@end")
    pivot = dfc.pivot(index="date", columns="chain", values="value")

    if chart_type=="Line":
        fig = go.Figure()
        for c in pivot.columns:
            fig.add_trace(go.Scatter(x=pivot.index, y=pivot[c], mode="lines", name=c.upper()))
        if show_bench:
            for _,r in load_benchmarks().query("benchmark==@metric_sel").iterrows():
                fig.add_hline(y=r.value, line_dash="dash", annotation_text=r.label)
    elif chart_type=="Area":
        fig = px.area(pivot, x=pivot.index, y=pivot.columns)
    else:
        monthly = (pivot.resample("M").mean().reset_index()
                   .melt(id_vars="date", var_name="chain", value_name="value"))
        fig = px.bar(monthly, x="date", y="value", color="chain", barmode="group")

    fig.update_layout(margin=dict(l=20,r=20,t=40,b=20), title=f"{metric_sel.replace('_',' ').title()} Comparison")
    st.plotly_chart(fig, use_container_width=True)

    summary = pivot.loc[start:end].agg(["first","last"]).T
    summary["% Change"] = (summary["last"]/summary["first"] - 1)*100
    st.dataframe(
        summary.rename(columns={"first":"Start","last":"End"})[["Start","End","% Change"]],
        use_container_width=True
    )
