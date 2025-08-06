from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Optional forecast
try:
    from prophet import Prophet
    has_prophet = True
except ImportError:
    has_prophet = False

# --- Data loaders ----------------------------------------------------------

@st.cache_data
def load_master():
    base = Path(__file__).parent
    for sub in ("data", "."):
        folder = base / sub
        if not folder.exists():
            continue
        files = list(folder.glob("master_chain_metrics_updated*.csv"))
        if not files:
            continue
        path = files[0]
        sample = pd.read_csv(path, nrows=0)
        date_cols = [c for c in sample.columns if "date" in c.lower()]
        df = pd.read_csv(path, parse_dates=date_cols if date_cols else None, low_memory=False)
        if date_cols:
            df = df.rename(columns={date_cols[0]:"date"})
            df["date"] = pd.to_datetime(df["date"]).dt.date
        df.columns = [c.lower().strip() for c in df.columns]
        return df
    st.error("master_chain_metrics_updated CSV not found.")
    return pd.DataFrame()

@st.cache_data
def load_regulatory():
    base = Path(__file__).parent
    for sub in ("data", "."):
        p = base / sub / "regulatory_milestones.csv"
        if p.exists():
            rdf = pd.read_csv(p, parse_dates=["Date"])
            rdf = rdf.rename(columns=lambda c: c.lower())
            rdf["date"] = pd.to_datetime(rdf["date"])
            return rdf
    return pd.DataFrame()

@st.cache_data
def load_sp500():
    base = Path(__file__).parent
    for sub in ("data", "."):
        folder = base / sub
        if not folder.exists(): continue
        paths = list(folder.glob("*S&P*500*Historical*.csv"))
        if paths:
            sp = pd.read_csv(paths[0], parse_dates=["Date"])
            sp = sp.rename(columns={"Date":"date","Close":"value"})
            sp = sp[["date","value"]].dropna()
            sp["date"] = pd.to_datetime(sp["date"]).dt.date
            sp["chain"], sp["metric"] = "sp500","price"
            return sp
    return pd.DataFrame()

# --- App setup ------------------------------------------------------------
st.set_page_config(page_title="Blockchain Dashboard", layout="wide")
menu = [
    "Research Objectives",
    "Regulatory Timeline",
    "Data Overview",
    "Insights",
    "Correlation",
]
if has_prophet:
    menu.append("Forecast")
menu.append("Comparison")
page = st.sidebar.radio("Go to", menu)

# --- Load data ------------------------------------------------------------
df_wide  = load_master()
reg_df   = load_regulatory()
sp500_df = load_sp500()

# --- Unpivot master into long ---------------------------------------------
chains = ["bitcoin","ethereum","solana"] + (["sp500"] if not sp500_df.empty else [])
records = []
if not df_wide.empty:
    for col in df_wide.columns:
        parts = col.rsplit("_",1)
        if len(parts)==2 and parts[1] in chains:
            metric, chain = parts
            tmp = df_wide[["date",col]].rename(columns={col:"value"}).dropna()
            tmp["chain"], tmp["metric"] = chain, metric
            records.append(tmp)
if not sp500_df.empty:
    records.append(sp500_df)
chain_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
if not chain_df.empty:
    chain_df["date"] = pd.to_datetime(chain_df["date"])

# --- Pages ---------------------------------------------------------------
if page=="Research Objectives":
    st.title("📋 Research Objectives")
    st.markdown("""
1. Core concepts: decentralization, consensus, smart contracts  
2. Applications: DeFi, trade finance, CBDCs, tokenization  
3. Benefits: cost, efficiency, transparency vs. legacy  
4. Challenges: regulatory, security, scalability, interoperability  
5. Adoption trends: on-chain usage & institutional uptake (2000–2025)
""")

elif page=="Regulatory Timeline":
    st.title("🗓️ Regulatory Timeline")
    if reg_df.empty:
        st.warning("No regulatory data.")
    else:
        fig = px.timeline(
            reg_df, x_start="date", x_end="date", y="milestone",
            title="Regulatory & Institutional Milestones"
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

elif page=="Data Overview":
    st.title("🔍 Data Overview")
    if df_wide.empty:
        st.warning("No master data.")
    else:
        st.write(f"**{df_wide['date'].nunique()} days × {len(df_wide.columns)-1} metrics**")
        st.dataframe(df_wide.head(), use_container_width=True)
        with st.expander("All columns"):
            st.write(df_wide.columns.tolist())

elif page=="Insights":
    st.title("💡 Dynamic Insights")
    if chain_df.empty:
        st.warning("No data.")
        st.stop()
    c1,c2 = st.columns(2)
    with c1:
        chain_sel = st.selectbox("Chain", sorted(chain_df["chain"].unique()))
    with c2:
        metric_sel = st.selectbox("Metric", sorted(chain_df["metric"].unique()))
    dfc = chain_df.query("chain==@chain_sel and metric==@metric_sel").set_index("date").sort_index()
    if dfc.empty:
        st.warning("No data for that selection.")
        st.stop()
    st.metric(f"{metric_sel} ({chain_sel})", f"{dfc['value'][-1]:.4f}")
    fig = px.line(dfc, y="value", title=f"{metric_sel} over time")
    st.plotly_chart(fig, use_container_width=True)

elif page=="Correlation":
    st.title("🔗 Variable Correlation")
    # build a wide pivot of all metric_chain pairs
    if chain_df.empty:
        st.warning("No data.")
        st.stop()
    pivot = chain_df.pivot(index="date", columns=["chain","metric"], values="value")
    options = [f"{c}_{m}" for c,m in pivot.columns]
    sel = st.multiselect("Select 2+ variables", options, default=options[:2])
    if len(sel)<2:
        st.info("Pick at least two.")
    else:
        # extract those series
        dfc = pd.DataFrame({v: pivot[c,m] for v in sel for c,m in [v.split("_",1)]})
        corr = dfc.corr()
        if len(sel)==2:
            x,y = sel
            r = corr.loc[x,y]
            st.write(f"Pearson r = **{r:.3f}**")
            fig = px.scatter(dfc, x=x, y=y, trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.imshow(corr, text_auto=True, title="Correlation matrix")
            st.plotly_chart(fig, use_container_width=True)

elif page=="Forecast" and has_prophet:
    st.title("🔮 BTC Price Forecast")
    if "price_bitcoin" in df_wide.columns:
        dfp = df_wide[["date","price_bitcoin"]].rename(columns={"date":"ds","price_bitcoin":"y"}).dropna()
        m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        m.fit(dfp)
        future = m.make_future_dataframe(periods=30)
        fc = m.predict(future)
        fig = px.line(fc, x="ds", y=["y","yhat","yhat_upper","yhat_lower"])
        st.plotly_chart(fig, use_container_width=True)

elif page=="Comparison":
    st.title("⚔️ Chain vs. Benchmarks")
    if chain_df.empty:
        st.warning("No data.")
        st.stop()
    metric_sel = st.selectbox("Metric", sorted(chain_df["metric"].unique()))
    dmin,dmax = chain_df.date.min(), chain_df.date.max()
    start,end = st.slider("Date Range", min_value=dmin, max_value=dmax, value=(dmin,dmax))
    chart_type = st.selectbox("Chart Type", ["Line","Area","Bar"])
    show_bench = st.checkbox("Show Benchmarks", True)
    dfc = chain_df.query("metric==@metric_sel and date>=@start and date<=@end")
    pivot = dfc.pivot(index="date", columns="chain", values="value")

    if chart_type=="Line":
        fig = go.Figure()
        for c in pivot.columns:
            fig.add_trace(go.Scatter(x=pivot.index, y=pivot[c], mode="lines", name=c))
        if show_bench:
            for _,r in load_benchmarks().query("benchmark==@metric_sel").iterrows():
                fig.add_hline(y=r.value, line_dash="dash", annotation_text=r.label)
    elif chart_type=="Area":
        fig = px.area(pivot, x=pivot.index, y=pivot.columns)
    else:
        monthly = pivot.resample("M").mean().reset_index().melt(id_vars="date",
                    var_name="chain", value_name="value")
        fig = px.bar(monthly, x="date", y="value", color="chain", barmode="group")

    fig.update_layout(title=metric_sel, margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(fig, use_container_width=True)
    summary = pivot.loc[start:end].agg(["first","last"]).T
    summary["% Change"] = (summary.last/summary.first-1)*100
    st.dataframe(summary.rename(columns={"first":"Start","last":"End"})[["Start","End","% Change"]])
