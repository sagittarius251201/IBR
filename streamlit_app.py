from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Optional forecasting
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
        if not folder.exists(): continue
        files = list(folder.glob("master_chain_metrics_updated*.csv"))
        if not files: continue
        df = pd.read_csv(files[0], low_memory=False)
        # unify date column
        date_cols = [c for c in df.columns if "date" in c.lower()]
        if date_cols:
            df = df.rename(columns={date_cols[0]: "date"})
        df["date"] = pd.to_datetime(df["date"])
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
            rdf.columns = [c.lower() for c in rdf.columns]
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
            sp["date"] = pd.to_datetime(sp["date"])
            sp["chain"], sp["metric"] = "sp500","price"
            return sp
    return pd.DataFrame()

@st.cache_data
def load_benchmarks():
    data = {
        "benchmark": [
            "visa_avg_tps","visa_peak_tps","mc_avg_tps","mc_peak_tps",
            "swift_settlement_days","dtcc_tplus1_adoption_pct","t2s_settlement_days",
            "daily_active_addresses","daily_transactions","fees","tvl","dex_volumes"
        ],
        "value": [
            1700,65000,5000,59000,1.25,95,0.10,
            1_000_000,5_000_000,1.0,1e12,2e8
        ]
    }
    df = pd.DataFrame(data)
    df["label"] = df["benchmark"].map({
        "visa_avg_tps":"Visa Avg TPS","visa_peak_tps":"Visa Peak TPS",
        "mc_avg_tps":"Mastercard Avg TPS","mc_peak_tps":"Mastercard Peak TPS",
        "swift_settlement_days":"SWIFT Settlement (days)",
        "dtcc_tplus1_adoption_pct":"DTCC T+1 Adoption (%)",
        "t2s_settlement_days":"ECB T2S Settlement (days)",
        "daily_active_addresses":"Legacy Daily Active Addrs",
        "daily_transactions":"Legacy Daily Txns",
        "fees":"Legacy Avg Fee (USD)",
        "tvl":"Legacy TVL (USD)",
        "dex_volumes":"Legacy DEX Volume (USD)"
    })
    return df

# --- App setup ------------------------------------------------------------
st.set_page_config(page_title="Blockchain Dashboard", layout="wide")
pages = ["Research Objectives","Regulatory Timeline","Data Overview","Insights","Correlation"]
if has_prophet: pages.append("Forecast")
pages.append("Comparison")
page = st.sidebar.radio("Go to", pages)

# --- Load data ------------------------------------------------------------
df_wide = load_master()
reg_df  = load_regulatory()
sp500_df = load_sp500()
bench_df = load_benchmarks()

# --- Unpivot to long form -----------------------------------------------
chains = ["bitcoin","ethereum","solana"] + (["sp500"] if not sp500_df.empty else [])
records = []
if not df_wide.empty:
    for col in df_wide.columns:
        metric_chain = col.rsplit("_",1)
        if len(metric_chain)==2 and metric_chain[1] in chains:
            m,c = metric_chain
            tmp = df_wide[["date",col]].rename(columns={col:"value"}).dropna()
            tmp["chain"], tmp["metric"] = c, m
            records.append(tmp)
if not sp500_df.empty:
    records.append(sp500_df)
chain_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()

# --- Pages ---------------------------------------------------------------
if page=="Research Objectives":
    st.title("📋 Research Objectives")
    st.markdown("""
1. Core Concepts: decentralization, consensus, smart contracts  
2. Applications: DeFi, trade finance, CBDCs, tokenization  
3. Benefits: cost, efficiency, transparency vs legacy  
4. Challenges: regulatory, security, scalability, interoperability  
5. Adoption Trends: on-chain usage & institutional uptake (2000–2025)
""")

elif page=="Regulatory Timeline":
    st.title("🗓️ Regulatory Timeline")
    if reg_df.empty:
        st.warning("No regulatory data.")
    else:
        df = reg_df.sort_values("date")
        fig = px.scatter(df, x="date", y="milestone", title="Regulatory Milestones")
        for _,r in df.iterrows():
            fig.add_shape({
                "type":"line",
                "x0":r.date,"x1":r.date,
                "y0":-0.5,"y1":len(df)-0.5,
                "line":{"color":"LightGray","dash":"dot"}
            })
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

elif page=="Data Overview":
    st.title("🔍 Data Overview")
    if df_wide.empty:
        st.warning("No data loaded.")
    else:
        st.write(f"**{df_wide['date'].nunique()} days × {len(df_wide.columns)-1} metrics**")
        st.dataframe(df_wide.head(), use_container_width=True)

elif page=="Insights":
    st.title("💡 Insights")
    if chain_df.empty:
        st.warning("No data."); st.stop()
    c1,c2 = st.columns(2)
    with c1: chain_sel = st.selectbox("Chain", sorted(chain_df.chain.unique()))
    with c2: metric_sel = st.selectbox("Metric", sorted(chain_df.metric.unique()))
    dfc = chain_df.query("chain==@chain_sel and metric==@metric_sel").set_index("date").sort_index()
    st.metric(f"{metric_sel} ({chain_sel})", f"{dfc.value.iloc[-1]:.4f}")
    st.plotly_chart(px.line(dfc, y="value", title=f"{metric_sel} over time"), use_container_width=True)

elif page=="Correlation":
    st.title("🔗 Correlation")
    if chain_df.empty:
        st.warning("No data."); st.stop()
    pivot = chain_df.pivot(index="date", columns=["chain","metric"], values="value")
    opts = [f"{c}_{m}" for c,m in pivot.columns]
    sel = st.multiselect("Variables (2+)", opts, default=opts[:2])
    if len(sel)<2:
        st.info("Select at least two.")
    else:
        dfc = pd.DataFrame({v: pivot[c,m] for v in sel for c,m in [v.split("_",1)]})
        corr = dfc.corr()
        if len(sel)==2:
            x,y = sel; r=corr.loc[x,y]
            st.write(f"Pearson r = {r:.3f}")
            st.plotly_chart(px.scatter(dfc, x=x, y=y, trendline="ols"),use_container_width=True)
        else:
            st.plotly_chart(px.imshow(corr, text_auto=True), use_container_width=True)

elif page=="Forecast":
    st.title("🔮 Forecast")
    if not has_prophet:
        st.warning("Install prophet.")
    else:
        price_cols = [c for c in df_wide.columns if c.startswith("price_")]
        if not price_cols:
            st.warning("No price series.")
        else:
            chains_av = [c.split("_",1)[1] for c in price_cols]
            chain_sel = st.selectbox("Chain to forecast", chains_av)
            col = f"price_{chain_sel}"
            dfp = df_wide[["date",col]].rename(columns={"date":"ds",col:"y"}).dropna()
            dfp["ds"] = pd.to_datetime(dfp["ds"])
            m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            m.fit(dfp)
            future = m.make_future_dataframe(periods=30)
            fc = m.predict(future)
            fc["ds_date"] = fc["ds"].dt.date
            dfp["ds_date"]  = dfp["ds"].dt.date
            merged = pd.merge(fc, dfp[["ds_date","y"]], on="ds_date", how="left")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=merged.ds, y=merged.yhat, mode="lines", name="Forecast"))
            fig.add_trace(go.Scatter(x=merged.ds, y=merged.yhat_upper, mode="lines", line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=merged.ds, y=merged.yhat_lower, mode="lines", fill="tonexty", line=dict(width=0), fillcolor="rgba(0,100,80,0.2)", name="CI"))
            fig.add_trace(go.Scatter(x=merged.ds, y=merged.y, mode="markers+lines", name="Actual"))
            fig.update_layout(title=f"{chain_sel.upper()} 30d Forecast", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

elif page == "Comparison":
    st.title("⚔️ Chain vs. Benchmarks")
    if chain_df.empty:
        st.warning("No data to compare.")
        st.stop()

    metric_sel = st.selectbox("Metric", sorted(chain_df["metric"].unique()))

    # Get native Python datetimes for slider
    dmin_np = chain_df["date"].min()
    dmax_np = chain_df["date"].max()
    dmin = pd.to_datetime(dmin_np).to_pydatetime()
    dmax = pd.to_datetime(dmax_np).to_pydatetime()

    start_dt, end_dt = st.slider(
        "Date Range",
        min_value=dmin,
        max_value=dmax,
        value=(dmin, dmax),
        format="YYYY-MM-DD"
    )

    # Now filter using pandas Timestamps: convert Python datetime back
    dfc = chain_df[
        (chain_df["metric"] == metric_sel) &
        (chain_df["date"] >= pd.to_datetime(start_dt)) &
        (chain_df["date"] <= pd.to_datetime(end_dt))
    ]

    pivot = dfc.pivot(index="date", columns="chain", values="value")

    chart = st.selectbox("Chart Type", ["Line", "Area", "Bar"])
    show_bench = st.checkbox("Show Benchmarks", True)

    if chart == "Line":
        fig = go.Figure()
        for c in pivot.columns:
            fig.add_trace(go.Scatter(
                x=pivot.index, y=pivot[c], mode="lines", name=c.upper()
            ))
        if show_bench:
            for _, r in bench_df[bench_df["benchmark"] == metric_sel].iterrows():
                fig.add_hline(
                    y=r.value,
                    line_dash="dash",
                    annotation_text=r.label,
                    annotation_position="top left"
                )

    elif chart == "Area":
        fig = px.area(pivot, x=pivot.index, y=pivot.columns)

    else:  # Bar
        monthly = (
            pivot
            .resample("M")
            .mean()
            .reset_index()
            .melt(id_vars="date", var_name="chain", value_name="value")
        )
        fig = px.bar(monthly, x="date", y="value", color="chain", barmode="group")

    fig.update_layout(
        title=f"{metric_sel.replace('_',' ').title()} Comparison",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary
    if not dfc.empty:
        # first/last values and percent change
        summary = (
            pivot
            .loc[pd.to_datetime(start_dt):pd.to_datetime(end_dt)]
            .agg(["first", "last"])
            .T
        )
        summary["% Change"] = (summary["last"] / summary["first"] - 1) * 100
        st.dataframe(
            summary.rename(columns={"first":"Start","last":"End"})[["Start","End","% Change"]],
            use_container_width=True
        )
    else:
        st.info("No data points in the selected date range to summarize.")
