import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.colors import qualitative

# ---------------------- Data Loading ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv('master_time_series_imputed.csv', parse_dates=['Date'])
    # Ensure Chain column exists for filtering
    if 'Chain' not in df.columns:
        df['Chain'] = 'All'
    # Deduplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    return df

df = load_data()

# ---------------------- App Config ------------------------
st.set_page_config(
    page_title='Blockchain Metrics Dashboard',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ---------------------- Sidebar --------------------------
st.sidebar.header('Filters & Settings')
# Chain filter
chains = st.sidebar.multiselect(
    'Select Chains', options=df['Chain'].unique(), default=list(df['Chain'].unique())
)
# Date range
start_date, end_date = st.sidebar.date_input(
    'Date Range', [df['Date'].min(), df['Date'].max()],
    min_value=df['Date'].min(), max_value=df['Date'].max()
)

# Traditional benchmarks selection
benchmarks = {
    'Visa Avg TPS': 1700,
    'Visa Peak TPS': 65000,
    'Mastercard Avg TPS': 5000,
    'Mastercard Peak TPS': 59000,
    'SWIFT Avg Settlement (days)': 1.25,
    'ECB T2S Avg Settlement (days)': 0.1
}
bench_to_plot = st.sidebar.multiselect('Select Traditional Benchmarks', list(benchmarks.keys()), default=list(benchmarks.keys())[:2])

# ------------------- Tabs Setup --------------------------
tab_objs, tab_price, tab_onchain, tab_defi, tab_sec, tab_cmp, tab_ins, tab_dl = st.tabs([
    'Objectives & Events',
    'Price Metrics',
    'On-Chain Value',
    'DeFi Metrics',
    'Network Security',
    'Compare Traditional',
    'Insights',
    'Download'
])

# Filter df
filtered = df[(df['Chain'].isin(chains)) & 
               (df['Date'] >= pd.to_datetime(start_date)) & 
               (df['Date'] <= pd.to_datetime(end_date))]

# ---------------- Tab 1: Objectives & Events ----------------
with tab_objs:
    st.title('Research Objectives & Key Events')
    st.markdown("""
### Objectives
- Explain core concepts of blockchain (decentralization, consensus, smart contracts)
- Document current & future financial services use cases (DeFi, CBDCs, trade finance)
- Quantify benefits: cost savings, throughput, transparency
- Assess regulatory & security challenges (AML/CFT, hacks, scalability)
- Analyze adoption trends & market sentiment across regions

### Key Regulatory Milestones & Impacts
Event | Date | Region | Impact
--- | --- | --- | ---
FATF 1st Guidance | 2018-10-01 | Global | Defined VASP AML/CFT rules
FATF Travel Rule | 2019-06-21 | Global | Introduced crypto Travel Rule
FATF Stablecoin/DeFi | 2021-10-28 | Global | Guidance on stablecoins & DeFi/NFTs
EU MiCA Entry | 2023-06-29 | EU | Unified crypto framework in force
Dubai VARA Framework | 2023-02-07 | UAE (Dubai) | Licensing & governance rules
ADGM Crypto Regime | 2018-06-25 | UAE (Abu Dhabi) | Sandbox for crypto firms
""")

# ---------------- Tab 2: Price Metrics ----------------
with tab_price:
    st.title('Price & Market Cap Metrics')
    metrics_price = [c for c in df.columns if c.lower().startswith('price') or 'market_cap' in c.lower()]
    sel_price = st.multiselect('Select Price/Market Cap Metrics', options=metrics_price, default=metrics_price)
    if sel_price:
        long_df = filtered.melt(id_vars=['Date','Chain'], value_vars=sel_price, var_name='Metric', value_name='Value')
        fig = px.line(long_df, x='Date', y='Value', color='Chain', facet_col='Metric', facet_col_wrap=2,
                      title='Price & Market Cap Over Time', color_discrete_sequence=qualitative.Plotly)
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Please select at least one price metric.')

# ---------------- Tab 3: On-Chain Value ----------------
with tab_onchain:
    st.title('On-Chain Value Metrics')
    metrics_val = ['Realized_Cap_B','MVRV_Ratio']
    sel_val = st.multiselect('Select On-Chain Metrics', options=metrics_val, default=metrics_val)
    if sel_val:
        long_val = filtered.melt(id_vars=['Date','Chain'], value_vars=sel_val, var_name='Metric', value_name='Value')
        fig = px.line(long_val, x='Date', y='Value', color='Chain', facet_col='Metric', facet_col_wrap=2,
                      title='On-Chain Value Trends', color_discrete_sequence=qualitative.Plotly)
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Please select on-chain value metrics.')

# ---------------- Tab 4: DeFi Metrics ----------------
with tab_defi:
    st.title('DeFi Usage Metrics')
    metrics_defi = ['DeFi_Protocols','Weekly_Contract_Calls','Lending_Util_%','Avg_DeFi_APR_%']
    sel_defi = st.multiselect('Select DeFi Metrics', options=metrics_defi, default=metrics_defi)
    if sel_defi:
        long_defi = filtered.melt(id_vars=['Date','Chain'], value_vars=sel_defi, var_name='Metric', value_name='Value')
        fig = px.line(long_defi, x='Date', y='Value', color='Chain', facet_col='Metric', facet_col_wrap=2,
                      title='DeFi Metrics Trends', color_discrete_sequence=qualitative.Plotly)
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Please select DeFi usage metrics.')

# ---------------- Tab 5: Network Security ----------------
with tab_sec:
    st.title('Network Security & Decentralization')
    metrics_sec = ['Difficulty_T','Full_Nodes','Orphan_Rate_%','Top10_Concentration_%']
    sel_sec = st.multiselect('Select Security Metrics', options=metrics_sec, default=metrics_sec)
    if sel_sec:
        long_sec = filtered.melt(id_vars=['Date','Chain'], value_vars=sel_sec, var_name='Metric', value_name='Value')
        fig = px.line(long_sec, x='Date', y='Value', color='Chain', facet_col='Metric', facet_col_wrap=2,
                      title='Security Metrics Trends', color_discrete_sequence=qualitative.Plotly)
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Please select security metrics.')

# ---------------- Tab 6: Comparison ----------------
with tab_cmp:
    st.title('Comparison vs Traditional Benchmarks')
    comp_metric = st.selectbox('Select Blockchain Metric', options=[c for c in df.columns if c not in ['Date','Chain']])
    chains_cmp = st.multiselect('Select Chains', options=df['Chain'].unique(), default=list(df['Chain'].unique()))
    comp_df = filtered[filtered['Chain'].isin(chains_cmp)]
    fig = px.line(comp_df, x='Date', y=comp_metric, color='Chain',
                  title=f'{comp_metric} vs Benchmarks', color_discrete_sequence=qualitative.Dark24)
    for name,val in benchmarks.items():
        if name in bench_to_plot:
            fig.add_hline(y=val, line_dash='dash', annotation_text=name, annotation_position='top left')
    fig.update_layout(hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Tab 7: Insights ----------------
with tab_ins:
    st.title('Automated Insights & Inferences')
    insight_metric = st.selectbox('Select Metric for Insights', options=[c for c in df.columns if c not in ['Date','Chain']])
    if insight_metric:
        temp = filtered[['Date','Chain', insight_metric]].dropna()
        temp['Pct_Change'] = temp.groupby('Chain')[insight_metric].pct_change()
        top = temp.nlargest(3, 'Pct_Change')[['Date','Chain', insight_metric, 'Pct_Change']]
        st.subheader(f'Top 3 Growth Periods for {insight_metric}')
        st.dataframe(top)
    else:
        st.info('Select a metric to generate insights.')

# ---------------- Tab 8: Download ----------------
with tab_dl:
    st.title('Download Filtered Data')
    dl_df = filtered.copy()
    csv = dl_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download CSV', data=csv, file_name='dashboard_data_export.csv', mime='text/csv')
