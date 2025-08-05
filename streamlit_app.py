import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.colors import qualitative

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv('master_time_series_imputed.csv', parse_dates=['Date'])
    df = df.loc[:, ~df.columns.duplicated()]
    return df

# App configuration
st.set_page_config(
    page_title='Blockchain Metrics Dashboard',
    layout='wide',
    initial_sidebar_state='expanded'
)

df = load_data()
has_chain = 'Chain' in df.columns  # Detect if Chain column exists

# Sidebar: Filters & Settings
with st.sidebar:
    st.header('Filters & Settings')
    # Chain selector if available
    if has_chain:
        chains = st.multiselect(
            'Select Chains', options=df['Chain'].unique(), default=list(df['Chain'].unique())
        )
    else:
        chains = None
    # Date range selector
    start_date, end_date = st.date_input(
        'Date Range', [df['Date'].min(), df['Date'].max()],
        min_value=df['Date'].min(), max_value=df['Date'].max()
    )
    st.markdown('---')
    # Numeric metric selection
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    metrics = st.multiselect(
        'Select Metrics', options=numeric_cols,
        default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
    )
    st.markdown('---')
    # Traditional benchmarks
    benchmarks = {
        'Visa Avg TPS': 1700,
        'Visa Peak TPS': 65000,
        'Mastercard Avg TPS': 5000,
        'Mastercard Peak TPS': 59000,
        'SWIFT Avg Settlement (days)': 1.25,
        'ECB T2S Avg Settlement (days)': 0.1
    }
    bench_to_plot = st.multiselect('Select Traditional Benchmarks', list(benchmarks.keys()),
                                   default=list(benchmarks.keys())[:2])
    st.markdown('---')
    # Chart type selector
    chart_type = st.selectbox('Chart Type', ['Line', 'Area', 'Bar', 'Scatter'])
    st.markdown('---')
    st.write('**Color Theme**')
    color_seq = qualitative.Plotly

# Filter data by date and chain
filtered = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
if has_chain:
    filtered = filtered[filtered['Chain'].isin(chains)]

# Create tabs
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

# Tab 1: Objectives & Events
with tab_objs:
    st.title('Research Objectives & Key Events')
    st.markdown("""
**Objectives:**
- Explain core blockchain concepts
- Document present & future applications in finance
- Quantify benefits: cost, efficiency, transparency
- Assess regulatory & security challenges
- Analyze adoption trends & market sentiment
    """)
    events = [
        ('2018-10-01', 'FATF 1st Guidance', 'Global', 'Defined VASP AML/CFT requirements'),
        ('2019-06-21', 'FATF Travel Rule', 'Global', 'Introduced the Travel Rule'),
        ('2021-10-28', 'FATF Updated Guidance', 'Global', 'Expanded to stablecoins, DeFi, NFTs'),
        ('2023-06-09', 'MiCA Published', 'EU', 'Unified regulation framework'),
        ('2023-06-29', 'MiCA In Force', 'EU', 'Start of regulated services'),
        ('2024-02-22', 'MiCA Delegated Acts', 'EU', 'Technical standards enacted'),
        ('2023-02-07', 'VARA Framework', 'Dubai', 'Licensing & governance rules'),
        ('2018-06-25', 'ADGM Regime', 'Abu Dhabi', 'First MENA crypto sandbox')
    ]
    ev_df = pd.DataFrame(events, columns=['Date', 'Event', 'Region', 'Impact'])
    ev_df['Date'] = pd.to_datetime(ev_df['Date'])
    st.dataframe(ev_df, use_container_width=True)

# Tab 2: Price Metrics
with tab_price:
    st.title('Price & Market Cap Metrics')
    metrics_price = [c for c in numeric_cols if 'price' in c.lower() or 'market_cap' in c.lower()]
    sel_price = st.multiselect('Select Price/Market Cap Metrics', options=metrics_price, default=metrics_price)
    if sel_price:
        id_vars = ['Date'] + (['Chain'] if has_chain else [])
        long_df = filtered.melt(id_vars=id_vars, value_vars=sel_price, var_name='Metric', value_name='Value')
        fig = px.line(long_df, x='Date', y='Value', color='Chain' if has_chain else None,
                      facet_col='Metric', facet_col_wrap=2,
                      title='Price & Market Cap Over Time',
                      color_discrete_sequence=color_seq)
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Please select at least one price metric.')

# Tab 3: On-Chain Value
with tab_onchain:
    st.title('On-Chain Value Metrics')
    metrics_val = ['Realized_Cap_B', 'MVRV_Ratio']
    sel_val = st.multiselect('Select On-Chain Metrics', options=metrics_val, default=metrics_val)
    if sel_val:
        id_vars = ['Date'] + (['Chain'] if has_chain else [])
        long_val = filtered.melt(id_vars=id_vars, value_vars=sel_val, var_name='Metric', value_name='Value')
        fig = px.line(long_val, x='Date', y='Value', color='Chain' if has_chain else None,
                      facet_col='Metric', facet_col_wrap=2,
                      title='On-Chain Value Trends',
                      color_discrete_sequence=color_seq)
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Please select at least one on-chain metric.')

# Tab 4: DeFi Metrics
with tab_defi:
    st.title('DeFi Usage Metrics')
    metrics_defi = ['DeFi_Protocols', 'Weekly_Contract_Calls', 'Lending_Util_%', 'Avg_DeFi_APR_%']
    sel_defi = st.multiselect('Select DeFi Metrics', options=metrics_defi, default=metrics_defi)
    if sel_defi:
        id_vars = ['Date'] + (['Chain'] if has_chain else [])
        long_defi = filtered.melt(id_vars=id_vars, value_vars=sel_defi, var_name='Metric', value_name='Value')
        fig = px.line(long_defi, x='Date', y='Value', color='Chain' if has_chain else None,
                      facet_col='Metric', facet_col_wrap=2,
                      title='DeFi Metrics Trends',
                      color_discrete_sequence=color_seq)
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Please select at least one DeFi metric.')

# Tab 5: Network Security
with tab_sec:
    st.title('Network Security & Decentralization')
    metrics_sec = ['Difficulty_T', 'Full_Nodes', 'Orphan_Rate_%', 'Top10_Concentration_%']
    sel_sec = st.multiselect('Select Security Metrics', options=metrics_sec, default=metrics_sec)
    if sel_sec:
        id_vars = ['Date'] + (['Chain'] if has_chain else [])
        long_sec = filtered.melt(id_vars=id_vars, value_vars=sel_sec, var_name='Metric', value_name='Value')
        fig = px.line(long_sec, x='Date', y='Value', color='Chain' if has_chain else None,
                      facet_col='Metric', facet_col_wrap=2,
                      title='Security Metrics Trends',
                      color_discrete_sequence=color_seq)
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Please select at least one security metric.')

# Tab 6: Comparison
with tab_cmp:
    st.title('Comparison vs Traditional Benchmarks')
    comp_metric = st.selectbox('Select Blockchain Metric for Comparison', options=metrics)
    chains_cmp = st.multiselect('Select Chains', options=df['Chain'].unique() if has_chain else [], default=list(df['Chain'].unique()) if has_chain else [])
    comp_df = filtered[filtered['Chain'].isin(chains_cmp)] if has_chain else filtered
    fig = px.line(comp_df, x='Date', y=comp_metric, color='Chain' if has_chain else None,
                  title=f'{comp_metric} vs Benchmarks', color_discrete_sequence=color_seq)
    for name, val in benchmarks.items():
        if name in bench_to_plot:
            fig.add_hline(y=val, line_dash='dash', annotation_text=name, annotation_position='top left')
    fig.update_layout(hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

# Tab 7: Insights
with tab_ins:
    st.title('Automated Insights & Inferences')
    insight_metric = st.selectbox('Select Metric for Insights', options=metrics)
    if insight_metric:
        temp = filtered[['Date'] + (['Chain'] if has_chain else []) + [insight_metric]].dropna()
        temp['Pct_Change'] = temp.groupby('Chain')[insight_metric].pct_change() if has_chain else temp[insight_metric].pct_change()
        top = temp.nlargest(3, 'Pct_Change')[['Date'] + (['Chain'] if has_chain else []) + [insight_metric, 'Pct_Change']]
        st.subheader(f'Top 3 Growth Periods for {insight_metric}')
        st.dataframe(top)
    else:
        st.info('Select a metric for insights.')

# Tab 8: Download
with tab_dl:
    st.title('Download Filtered Data')
    dl_df = filtered[['Date'] + (['Chain'] if has_chain else []) + metrics]
    csv = dl_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download CSV', data=csv, file_name='dashboard_data_export.csv', mime='text/csv')

