import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.colors import qualitative

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv('master_time_series_imputed.csv', parse_dates=['Date'])
    df = df.loc[:, ~df.columns.duplicated()]  # drop duplicate columns
    return df

# App configuration
st.set_page_config(page_title='Blockchain Metrics Dashboard', layout='wide', initial_sidebar_state='expanded')

df = load_data()

# Sidebar: Filters & Settings
with st.sidebar:
    st.header('Filters & Settings')
    # Chain selector
    chains = st.multiselect(
        'Select Chains', options=df['Chain'].unique(), default=list(df['Chain'].unique())
    )
    # Date range selector
    start_date, end_date = st.date_input(
        'Date Range', [df['Date'].min(), df['Date'].max()],
        min_value=df['Date'].min(), max_value=df['Date'].max()
    )
    st.markdown('---')
    # Metric selection
    available_metrics = [c for c in df.columns if c not in ['Date', 'Chain']]
    metrics = st.multiselect(
        'Select Metrics', options=available_metrics,
        default=available_metrics[:2]
    )
    st.markdown('---')
    # Traditional benchmarks
    trad_defaults = {
        'Visa Avg TPS': 1700,
        'Visa Peak TPS': 65000,
        'SWIFT Avg Settlement (days)': 1.25
    }
    trad_metrics = st.multiselect(
        'Traditional Benchmarks', options=list(trad_defaults.keys()),
        default=list(trad_defaults.keys())
    )
    st.markdown('---')
    # Chart type
    chart_type = st.selectbox('Chart Type', ['Line', 'Area', 'Bar', 'Scatter'])
    st.markdown('---')
    st.write('**Color Theme**')
    color_seq = qualitative.Plotly

# Filter data
filtered = df[
    (df['Chain'].isin(chains)) &
    (df['Date'] >= pd.to_datetime(start_date)) &
    (df['Date'] <= pd.to_datetime(end_date))
]

# Create tabs
tabs = st.tabs([
    'Objectives & Events',
    'Time Series',
    'Comparison',
    'Insights',
    'Download'
])

# Tab 1: Objectives & Events
with tabs[0]:
    st.title('Research Objectives & Key Events')
    st.markdown(
        '''
**Objectives:**
- Explain core blockchain concepts
- Document present & future applications in finance
- Quantify benefits: cost, efficiency, transparency
- Assess regulatory & security challenges
- Analyze adoption trends & market sentiment
        '''
    )
    st.subheader('Key Regulatory Milestones')
    events = [
        ('2018-10-01', 'FATF 1st Guidance', 'Global', 'Defined VASP AML/CFT requirements'),
        ('2019-06-21', 'FATF VASP Guidance', 'Global', 'Introduced the Travel Rule'),
        ('2021-10-28', 'FATF Updated Guidance', 'Global', 'Expanded to stablecoins, DeFi, NFTs'),
        ('2023-06-09', 'MiCA Published', 'EU', 'Unified regulation framework'),
        ('2023-06-29', 'MiCA In Force', 'EU', 'Start of regulated services'),
        ('2024-02-22', 'MiCA Delegated Acts', 'EU', 'Technical standards enacted'),
        ('2023-02-07', 'VARA Framework', 'Dubai', 'Licensing and governance rules'),
        ('2018-06-25', 'ADGM Regime', 'Abu Dhabi', 'First MENA crypto sandbox')
    ]
    ev_df = pd.DataFrame(events, columns=['Date', 'Event', 'Region', 'Impact'])
    ev_df['Date'] = pd.to_datetime(ev_df['Date'])
    st.dataframe(ev_df, use_container_width=True)

# Tab 2: Time Series
with tabs[1]:
    st.title('Time Series Analysis')
    if metrics:
        for m in metrics:
            if chart_type == 'Line':
                fig = px.line(
                    filtered, x='Date', y=m, color='Chain',
                    title=m, color_discrete_sequence=color_seq
                )
            elif chart_type == 'Area':
                fig = px.area(
                    filtered, x='Date', y=m, color='Chain',
                    title=m, color_discrete_sequence=color_seq
                )
            elif chart_type == 'Bar':
                fig = px.bar(
                    filtered, x='Date', y=m, color='Chain',
                    barmode='group', title=m
                )
            else:
                fig = px.scatter(filtered, x='Date', y=m, color='Chain', title=m)
            fig.update_layout(hovermode='x unified', template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning('Please select at least one metric.')

# Tab 3: Comparison
with tabs[2]:
    st.title('Blockchain vs Traditional Benchmarks')
    if metrics:
        comp = st.selectbox('Choose Blockchain Metric', options=metrics)
        fig = px.line(
            filtered, x='Date', y=comp, color='Chain',
            title=f'{comp} vs Benchmarks', color_discrete_sequence=color_seq
        )
        for name, val in trad_defaults.items():
            if name in trad_metrics:
                fig.add_hline(y=val, line_dash='dash', annotation_text=name)
        fig.update_layout(hovermode='x unified', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning('Please select metrics in the sidebar.')

# Tab 4: Insights
with tabs[3]:
    st.title('Insights & Inferences')
    if metrics:
        primary = metrics[0]
        filtered['Pct_Change'] = filtered.groupby('Chain')[primary].pct_change()
        top3 = filtered.nlargest(3, 'Pct_Change')[['Date', 'Chain', primary, 'Pct_Change']]
        st.subheader(f'Top 3 Growth Weeks: {primary}')
        st.dataframe(top3)
        if len(metrics) > 1:
            corr = filtered.pivot_table(index='Date', columns='Chain', values=metrics).corr()
            st.subheader('Correlation Matrix')
            st.dataframe(corr)
            st.write('_High correlation suggests similar movement patterns._')
    else:
        st.warning('Select metrics for insights.')

# Tab 5: Download
with tabs[4]:
    st.title('Download Filtered Data')
    download_df = filtered[['Date', 'Chain'] + metrics]
    csv = download_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download CSV', data=csv, file_name='dashboard_export.csv', mime='text/csv')
