import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.colors import qualitative

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv('master_time_series_imputed.csv', parse_dates=['Date'])
    # Rename internal columns to user-friendly labels
    df = df.rename(columns={
        'Price_USD': 'Price (USD)',
        'Market_Cap': 'Market Cap (USD)',
        'TVL (USD)': 'TVL (USD)',
        'Mining Difficulty': 'Mining Difficulty',
        'Full_Nodes': 'Full Nodes',
        'Energy_TWh': 'Energy (TWh)',
        'DeFi Protocol Count': 'DeFi Protocol Count',
        'Realized_Cap_B': 'Realized Cap (B USD)',
        'Exchange Netflow': 'Exchange Netflow'
    })
    return df

# Main application
st.set_page_config(page_title='Blockchain Metrics Dashboard', layout='wide', initial_sidebar_state='expanded')

df = load_data()

# Sidebar filters
with st.sidebar:
    st.header('Configure Filters')
    chains = st.multiselect('Chains', options=df['Chain'].unique(), default=list(df['Chain'].unique()))
    metrics = st.multiselect(
        'Metrics', options=[c for c in df.columns if c not in ['Date', 'Chain']],
        default=['Price (USD)', 'Market Cap (USD)']
    )
    start_date, end_date = st.date_input(
        'Date Range', [df['Date'].min(), df['Date'].max()], min_value=df['Date'].min(), max_value=df['Date'].max()
    )
    trad_defaults = ['Visa Avg TPS', 'Visa Peak TPS', 'SWIFT Avg Settlement (days)']
    trad_metrics = st.multiselect('Traditional Benchmarks', trad_defaults, default=trad_defaults)

# Define color sequence
color_seq = qualitative.Plotly

# Filter data
mask = (
    df['Chain'].isin(chains)
    & (df['Date'] >= pd.to_datetime(start_date))
    & (df['Date'] <= pd.to_datetime(end_date))
)
filtered = df.loc[mask]

# Create tabs
tab_objs, tab_ts, tab_cmp, tab_ins, tab_dl = st.tabs([
    'Objectives & Reg Events',
    'Time Series',
    'Compare Traditional',
    'Insights',
    'Download Data'
])

# Tab 1: Objectives & Regulatory Events
with tab_objs:
    st.title('Research Objectives')
    st.markdown(
        '''
        - Explain fundamental concepts of blockchain
        - Document current and future financial services use cases
        - Quantify benefits: cost, efficiency, transparency
        - Assess regulatory & security challenges
        - Analyze adoption trends and market sentiment
        '''
    )
    st.subheader('Key Regulatory Milestones')
    milestones = pd.DataFrame({
        'Date': [
            '2018-10-01', '2019-06-21', '2021-10-28',
            '2023-06-09', '2023-06-29', '2024-02-22',
            '2023-02-07', '2018-06-25'
        ],
        'Event': [
            'FATF 1st Guidance', 'FATF VASP Guidance', 'FATF Updated Guidance',
            'MiCA Published', 'MiCA In Force', 'MiCA Delegated Acts',
            'VARA Framework', 'ADGM Crypto Reg'
        ]
    })
    milestones['Date'] = pd.to_datetime(milestones['Date'])
    st.dataframe(milestones, use_container_width=True)

# Tab 2: Time Series
with tab_ts:
    st.title('Time Series Analysis')
    for metric in metrics:
        fig = px.line(
            filtered, x='Date', y=metric, color='Chain',
            title=metric, color_discrete_sequence=color_seq
        )
        fig.update_layout(hovermode='x unified', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Compare Traditional
with tab_cmp:
    st.title('Blockchain vs Traditional Benchmarks')
    b_metric = st.selectbox('Select Blockchain Metric', metrics)
    fig = px.line(
        filtered, x='Date', y=b_metric, color='Chain',
        title=f'{b_metric} vs Benchmarks', color_discrete_sequence=color_seq
    )
    # Add standard benchmark lines
    stds = {
        'Visa Avg TPS': 1700,
        'Visa Peak TPS': 65000,
        'SWIFT Avg Settlement (days)': 1.25
    }
    for name, val in stds.items():
        fig.add_hline(y=val, line_dash='dash', annotation_text=name)
    st.plotly_chart(fig, use_container_width=True)

# Tab 4: Insights
with tab_ins:
    st.title('Automated Insights & Inferences')
    if metrics:
        m0 = metrics[0]
        filtered['Pct_Change'] = filtered.groupby('Chain')[m0].pct_change()
        top_growth = filtered.nlargest(3, 'Pct_Change')[['Date', 'Chain', m0, 'Pct_Change']]
        st.subheader(f'Top 3 Weekly Growth in {m0}')
        st.dataframe(top_growth)
    if len(metrics) > 1:
        corr = filtered[metrics].corr()
        st.subheader('Correlation Matrix')
        st.dataframe(corr)
        st.markdown('**Inference:** High correlation indicates similar movement patterns among metrics.')

# Tab 5: Download Data
with tab_dl:
    st.title('Download Filtered Dataset')
    download_df = filtered[['Date', 'Chain'] + metrics]
    csv = download_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        'Download CSV', data=csv, file_name='filtered_data.csv', mime='text/csv'
    )
