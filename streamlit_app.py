import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.colors import qualitative

# Caching data
@st.cache_data
def load_data():
    df = pd.read_csv('master_time_series_imputed.csv', parse_dates=['Date'])
    # Rename columns for user-friendly labels
    df = df.rename(columns={
        'Price_USD': 'Price (USD)',
        'Market_Cap': 'Market Cap (USD)',
        'TVL_USD': 'TVL (USD)',
        'Difficulty_T': 'Mining Difficulty',
        'Full_Nodes': 'Full Nodes',
        'Energy_TWh': 'Energy (TWh)',
        'DeFi_Protocols': 'DeFi Protocol Count',
        'Realized_Cap_B': 'Realized Cap (B USD)',
        'Exchange_Netflow': 'Exchange Netflow'
    })
    return df

df = load_data()
st.set_page_config(page_title='Blockchain Metrics Dashboard', layout='wide', initial_sidebar_state='expanded')
# Sidebar configuration
st.sidebar.header('Configure Filters')
chains = st.sidebar.multiselect('Chains', options=df['Chain'].unique(), default=df['Chain'].unique())
metrics = st.sidebar.multiselect('Metrics', options=[c for c in df.columns if c not in ['Date','Chain']], default=['Price (USD)', 'Market Cap (USD)'])
start, end = st.sidebar.date_input('Date Range', [df['Date'].min(), df['Date'].max()])
# Traditional benchmarks defaults
trad_metrics = st.sidebar.multiselect('Traditional Metrics', ['Visa Avg TPS','Visa Peak TPS','SWIFT Avg Settlement (days)'], default=['Visa Avg TPS'])
# Color scheme
color_seq = qualitative.Plotly

# Filter
mask = (df['Chain'].isin(chains)) & (df['Date']>=pd.to_datetime(start)) & (df['Date']<=pd.to_datetime(end))
data = df.loc[mask]

# Tabs
tabs = st.tabs(['Objectives & Reg Events','Time Series','Compare Traditional','Insights','Download Data'])

# Tab 1: Objectives & Regulatory Events
with tabs[0]:
    st.title('Research Objectives')
    st.markdown(
        '- Explain fundamental concepts of blockchain
        - Document current and future financial services use cases
        - Quantify benefits: cost, efficiency, transparency
        - Assess regulatory & security challenges
        - Analyze adoption trends and market sentiment'
    )
    st.subheader('Key Regulatory Milestones')
    reg = pd.DataFrame({
        'Date': ['2018-10-01','2019-06-21','2021-10-28','2023-06-09','2023-06-29','2024-02-22','2023-02-07','2018-06-25'],
        'Event': [
            'FATF 1st Guidance','FATF VASP Guidance','FATF Updated Guidance','MiCA Published','MiCA In Force','MiCA Delegated Acts',
            'VARA Framework','ADGM Crypto Reg'
        ]
    })
    st.dataframe(reg)

# Tab 2: Time Series
with tabs[1]:
    st.title('Time Series Analysis')
    for metric in metrics:
        fig = px.line(data, x='Date', y=metric, color='Chain', title=metric, color_discrete_sequence=color_seq)
        fig.update_layout(hovermode='x unified', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Compare Traditional
with tabs[2]:
    st.title('Blockchain vs Traditional Benchmarks')
    std = {
        'Visa Avg TPS':1700,'Visa Peak TPS':65000,'SWIFT Avg Settlement (days)':1.25
    }
    metric = st.selectbox('Select Blockchain Metric', metrics)
    fig = px.line(data, x='Date', y=metric, color='Chain', title=f'{metric} vs Benchmarks', color_discrete_sequence=color_seq)
    for name,val in std.items(): fig.add_hline(y=val, line_dash='dash', annotation_text=name)
    st.plotly_chart(fig, use_container_width=True)

# Tab 4: Insights
with tabs[3]:
    st.title('Automated Insights & Inferences')
    # Show top 3 weeks by growth for selected metric
    if len(metrics)>0:
        m = metrics[0]
        data['Pct_Change'] = data.groupby('Chain')[m].pct_change()
        top = data.nlargest(3, 'Pct_Change')[['Date','Chain',m,'Pct_Change']]
        st.subheader(f'Top 3 Weekly Growth for {m}')
        st.dataframe(top)

    # Correlation summary
    if len(metrics)>1:
        corr = data[metrics].corr()
        st.subheader('Correlation Matrix')
        st.write(corr)
        st.markdown('**Inference:** High correlation indicates similar movement patterns among selected metrics.')

# Tab 5: Download
with tabs[4]:
    st.title('Download Data')
    df_download = data[['Date','Chain'] + metrics]
    csv = df_download.to_csv(index=False).encode('utf-8')
    st.download_button('Download Current View', data=csv, file_name='dashboard_data.csv')
