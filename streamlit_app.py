import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.colors import qualitative

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv('master_time_series_imputed.csv', parse_dates=['Date'])
    # Optional: rename columns for readability if needed
    return df

# Main application
st.set_page_config(page_title='Blockchain Metrics Dashboard', layout='wide', initial_sidebar_state='expanded')

df = load_data()

# Sidebar filters
with st.sidebar:
    st.header('Configure Filters')
    # Date range selector
    start_date, end_date = st.date_input(
        'Date Range', [df['Date'].min(), df['Date'].max()],
        min_value=df['Date'].min(), max_value=df['Date'].max()
    )
    # Metric selector (including chain-specific columns)
    metrics = st.multiselect(
        'Select Metrics', options=[c for c in df.columns if c != 'Date'],
        default=[c for c in df.columns if c not in ['Date']][:3]
    )
    # Traditional benchmarks selection
    trad_defaults = ['Visa Avg TPS', 'Visa Peak TPS', 'SWIFT Avg Settlement (days)']
    trad_metrics = st.multiselect('Traditional Benchmarks', options=trad_defaults, default=trad_defaults)

# Filter data by date
filtered = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

# Color sequence
color_seq = qualitative.Plotly

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    'Objectives & Reg Events',
    'Time Series',
    'Compare Traditional',
    'Insights',
    'Download Data'
])

# Tab1: Objectives & Regulatory Events
with tab1:
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
        'Date': pd.to_datetime([
            '2018-10-01','2019-06-21','2021-10-28',
            '2023-06-09','2023-06-29','2024-02-22',
            '2023-02-07','2018-06-25'
        ]),
        'Event': [
            'FATF 1st Guidance','FATF VASP Guidance','FATF Updated Guidance',
            'MiCA Published','MiCA In Force','MiCA Delegated Acts',
            'VARA Framework','ADGM Crypto Reg'
        ]
    })
    st.dataframe(milestones, use_container_width=True)

# Tab2: Time Series
with tab2:
    st.title('Time Series Analysis')
    if metrics:
        for m in metrics:
            fig = px.line(filtered, x='Date', y=m, title=m, color_discrete_sequence=color_seq)
            fig.update_layout(hovermode='x unified', template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Select at least one metric to display time series.')

# Tab3: Compare Traditional
with tab3:
    st.title('Compare with Traditional Benchmarks')
    blockchain_metric = st.selectbox('Blockchain Metric', options=metrics)
    if blockchain_metric:
        fig = px.line(filtered, x='Date', y=blockchain_metric, title=f'{blockchain_metric} vs Standards', color_discrete_sequence=color_seq)
        # Add benchmark lines
        standards = {
            'Visa Avg TPS':1700, 'Visa Peak TPS':65000, 'SWIFT Avg Settlement (days)':1.25
        }
        for name, val in standards.items():
            if name in trad_metrics:
                fig.add_hline(y=val, line_dash='dash', annotation_text=name)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Select a blockchain metric in the sidebar.')

# Tab4: Insights
with tab4:
    st.title('Insights & Inferences')
    if metrics:
        primary = metrics[0]
        filtered['Pct_Change'] = filtered[primary].pct_change()
        top3 = filtered.nlargest(3, 'Pct_Change')[['Date', primary, 'Pct_Change']]
        st.subheader(f'Top 3 Weekly Growth for {primary}')
        st.dataframe(top3)
    if len(metrics) > 1:
        corr = filtered[metrics].corr()
        st.subheader('Correlation Matrix')
        st.dataframe(corr)
        st.markdown('**High correlation indicates related movement among metrics.**')

# Tab5: Download Data
with tab5:
    st.title('Download Current View')
    download_df = filtered[['Date'] + metrics]
    csv = download_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download CSV', data=csv, file_name='filtered_data.csv', mime='text/csv')
