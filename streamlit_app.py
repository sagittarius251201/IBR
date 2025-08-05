import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.colors import qualitative

# Caching data
@st.cache_data
def load_data():
    df = pd.read_csv('master_time_series_imputed.csv', parse_dates=['Date'])
    # Drop duplicate columns
    df = df.loc[:,~df.columns.duplicated()]
    return df

# App config
st.set_page_config(page_title='Blockchain Metrics Dashboard', layout='wide', initial_sidebar_state='expanded')
df = load_data()

# Sidebar
with st.sidebar:
    st.header('Filters & Settings')
    # Chain filter
    chains = st.multiselect('Select Chains', options=df['Chain'].unique(), default=list(df['Chain'].unique()))
    # Date range
    start_date, end_date = st.date_input('Date Range', [df['Date'].min(), df['Date'].max()],
                                         min_value=df['Date'].min(), max_value=df['Date'].max())
    # Metric selection
    available_metrics = [c for c in df.columns if c not in ['Date','Chain']]
    metrics = st.multiselect('Select Metrics', options=available_metrics,
                             default=['Price (USD)','Market Cap (USD)'] if 'Price (USD)' in available_metrics and 'Market Cap (USD)' in available_metrics else available_metrics[:2])
    # Traditional benchmarks
    trad_options = {'Visa Avg TPS':1700,'Visa Peak TPS':65000,'SWIFT Avg Settlement (days)':1.25}
    trad_metrics = st.multiselect('Traditional Benchmarks', options=list(trad_options.keys()), default=list(trad_options.keys()))
    # Chart type
    chart_type = st.selectbox('Chart Type', ['Line','Area','Bar','Scatter'])
    st.markdown('---')
    st.write('**Color Theme**')
    color_seq = qualitative.Plotly

# Filter data
data = df[(df['Chain'].isin(chains)) & (df['Date']>=pd.to_datetime(start_date)) & (df['Date']<=pd.to_datetime(end_date))]

# Tabs
tabs = st.tabs(['Objectives & Events','Time Series','Comparison','Insights','Download'])

# Tab: Objectives & Events
with tabs[0]:
    st.title('Research Objectives & Key Events')
    st.write('''
**Objectives:**
- Explain core blockchain concepts
- Document present & future applications in finance
- Quantify benefits: cost, efficiency, transparency
- Assess regulatory & security challenges
- Analyze adoption trends & market sentiment
    ''')
    st.subheader('Important Regulatory Events & Impact')
    event_data = [
        ('2018-10-01','FATF 1st Guidance','Global','Defined VASP AML/CFT requirements, kickstarting global regulation'),
        ('2019-06-21','FATF VASP Guidance','Global','Introduced “Travel Rule”, aligning crypto transfers with banking norms'),
        ('2021-10-28','FATF Updated Guidance','Global','Expanded to stablecoins, DeFi & NFTs, driving industry self-regulation'),
        ('2023-06-09','MiCA Published','EU','Unified EU crypto framework, clarifying asset definitions'),
        ('2023-06-29','MiCA In Force','EU','Marked start of regulated crypto services in Europe'),
        ('2024-02-22','MiCA Delegated Acts','EU','Enacted technical standards for stablecoins & oversight'),
        ('2023-02-07','VARA Framework','Dubai','Established Dubai as global crypto hub with clear licensing'),
        ('2018-06-25','ADGM Regime','Abu Dhabi','First MENA region regulatory sandbox for crypto-asset activities')
    ]
    events_df = pd.DataFrame(event_data, columns=['Date','Event','Region','Impact'])
    events_df['Date'] = pd.to_datetime(events_df['Date'])
    st.dataframe(events_df, use_container_width=True)

# Tab: Time Series
with tabs[1]:
    st.title('Time Series Analysis')
    if metrics:
        for m in metrics:
            if chart_type=='Line': fig = px.line(data, x='Date', y=m, color='Chain', title=m, color_discrete_sequence=color_seq)
            elif chart_type=='Area': fig = px.area(data, x='Date', y=m, color='Chain', title=m, color_discrete_sequence=color_seq)
            elif chart_type=='Bar': fig = px.bar(data, x='Date', y=m, color='Chain', barmode='group', title=m)
            else: fig = px.scatter(data, x='Date', y=m, color='Chain', title=m)
            fig.update_layout(hovermode='x unified', template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning('Select at least one metric!')

# Tab: Comparison
with tabs[2]:
    st.title('Blockchain vs Traditional Benchmarks')
    compare_metric = st.selectbox('Choose Metric', options=metrics)
    fig = px.line(data, x='Date', y=compare_metric, color='Chain', title=f'{compare_metric} vs Benchmarks', color_discrete_sequence=color_seq)
    for name,val in trad_options.items():
        if name in trad_metrics: fig.add_hline(y=val, line_dash='dash', annotation_text=name)
    fig.update_layout(hovermode='x unified', template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

# Tab: Insights
with tabs[3]:
    st.title('Insights & Inferences')
    if metrics:
        primary=metrics[0]
        data['Pct_Change']=data.groupby('Chain')[primary].pct_change()
        top3=data.nlargest(3,'Pct_Change')[['Date','Chain',primary,'Pct_Change']]
        st.subheader(f'Top 3 Weekly Growth: {primary}')
        st.dataframe(top3)
    if len(metrics)>1:
        corr=data[metrics].corr()
        st.subheader('Correlation Matrix')
        st.dataframe(corr)
        st.write('_High correlation suggests similar trends between metrics_')

# Tab: Download
with tabs[4]:
    st.title('Download Filtered Data')
    dl_df=data[['Date','Chain']+metrics]
    csv=dl_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download CSV', data=csv, file_name='dashboard_export.csv', mime='text/csv')
