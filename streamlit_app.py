import streamlit as st
import pandas as pd
import plotly.express as px

# Cache the data load for performance
@st.cache_data
# Load the time series data; ensure the CSV is in the same folder as this script
def load_data():
    df = pd.read_csv('master_time_series_imputed.csv', parse_dates=['Date'])
    return df

# Main
st.set_page_config(page_title='Blockchain Metrics Dashboard', layout='wide')
df = load_data()

# Sidebar controls
st.sidebar.title('Configuration')
# Date range selector
date_min, date_max = df['Date'].min(), df['Date'].max()
start_date, end_date = st.sidebar.date_input('Date Range', [date_min, date_max], min_value=date_min, max_value=date_max)
# Metric selector
metrics = st.sidebar.multiselect('Select Metrics', options=[c for c in df.columns if c != 'Date'], default=[c for c in df.columns if c != 'Date'][:3])
# Chart type selector
chart_type = st.sidebar.selectbox('Chart Type', ['Line', 'Area', 'Bar', 'Scatter'])

# Filter data by date
mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
data = df.loc[mask]

# Tabs
tab1, tab2, tab3 = st.tabs(['Time Series', 'Correlation', 'Download Data'])

with tab1:
    st.header('Time Series Analysis')
    for m in metrics:
        if chart_type == 'Line':
            fig = px.line(data, x='Date', y=m, title=m)
        elif chart_type == 'Area':
            fig = px.area(data, x='Date', y=m, title=m)
        elif chart_type == 'Bar':
            fig = px.bar(data, x='Date', y=m, title=m)
        else:
            fig = px.scatter(data, x='Date', y=m, title=m)
        # Unified hover for better insights
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header('Correlation Matrix')
    if len(metrics) > 1:
        corr = data[metrics].corr()
        fig = px.imshow(corr, text_auto=True, title='Correlation Heatmap')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Select at least two metrics to view correlation.')

with tab3:
    st.header('Download Filtered Data')
    download_df = data[['Date'] + metrics]
    csv = download_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download CSV', data=csv, file_name='filtered_data.csv', mime='text/csv')
