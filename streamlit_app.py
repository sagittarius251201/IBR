
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

# Caching data load
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('master_time_series_imputed.csv', parse_dates=['Date'])
    return df

df = load_data()

st.set_page_config(page_title='Blockchain Metrics Dashboard', layout='wide')

# Sidebar
st.sidebar.title('Configuration')
chains = st.sidebar.multiselect('Select Chains', options=df['Chain'].unique(), default=df['Chain'].unique()[:3])
start_date, end_date = st.sidebar.date_input('Date Range', [df['Date'].min(), df['Date'].max()])
metrics = st.sidebar.multiselect('Select Metrics', options=[col for col in df.columns if col not in ['Date','Chain']], default=['Difficulty_T'])

chart_type = st.sidebar.selectbox('Chart Type', ['Line', 'Area', 'Bar', 'Scatter'])

# Filter data
data = df[df['Chain'].isin(chains)]
data = data[(data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))]

# Tabs
tab1, tab2, tab3 = st.tabs(['Time Series', 'Correlation', 'Data Download'])

with tab1:
    st.header('Time Series Analysis')
    for metric in metrics:
        if chart_type == 'Line':
            fig = px.line(data, x='Date', y=metric, color='Chain', title=f'{metric} over Time', hover_data={'Chain':True})
        elif chart_type == 'Area':
            fig = px.area(data, x='Date', y=metric, color='Chain', title=f'{metric} over Time', hover_data={'Chain':True})
        elif chart_type == 'Bar':
            fig = px.bar(data, x='Date', y=metric, color='Chain', barmode='group', title=f'{metric} over Time')
        else:
            fig = px.scatter(data, x='Date', y=metric, color='Chain', title=f'{metric} over Time', hover_data=['Chain'])
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header('Correlation Matrix')
    corr_df = data.pivot(index='Date', columns='Chain', values=metrics).corr()
    fig_corr = px.imshow(corr_df, text_auto=True, title='Correlation Heatmap')
    st.plotly_chart(fig_corr, use_container_width=True)

with tab3:
    st.header('Download Filtered Data')
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button('Download CSV', data=csv, file_name='filtered_data.csv', mime='text/csv')
