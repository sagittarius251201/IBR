from pathlib import Path
import os

import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def load_chain_data():
    return pd.read_csv(Path(__file__).parent / 'data' / 'master_chain_metrics_updated.csv', parse_dates=['date'])

@st.cache_data
def load_regulatory_data():
    return pd.read_csv(Path(__file__).parent / 'data' / 'regulatory_milestones.csv', parse_dates=['Date'])

st.set_page_config(page_title='Blockchain Dashboard', layout='wide')

st.sidebar.title("Navigation")
tabs = ["Research Objectives", "Regulatory Decisions", "Data Overview", "Insights", "Comparison"]
choice = st.sidebar.radio("Go to", tabs)

chain_data = load_chain_data()
reg_data = load_regulatory_data()

if choice == "Research Objectives":
    st.title("Research Objectives")
    st.markdown("""

1. Explain core blockchain concepts.
2. Document current and future applications in finance.
3. Quantify benefits: cost, efficiency, transparency.
4. Examine regulatory & security challenges.
5. Analyze adoption rates & market trends.
    """)

elif choice == "Regulatory Decisions":
    st.title("Regulatory & Institutional Milestones")
    st.dataframe(reg_data)

elif choice == "Data Overview":
    st.title("Data Overview")
    st.write(chain_data.head())
    st.write("Data loaded with {} rows and {} columns.".format(*chain_data.shape))

elif choice == "Insights":
    st.title("Dynamic Insights")
    metric = st.selectbox("Select Metric", [col for col in chain_data.columns if col not in ['date','chain']])
    chain = st.selectbox("Select Chain", chain_data['chain'].unique())
    df = chain_data[chain_data['chain']==chain]
    latest = df.iloc[-1][metric]
    st.metric(label=f"{metric} ({chain}) Latest", value=round(latest,4))
    st.line_chart(df.set_index('date')[metric])

elif choice == "Comparison":
    st.title("Chain Comparison")
    metric = st.selectbox("Metric to Compare", [col for col in chain_data.columns if col not in ['date','chain']])
    df_pivot = chain_data.pivot(index='date', columns='chain', values=metric)
    fig = px.line(df_pivot, labels={'value':metric, 'date':'Date'})
    st.plotly_chart(fig, use_container_width=True)
