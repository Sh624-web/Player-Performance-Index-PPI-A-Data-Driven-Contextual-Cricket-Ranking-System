# ppi_dashboard.py

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# Load data and model
df = pd.read_csv("your_cleaned_ppi_data.csv")
model = joblib.load("xgb_model.pkl")

# Title
st.title("🏏 Player Performance Index (PPI) Dashboard")

# Sidebar
st.sidebar.header("Navigation")
option = st.sidebar.radio("Select a Page:", ["Summary", "Visualizations", "Model Predictions", "SHAP Explainability"])

# Pages
if option == "Summary":
    st.header("Data Overview")
    st.dataframe(df.head(50))

    st.subheader("Top Batsmen by Runs")
    top_batsmen = df.groupby('batsman')['runs_batsman'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_batsmen)

    st.subheader("Top Bowlers by Wickets")
    top_bowlers = df[df['wicket'] == 1].groupby('bowler').size().sort_values(ascending=False).head(10)
    st.bar_chart(top_bowlers)

elif option == "Visualizations":
    st.header("Visualizations")

    st.subheader("Average Run Rate by Ball")
    run_rate = df.groupby(df['ball'].astype(int))['runs_total'].mean()
    st.line_chart(run_rate)

    st.subheader("Wickets by Ball Number")
    wickets = df[df['wicket'] == 1].groupby(df['ball'].astype(int)).size()
    st.bar_chart(wickets)

    st.subheader("Match Result Distribution")
    result_counts = df['result'].value_counts()
    st.bar_chart(result_counts)

elif option == "Model Predictions":
    st.header("Model Predictions")

    sample = df[['runs_batsman', 'runs_extras', 'runs_total', 'ball', 'wicket']].sample(5)
    preds = model.predict(sample)

    st.write("Sample Predictions:")
    st.dataframe(sample.assign(predicted_result=preds))

elif option == "SHAP Explainability":
    st.header("SHAP Feature Importance")

    sample = df[['runs_batsman', 'runs_extras', 'runs_total', 'ball', 'wicket']].sample(100)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    st.pyplot(fig)

# Footer
st.sidebar.info("Built with ❤️ by Shardul Bhandari - PPI Project")
