import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb

# Load the data and model
df = pd.read_csv("your_cleaned_ppi_data.csv")
xgb_model = joblib.load("xgb_model.pkl")

st.set_page_config(page_title="🏏 Player Performance Index Dashboard", layout="wide")

st.title("🏏 Player Performance Index (PPI) Dashboard")
st.markdown("---")

# Sidebar Navigation
page = st.sidebar.selectbox("Select Page", [
    "Home",
    "Player Analysis",
    "Model Insights",
    "Visual Analytics"
])

# --- Home Page ---
if page == "Home":
    st.header("Project Overview")
    st.markdown("""
    This dashboard provides a **data-driven cricket ranking system** that goes beyond traditional statistics.
    
    **Features:**
    - Context-based metrics like Clutch Factor and Win Contribution
    - Machine Learning Models: Random Forest, XGBoost, Neural Network
    - SHAP explainability for model transparency
    - Visual Insights on Player and Team Performances
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Cricket_pitch.svg/1920px-Cricket_pitch.svg.png", caption="Cricket Analytics", use_container_width=True)

# --- Player Analysis Page ---
elif page == "Player Analysis":
    st.header("Player Performance Analysis")
    player = st.selectbox("Select a Batsman", df['batsman'].unique())
    player_data = df[df['batsman'] == player]

    st.subheader(f"Statistics for {player}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Runs", int(player_data['runs_batsman'].sum()))
    col2.metric("Clutch Runs", int(player_data['clutch'].sum()))
    col3.metric("Win Contribution", int(player_data['win_contribution'].sum()))

    st.subheader("Top 5 Innings (by Runs)")
    st.dataframe(player_data[['match_id', 'runs_batsman', 'ball', 'wicket']].sort_values(by='runs_batsman', ascending=False).head(5))

# --- Model Insights Page ---
elif page == "Model Insights":
    st.header("Model Performance and Interpretability")

    st.subheader("Accuracy Comparison")
    accuracies = {
        'Random Forest': 0.73,
        'XGBoost': 0.77,
        'Neural Network': 0.75
    }
    fig, ax = plt.subplots()
    ax.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'purple'])
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("SHAP Feature Importance (XGBoost Model)")

    # Sample a subset of data
    sample_size = 500
    X_sampled = df[['runs_batsman', 'runs_extras', 'runs_total', 'ball', 'wicket']].sample(n=sample_size, random_state=42)

    # SHAP explainer
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sampled)

    # Capture the figure properly
    shap.summary_plot(shap_values, X_sampled, show=False)
    fig = plt.gcf()  # "Get Current Figure"
    st.pyplot(fig)
    plt.close(fig)
# --- Visual Analytics Page ---
elif page == "Visual Analytics":
    st.header("Visual Analytics")

    st.subheader("Top 10 Clutch Players")
    clutch_df = df[df['clutch'] > 0].groupby('batsman')['clutch'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(clutch_df)

    st.subheader("Top 10 Win Contributors")
    win_contrib_df = df[df['win_contribution'] > 0].groupby('batsman')['win_contribution'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(win_contrib_df)

    st.subheader("Batsman vs Bowler Heatmap")
    pivot = df.pivot_table(index='batsman', columns='bowler', values='runs_batsman', aggfunc='sum').fillna(0)
    top_batsmen = pivot.sum(axis=1).sort_values(ascending=False).head(15).index
    top_bowlers = pivot.sum(axis=0).sort_values(ascending=False).head(15).index
    filtered_pivot = pivot.loc[top_batsmen, top_bowlers]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(filtered_pivot, cmap="coolwarm", annot=True, fmt=".0f", ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Match Result Distribution")
    result_counts = df['result'].value_counts()

    fig2, ax2 = plt.subplots()
    wedges, _ = ax2.pie(
        result_counts,
        startangle=140,
        radius=1,
        wedgeprops=dict(width=0.5)
    )
    centre_circle = plt.Circle((0, 0), 0.75, fc='white')
    fig2.gca().add_artist(centre_circle)
    ax2.axis('equal')

    labels_with_pct = [
        f"{country} - {percent:.1f}%" for country, percent in 
        zip(result_counts.index, 100*result_counts/result_counts.sum())
    ]
    ax2.legend(
        wedges,
        labels_with_pct,
        title="Match Results",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=8,
        title_fontsize=10
    )
    plt.title("Match Result Distribution")
    st.pyplot(fig2)
    plt.close(fig2)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, SHAP, and XGBoost.")
