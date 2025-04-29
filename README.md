# 🏏 Player Performance Index (PPI) – A Data-Driven Contextual Cricket Ranking System

This repository contains the full implementation of the MSc Data Science final project titled:  
**“Player Performance Index (PPI): A Data-Driven Contextual Cricket Ranking System”**, conducted at the University of Hertfordshire.

📍 GitHub Repo: https://github.com/Sh624-web/Player-Performance-Index-PPI-A-Data-Driven-Contextual-Cricket-Ranking-System

---

## 📘 Project Summary

Traditional cricket metrics like batting average and strike rate fail to reflect the situational value of a player's performance. This project introduces a **context-aware ranking system** that evaluates players based on **Clutch Factor**, **Win Contribution**, and **Game-Changing Innings**, using:

- 🧠 Machine Learning Models: **Random Forest**, **XGBoost**, **Neural Network**
- 🛠️ Model Interpretability: **SHAP values**
- 🌐 Visualization: **Streamlit dashboard**

---

## 🧱 Repository Structure

│ ├── data/ # Processed cricket datasets (CSV format) ├── notebooks/ # Jupyter Notebooks for feature engineering & modelling ├── dashboard/ # Streamlit app for PPI exploration ├── outputs/ # Visualizations, SHAP plots, confusion matrices ├── utils/ # Custom functions (preprocessing, feature creation) ├── requirements.txt # Python dependencies ├── main.py # Main script to execute training and evaluation └── README.md # Project overview

yaml

---

## ⚙️ Installation & Usage

1. **Clone the Repository**
```
git clone https://github.com/Sh624-web/Player-Performance-Index-PPI-A-Data-Driven-Contextual-Cricket-Ranking-System
cd Player-Performance-Index-PPI
Install Dependencies


pip install -r requirements.txt
Run Model Pipeline


python main.py
Launch Dashboard (optional)


streamlit run dashboard/app.py
📊 Features
Clutch Factor: Player performance in pressure moments

Win Contribution: Influence on team’s win probability

Game-Changing Innings: Key innings that reverse match momentum

SHAP Analysis: Feature contribution and local/global interpretation

Streamlit Dashboard: Explore PPI rankings, filters by team/match

🔍 Data Source
Ball-by-ball cricket data was sourced from Cricsheet.org in YAML format. Python scripts convert this into structured datasets for analysis.

📁 Output Highlights
Confusion Matrices for each ML model

SHAP value summary plots

Comparison bar chart for model accuracy

Top performers by context metrics

All visualizations are included in the report and the outputs/ directory.

🧠 Models Used

Model	Accuracy	Precision	Recall	F1-Score
Random Forest	22%	13%	22%	17%
XGBoost	23%	14%	23%	18%
Neural Network	23%	14%	23%	18%
Neural Network achieved the best overall performance.

✅ Dependencies
Python 3.8+

pandas, numpy

scikit-learn

xgboost

keras / tensorflow

shap

streamlit

matplotlib / seaborn

👨‍💻 Author
Shardul Bhandari
MSc Data Science, University of Hertfordshire
Supervisor: Darshan Kakkad
Date: April 2025

📜 License
This repository is for educational and academic research use only.

📂 Report
The full project report (with figures and results) is submitted on Canvas. It includes methodology, analysis, results, and future recommendations.

📝 Acknowledgements
Cricsheet for open-source cricket data

University of Hertfordshire for academic support

Libraries: SHAP, Scikit-learn, Keras, XGBoost, Streamlit

