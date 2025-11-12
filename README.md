🏀 NBA Prop Model – Streamlit App


A professional-grade NBA Player Prop Model built for PRA (Points + Rebounds + Assists) and other markets, combining data-driven projections, risk management, and real-time analytics.
Designed for long-term profitability through advanced statistical modeling, bankroll discipline, and market calibration.

🚀 Features
🎯 Core Model

Monte Carlo engine (bootstrapped resampling, 1 000 runs) for realistic PRA/PTS/REB/AST distributions

Usage-, pace-, and opponent-adjusted projections (15-game rolling window)

Dual-source NBA API + SportsMetrics backup for automatic data refresh

Manual line input (for PrizePicks, Underdog, or sportsbook props)

📈 Analytics Engine

Expected Value (EV) and Closing Line Value (CLV) tracking

Fractional Kelly staking with a built-in 5 % max-loss daily cap

Dynamic variance and correlation weighting for combo bets

Automatic model calibration using hit-rate vs. EV-bucket performance

🧮 Advanced Tabs
Tab	Description
Model	Select players, enter lines, and run simulations
Results	View logs, EV trend, rolling hit-rate, and monthly performance
Calibration	Evaluate EV buckets, hit-rate stability, and receive tuning recommendations
Insights	Track edge sustainability, CLV/EV decay, correlation learning, and model health
📊 Key Metrics Explained
Metric	Meaning	Goal
EV (%)	Expected Value = (avg Payout – Cost) ÷ Cost × 100	Positive EV > 10 % = edge
CLV	Closing Line Value – how much better your entry was than the final market	Positive = beat market
Kelly Stake	Optimal bet sizing based on bankroll and EV	Fractional Kelly = safer growth
Variance & Skewness	Measure distribution volatility	Lower variance = more stable
Hit Rate	% of positive EV bets hitting > expected	Aim > 55 % for strong edge
💾 Data Logging

All results automatically save to a local file:

results_log.csv


Each row stores:

Player / Market / Line / Projection / EV / CLV / Variance / Stake / Skewness / P25 / P75 / Simulation Mean

Timestamped for trend, calibration, and insight charts

⚙️ Setup & Deployment
🔧 Local Setup
pip install -r requirements.txt
streamlit run app.py

☁️ Streamlit Cloud

Upload all files to GitHub

Go to share.streamlit.io
 and deploy your repo

Make sure .streamlit/config.toml and results_log.csv exist in the root

🧠 Model Evolution Roadmap

 Multi-API data sync (NBA API + SportsMetrics)

 Bootstrapped Monte Carlo simulation

 Live auto-refresh (24 h cycle)

 CLV / EV trend tracking

 Advanced calibration + feedback engine

 Edge sustainability / correlation learning

 Reinforcement learning for automated parameter tuning

 Player image & role-based impact adjustments

⚠️ Disclaimer

This application is for educational and entertainment purposes only.
No guarantee of profitability is made.
Always wager responsibly and within your means.
---

_Last updated: November 2025 • Version 1.0.0_
