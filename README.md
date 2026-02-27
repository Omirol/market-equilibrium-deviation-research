# Market Equilibrium Deviation Research (Project Y)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Numba](https://img.shields.io/badge/Numba-JIT%20Optimized-orange.svg)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Keras%20%7C%20Transformers-red.svg)
![Binance](https://img.shields.io/badge/Exchange-Binance%20Futures-yellow.svg)

## 📌 Overview
**Project Y** is an advanced quantitative research and algorithmic trading framework focused on discovering and exploiting market microstructure anomalies in high-frequency data streams (Binance USDⓈ-M Futures). 

This repository documents the transition from theoretical statistical edge formulation ("Step Level" anomaly detection) to a robust, low-latency execution engine. The core philosophy treats the market order book not just as a financial ledger, but as a physical environment influenced by pressure, gravity, and velocity.

## 🔬 Core Research Concepts

### 1. Step Level Anomalies & "Streaks"
Extensive backtesting and statistical analysis were conducted to evaluate the parity between trend continuation and reversal across micro-structural steps.
- **Hypothesis:** Specific "Step Levels" provide a measurable, long-term statistical edge.
- **Findings:** Traditional Machine Learning models (e.g., Random Forest) fail to consistently outperform random chance in this specific microstructure environment. This necessitated the development of proprietary sequence modeling and custom anomaly detection techniques.

### 2. Live Physics Engine for Market Microstructure
Instead of relying solely on traditional technical indicators, the execution engine models the order book using physical concepts:
- **Gravity & Pressure:** Calculating the magnetic pull of large limit orders.
- **Velocity & Walls:** Measuring the speed of tape execution against resistance walls to predict micro-breakouts.

### 3. Market Data Tokenization & Transformers
Adapting NLP concepts for financial time-series:
- Raw tick data is categorized into discrete `speed_bins`, `vol_bins`, and `zone_bins`.
- This "Market Vocabulary" is fed into custom **Transformer** and **LSTM** architectures to process order-flow context the same way an LLM processes natural language.
- Implementation of **Focal Loss** to mitigate extreme class imbalances inherent in market movements.

## ⚙️ System Architecture & Engineering

The execution framework is built for resilience, speed, and autonomy under extreme market volatility.

* **Low-Latency Execution:** Critical path trading logic is optimized using `Numba (@jit)` to achieve near C-level execution speeds in Python.
* **Asynchronous Data Pipeline:** Robust WebSocket connection management handling real-time tick data, featuring automatic fallback mechanisms and drop-recovery protocols.
* **Distributed Orchestration (Telegram Manager):** A custom backend manager utilizing `asyncio` and the Telegram API for remote server monitoring, multi-bot orchestration, and process memory management (`psutil`).
* **Multi-Layer Risk Management:** "Paranoid Sync" concepts ensuring local bot states strictly match exchange server states, preventing ghost positions and executing emergency SL/TP dynamically.

## 📊 Current Status & Motivation

This project is currently paused for strategic reassessment and community feedback. 

**Author's Note:** The R&D and engineering of this system have been conducted under extreme geopolitical conditions in Ukraine. The necessity to build fully autonomous, fault-tolerant systems is not just a theoretical preference, but a practical requirement dictated by reality (including infrastructure disruptions and ballistic threats). 

The primary motivation behind releasing this documentation is to preserve the R&D results and seek critical, professional feedback from the quantitative finance and algorithmic trading community. 

## 🛠 Tech Stack
- **Core:** Python, Numba, AsyncIO, Pandas, Numpy.
- **ML/AI:** Keras, TensorFlow (Transformers, LSTM), LightGBM, CatBoost, Scikit-learn, Optuna.
- **Infrastructure:** Binance UM-Futures API (REST & WebSockets), python-telegram-bot, psutil.

## 🤝 Let's Connect
Constructive critique, technical advice, or strategic guidance are highly appreciated. 

**Denys Ivanok** Quant Researcher & System Architect | ex-CFO  
[LinkedIn Profile](https://www.linkedin.com/in/denysivanok/)
