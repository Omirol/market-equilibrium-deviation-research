# Market Equilibrium & Statistical Deviation in Crypto Microstructure

**Repository Type:** Research Archive / Empirical Study  
**Status:** Paused for Analysis & Calibration  
**License:** All Rights Reserved  

---

## 1. Abstract
This repository documents an engineering investigation into the hypothesis of **Global Market Equilibrium** applied to cryptocurrency perpetual futures. The primary objective was to determine if high-volatility assets (BTC, ETH, SOL) exhibit exploitable statistical deviations ("Managed Imbalance") within a predominantly efficient, maximum-entropy market structure.

This project is **not** a commercial trading bot product. It is a documented attempt to structure market chaos using Event-Driven architecture, outlining both the methodological successes and the limitations of Machine Learning in this domain.

## 2. The Core Thesis
> **Hypothesis:** While crypto markets converge to a 50/50 directional probability over macro horizons, volatility clustering creates temporary, path-dependent "Regimes" where entropy decreases.

**Key Findings:**
* **Equilibrium Dominance:** Standard time-based series demonstrate efficient balancing mechanisms. Directional probability reverts to the mean (~50%) across large sample sizes.
* **ML Failure:** Traditional models (LSTM, Transformers, Random Forest) trained on time-series data failed to consistently outperform random chance, suffering from noise overfitting.
* **The "Event" Pivot:** Transitioning from *Time-Based* to *Volatility-Based* (Step Level) representation revealed statistically significant deviations.

## 3. Methodology
The research moved away from time-based candles to an **Event-Driven Architecture**:

1.  **Adaptive Step Levels:** Discarding time to focus solely on price vector changes relative to volatility (ATR-based dynamic grids).
2.  **Behavioral Classification:** Categorizing market states not just by price direction, but by structural integrity (Continuation vs. Reversal).
3.  **Rolling Window Entropy:** Measuring the probability drift within specific event windows to identify "Fat Tail" events.

## 4. Empirical Results (Backtest Sample)
*Data Period: Volatility Clusters 2024-2025* *Method: Event-Driven Step Logic (No ML)*

| Asset | Events Sample | Win Rate | Net Edge (w/o fees) |
|-------|---------------|----------|---------------------|
| **BTCUSDT** | 654 | **58.9%** | **17.7%** |
| **ETHUSDT** | 643 | **56.6%** | **13.2%** |
| **SOLUSDT** | 1,285 | **56.0%** | **12.1%** |
| **DOTUSDT** | 773 | 56.0% | 12.0% |
| **AVAXUSDT**| 1104 | 53.2% | 6.3% |

*Note: Results are sensitive to execution latency and micro-liquidity (wick) variance.*

## 5. Technical Architecture
The provided code (`new_logic_2.py`) represents the execution engine built to test these theories in a live environment:
* **Orchestrator:** Python 3.11 `asyncio` core for zero-latency WebSocket processing.
* **Risk Protocol:** Hard-coded 3-Tier Circuit Breaker (Drawdown protection).
* **Infrastructure:** AWS deployment with Telegram-based telemetry.

## 6. Current Limitations & Open Questions
Despite the identified edge, the system faces challenges that halted commercial scaling:
1.  **Liquidity Friction:** Theoretical step-levels often suffer from execution slippage during high-volatility impulses.
2.  **Regime Shifts:** The optimal window size is not static. We are investigating methods for **Dynamic Window Governance** without falling into the trap of curve-fitting.
3.  **Fee Drag:** The edge narrows significantly when factoring in standard exchange fee tiers for high-frequency rotation.

## 7. Contact & Feedback
This research was conducted in Chernihiv, Ukraine, under constraints that prioritized robust, autonomous engineering.

I am releasing this archive to seek **rigorous critique** from the quantitative finance community regarding:
* Methodologies for regime classification without look-ahead bias.
* Statistical validity of the identified "Fat Tails".

**Docs:** See [Project_Y_Research_Report.pdf](Project_Y_Research_Report.pdf) for full charts and data.
