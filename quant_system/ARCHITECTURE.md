# Quant System v1.0 — Production Architecture

## System Architecture (Text Diagram)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         QUANT SYSTEM v1.0 — END TO END                          │
│                                                                                  │
│  "If this system fails, here is exactly why — and how to detect it early."      │
└─────────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────────────┐
                    │     DATA INGESTION LAYER          │
                    │                                   │
                    │  ┌─────────┐  ┌──────────────┐   │
                    │  │Real-time│  │  Historical   │   │
                    │  │ Odds    │  │  SG / Stats   │   │
                    │  │ Polling │  │  (Parquet)    │   │
                    │  └────┬────┘  └──────┬───────┘   │
                    │       │              │            │
                    │  ┌────┴──────────────┴──────┐    │
                    │  │    Line Tracker           │    │
                    │  │  (every 5-15 min)         │    │
                    │  │  Stores: LineSnapshot DB   │    │
                    │  └────────────┬──────────────┘    │
                    └───────────────┼───────────────────┘
                                    │
                    ┌───────────────┼───────────────────┐
                    │     FEATURE PIPELINE              │
                    │                                   │
                    │  NBA: Per-min, B2B, refs, pace    │
                    │  Golf: SG, course fit, form       │
                    │                                   │
                    │  Features versioned + validated    │
                    │  Snapshot saved with every bet     │
                    └───────────────┼───────────────────┘
                                    │
                    ┌───────────────┼───────────────────┐
                    │     MODELING LAYER                 │
                    │                                   │
                    │  NBA: Bootstrap MC (10K sims)     │
                    │  Golf: 5-engine ensemble          │
                    │                                   │
                    │  Output: model_prob, projection    │
                    │          model_std, confidence     │
                    └───────────────┼───────────────────┘
                                    │
                    ┌───────────────▼───────────────────┐
                    │                                   │
                    │     ╔═══════════════════════╗     │
                    │     ║   QUANT ENGINE        ║     │
                    │     ║   (Master Orchestrator)║     │
                    │     ╚═══════════╤═══════════╝     │
                    │                 │                  │
                    │    ┌────────────┼────────────┐    │
                    │    │            │            │    │
                    │    ▼            ▼            ▼    │
                    │ ┌──────┐  ┌─────────┐  ┌──────┐  │
                    │ │Sharp │  │ Edge    │  │Kelly │  │
                    │ │Money │  │Validator│  │Sizing│  │
                    │ │Check │  │         │  │      │  │
                    │ └──┬───┘  └────┬────┘  └──┬───┘  │
                    │    │           │           │      │
                    │    └─────┬─────┘    ┌─────┘      │
                    │          │          │             │
                    │          ▼          ▼             │
                    │    ┌───────────────────────┐      │
                    │    │   BET DECISION        │      │
                    │    │   approved: bool      │      │
                    │    │   stake: $X.XX        │      │
                    │    │   rejection_reason    │      │
                    │    └──────────┬────────────┘      │
                    │               │                   │
                    └───────────────┼───────────────────┘
                                    │
                         ┌──────────┼──────────┐
                         │    IF APPROVED       │
                         │                      │
                         ▼                      ▼
                ┌────────────────┐    ┌─────────────────┐
                │  BET LOGGER    │    │ EXPOSURE CHECK   │
                │  (Immutable)   │    │ (Portfolio-level)│
                │                │    │                  │
                │  Logs:         │    │  Player cap: 5%  │
                │  - timestamp   │    │  Total cap: 25%  │
                │  - model_prob  │    │  Daily cap: 8%   │
                │  - market_prob │    │  Correlation adj  │
                │  - edge        │    │                  │
                │  - features    │    └─────────────────┘
                │  - stake       │
                └────────┬───────┘
                         │
            ╔════════════╧════════════════════════════════════════╗
            ║                AFTER SETTLEMENT                     ║
            ╠════════════════════════════════════════════════════╝
            │
            ▼
    ┌───────────────┐     ┌──────────────────┐     ┌─────────────────┐
    │ CLV TRACKER   │     │ CALIBRATION      │     │ FAILURE         │
    │               │     │ MONITOR          │     │ PROTECTION      │
    │ beat_close?   │     │                  │     │                 │
    │ clv_cents     │     │ predicted vs     │     │ Drawdown: 15/30 │
    │ clv_raw       │     │ actual win rate  │     │ Streak: 8/12/18 │
    │               │     │ Brier score      │     │ Variance: 2x/3x │
    │ Rolling:      │     │ MAE tracking     │     │ Daily limits    │
    │  50/100/250   │     │                  │     │                 │
    └───────┬───────┘     └────────┬─────────┘     └───────┬─────────┘
            │                      │                       │
            └──────────┬───────────┘                       │
                       │                                   │
                       ▼                                   │
            ┌──────────────────────┐                       │
            │   EDGE VALIDATOR     │◄──────────────────────┘
            │                      │
            │  "Do I have edge?"   │
            │                      │
            │  → ACTIVE            │
            │  → REDUCED  (50%)    │
            │  → SUSPENDED (stop)  │
            │  → KILLED  (off)     │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │   FEEDBACK LOOP      │
            │   (Self-Learning)    │
            │                      │
            │  1. Model drift      │
            │  2. Feature health   │
            │  3. Auto-reweight    │
            │  4. Recommendations  │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │   DASHBOARD          │
            │                      │
            │  CLV (most important)│
            │  ROI vs expected     │
            │  Drawdown curve      │
            │  Calibration plots   │
            │  Feature performance │
            │  MC projections      │
            │  Circuit breakers    │
            └──────────────────────┘
```

## Module Map

```
quant_system/
├── engine.py                 ← MASTER ORCHESTRATOR (start here)
├── core/
│   ├── types.py              ← All dataclasses (BetRecord, CLVResult, etc.)
│   ├── bet_logger.py         ← Immutable bet audit trail
│   ├── clv_tracker.py        ← Closing Line Value tracking
│   ├── edge_validator.py     ← Daily "do I have edge?" check
│   └── calibration.py        ← Predicted vs actual monitoring
├── risk/
│   ├── kelly_adaptive.py     ← Dynamic Kelly (1/10 base, adaptive)
│   ├── bankroll_manager.py   ← P&L, drawdown, limits
│   ├── failure_protection.py ← Circuit breakers (cannot be overridden)
│   └── exposure_manager.py   ← Correlation-aware portfolio limits
├── backtest/
│   ├── walk_forward.py       ← Walk-forward validation (no data leakage)
│   ├── mc_bankroll.py        ← 10,000 path Monte Carlo simulation
│   └── feature_ablation.py   ← Feature importance by systematic removal
├── market/
│   ├── line_tracker.py       ← Line movement recording
│   └── sharp_detector.py     ← Sharp vs soft book divergence
├── learning/
│   ├── model_drift.py        ← PSI, KS test, accuracy degradation
│   ├── feature_monitor.py    ← Per-feature health tracking
│   └── feedback_loop.py      ← Self-improvement orchestrator
├── dashboard/
│   └── reporting.py          ← Full dashboard data generation
└── db/
    └── schema.py             ← SQLAlchemy models (6 tables)
```

## Database Schema (6 tables)

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `bet_log` | Every bet ever placed | bet_id, model_prob, market_prob, edge, stake, pnl, features_snapshot |
| `line_snapshots` | Time-series of all line movements | player, stat_type, source, line, captured_at, is_opening, is_closing |
| `clv_log` | CLV measurement per settled bet | bet_id, clv_raw, clv_cents, beat_close |
| `calibration_log` | Periodic calibration snapshots | bucket_label, predicted_avg, actual_rate, calibration_error |
| `system_state_log` | Audit trail of state changes | previous_state, new_state, reason, clv_at_change |
| `feature_log` | Feature importance over time | feature_name, importance_score, is_degraded |

## Threshold Configuration

### CLV Thresholds
- CLV < 0 over 100 bets → **REDUCED** (halve bet sizes)
- CLV < 0 over 250 bets → **SUSPENDED** (stop betting)
- CLV positive with p < 0.05 → **confirmed edge**

### Drawdown Circuit Breakers
- 15% drawdown → **REDUCED**
- 30% drawdown → **SUSPENDED**
- 45% drawdown → **KILLED**

### Calibration
- MAE < 4% → well calibrated (boost Kelly 15%)
- MAE > 8% → retrain required
- MAE > 15% → suspend immediately

### Kelly Sizing
- Base: **1/10 Kelly** (not 1/4, not 1/2)
- Floor: 1/50 Kelly minimum
- Ceiling: 1/4 Kelly maximum
- Hard cap: **3% of bankroll per bet** (never exceeded)
- Daily exposure: 15% maximum
- Player exposure: 5% maximum

### Recovery Requirements
- SUSPENDED → ACTIVE: 30 bets of positive CLV + 24h cooldown
- KILLED → review: 72h minimum cooldown + manual approval

## Implementation Priority

### Phase 1: Foundation (Week 1-2) — BUILT
1. ✅ Database schema (6 tables)
2. ✅ Bet logger (immutable audit trail)
3. ✅ CLV tracker (the most important metric)
4. ✅ Edge validator (daily health check)
5. ✅ Adaptive Kelly (dynamic sizing)
6. ✅ Bankroll manager (P&L tracking)
7. ✅ Failure protection (circuit breakers)

### Phase 2: Intelligence (Week 3-4) — BUILT
8. ✅ Line tracker (movement recording)
9. ✅ Sharp money detector
10. ✅ Calibration monitor
11. ✅ Exposure manager

### Phase 3: Learning (Week 5-6) — BUILT
12. ✅ Model drift detector
13. ✅ Feature monitor
14. ✅ Feedback loop (self-improvement)
15. ✅ Walk-forward validator
16. ✅ MC bankroll simulator
17. ✅ Feature ablation

### Phase 4: Integration (Week 7-8) — NEXT
18. □ Wire into GOLFAPP dashboard.py
19. □ Wire into NBA-PRA-APP app.py
20. □ Add line snapshot recording to scrapers
21. □ Add auto-settlement from results scrapers
22. □ Streamlit dashboard components
23. □ Backfill historical bets for calibration

### Phase 5: Validation (Week 9-12)
24. □ Run walk-forward on 2024-2025 golf data
25. □ Run walk-forward on 2024-2025 NBA data
26. □ Feature ablation on both engines
27. □ MC simulation with real parameters
28. □ Paper trading (log bets without real money)

## What Would Still Cause This System to Fail

### 1. Data Quality (Highest Risk)
**Why:** If PrizePicks lines change and we don't capture the closing line,
CLV is meaningless. If SG data has errors, projections are wrong.
**Detection:** Line snapshot gaps > 30 min flagged. Feature distribution
monitoring via PSI. Missing data alerts.
**Mitigation:** Multiple data sources, fallback layers, data validation.

### 2. Market Adaptation
**Why:** If your bets are large enough, books will adjust. If you're
finding the same inefficiencies as other quant systems, the edge shrinks.
**Detection:** CLV decay over time. Edge decay detection. Increasing
calibration error.
**Mitigation:** Continuous model improvement. Feature innovation. Diverse
bet types and markets.

### 3. Regime Changes
**Why:** Rule changes (NBA), course modifications (Golf), major injury
patterns, or market structure changes (PrizePicks changing payouts).
**Detection:** PSI on feature distributions. Prediction shift detection.
Sudden calibration breaks.
**Mitigation:** Fast detection → fast response. Auto-suspend when drift
detected. Human review before recovery.

### 4. Overfitting in Backtests
**Why:** Walk-forward prevents most leakage, but parameter selection for
the walk-forward config itself can overfit (choosing the best window size).
**Detection:** Out-of-sample season holdout. Compare multiple configs.
**Mitigation:** Use conservative configs. Validate on unseen data.

### 5. Psychological Override
**Why:** User sees the system say "SUSPENDED" after 10 losses but
believes the model is right and overrides.
**Detection:** State override logging. Manual bet tracking.
**Mitigation:** Hard circuit breakers that CANNOT be overridden.
The failure_protection module is the final line of defense.

### 6. Execution Risk
**Why:** Lines move between evaluation and placement. PrizePicks
goes down. API failures during critical windows.
**Detection:** Timestamped evaluation vs execution. Stale line detection.
**Mitigation:** Record line at evaluation time. If line moved > 0.5
points, re-evaluate. Retry with exponential backoff.
