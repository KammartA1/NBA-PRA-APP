# NBA Prop Alpha Engine

A quantitative player proposition projection system built for PrizePicks, Underdog Fantasy, and Sleeper. The engine combines possession-level game simulation, Monte Carlo probability modeling, Bayesian calibration, and institutional-grade risk management to identify mispriced player props across DFS sportsbooks.

---

## Architecture Overview

```
                          +------------------+
                          |   Streamlit UI   |
                          |    (app.py)      |
                          +--------+---------+
                                   |
                    +--------------+--------------+
                    |                             |
           +--------+--------+          +---------+---------+
           |  NBA Engine     |          |  Quant System     |
           |  (nba_engine.py)|          |  v1.0             |
           +--------+--------+          +---------+---------+
                    |                             |
        +-----------+-----------+      +----------+----------+
        |           |           |      |          |          |
   +----+----+ +----+----+ +---+---+  +---+--+ +-+------+ +-+------+
   |Simulation| |Edge     | |Data   |  |Risk  | |Market  | |Learning|
   |Engine    | |Analysis | |Sources|  |Mgmt  | |Intel   | |Loop    |
   +----------+ +---------+ +-------+  +------+ +--------+ +--------+
```

### Core Systems

| System | Location | Purpose |
|--------|----------|---------|
| **NBA Engine** | `nba_engine.py` | Projection pipeline: data ingestion, context adjustment, Monte Carlo simulation, probability estimation |
| **Quant System** | `quant_system/` | Master orchestrator: bet evaluation, placement, settlement, daily feedback loop |
| **Simulation Engine** | `simulation/` | Possession-level NBA game simulation with fatigue, fouls, game script, lineup rotation |
| **Edge Analysis** | `edge_analysis/` | 10 independent signal sources with correlation independence testing |
| **Risk Management** | `quant_system/risk/` | Adaptive Kelly sizing, exposure limits, drawdown controls |
| **Workers** | `workers/` | Background scheduling: odds polling, signal generation, CLV settlement, model retraining |

---

## Projection Pipeline

### 1. Data Ingestion

The engine pulls real-time data from multiple sources:

- **NBA API** (`nba_api`) -- Current-season game logs, per-minute production, usage rates
- **PrizePicks** (`services/prizepicks_scraper.py`) -- Live prop lines via 24/7 polling service with PerimeterX bypass
- **Underdog Fantasy** -- Higher/lower picks with rival play integration
- **Sleeper** -- Pick'em market coverage with insured entries
- **The Odds API** -- Sharp and soft book line comparison

Player resolution is automatic: each name maps to an NBA player ID, pulls the last N games (configurable), and computes per-minute production with season-aware rollover.

### 2. Context Engine (15+ Adjustment Factors)

Every projection is adjusted for game context:

| Factor | Description |
|--------|-------------|
| Opponent pace | Faster opponents increase possession count |
| Defensive rating | Stronger defenses reduce per-minute production |
| Rest days | Back-to-back fatigue, extended rest bonuses |
| Altitude | Denver altitude adjustment |
| Referee tendencies | Ref-specific foul and pace patterns |
| Home/away splits | Normalized home-court advantage |
| Injury/lineup | Usage redistribution when key teammates are out |
| Pace differential | Team pace vs league average |
| Game script | Blowout risk, garbage time, close-game minutes |
| Recency weighting | Recent games weighted higher than older samples |
| Minutes model | Pace-adjusted, blowout-adjusted, injury-adjusted minutes |
| Usage redistribution | On/off usage boost system for teammate absences |

### 3. Monte Carlo Simulation

**Per-Leg Simulation (10,000 iterations):**
- Recency-weighted bootstrap sampling of recent game logs
- Per-minute production adjusted via defensive and usage multipliers
- Minutes adjusted via pace and blowout multipliers
- Lognormal noise injection for heavy tails and realistic variance
- Output: mean, standard deviation, probability over/under line

**Joint 2-Pick Monte Carlo (Parlays):**
- Correlation estimation from team overlap, minutes, market pair, opponent context
- Correlated normal draws for joint probability estimation
- Blend of joint simulation with naive product for stability
- EV and Kelly stake computation from joint probability

**Possession-Level Game Simulation (`simulation/`):**
- Full game engine with alternating possessions
- Dynamic lineup management, fatigue accumulation, foul modeling
- Game script tracking (close game vs blowout)
- Transition play, hot/cold streaks, substitution patterns
- Output: full stat distributions with percentiles (p5, p10, p25, p50, p75, p90, p95)

### 4. Bayesian Calibration

The self-learning engine continuously improves probability estimates:

- Bets are bucketed by EV range (< 0%, 0-5%, 5-10%, 10-20%, 20%+)
- Predicted win rate vs actual hit rate comparison per bucket
- Calibration multiplier derived to shrink/expand distance from 50%
- Small EV shift applied based on historical CLV performance
- Model recalibration triggered when MAE > 8%; suspension when MAE > 15%

---

## Edge Detection

### 10 Independent Signal Sources

Located in `edge_analysis/sources/`:

| # | Source | Signal |
|---|--------|--------|
| 1 | Defensive Matchup | Opponent pace, defensive rating, rebound % |
| 2 | Minutes Distribution | Pace-adjusted minutes model |
| 3 | Game Script | Context effects from game flow |
| 4 | Referee Tendencies | Ref-specific foul and pace patterns |
| 5 | Home/Away | Normalized home-court factor |
| 6 | Recency Weighting | Preference for recent performance |
| 7 | Pace Differential | Pace vs league average |
| 8 | Usage Redistribution | On/off usage boost |
| 9 | Rest Effects | Rest days and back-to-back impact |
| 10 | Lineup Effects | Injury/lineup usage-rate-scaled boost |

**Source Quality Gating:**
- Each source is ranked by standalone Sharpe ratio
- Correlation independence matrix rejects redundant signals
- Sources failing p-value tests are automatically excluded
- Edge attribution shows which sources contributed to each signal

### Edge Decomposition

The `edge_analysis/decomposer.py` breaks down total edge into:
- **Predictive edge** -- Model accuracy vs market
- **Market inefficiency** -- Structural pricing errors
- **Informational edge** -- Speed/depth of data advantage
- **Execution edge** -- Line timing and platform selection

---

## Risk Management

### Adaptive Kelly Sizing

```
Base:     1/10 Kelly (conservative starting point)
Floor:    1/50 Kelly (minimum during drawdowns)
Ceiling:  1/4 Kelly (maximum during strong CLV)
Hard cap: 3% of bankroll per bet (NEVER exceeded)
```

**Dynamic Adjustments:**
- CLV > +2 cents: 30% Kelly boost
- CLV < -0.5 cents: 50% Kelly reduction
- Calibration MAE < 4%: 15% boost
- Calibration MAE > 10%: 50% reduction
- Drawdown-scaled reductions at 10%, 20%, 30% thresholds

**Exposure Limits:**
- Max daily exposure: 15% of bankroll
- Max player exposure: 5% of bankroll
- Minimum edge to bet: 4%
- Correlation-aware portfolio constraints

### Kill Switches (Non-Overridable)

Six circuit breakers that cannot be bypassed:

| Condition | Trigger | Action |
|-----------|---------|--------|
| CLV negative over 250 bets | CLV <= 0 sustained | **KILLED** |
| Model worse than market | Brier > market Brier over 100+ bets | **SUSPENDED** |
| Edge decay detected | CLV declining 3 consecutive 100-bet windows | **REDUCED** |
| Execution destroys edge | Post-execution ROI < 0 for 50+ bets | **SUSPENDED** |
| 50% drawdown from peak | Bankroll drops 50% | **KILLED** |
| Calibration broken | MAE > 15% for 100+ bets | **SUSPENDED** |

**System States:**
- **ACTIVE** -- Normal operation (1.0x Kelly)
- **REDUCED** -- Halve bet sizes (0.5x Kelly)
- **SUSPENDED** -- Stop all new bets (0.0x Kelly)
- **KILLED** -- System shutdown, requires manual intervention

### Closing Line Value (CLV) Tracking

CLV is the single strongest predictor of long-term profitability:

- Every bet is timestamped at entry and compared against closing line
- Measured in probability space and cents per dollar
- Rolling windows: 50, 100, 250, 500 bets
- Positive CLV = beating the market's final assessment = genuine edge
- Negative CLV trends trigger automatic position reduction and kill switches
- All results independently verifiable via immutable audit trail

---

## Platform Integration

### PrizePicks
- 24/7 scraper service with 30-minute polling intervals
- PerimeterX bypass via `curl_cffi` or `cloudscraper`
- Power play and flex entry support
- Goblin mode for correlated combo detection
- Full PRA (Points/Rebounds/Assists) market coverage

### Underdog Fantasy
- Higher/lower picks with rival play integration
- Optimized for 2-6 leg entries
- Correlation analysis across legs

### Sleeper
- Pick'em market coverage
- Insured and non-insured entry optimization
- EV-maximized pick count selection

### Cross-Platform
- Line comparison across all platforms identifies sharpest number per prop
- Platform-specific payout structure factored into EV calculation
- Unified scanner surfaces best opportunities across all sportsbooks

---

## Sharp Money Detection

Located in `quant_system/market/sharp_detector.py`:

**Sharp Sources:** Pinnacle, Circa, Bookmaker, BetCris
**Soft Sources:** DraftKings, FanDuel, BetMGM, Caesars, PointsBet, PrizePicks

- Sharp vs soft book divergence detection
- Confidence adjustments: 0.5x (disagree with sharps) to 1.3x (agree with sharps)
- Line movement tracking via `quant_system/market/line_tracker.py`

---

## Background Workers

The `workers/` system runs continuously via APScheduler:

| Worker | Interval | Purpose |
|--------|----------|---------|
| Odds Worker | 5 min | Poll for new lines from all platforms |
| Signal Worker | On new data | Generate trading signals when lines arrive |
| Closing Worker | 5 min | Settle bets and calculate CLV |
| Model Worker | Weekly (Mon 4 AM) | Model retraining with latest data |
| Report Worker | Daily | Generate performance dashboards |
| Stats Worker | Periodic | Player stats updates from NBA API |
| Data Audit Worker | Periodic | Data quality and integrity checks |

---

## Database & Persistence

### Local Storage (SQLAlchemy)
Located in `database/`:
- `bets` -- Every bet with full context snapshot
- `signals` -- Trading signals with edge and sharpness
- `lines` -- Line snapshots from all sources
- `events` -- NBA games and outcomes
- `model_versions` -- Model metadata and performance
- `worker_status` -- Background worker health

### Cloud Storage (Supabase/Postgres)
Configured via `supabase_store.py` and `supabase_schema.sql`:
- `bet_history` -- Synchronized bet records
- `user_settings` -- Bankroll and preferences
- `prop_line_history` -- Historical line movements
- `opening_lines` -- Opening line capture for CLV
- Falls back to local file storage if Supabase is not configured

---

## Governance & Quality

Located in `governance/`:

- **Feature Importance** -- Ranking and health monitoring for all model features
- **Performance Tracker** -- System-wide metrics and trend analysis
- **Rollback** -- Feature and model version rollback capability
- **Simplicity Audit** -- Code complexity analysis to prevent overengineering
- **Version Control** -- Model version tracking with full audit trail
- **Auto Cleanup** -- Automatic data cleanup and maintenance

---

## Immutable Audit Trail

Every bet placed through the system is logged with a complete context snapshot before execution:

- Timestamp, model probability, market probability
- Edge magnitude, stake amount, Kelly fraction
- All feature values at time of bet
- System state (ACTIVE/REDUCED/SUSPENDED)
- Model version and calibration parameters
- Closing line and CLV computed post-settlement

This audit trail cannot be modified after creation and provides full transparency for performance verification.

---

## Dashboard (Streamlit)

The main interface (`app.py`) provides:

- **Model Tab** -- Run projections, view Monte Carlo distributions
- **Scanner** -- Live edge detection across all platforms
- **Platforms** -- PrizePicks, Underdog, Sleeper integration panels
- **History** -- Bet log with P&L tracking
- **Calibration** -- Predicted vs actual win rate curves
- **CLV** -- Closing line value tracking and analysis
- **Kill Switch** -- Circuit breaker status display
- **Edge Sources** -- 10-signal attribution dashboard
- **Edge Decomposition** -- Predictive, market, informational, execution breakdown
- **Capital** -- Kelly calculator, risk metrics, portfolio optimization
- **Quant Dashboard** -- Full quant engine status
- **Settings** -- Bankroll, API keys, Kelly fraction, preferences

---

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `ODDS_API_KEY` | The Odds API key for line data |
| `ANTHROPIC_API_KEY` | Claude AI integration for edge explanations |
| `SUPABASE_URL` | Supabase project URL (optional) |
| `SUPABASE_KEY` | Supabase API key (optional) |
| `SCRAPER_API_KEY` | PrizePicks scraper API key |

### Requirements

- Python 3.10+
- Streamlit
- nba_api
- pandas, numpy, scipy
- SQLAlchemy
- APScheduler
- requests, curl_cffi
- Anthropic SDK (optional, for AI explanations)
