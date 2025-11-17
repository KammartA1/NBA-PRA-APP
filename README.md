3. How the Model Works
3.1 Data Inputs (Auto-Pulled)

When you click Run Model:

The app resolves each player name to an NBA ID via nba_api.

It pulls that player's current-season game logs and uses the last N games (slider in the sidebar).

For each game it computes:

Minutes played

Stat totals for the chosen market (Points / Rebounds / Assists / PRA)

Per-minute production

It also pulls team-level opponent context:

Pace

Defensive rating

Rebound %, defensive rebound %, assist %

This gives a clean sample of recent per-minute and minute distributions that are season-aware and automatically roll into the new season each year.

3.2 Defensive Matchup Engine

A context multiplier is computed per opponent + market, based on:

Opponent pace vs league average

Opponent defensive rating vs league average

Opponent rebound % and assist % (for Rebounds / Assists markets)

This multiplier feeds into both the projection and the Monte Carlo engine so every sample is defense-adjusted.

3.3 Pace-Adjusted Minutes Model

Minutes are adjusted based on:

Opponent pace vs league

Blowout risk flag

Teammate injury flag (for extra opportunity)

Rolling average minutes for that player

Faster games slightly increase expected minutes; slower games reduce them. Blowout flags trim minutes; injury flags boost them.

3.4 Usage & On/Off Boost System

When "key teammate out" is checked:

Per-minute production is boosted to reflect increased usage.

Minutes expectation is bumped.

Volatility is slightly increased.

This approximates an on/off usage system without needing full play-by-play data.

3.5 Empirical Bootstrap Monte Carlo (Per Leg)

For each leg:

The last N games' per-minute and minutes are stored.

A recency-weighted sampler prefers more recent games.

For each of 10,000 simulations:

Sample a game index (recency-weighted).

Adjust per-minute production via defensive & usage multipliers.

Adjust minutes via pace & blowout multipliers.

Multiply per-minute × minutes × lognormal noise (for heavy tails).

The simulated distribution yields:

MC mean

MC standard deviation

Probability that the stat goes OVER your line

This probability is then calibrated by the self-learning engine before being displayed.

3.6 Joint 2-Pick Monte Carlo

For the combo section:

A correlation coefficient is estimated from:

Team overlap

Minutes

Market pair (Points vs Assists vs Rebounds vs PRA)

Opponent context alignment

Two correlated normal distributions are generated using each leg's MC mean & SD.

10,000 joint draws are simulated to estimate the joint probability that both legs go OVER their lines.

This joint probability is blended with the naive product of the two calibrated single-leg probabilities for stability.

EV and fractional Kelly stake are computed from the joint probability and payout multiple.

3.7 Bankroll & Risk Controls

A Fractional Kelly slider controls risk level (0–100% Kelly).

A hard cap of 3% of bankroll per play is enforced.

A Max Daily Loss (% of bankroll) slider is applied using your logged results:

If realized PnL for today breaches the cap, suggested stake is automatically set to $0 and a warning is shown.

3.8 Self-Learning Calibration Engine

The Calibration tab analyzes your historical bets:

Buckets bets by EV:

EV < 0

0–5%

5–10%

10–20%

20%+

For each bucket it computes:

Count

Hit rate

Average EV

Average CLV

It also compares:

Predicted win rate (from EV)

Actual hit rate

From these, it derives:

A probability multiplier (shrink/expand distance from 50%)

A small EV shift

These calibration parameters are applied back into the main Model tab each time you run the model, so the system learns from your performance over time.

If the model is too optimistic / pessimistic or if average CLV turns negative, the tab recommends cutting volume or tightening edges.
