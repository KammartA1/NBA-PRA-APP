# New Sport Quant Engine — Build Prompt Template

Copy and paste this prompt to Claude when adding a new sport to the Sports Quant Engine. Replace `[SPORT]` with the sport name (e.g., WNBA, NFL, NHL, UFC, etc.).

---

## Prompt

```
Build a comprehensive [SPORT] quant engine for the Sports Quant Engine app. Follow the exact same architecture as the MLB engine in simulation/mlb/.

### Step 1: Research
Do intensive research of the most analytical, mathematical, probability-based approaches used by the top [SPORT] bettors and quantitative betting syndicates. Find what works for every single one of them. Focus on:

1. **Projection systems**: What statistical models project individual player performance? What regression/shrinkage methods do they use? What is the optimal sample size before trusting a stat?

2. **Matchup modeling**: How do sharp bettors model player-vs-opponent matchups? What is the best mathematical method for combining player and opponent rates?

3. **Venue/environmental factors**: What venue-specific adjustments are used? How does environment (weather, altitude, indoor/outdoor, surface type) affect outcomes?

4. **Game context**: How do sharp bettors account for lineup/rotation position, rest days, travel, pace, game script, garbage time, blowouts?

5. **Key betting-specific metrics**: Which stats stabilize fastest? Which correlate best with props? What advanced analytics (tracking data, expected stats) matter?

6. **Edge identification**: What EV thresholds do pros use? How do they set their own lines vs the market? What Kelly fraction is typical?

7. **Simulation approach**: How do the best models simulate games and derive player prop distributions? What level of granularity produces the best results?

8. **Known pitfalls**: What are the most common mistakes in [SPORT] prop betting models?

### Step 2: Build the Engine
Create these files following the MLB engine pattern:

1. **simulation/[sport]/__init__.py** — Package init
2. **simulation/[sport]/config.py** — League constants, venue factors, engine config
3. **simulation/[sport]/profiles.py** — Player profiles with per-unit outcome rates
4. **simulation/[sport]/engine.py** — Monte Carlo game simulator with proper game flow
5. **simulation/[sport]/data_loader.py** — Real player profiles from public APIs with:
   - Multi-year weighting (Marcel method: 5/4/3 across 3 seasons)
   - Per-stat stabilization shrinkage toward league average
   - Matchup-specific splits (equivalent to platoon splits)
   - Recent-form weighting (last N games)
   - Caching for performance
6. **simulation/[sport]/projection.py** — End-to-end projection pipeline with:
   - Market code normalization (display names -> codes)
   - Context resolution (opponent, venue, schedule)
   - Confidence scoring
   - Full probability distribution output

### Step 3: Integrate with the App
Add the sport to the app by modifying:

1. **core/sports.py** — Add to SPORTS dict and ENABLED_SPORTS list with:
   - display_name, icon, subtitle, engine type, markets dict, season_label
   - Markets dict maps display names to engine codes (e.g., "Points": "PTS")

2. **app.py** — Add these integrations:
   - Import the projection function (compute_leg_projection_[sport])
   - Add wrapper function `compute_leg_projection_[sport]()` near line 7410 (next to MLB)
   - Add `_recompute_pricing_[sport]()` function with sport-appropriate gating
   - Route the scanner dispatch in `_dispatch_candidate()` for the new sport
   - Route `recompute_pricing_fields()` for the new sport
   - Add stat types to `map_platform_stat_to_market()` for PrizePicks parsing
   - Add sport-specific scanner market defaults
   - Skip NBA-specific player pre-filter for the new sport
   - Make bet logging sport-aware

3. **app.py — PrizePicks parser** — Add league filter:
   - Create `_pp_league_is_[sport]()` helper
   - Add sport branch in `_parse_pp_response_all()`

### Step 4: Test
1. Run a projection for 3 players across different markets
2. Test with ThreadPoolExecutor (like the scanner)
3. Verify PrizePicks lines are parsed correctly for the sport
4. Run a full scanner scan and verify edges are found or specific gate reasons shown

### Architecture Principles
- Use log5/odds-ratio for matchup combination wherever applicable
- Apply per-stat stabilization shrinkage (stat-specific sample sizes)
- Marcel-weight multi-year data (current 5x, prev 4x, 2 years ago 3x)
- Include recent-form weighting with stat-specific trust factors
- Cache API calls aggressively (module-level dict cache)
- Use ThreadPoolExecutor-safe code (no shared mutable state)
- Return distributions, not just point estimates
- Include confidence tier in output
- Gate on EV >= 3% AND edge >= 4% for PrizePicks
- Use 5000 sims for scanner (fast), 10000+ for MODEL tab (accurate)
```

---

## Existing Sports Reference

| Sport | Engine Files | Data Source | Key Matchup Method |
|-------|-------------|-------------|-------------------|
| NBA | app.py (built-in) | nba_api | Historical game logs + Bayesian |
| MLB | simulation/mlb/ | statsapi | Log5 odds-ratio + Monte Carlo |

## Key Files to Modify for Any New Sport

| File | What to Add |
|------|------------|
| `core/sports.py` | SPORTS entry, markets dict, ENABLED_SPORTS |
| `app.py` | Wrapper function, pricing function, scanner dispatch, stat mapping, PP parser |
| `simulation/[sport]/` | Full engine (config, profiles, engine, data_loader, projection) |
