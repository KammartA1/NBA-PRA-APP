"""
Smoke test for all NBA PRA APP markets.
Extracts the first ~7600 lines of app.py (before UI code) and tests all market pipelines.
"""
import sys, os, time, traceback
import pandas as pd, numpy as np

print("=" * 70)
print("NBA PRA APP — MARKET SMOKE TEST")
print("=" * 70)

# ─── Read app.py up to the UI entry point ────────────────────────────
print("\n[1/6] Loading app.py data structures and functions …")
t0 = time.time()

# Read the source file
with open("/home/user/NBA-PRA-APP/app.py", "r") as f:
    lines = f.readlines()

# Find where module-level UI code starts (line with _auth_username)
cutoff = None
for i, line in enumerate(lines):
    if line.strip().startswith("_auth_username = st.session_state"):
        cutoff = i
        break

if cutoff is None:
    print("  FAIL: Could not find UI entry point in app.py")
    sys.exit(1)

print(f"  Using first {cutoff} lines (before UI entry at line {cutoff+1})")

# Create the truncated source
source = "".join(lines[:cutoff])

# ─── Set up mock environment ─────────────────────────────────────────
import streamlit as st

class _MockSessionState(dict):
    def __getitem__(self, key):
        return self.get(key, None)
    def __getattr__(self, key):
        return self.get(key, None)
    def __contains__(self, key):
        return True
    def pop(self, key, *args):
        return dict.pop(self, key, None) if key in dict.keys(self) else None

st.session_state = _MockSessionState({
    "_auth_user": "smoke_test", "user_id": "smoke_test",
    "bankroll": 1000, "n_games": 10, "frac_kelly": 0.25,
})

class _MockSecrets(dict):
    def __getitem__(self, key): return self.get(key, "")
    def __getattr__(self, key): return self.get(key, "")
    def __contains__(self, key): return True
st.secrets = _MockSecrets()

_noop = lambda *a, **kw: None
class _MockCtx:
    def __getattr__(self, name): return _noop
    def __call__(self, *a, **kw): return self
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __iter__(self): return iter([self])
    def __bool__(self): return True

st.sidebar = _MockCtx()
for fn in ['error','warning','info','success','write','markdown','header',
           'subheader','title','caption','divider','text','metric',
           'dataframe','table','plotly_chart','html','stop','rerun',
           'set_page_config','spinner','toast','balloons','snow',
           'text_input','number_input','selectbox','multiselect','slider',
           'checkbox','radio','toggle','button','download_button']:
    if fn in ('stop', 'rerun') or not hasattr(st, fn):
        setattr(st, fn, _noop)
st.stop = _noop; st.rerun = _noop
st.columns = lambda *a, **kw: [_MockCtx() for _ in range(a[0] if a and isinstance(a[0], int) else len(a[0]) if a and isinstance(a[0], (list,tuple)) else 2)]
st.tabs = lambda labels, **kw: [_MockCtx() for _ in labels]
st.expander = lambda *a, **kw: _MockCtx()
st.container = lambda *a, **kw: _MockCtx()
st.empty = lambda *a, **kw: _MockCtx()
st.form = lambda *a, **kw: _MockCtx()
st.status = lambda *a, **kw: _MockCtx()
st.popover = lambda *a, **kw: _MockCtx()

def _passthrough(*args, **kwargs):
    if args and callable(args[0]): return args[0]
    def w(fn): return fn
    return w
st.cache_data = _passthrough
st.cache_resource = _passthrough

# ─── Execute the truncated source in a module namespace ──────────────
import types
A = types.ModuleType("app")
A.__file__ = "/home/user/NBA-PRA-APP/app.py"
sys.modules["app"] = A

# Add app directory to path
sys.path.insert(0, "/home/user/NBA-PRA-APP")

try:
    exec(compile(source, "/home/user/NBA-PRA-APP/app.py", "exec"), A.__dict__)
    print(f"  OK — loaded in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"  PARTIAL — loaded with error: {e}")
    if not hasattr(A, 'ODDS_MARKETS'):
        print("  FATAL — ODDS_MARKETS not available")
        traceback.print_exc()
        sys.exit(1)

# ─── Collect markets ────────────────────────────────────────────────
all_markets = set(A.ODDS_MARKETS.keys())
print(f"  Total markets in ODDS_MARKETS: {len(all_markets)}")
for m in sorted(all_markets):
    print(f"    {m}")

# ═════════════════════════════════════════════════════════════════════
# [2/6] map_platform_stat_to_market
# ═════════════════════════════════════════════════════════════════════
print("\n[2/6] Verifying map_platform_stat_to_market() — all ODDS_MARKETS keys …")
mapping_failures = []
for mkt in sorted(all_markets):
    result = A.map_platform_stat_to_market(mkt)
    if result is None:
        mapping_failures.append(mkt)
        print(f"  FAIL: '{mkt}' → None (unmapped)")
    else:
        print(f"  OK:   '{mkt}' → '{result}'")

# ═════════════════════════════════════════════════════════════════════
# [3/6] STAT_FIELDS
# ═════════════════════════════════════════════════════════════════════
print("\n[3/6] Verifying STAT_FIELDS coverage …")
skip_sf = {"Alt Points", "Alt Rebounds", "Alt Assists", "Alt 3PM",
           "Double Double", "Triple Double", "First Basket"}
stat_field_failures = []
for mkt in sorted(all_markets):
    if mkt in skip_sf:
        continue
    base = mkt
    for pfx in ("H1 ", "H2 ", "Q1 "):
        if mkt.startswith(pfx):
            base = mkt[len(pfx):]
            break
    if base in A.STAT_FIELDS or mkt in A.STAT_FIELDS:
        sf = A.STAT_FIELDS.get(mkt, A.STAT_FIELDS.get(base))
        print(f"  OK:   '{mkt}' → {sf}")
    else:
        stat_field_failures.append(mkt)
        print(f"  FAIL: '{mkt}' (base='{base}') — not in STAT_FIELDS")

# ═════════════════════════════════════════════════════════════════════
# [4/6] HALF_FACTOR
# ═════════════════════════════════════════════════════════════════════
print("\n[4/6] Verifying HALF_FACTOR for H1/H2/Q1 markets …")
half_factor_failures = []
for mkt in sorted(all_markets):
    if mkt.startswith(("H1 ", "H2 ", "Q1 ")):
        if mkt in A.HALF_FACTOR:
            print(f"  OK:   '{mkt}' → {A.HALF_FACTOR[mkt]}")
        else:
            half_factor_failures.append(mkt)
            print(f"  FAIL: '{mkt}' — no HALF_FACTOR entry")

# ═════════════════════════════════════════════════════════════════════
# [5/6] compute_stat_from_gamelog with mock data
# ═════════════════════════════════════════════════════════════════════
print("\n[5/6] Testing compute_stat_from_gamelog with mock gamelog …")
np.random.seed(42)
mock_gl = pd.DataFrame({
    "PTS":  np.random.poisson(22, 10).astype(float),
    "REB":  np.random.poisson(8, 10).astype(float),
    "AST":  np.random.poisson(5, 10).astype(float),
    "FG3M": np.random.poisson(2, 10).astype(float),
    "BLK":  np.random.poisson(1, 10).astype(float),
    "STL":  np.random.poisson(1, 10).astype(float),
    "TOV":  np.random.poisson(3, 10).astype(float),
    "FGM":  np.random.poisson(8, 10).astype(float),
    "FGA":  np.random.poisson(18, 10).astype(float),
    "FG3A": np.random.poisson(6, 10).astype(float),
    "FTM":  np.random.poisson(4, 10).astype(float),
    "FTA":  np.random.poisson(5, 10).astype(float),
    "PF":   np.random.poisson(3, 10).astype(float),
    "MIN":  np.random.normal(32, 3, 10).round(1),
})

stat_failures = []
test_stats = [
    "Points", "Rebounds", "Assists", "3PM", "PRA", "PR", "PA", "RA",
    "Blocks", "Steals", "Turnovers", "Stocks",
    "FGM", "FGA", "3PA", "FTM", "FTA", "2PA", "Personal Fouls",
    "Fantasy Score", "H1 Fantasy Score", "H2 Fantasy Score",
]
for mkt in test_stats:
    try:
        s = A.compute_stat_from_gamelog(mock_gl, mkt)
        if s is None or (hasattr(s, 'empty') and s.empty) or s.isna().all():
            stat_failures.append((mkt, "empty/NaN"))
            print(f"  FAIL: '{mkt}' — returned empty/NaN")
        else:
            m = float(s.dropna().mean())
            print(f"  OK:   '{mkt}' — mean={m:.2f}, n={len(s.dropna())}")
    except Exception as e:
        stat_failures.append((mkt, str(e)))
        print(f"  FAIL: '{mkt}' — {e}")

# ═════════════════════════════════════════════════════════════════════
# [6/6] Full pipeline: stat → scale → project → probability
# ═════════════════════════════════════════════════════════════════════
print("\n[6/6] Full projection pipeline test …")
from scipy.stats import norm

pipeline_failures = []
pipeline_ok = []

for mkt in sorted(all_markets):
    if mkt == "First Basket":
        print(f"  SKIP: '{mkt}' (non-projectable)")
        continue
    try:
        # Determine base market
        base = mkt
        hf = A.HALF_FACTOR.get(mkt, 1.0)
        for pfx in ("H1 ", "H2 ", "Q1 ", "Alt "):
            if mkt.startswith(pfx):
                base = mkt[len(pfx):]
                break

        # Compute stat from gamelog
        s = A.compute_stat_from_gamelog(mock_gl, base)
        if (s is None or s.empty or s.isna().all()) and mkt in A.STAT_FIELDS:
            s = A.compute_stat_from_gamelog(mock_gl, mkt)

        if s is None or s.empty or s.isna().all():
            if mkt in skip_sf:
                # DD/TD use special logic, skip
                print(f"  SKIP: '{mkt}' (special)")
                continue
            pipeline_failures.append((mkt, "no stat data"))
            print(f"  FAIL: '{mkt}' — no stat data from gamelog")
            continue

        # Scale for period markets
        scaled = s * hf

        # Projection
        proj = float(scaled.dropna().mean())
        sigma = float(scaled.dropna().std(ddof=1)) if len(scaled.dropna()) > 1 else max(proj * 0.3, 0.5)

        if proj <= 0 and mkt not in ("Double Double", "Triple Double"):
            pipeline_failures.append((mkt, f"proj={proj:.3f}"))
            print(f"  WARN: '{mkt}' — proj={proj:.3f}")
            continue

        # Test line at 90% of projection
        line = max(0.5, round(proj * 0.9, 1))

        # Probability via normal CDF
        if sigma > 0:
            p_over = 1.0 - norm.cdf(line + 0.5, loc=proj, scale=sigma)
            p_under = norm.cdf(line - 0.5, loc=proj, scale=sigma)
        else:
            p_over = 0.5
            p_under = 0.5

        # Validate
        ok = True
        if np.isnan(proj): ok = False
        if not (0 <= p_over <= 1): ok = False
        if not (0 <= p_under <= 1): ok = False

        if ok:
            side = "OVER" if p_over >= p_under else "UNDER"
            prob = max(p_over, p_under)
            pipeline_ok.append(mkt)
            print(f"  OK:   '{mkt}' — proj={proj:.2f}, line={line}, "
                  f"p_over={p_over:.3f}, p_under={p_under:.3f} → {side} @ {prob:.1%}")
        else:
            pipeline_failures.append((mkt, "invalid output"))
            print(f"  FAIL: '{mkt}' — invalid proj/prob")

    except Exception as e:
        pipeline_failures.append((mkt, str(e)))
        print(f"  FAIL: '{mkt}' — {e}")

# ═════════════════════════════════════════════════════════════════════
# EXTRA CHECKS
# ═════════════════════════════════════════════════════════════════════

# POSITIONAL_PRIORS
print("\n[EXTRA] POSITIONAL_PRIORS coverage …")
prior_failures = []
skip_pp = {"Alt Points", "Alt Rebounds", "Alt Assists", "Alt 3PM",
           "First Basket", "Double Double", "Triple Double"}
for mkt in sorted(all_markets):
    if mkt in skip_pp:
        continue
    found = any(mkt in A.POSITIONAL_PRIORS.get(p, {}) for p in ("Guard","Wing","Big","Unknown"))
    if found:
        print(f"  OK:   '{mkt}'")
    else:
        prior_failures.append(mkt)
        print(f"  FAIL: '{mkt}' — not in any position bucket")

# LAMBDA_DECAY_BY_STAT
print("\n[EXTRA] LAMBDA_DECAY_BY_STAT coverage …")
lambda_failures = []
for mkt in sorted(all_markets):
    if mkt in skip_pp:
        continue
    if mkt in A.LAMBDA_DECAY_BY_STAT:
        print(f"  OK:   '{mkt}' → λ={A.LAMBDA_DECAY_BY_STAT[mkt]:.3f}")
    else:
        lambda_failures.append(mkt)
        print(f"  FAIL: '{mkt}' — missing")

# PP alias mapping
print("\n[EXTRA] PrizePicks API alias mapping …")
pp_aliases = [
    ("Points", "Points"), ("Rebounds", "Rebounds"), ("Assists", "Assists"),
    ("3-Pointers Made", "3PM"), ("3-PT Made", "3PM"), ("3PT Made", "3PM"),
    ("Pts+Reb+Ast", "PRA"), ("Pts+Reb", "PR"), ("Pts+Ast", "PA"), ("Reb+Ast", "RA"),
    ("Blocked Shots", "Blocks"), ("Steals", "Steals"), ("Turnovers", "Turnovers"),
    ("Blks+Stls", "Stocks"),
    ("Field Goals Made", "FGM"), ("Field Goals Attempted", "FGA"),
    ("Free Throws Made", "FTM"), ("Free Throws Attempted", "FTA"),
    ("3-Pt Attempts", "3PA"), ("Fantasy Score", "Fantasy Score"),
    ("H1 Points", "H1 Points"), ("1H Points", "H1 Points"), ("1st Half Points", "H1 Points"),
    ("H1 Rebounds", "H1 Rebounds"), ("H1 Assists", "H1 Assists"),
    ("H1 3PM", "H1 3PM"), ("H1 PRA", "H1 PRA"),
    ("H2 Points", "H2 Points"), ("H2 Rebounds", "H2 Rebounds"),
    ("H2 Assists", "H2 Assists"), ("H2 PRA", "H2 PRA"),
    ("Q1 Points", "Q1 Points"), ("Q1 Rebounds", "Q1 Rebounds"), ("Q1 Assists", "Q1 Assists"),
    ("Q1 3-PT Made", "Q1 3PM"), ("Q1 3PM", "Q1 3PM"),
    ("Q1 Free Throws Made", "Q1 FTM"), ("Q1 FTM", "Q1 FTM"),
    ("H1 Fantasy Score", "H1 Fantasy Score"), ("H2 Fantasy Score", "H2 Fantasy Score"),
    ("H2 3PM", "H2 3PM"),
    ("Personal Fouls", "Personal Fouls"), ("Fouls", "Personal Fouls"), ("PF", "Personal Fouls"),
    ("Two Pointers Attempted", "2PA"), ("2PA", "2PA"),
    ("Double Double", "Double Double"), ("Triple Double", "Triple Double"),
    ("Points (Combo)", "Points (Combo)"), ("Rebounds (Combo)", "Rebounds (Combo)"),
    ("Assists (Combo)", "Assists (Combo)"), ("3PM (Combo)", "3PM (Combo)"),
    ("H1 3-PT Made", "H1 3PM"), ("H1 3PT Made", "H1 3PM"),
]
alias_failures = []
for alias, expected in pp_aliases:
    result = A.map_platform_stat_to_market(alias)
    if result is None:
        alias_failures.append(alias)
        print(f"  FAIL: '{alias}' → None (expected '{expected}')")
    elif result != expected:
        alias_failures.append(alias)
        print(f"  FAIL: '{alias}' → '{result}' (expected '{expected}')")
    else:
        print(f"  OK:   '{alias}' → '{result}'")

# Combo engine
print("\n[EXTRA] Combo market engine …")
combo_ok = True
for cm in sorted(A.COMBO_MARKETS):
    base = A.COMBO_BASE_MAP.get(cm)
    if base and base in A.STAT_FIELDS:
        print(f"  OK:   '{cm}' → base='{base}'")
    else:
        print(f"  FAIL: '{cm}' — base='{base}' missing")
        combo_ok = False
if A.is_combo_market("Points (Combo)") and not A.is_combo_market("Points"):
    print("  OK:   is_combo_market() logic correct")
else:
    print("  FAIL: is_combo_market() logic error")
    combo_ok = False

# ═════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SMOKE TEST SUMMARY")
print("=" * 70)

all_ok = True
checks = [
    ("map_platform_stat_to_market", mapping_failures),
    ("STAT_FIELDS coverage", stat_field_failures),
    ("HALF_FACTOR coverage", half_factor_failures),
    ("compute_stat_from_gamelog", [f[0] for f in stat_failures]),
    ("Full projection pipeline", [f[0] for f in pipeline_failures]),
    ("POSITIONAL_PRIORS", prior_failures),
    ("LAMBDA_DECAY_BY_STAT", lambda_failures),
    ("PP alias mapping", alias_failures),
]

for label, fails in checks:
    if fails:
        all_ok = False
        print(f"  FAIL: {label} — {len(fails)} issues: {fails}")
    else:
        print(f"  OK:   {label}")

if not combo_ok:
    all_ok = False
    print(f"  FAIL: Combo engine")
else:
    print(f"  OK:   Combo engine")

print(f"\nMarkets: {len(all_markets)} total, {len(pipeline_ok)} pipeline OK, {len(pipeline_failures)} pipeline issues")

if all_ok:
    print("\nALL SMOKE TESTS PASSED")
else:
    print("\nSOME TESTS FAILED — see details above")

sys.exit(0 if all_ok else 1)
