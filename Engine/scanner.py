# =========================================================
#  TIER C MODULE 12 — LIVE EDGE SCANNER
# =========================================================

import numpy as np
import pandas as pd

def compute_edge(prob, price):
    """
    EV = prob*(price-1) - (1-prob)
    """
    return float(prob*(price-1) - (1-prob))

def scan_edges(df, ensemble_func, ctx_func, threshold=0.65):
    """
    df: normalized props (player, market, line, over_price, under_price, bookmaker)
    ensemble_func: function(samples, line, ctx_mult, gs_prob) -> prob
    ctx_func: function(row)-> context multipliers
    threshold: minimum EV requirement
    """
    results = []
    for idx,row in df.iterrows():
        try:
            # Extract sample placeholder (Tier C would use real game logs)
            samples = np.random.normal(row['line'], 4, size=30)

            ctx = ctx_func(row)
            prob = ensemble_func(samples, row['line'], ctx.get('ctx_mult',1), ctx.get('gs_prob',0.5))

            price = row.get('over_price', None)
            if price is None: 
                continue

            ev = compute_edge(prob, price)
            if ev >= threshold:
                results.append({
                    "player": row['player'],
                    "market": row['market'],
                    "line": row['line'],
                    "price": price,
                    "prob": prob,
                    "edge": ev,
                    "book": row['bookmaker'],
                    "ctx": ctx
                })
        except Exception:
            continue

    return pd.DataFrame(results)
