# =========================================================
#  TIER C MODULE 2 — NORMALIZATION ENGINE
# =========================================================

import pandas as pd
import numpy as np
from datetime import datetime

# Maps Odds API market keys to readable names
MARKET_MAP = {
    "player_points": "Points",
    "player_rebounds": "Rebounds",
    "player_assists": "Assists",
    "player_points_rebounds_assists": "PRA"
}

def normalize_props(odds_data):
    """Converts raw Odds API JSON into a clean DataFrame.

    Output Columns:
      - player
      - market
      - line
      - over_price
      - under_price
      - bookmaker
      - home
      - away
      - game_time
      - game_timestamp
    """
    rows = []
    if not odds_data:
        return pd.DataFrame()

    for game in odds_data:
        home = game.get("home_team","")
        away = game.get("away_team","")
        game_time = game.get("commence_time","")
        try:
            ts = pd.to_datetime(game_time)
        except:
            ts = None

        for bm in game.get("bookmakers",[]):
            book = bm.get("key","")

            for market in bm.get("markets",[]):
                mkey = market.get("key","")
                market_name = MARKET_MAP.get(mkey, mkey)

                # Odds API gives separate outcomes for over/under
                over_price = None
                under_price = None
                line = None
                player = None

                for out in market.get("outcomes",[]):
                    desc = out.get("description","")
                    price = out.get("price")
                    pt = out.get("point")

                    if pt is not None:
                        line = float(pt)

                    if "Over" in desc:
                        over_price = float(price)
                        player = desc.replace("Over ","")
                    elif "Under" in desc:
                        under_price = float(price)
                        player = desc.replace("Under ","")

                if player and line is not None:
                    rows.append({
                        "player": player,
                        "market": market_name,
                        "line": line,
                        "over_price": over_price,
                        "under_price": under_price,
                        "bookmaker": book,
                        "home": home,
                        "away": away,
                        "game_time": game_time,
                        "game_timestamp": ts
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Clean player names
    df["player"] = df["player"].str.replace("Over ","").str.replace("Under ","").str.strip()

    # Compute implied probabilities
    df["imp_over"] = 1 / df["over_price"].replace(0,np.nan)
    df["imp_under"] = 1 / df["under_price"].replace(0,np.nan)

    return df
