-- ============================================================
-- NBA-PRA-APP — Results Tracker tables
-- ============================================================
-- HOW TO RUN:
--   1. Open your Supabase project dashboard
--   2. Left sidebar → "SQL Editor"
--   3. Click "+ New query"
--   4. Paste this ENTIRE file
--   5. Click "Run" (or Ctrl/Cmd + Enter)
--
-- Safe to run multiple times — everything uses IF NOT EXISTS.
-- Depends on 001_scanner_tables.sql having been run first.
-- ============================================================

-- 1. Add a "graded" flag to scan_results so the grader knows what's
--    already been settled. Race-condition safe: we grade off game_date
--    + this flag, never off is_active (which the next scan flips).
ALTER TABLE scan_results ADD COLUMN IF NOT EXISTS graded boolean DEFAULT false;
CREATE INDEX IF NOT EXISTS idx_sr_graded ON scan_results(graded, scanned_at DESC);

-- 2. Bet Results (the grader writes one row per settled prop here) --
CREATE TABLE IF NOT EXISTS bet_results (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    scan_result_id  bigint,            -- links back to scan_results.id (latest row in the group)
    scan_id         text,
    sport           text        NOT NULL DEFAULT 'NBA',
    graded_at       timestamptz NOT NULL DEFAULT now(),
    game_date       date,              -- date the game was played (ET)
    player          text        NOT NULL,
    team            text        DEFAULT '',
    opp             text        DEFAULT '',
    market          text        NOT NULL,
    line            float8,
    side            text        DEFAULT 'Over',
    proj            float8,            -- what we projected
    p_cal           float8,            -- our predicted probability (for calibration)
    ev_pct          float8,
    edge_cat        text        DEFAULT '',
    actual_value    float8,            -- actual stat the player produced (NULL if void/DNP)
    result          text        NOT NULL,   -- 'win' | 'loss' | 'push' | 'void'
    hit             boolean,           -- TRUE=win, FALSE=loss, NULL=push/void (for calibration)
    profit_units    float8      DEFAULT 0,  -- flat 1u stake at -110: win=+0.909, loss=-1.0, else 0
    src             text        DEFAULT 'PrizePicks',
    -- one settled row per logical prop (player+market+line+side on a given game date)
    CONSTRAINT uq_bet_prop UNIQUE (sport, game_date, player, market, line, side)
);
CREATE INDEX IF NOT EXISTS idx_br_graded  ON bet_results(sport, graded_at DESC);
CREATE INDEX IF NOT EXISTS idx_br_result  ON bet_results(result);
CREATE INDEX IF NOT EXISTS idx_br_market  ON bet_results(market);
CREATE INDEX IF NOT EXISTS idx_br_gamedate ON bet_results(game_date DESC);
CREATE INDEX IF NOT EXISTS idx_br_pcal    ON bet_results(p_cal);

-- Row Level Security + permissive policy ---------------------------
ALTER TABLE bet_results ENABLE ROW LEVEL SECURITY;
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='bet_results' AND policyname='allow_all_br') THEN
        CREATE POLICY "allow_all_br" ON bet_results FOR ALL USING (true) WITH CHECK (true);
    END IF;
END $$;

-- Done. You should see "Success. No rows returned".
