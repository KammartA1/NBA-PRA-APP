-- =============================================================
-- Supabase Schema for NBA-PRA-APP Persistent Storage
-- Run this in Supabase Dashboard > SQL Editor
-- =============================================================

-- 1. Bet History (most critical — CLV tracking)
CREATE TABLE IF NOT EXISTS bet_history (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ts timestamptz NOT NULL DEFAULT now(),
    user_id text NOT NULL DEFAULT 'default',
    legs jsonb NOT NULL DEFAULT '[]'::jsonb,
    n_legs int4 NOT NULL DEFAULT 1,
    leg_results jsonb DEFAULT '["Pending"]'::jsonb,
    result text NOT NULL DEFAULT 'Pending',
    decision text NOT NULL DEFAULT 'BET',
    notes text DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_bh_user ON bet_history(user_id);
CREATE INDEX IF NOT EXISTS idx_bh_ts ON bet_history(ts DESC);
CREATE INDEX IF NOT EXISTS idx_bh_result ON bet_history(user_id, result);

-- 2. User Settings (bankroll, kelly fraction, preferences)
CREATE TABLE IF NOT EXISTS user_settings (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id text UNIQUE NOT NULL,
    settings jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at timestamptz DEFAULT now()
);

-- 3. Prop Line History (line/price snapshots for CLV)
CREATE TABLE IF NOT EXISTS prop_line_history (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ts timestamptz NOT NULL DEFAULT now(),
    player text NOT NULL,
    market text NOT NULL,
    line float8,
    price float8,
    book text,
    event_id text
);
CREATE INDEX IF NOT EXISTS idx_plh_player ON prop_line_history(player, market);
CREATE INDEX IF NOT EXISTS idx_plh_ts ON prop_line_history(ts DESC);

-- 4. Opening Lines (first-seen line per player/market/side/date)
CREATE TABLE IF NOT EXISTS opening_lines (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    lookup_key text UNIQUE NOT NULL,
    line float8,
    price float8,
    ts timestamptz,
    date_str text
);

-- 5. User Auth (login credentials)
CREATE TABLE IF NOT EXISTS user_auth (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    username text UNIQUE NOT NULL,
    pw_hash text NOT NULL,
    email text DEFAULT '',
    created_at timestamptz DEFAULT now()
);

-- 6. Watchlist (per-user player watchlists)
CREATE TABLE IF NOT EXISTS watchlist (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id text UNIQUE NOT NULL,
    players jsonb NOT NULL DEFAULT '[]'::jsonb,
    updated_at timestamptz DEFAULT now()
);

-- Disable RLS for single-user server-side app (service_role key bypasses anyway)
ALTER TABLE bet_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_settings ENABLE ROW LEVEL SECURITY;
ALTER TABLE prop_line_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE opening_lines ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_auth ENABLE ROW LEVEL SECURITY;
ALTER TABLE watchlist ENABLE ROW LEVEL SECURITY;

-- Permissive policies (service_role key bypasses, but add for safety)
CREATE POLICY "allow_all" ON bet_history FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all" ON user_settings FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all" ON prop_line_history FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all" ON opening_lines FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all" ON user_auth FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all" ON watchlist FOR ALL USING (true) WITH CHECK (true);
