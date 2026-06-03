-- ============================================================
-- NBA-PRA-APP — Scanner / background-worker tables
-- ============================================================
-- HOW TO RUN:
--   1. Open your Supabase project dashboard
--   2. Left sidebar → "SQL Editor"
--   3. Click "+ New query"
--   4. Paste this ENTIRE file
--   5. Click "Run" (or Ctrl/Cmd + Enter)
--
-- Safe to run multiple times — everything uses IF NOT EXISTS.
-- ============================================================

-- 1. Scan Results (background worker writes the +EV props here) -----
CREATE TABLE IF NOT EXISTS scan_results (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    scan_id     text        NOT NULL,
    sport       text        NOT NULL DEFAULT 'NBA',
    scanned_at  timestamptz NOT NULL DEFAULT now(),
    player      text        NOT NULL,
    team        text        DEFAULT '',
    opp         text        DEFAULT '',
    market      text        NOT NULL,
    line        float8,
    side        text        DEFAULT 'Over',
    proj        float8,
    p_cal       float8,
    ev_pct      float8,
    edge_cat    text        DEFAULT '',
    l5_avg      float8,
    l10_avg     float8,
    src         text        DEFAULT 'PrizePicks',
    is_active   boolean     DEFAULT true
);
CREATE INDEX IF NOT EXISTS idx_sr_active  ON scan_results(sport, is_active);
CREATE INDEX IF NOT EXISTS idx_sr_scan    ON scan_results(scan_id);
CREATE INDEX IF NOT EXISTS idx_sr_scanned ON scan_results(scanned_at DESC);

-- 2. Worker Run Log (heartbeat — proves the cron ran) --------------
CREATE TABLE IF NOT EXISTS worker_runs (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    worker_name text        NOT NULL,
    status      text        NOT NULL,
    ran_at      timestamptz NOT NULL DEFAULT now(),
    details     jsonb       DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_wr_worker ON worker_runs(worker_name, ran_at DESC);

-- 3. Notification Log (record of Telegram alerts sent) -------------
CREATE TABLE IF NOT EXISTS notification_log (
    id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    sent_at   timestamptz NOT NULL DEFAULT now(),
    type      text        NOT NULL,
    message   text        DEFAULT '',
    delivered boolean     DEFAULT true
);
CREATE INDEX IF NOT EXISTS idx_nl_type ON notification_log(type, sent_at DESC);

-- Row Level Security + permissive policies -------------------------
ALTER TABLE scan_results     ENABLE ROW LEVEL SECURITY;
ALTER TABLE worker_runs      ENABLE ROW LEVEL SECURITY;
ALTER TABLE notification_log ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='scan_results' AND policyname='allow_all_sr') THEN
        CREATE POLICY "allow_all_sr" ON scan_results FOR ALL USING (true) WITH CHECK (true);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='worker_runs' AND policyname='allow_all_wr') THEN
        CREATE POLICY "allow_all_wr" ON worker_runs FOR ALL USING (true) WITH CHECK (true);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='notification_log' AND policyname='allow_all_nl') THEN
        CREATE POLICY "allow_all_nl" ON notification_log FOR ALL USING (true) WITH CHECK (true);
    END IF;
END $$;

-- Done. You should see "Success. No rows returned".
