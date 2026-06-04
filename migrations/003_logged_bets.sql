-- ============================================================
-- NBA-PRA-APP — Logged Bets table (user's actual placed bets)
-- ============================================================
-- HOW TO RUN:
--   1. Open your Supabase project dashboard
--   2. Left sidebar → "SQL Editor"
--   3. Click "+ New query"
--   4. Paste this ENTIRE file
--   5. Click "Run" (or Ctrl/Cmd + Enter)
--
-- Safe to run multiple times — everything uses IF NOT EXISTS.
-- This lets the background grader auto-grade YOUR logged bets each morning,
-- even while the Streamlit app is asleep.
-- ============================================================

CREATE TABLE IF NOT EXISTS logged_bets (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    bet_id       text        NOT NULL,           -- stable UUID from the app
    user_id      text        NOT NULL DEFAULT 'default',
    sport        text        NOT NULL DEFAULT 'NBA',
    logged_at    timestamptz NOT NULL DEFAULT now(),
    game_date    date,                           -- ET date the bet's games are played
    legs         jsonb       NOT NULL DEFAULT '[]'::jsonb,  -- [{player, player_id, market, line, side, p_cal, ...}]
    n_legs       int         DEFAULT 0,
    leg_results  jsonb       DEFAULT '[]'::jsonb, -- ["HIT","MISS","PUSH","Pending"]
    result       text        DEFAULT 'Pending',   -- Pending | HIT | MISS | PUSH | SKIP
    decision     text        DEFAULT 'BET',       -- BET | PASS
    graded       boolean     DEFAULT false,
    notes        text        DEFAULT '',
    CONSTRAINT uq_logged_bet UNIQUE (bet_id)
);
CREATE INDEX IF NOT EXISTS idx_lb_user    ON logged_bets(user_id, logged_at DESC);
CREATE INDEX IF NOT EXISTS idx_lb_pending ON logged_bets(graded, result);
CREATE INDEX IF NOT EXISTS idx_lb_gamedate ON logged_bets(game_date DESC);

ALTER TABLE logged_bets ENABLE ROW LEVEL SECURITY;
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='logged_bets' AND policyname='allow_all_lb') THEN
        CREATE POLICY "allow_all_lb" ON logged_bets FOR ALL USING (true) WITH CHECK (true);
    END IF;
END $$;

-- Done. You should see "Success. No rows returned".
