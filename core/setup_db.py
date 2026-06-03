#!/usr/bin/env python3
"""
core/setup_db.py — Create all Supabase tables via direct Postgres connection.
Run once: python -m core.setup_db

Requires env vars: SUPABASE_DB_HOST, SUPABASE_DB_PASSWORD
(or falls back to SUPABASE_URL to derive the host)
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("setup_db")

SQL_STATEMENTS = [
    # 1. Bet History
    """CREATE TABLE IF NOT EXISTS bet_history (
        id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        ts timestamptz NOT NULL DEFAULT now(),
        user_id text NOT NULL DEFAULT 'default',
        legs jsonb NOT NULL DEFAULT '[]'::jsonb,
        n_legs int4 NOT NULL DEFAULT 1,
        leg_results jsonb DEFAULT '["Pending"]'::jsonb,
        result text NOT NULL DEFAULT 'Pending',
        decision text NOT NULL DEFAULT 'BET',
        notes text DEFAULT ''
    )""",
    "CREATE INDEX IF NOT EXISTS idx_bh_user ON bet_history(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_bh_ts ON bet_history(ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_bh_result ON bet_history(user_id, result)",

    # 2. User Settings
    """CREATE TABLE IF NOT EXISTS user_settings (
        id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        user_id text UNIQUE NOT NULL,
        settings jsonb NOT NULL DEFAULT '{}'::jsonb,
        updated_at timestamptz DEFAULT now()
    )""",

    # 3. Prop Line History
    """CREATE TABLE IF NOT EXISTS prop_line_history (
        id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        ts timestamptz NOT NULL DEFAULT now(),
        player text NOT NULL,
        market text NOT NULL,
        line float8,
        price float8,
        book text,
        event_id text
    )""",
    "CREATE INDEX IF NOT EXISTS idx_plh_player ON prop_line_history(player, market)",
    "CREATE INDEX IF NOT EXISTS idx_plh_ts ON prop_line_history(ts DESC)",

    # 4. Opening Lines
    """CREATE TABLE IF NOT EXISTS opening_lines (
        id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        lookup_key text UNIQUE NOT NULL,
        line float8,
        price float8,
        ts timestamptz,
        date_str text
    )""",

    # 5. User Auth
    """CREATE TABLE IF NOT EXISTS user_auth (
        id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        username text UNIQUE NOT NULL,
        pw_hash text NOT NULL,
        email text DEFAULT '',
        created_at timestamptz DEFAULT now()
    )""",

    # 6. Watchlist
    """CREATE TABLE IF NOT EXISTS watchlist (
        id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        user_id text UNIQUE NOT NULL,
        players jsonb NOT NULL DEFAULT '[]'::jsonb,
        updated_at timestamptz DEFAULT now()
    )""",

    # 7. Scan Results (background worker writes here)
    """CREATE TABLE IF NOT EXISTS scan_results (
        id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        scan_id text NOT NULL,
        sport text NOT NULL DEFAULT 'NBA',
        scanned_at timestamptz NOT NULL DEFAULT now(),
        player text NOT NULL,
        team text DEFAULT '',
        opp text DEFAULT '',
        market text NOT NULL,
        line float8,
        side text DEFAULT 'Over',
        proj float8,
        p_cal float8,
        ev_pct float8,
        edge_cat text DEFAULT '',
        l5_avg float8,
        l10_avg float8,
        src text DEFAULT 'PrizePicks',
        is_active boolean DEFAULT true
    )""",
    "CREATE INDEX IF NOT EXISTS idx_sr_active ON scan_results(sport, is_active)",
    "CREATE INDEX IF NOT EXISTS idx_sr_scan ON scan_results(scan_id)",
    "CREATE INDEX IF NOT EXISTS idx_sr_scanned ON scan_results(scanned_at DESC)",

    # 8. Worker Run Log
    """CREATE TABLE IF NOT EXISTS worker_runs (
        id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        worker_name text NOT NULL,
        status text NOT NULL,
        ran_at timestamptz NOT NULL DEFAULT now(),
        details jsonb DEFAULT '{}'::jsonb
    )""",
    "CREATE INDEX IF NOT EXISTS idx_wr_worker ON worker_runs(worker_name, ran_at DESC)",

    # 9. Notification Log
    """CREATE TABLE IF NOT EXISTS notification_log (
        id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        sent_at timestamptz NOT NULL DEFAULT now(),
        type text NOT NULL,
        message text DEFAULT '',
        delivered boolean DEFAULT true
    )""",
    "CREATE INDEX IF NOT EXISTS idx_nl_type ON notification_log(type, sent_at DESC)",

    # RLS + Policies
    "ALTER TABLE bet_history ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE user_settings ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE prop_line_history ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE opening_lines ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE user_auth ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE watchlist ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE scan_results ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE worker_runs ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE notification_log ENABLE ROW LEVEL SECURITY",
]

RLS_POLICIES = [
    ("bet_history", "allow_all_bh"),
    ("user_settings", "allow_all_us"),
    ("prop_line_history", "allow_all_plh"),
    ("opening_lines", "allow_all_ol"),
    ("user_auth", "allow_all_ua"),
    ("watchlist", "allow_all_wl"),
    ("scan_results", "allow_all_sr"),
    ("worker_runs", "allow_all_wr"),
    ("notification_log", "allow_all_nl"),
]


# AWS regions Supabase commonly hosts in — used to find the IPv4 pooler.
POOLER_REGIONS = [
    "us-east-1", "us-west-1", "us-east-2", "us-west-2",
    "eu-west-1", "eu-west-2", "eu-central-1", "eu-central-2",
    "ap-southeast-1", "ap-southeast-2", "ap-south-1",
    "ap-northeast-1", "ap-northeast-2", "ca-central-1", "sa-east-1",
]


def _try_connect(host, user, password, port=5432):
    """Try one connection target. Returns conn or None."""
    try:
        import psycopg2
        return psycopg2.connect(
            host=host, port=port, dbname="postgres", user=user,
            password=password, sslmode="require", connect_timeout=10,
        )
    except ImportError:
        import pg8000
        return pg8000.connect(
            host=host, port=port, database="postgres", user=user,
            password=password, ssl_context=True, timeout=10,
        )


def main():
    supa_url = os.environ.get("SUPABASE_URL", "")
    db_host = os.environ.get("SUPABASE_DB_HOST", "")
    db_password = os.environ.get("SUPABASE_DB_PASSWORD", "")
    db_url = os.environ.get("SUPABASE_DB_URL", "")  # full pooler URI overrides all
    pooler_region = os.environ.get("SUPABASE_POOLER_REGION", "")

    ref = ""
    if supa_url:
        ref = supa_url.replace("https://", "").split(".")[0]
    if not db_host and ref:
        db_host = f"db.{ref}.supabase.co"

    if db_url:
        # User supplied the exact connection string from the dashboard.
        import urllib.parse as up
        p = up.urlparse(db_url)
        candidates = [(p.hostname, p.username, p.password or db_password, p.port or 5432)]
    else:
        if not db_password or (not db_host and not ref):
            log.error("Set SUPABASE_DB_PASSWORD and SUPABASE_URL (or SUPABASE_DB_HOST).")
            sys.exit(1)
        # 1) direct host (IPv6 on new projects — works locally, often NOT on CI)
        # 2) session pooler (IPv4) across candidate regions
        candidates = [(db_host, "postgres", db_password, 5432)]
        regions = [pooler_region] if pooler_region else POOLER_REGIONS
        for region in regions:
            if not region:
                continue
            candidates.append(
                (f"aws-0-{region}.pooler.supabase.com", f"postgres.{ref}", db_password, 5432)
            )

    conn = None
    for host, user, password, port in candidates:
        log.info("Trying %s:%s as %s ...", host, port, user)
        try:
            conn = _try_connect(host, user, password, port)
            log.info("Connected via %s", host)
            break
        except Exception as e:
            log.warning("  failed: %s", str(e).splitlines()[0] if str(e) else e)

    if conn is None:
        log.error(
            "Could not connect to Postgres by any route. "
            "Easiest fix: open Supabase Dashboard -> SQL Editor and run "
            "migrations/001_scanner_tables.sql by hand."
        )
        sys.exit(1)

    conn.autocommit = True
    cur = conn.cursor()
    log.info("Connected!")

    # Create tables and indexes
    ok = 0
    fail = 0
    for sql in SQL_STATEMENTS:
        try:
            cur.execute(sql)
            ok += 1
        except Exception as e:
            log.warning("Statement failed (may already exist): %s — %s", sql[:60], e)
            fail += 1

    # Create RLS policies (ignore if already exist)
    for table, policy_name in RLS_POLICIES:
        try:
            cur.execute(
                f"CREATE POLICY \"{policy_name}\" ON {table} FOR ALL USING (true) WITH CHECK (true)"
            )
            ok += 1
        except Exception as e:
            if "already exists" in str(e).lower():
                ok += 1
            else:
                log.warning("Policy %s on %s: %s", policy_name, table, e)
                fail += 1

    # Verify
    cur.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='public' ORDER BY table_name"
    )
    tables = [r[0] for r in cur.fetchall()]

    conn.close()

    log.info("Done: %d succeeded, %d failed", ok, fail)
    log.info("Tables in database: %s", tables)

    required = {"scan_results", "worker_runs", "notification_log", "bet_history",
                "user_settings", "prop_line_history", "opening_lines"}
    missing = required - set(tables)
    if missing:
        log.error("MISSING tables: %s", missing)
        sys.exit(1)
    else:
        log.info("All required tables exist!")


if __name__ == "__main__":
    main()
