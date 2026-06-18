"""
Supabase schema setup
=====================
Creates the two tables used by the signal monitor / trade updater:
  - signal_log : one row per monitor run (full run history)
  - trade_log  : one row per fired signal (PENDING -> CLOSED lifecycle)

Run once before using signal_monitor.py / trade_update.py:
    python setup_supabase.py

This uses the Supabase Python client to execute the DDL via an `exec_sql`
RPC. If that helper function does not exist in your project yet (it does not by
default, and the anon key cannot run DDL through PostgREST), the script prints
the exact SQL to paste into the Supabase dashboard SQL Editor instead.
"""

import sys

from supabase_client import get_client

SCHEMA_SQL = """
create table if not exists signal_log (
  id bigint generated always as identity primary key,
  run_date date not null,
  signal_date date not null,
  signal_type text not null,
  regime text not null,
  spx_close numeric not null,
  change_pct numeric not null,
  ma_200 numeric not null,
  triggered boolean not null default false,
  entry_date date,
  exit_date date,
  realistic_ev_pct numeric,
  win_rate_pct numeric,
  created_at timestamptz default now()
);

create table if not exists trade_log (
  id bigint generated always as identity primary key,
  signal_date date not null,
  signal_type text not null,
  regime text not null,
  entry_date date,
  exit_date date,
  entry_price numeric,
  exit_price numeric,
  actual_return_pct numeric,
  status text not null default 'PENDING',
  created_at timestamptz default now()
);
""".strip()

# One-time helper that lets the anon/service client run raw SQL via RPC.
EXEC_SQL_BOOTSTRAP = (
    "create or replace function public.exec_sql(sql text)\n"
    "returns void language plpgsql security definer as $$\n"
    "begin execute sql; end;\n"
    "$$;"
)


def tables_exist(client) -> bool:
    """Return True only if both tables are queryable."""
    for table in ("signal_log", "trade_log"):
        try:
            client.table(table).select("id").limit(1).execute()
        except Exception:
            return False
    return True


def manual_instructions():
    print("\nCould not create tables automatically with the current key.")
    print("The anon key cannot run DDL through the API, and no `exec_sql` RPC was found.")
    print("\nTo finish setup, open the Supabase SQL Editor:")
    print("  https://supabase.com/dashboard/project/_/sql/new")
    print("and run the following SQL once:\n")
    print(SCHEMA_SQL)
    print("\n(Optional) To enable fully automated setup on future runs, also run:\n")
    print(EXEC_SQL_BOOTSTRAP)


def main():
    try:
        client = get_client()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    # Preferred path: run the DDL through an exec_sql RPC if it exists.
    try:
        client.rpc("exec_sql", {"sql": SCHEMA_SQL}).execute()
        print("Supabase tables created successfully")
        return
    except Exception as exc:
        # RPC missing or not permitted — fall back to verifying / instructing.
        if tables_exist(client):
            print("Supabase tables created successfully")
            return
        print(f"Automated DDL not available ({str(exc)[:120]})")
        manual_instructions()
        sys.exit(1)


if __name__ == "__main__":
    main()
