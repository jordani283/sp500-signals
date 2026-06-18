"""
Shared Supabase helper
======================
Loads SUPABASE_URL / SUPABASE_KEY and returns an initialised supabase-py client.

Credentials are read from environment variables. Locally these come from a .env
file (loaded via python-dotenv); in CI (e.g. GitHub Actions) there is no .env
file and the variables are injected directly — load_dotenv() is a no-op then.
The same code therefore works in both environments with no changes.
"""

import os

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()  # loads .env locally; no-op if the file is absent (e.g. in CI)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


def get_client() -> Client:
    """Create a Supabase client from environment variables.

    Raises a clear RuntimeError if the credentials are missing or the client
    cannot be constructed."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "SUPABASE_URL / SUPABASE_KEY not found. Locally, create a .env file "
            "in the project root with those two values; in GitHub Actions, set "
            "them as repository secrets (see README)."
        )
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as exc:  # malformed url/key, network stack issues, etc.
        raise RuntimeError(f"Could not initialise Supabase client: {exc}") from exc
