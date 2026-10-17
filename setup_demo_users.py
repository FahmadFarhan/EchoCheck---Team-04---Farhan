"""
setup_demo_users.py - Idempotent demo account provisioner for SentinelVoice / EchoCheck AI.

Creates one demo organisation and three role-based demo accounts.
Safe to re-run: it will NOT overwrite existing passwords or recreate
conflicting records. If a username already exists the account is left
unchanged and a warning is printed.

Usage:
    python setup_demo_users.py

Or from PowerShell inside the .venv:
    .venv\\Scripts\\python.exe setup_demo_users.py

Environment variables (optional — override defaults):
    DEMO_ORG_NAME       Organisation display name   (default: "EchoCheck Demo Forensics Lab")
    DEMO_ADMIN_USER     Admin username              (default: "admin.demo")
    DEMO_ADMIN_PASS     Admin password              (default: auto-generated & printed once)
    DEMO_ANALYST_USER   Analyst username            (default: "analyst.demo")
    DEMO_ANALYST_PASS   Analyst password            (default: auto-generated & printed once)
    DEMO_REVIEWER_USER  Senior Reviewer username    (default: "reviewer.demo")
    DEMO_REVIEWER_PASS  Senior Reviewer password    (default: auto-generated & printed once)

SECURITY NOTE: Never use auto-generated demo passwords in production.
               All passwords are stored as PBKDF2-HMAC-SHA256 hashes.
               API keys are never written here.
"""

from __future__ import annotations

import os
import secrets
import sys

from dotenv import load_dotenv

load_dotenv()

# Import after load_dotenv so SENTINEL_DB_PATH is picked up
import database as db
from auth import hash_password

# ---------------------------------------------------------------------------
# Configuration — read from environment or use safe defaults
# ---------------------------------------------------------------------------

ORG_NAME = os.getenv("DEMO_ORG_NAME", "EchoCheck Demo Forensics Lab")

_ACCOUNTS = [
    {
        "env_user": "DEMO_ADMIN_USER",
        "env_pass": "DEMO_ADMIN_PASS",
        "default_username": "admin.demo",
        "role": "admin",
        "display_name": "Demo Administrator",
    },
    {
        "env_user": "DEMO_ANALYST_USER",
        "env_pass": "DEMO_ANALYST_PASS",
        "default_username": "analyst.demo",
        "role": "analyst",
        "display_name": "Demo Forensic Analyst",
    },
    {
        "env_user": "DEMO_REVIEWER_USER",
        "env_pass": "DEMO_REVIEWER_PASS",
        "default_username": "reviewer.demo",
        "role": "senior_reviewer",
        "display_name": "Demo Senior Reviewer",
    },
]

# Fixed demo org_id so it is stable across re-runs
DEMO_ORG_ID = "org-echocheck-demo-001"


def _generate_password() -> str:
    """Generates a cryptographically random 16-character password."""
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
    return "".join(secrets.choice(alphabet) for _ in range(16))


def setup() -> None:
    """Main provisioning function. Idempotent — safe to run multiple times."""
    db.init_db()

    # --- Organisation ---
    existing_org = db.get_organisation(DEMO_ORG_ID)
    if existing_org is None:
        db.create_organisation(name=ORG_NAME, org_id=DEMO_ORG_ID)
        print(f"✅ Organisation created: '{ORG_NAME}' (org_id={DEMO_ORG_ID})")
    else:
        print(f"ℹ️  Organisation already exists: '{existing_org['name']}' (org_id={DEMO_ORG_ID})")

    print()
    print("=" * 60)
    print("DEMO ACCOUNT CREDENTIALS")
    print("=" * 60)

    for account in _ACCOUNTS:
        username = os.getenv(account["env_user"], account["default_username"]).strip().lower()
        password_env = os.getenv(account["env_pass"], "").strip()

        if db.user_exists(username):
            print(
                f"⚠️  [{account['role'].upper():16s}] '{username}' already exists — "
                f"password NOT changed."
            )
            continue

        # Use provided password or generate one
        if password_env:
            password = password_env
            password_source = "(from environment)"
        else:
            password = _generate_password()
            password_source = "(auto-generated — copy now, not shown again)"

        pw_hash = hash_password(password)
        db.create_user(
            username=username,
            password_hash=pw_hash,
            role=account["role"],
            org_id=DEMO_ORG_ID,
            display_name=account["display_name"],
        )

        print(
            f"✅ [{account['role'].upper():16s}] username={username!r}  "
            f"password={password!r}  {password_source}"
        )

    print()
    print("=" * 60)
    print("Setup complete.")
    print()
    print("Start the application with:")
    print("    .venv\\Scripts\\python.exe -m streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    setup()
