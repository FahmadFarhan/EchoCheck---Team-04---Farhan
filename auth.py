"""
auth.py - Authentication and Role-Based Access Control (RBAC) for SentinelVoice / EchoCheck AI.

Provides:
- Password hashing/verification using hashlib PBKDF2-HMAC-SHA256 (no extra deps)
- Session management via Streamlit session_state
- Permission matrix enforcement
- Case-level and organisation-level access checks

Roles: admin, analyst, senior_reviewer
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import streamlit as st

import database as db

# ---------------------------------------------------------------------------
# Permission matrix
# ---------------------------------------------------------------------------

# Each key is an action name; value is a set of roles that may perform it.
_PERMISSIONS: Dict[str, set] = {
    "view_org_cases":           {"admin"},
    "view_assigned_cases":      {"admin", "analyst", "senior_reviewer"},
    "create_case":              {"admin", "analyst", "senior_reviewer"},
    "assign_case_members":      {"admin"},
    "upload_audio":             {"admin", "analyst"},
    "run_deepfake_detection":   {"admin", "analyst"},
    "enroll_verify_speaker":    {"admin", "analyst"},
    "view_analysis_results":    {"admin", "analyst", "senior_reviewer"},
    "generate_draft_report":    {"admin", "analyst", "senior_reviewer"},
    "approve_reject_report":    {"senior_reviewer"},
    "manage_users":             {"admin"},
    "view_org_audit_log":       {"admin"},
    "view_case_audit_history":  {"admin", "analyst", "senior_reviewer"},
    "change_api_config":        {"admin"},
}


def has_permission(role: str, action: str) -> bool:
    """Returns True if the given role is allowed to perform the action."""
    allowed = _PERMISSIONS.get(action, set())
    return role in allowed


# ---------------------------------------------------------------------------
# Password hashing (PBKDF2-HMAC-SHA256, no external deps)
# ---------------------------------------------------------------------------

_ITERATIONS = 260_000
_HASH_ALG = "sha256"


def hash_password(password: str) -> str:
    """
    Hashes a password with a random salt using PBKDF2-HMAC-SHA256.
    Returns a string in the form  salt_hex:hash_hex  that is safe to store.
    """
    salt = secrets.token_hex(32)  # 256-bit salt
    pw_hash = hashlib.pbkdf2_hmac(
        _HASH_ALG,
        password.encode("utf-8"),
        salt.encode("utf-8"),
        _ITERATIONS,
    ).hex()
    return f"{salt}:{pw_hash}"


def verify_password(password: str, stored_hash: str) -> bool:
    """
    Verifies a plaintext password against a stored hash string (salt:hash).
    Returns True on match, False otherwise.  Constant-time safe comparison.
    """
    try:
        salt, expected = stored_hash.split(":", 1)
    except (ValueError, AttributeError):
        return False
    candidate = hashlib.pbkdf2_hmac(
        _HASH_ALG,
        password.encode("utf-8"),
        salt.encode("utf-8"),
        _ITERATIONS,
    ).hex()
    return secrets.compare_digest(candidate, expected)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

_SESSION_KEY = "_sv_auth"


def _session() -> Dict[str, Any]:
    """Returns the auth session dict from st.session_state, creating it if needed."""
    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = {}
    return st.session_state[_SESSION_KEY]


def is_logged_in() -> bool:
    """Returns True if the current session has an authenticated user."""
    sess = _session()
    return bool(sess.get("user_id") and sess.get("role"))


def get_current_user() -> Optional[Dict[str, Any]]:
    """
    Returns a copy of the authenticated user's session data, or None.

    Keys: user_id, username, display_name, role, org_id, org_name
    """
    if not is_logged_in():
        return None
    sess = _session()
    return {
        "user_id":      sess["user_id"],
        "username":     sess["username"],
        "display_name": sess.get("display_name", sess["username"]),
        "role":         sess["role"],
        "org_id":       sess["org_id"],
        "org_name":     sess.get("org_name", ""),
    }


def login(username: str, password: str) -> tuple[bool, str]:
    """
    Authenticates a user.  Sets session state on success.
    Returns (success: bool, message: str).
    """
    username = (username or "").strip().lower()
    if not username or not password:
        return False, "Username and password are required."

    user = db.get_user_by_username(username)
    if user is None:
        _log_auth_event(username=username, outcome="FAILED_NO_USER")
        return False, "Invalid credentials."

    if not user.get("is_active"):
        _log_auth_event(username=username, user_id=user["user_id"], outcome="FAILED_INACTIVE")
        return False, "This account has been deactivated. Contact your administrator."

    if not verify_password(password, user["password_hash"]):
        _log_auth_event(username=username, user_id=user["user_id"], outcome="FAILED_WRONG_PASSWORD")
        return False, "Invalid credentials."

    # Fetch org name
    org = db.get_organisation(user["org_id"])
    org_name = org["name"] if org else ""

    # Populate session
    sess = _session()
    sess["user_id"]      = user["user_id"]
    sess["username"]     = user["username"]
    sess["display_name"] = user.get("display_name") or user["username"]
    sess["role"]         = user["role"]
    sess["org_id"]       = user["org_id"]
    sess["org_name"]     = org_name

    _log_auth_event(username=username, user_id=user["user_id"], org_id=user["org_id"], outcome="SUCCESS")
    db.log_app_audit_event(
        action="LOGIN",
        user_id=user["user_id"],
        org_id=user["org_id"],
        outcome="SUCCESS",
        metadata={"username": username},
    )
    return True, "Login successful."


def logout() -> None:
    """Clears the authentication session."""
    sess = _session()
    user = get_current_user()
    if user:
        db.log_app_audit_event(
            action="LOGOUT",
            user_id=user["user_id"],
            org_id=user["org_id"],
            outcome="SUCCESS",
            metadata={"username": user["username"]},
        )
    sess.clear()
    # Also clear latest_case_id so next login starts fresh
    if "latest_case_id" in st.session_state:
        del st.session_state["latest_case_id"]


def require_login() -> bool:
    """
    Call at the top of every page.  Shows login form if not authenticated.
    Returns True if the user is logged in, False if the login form was shown.
    """
    if is_logged_in():
        return True
    _render_login_form()
    return False


def require_permission(action: str, silent: bool = False) -> bool:
    """
    Checks if the current user has permission to perform *action*.
    If not, displays an error (unless silent=True) and returns False.
    """
    user = get_current_user()
    if user is None:
        if not silent:
            st.error("🔐 You must be logged in to perform this action.", icon="🔒")
        return False
    if not has_permission(user["role"], action):
        if not silent:
            st.error(
                f"🚫 **Access Denied**: Your role (`{user['role']}`) does not have permission to perform this action.",
                icon="🚫",
            )
        return False
    return True


# ---------------------------------------------------------------------------
# Case-level access checks
# ---------------------------------------------------------------------------

def can_access_case(case: Dict[str, Any]) -> bool:
    """
    Returns True if the current user may view/interact with the given case dict.

    Rules (backend-enforced):
    1. User must be logged in.
    2. User's org_id must match the case's org_id.
    3. Admin may access all cases in their org.
    4. Analyst/senior_reviewer may access only cases they are assigned to.
    """
    user = get_current_user()
    if user is None:
        return False

    # Organisation isolation
    case_org = case.get("org_id")
    if case_org and case_org != user["org_id"]:
        return False

    # Admin sees all org cases
    if user["role"] == "admin":
        return True

    # Analyst/reviewer: must be an assigned member
    return db.is_case_member(case["case_id"], user["user_id"])


def assert_case_access(case: Optional[Dict[str, Any]]) -> bool:
    """
    Like can_access_case but also renders an error if access is denied.
    Returns True on success.
    """
    if case is None:
        st.error("Case not found.", icon="❌")
        return False
    if not can_access_case(case):
        st.error(
            "🚫 **Access Denied**: You are not authorised to access this case. "
            "Only assigned team members may view this evidence record.",
            icon="🚫",
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Login form UI
# ---------------------------------------------------------------------------

def _render_login_form() -> None:
    """Renders the authentication login form in the main content area."""
    st.markdown(
        """
        <div style="max-width: 420px; margin: 3rem auto 0 auto;">
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <div style="font-size: 2.5rem;">⚖️</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #0f172a; letter-spacing: -0.01em;">
                    SENTINEL<span style="color: #2563eb;">VOICE</span>
                </div>
                <div style="font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.2rem;">
                    Forensic Voice Evidence Workstation
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_l, col_mid, col_r = st.columns([1, 2, 1])
    with col_mid:
        st.markdown(
            """
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px;
                        padding: 1.5rem 1.5rem 1rem 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
                <div style="font-size: 1rem; font-weight: 700; color: #1e293b; margin-bottom: 0.1rem;">Sign In</div>
                <div style="font-size: 0.78rem; color: #64748b; margin-bottom: 0.75rem;">
                    Authorised personnel only. All access is logged.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("sv_login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="your.username")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submit = st.form_submit_button("Sign In", use_container_width=True, type="primary")

            if submit:
                ok, msg = login(username, password)
                if ok:
                    st.success("✅ " + msg)
                    st.rerun()
                else:
                    st.error("❌ " + msg)

        st.markdown(
            """
            <div style="text-align: center; font-size: 0.72rem; color: #94a3b8; margin-top: 0.75rem;">
                Access restricted to authorised personnel only.<br/>
                All authentication events are audit-logged.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_auth_event(
    username: str,
    user_id: Optional[str] = None,
    org_id: Optional[str] = None,
    outcome: str = "UNKNOWN",
) -> None:
    """Logs authentication attempt to the app audit log. Errors are suppressed."""
    try:
        db.log_app_audit_event(
            action="LOGIN_ATTEMPT",
            user_id=user_id,
            org_id=org_id,
            outcome=outcome,
            metadata={"username": username},
        )
    except Exception:
        pass
