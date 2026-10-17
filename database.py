"""
database.py - SQLite storage and investigation records for SentinelVoice.

Maintains an audit-ready trail of all account recovery audio assessments,
recommendations, analyst notes, and status workflows.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = os.getenv("SENTINEL_DB_PATH", "sentinel_voice.db")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Creates a database connection with row factory enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initializes the database schema if it does not already exist."""
    schema = """
    CREATE TABLE IF NOT EXISTS cases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id TEXT UNIQUE NOT NULL,
        case_reference TEXT NOT NULL,
        account_action TEXT NOT NULL,
        customer_identifier TEXT,
        description TEXT,
        audio_filename TEXT NOT NULL,
        audio_filesize_bytes INTEGER,
        audio_duration_seconds REAL,
        api_status TEXT NOT NULL,
        api_error_message TEXT,
        request_id TEXT,
        operation_id TEXT,
        verdict TEXT NOT NULL,
        label TEXT,
        is_synthetic INTEGER,
        score REAL,
        threshold REAL,
        risk_level TEXT NOT NULL,
        recommendation_code TEXT NOT NULL,
        recommendation_text TEXT NOT NULL,
        analyst_decision TEXT,
        analyst_notes TEXT,
        case_status TEXT NOT NULL DEFAULT 'NEW',
        raw_response_json TEXT,
        sha256_hash TEXT,
        speaker_id TEXT,
        speaker_display_name TEXT,
        speaker_verify_decision TEXT,
        speaker_similarity REAL,
        speaker_threshold REAL,
        speaker_verify_request_id TEXT,
        audio_integrity_json TEXT,
        record_type TEXT NOT NULL DEFAULT 'forensic',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS speakers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        speaker_id TEXT UNIQUE NOT NULL,
        display_name TEXT,
        case_reference TEXT,
        provenance_notes TEXT,
        samples_count INTEGER DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS case_audit_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        event_description TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    -- RBAC: Organisations
    CREATE TABLE IF NOT EXISTS organisations (
        org_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    -- RBAC: Users
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        display_name TEXT,
        role TEXT NOT NULL CHECK(role IN ('admin', 'analyst', 'senior_reviewer')),
        org_id TEXT NOT NULL,
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (org_id) REFERENCES organisations(org_id)
    );

    -- RBAC: Case membership (analyst/reviewer assignment)
    CREATE TABLE IF NOT EXISTS case_members (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        assigned_by TEXT,
        assigned_at TEXT NOT NULL,
        UNIQUE(case_id, user_id)
    );

    -- Reports workflow (draft -> submitted -> approved/rejected)
    CREATE TABLE IF NOT EXISTS reports (
        report_id TEXT PRIMARY KEY,
        case_id TEXT NOT NULL,
        org_id TEXT NOT NULL,
        author_user_id TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'draft'
            CHECK(status IN ('draft', 'submitted', 'approved', 'rejected')),
        reviewer_user_id TEXT,
        reviewer_comment TEXT,
        report_version INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    -- Application-wide append-only audit log
    CREATE TABLE IF NOT EXISTS app_audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT UNIQUE NOT NULL,
        action TEXT NOT NULL,
        user_id TEXT,
        org_id TEXT,
        case_id TEXT,
        outcome TEXT NOT NULL DEFAULT 'SUCCESS',
        metadata_json TEXT,
        created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_cases_case_id ON cases(case_id);
    CREATE INDEX IF NOT EXISTS idx_cases_reference ON cases(case_reference);
    CREATE INDEX IF NOT EXISTS idx_cases_verdict ON cases(verdict);
    CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(case_status);
    CREATE INDEX IF NOT EXISTS idx_cases_created ON cases(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_speakers_speaker_id ON speakers(speaker_id);
    CREATE INDEX IF NOT EXISTS idx_audit_case_id ON case_audit_events(case_id);
    CREATE INDEX IF NOT EXISTS idx_audit_created ON case_audit_events(created_at ASC);
    CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
    CREATE INDEX IF NOT EXISTS idx_users_org ON users(org_id);
    CREATE INDEX IF NOT EXISTS idx_case_members_case ON case_members(case_id);
    CREATE INDEX IF NOT EXISTS idx_case_members_user ON case_members(user_id);
    CREATE INDEX IF NOT EXISTS idx_reports_case ON reports(case_id);
    CREATE INDEX IF NOT EXISTS idx_app_audit_action ON app_audit_log(action);
    CREATE INDEX IF NOT EXISTS idx_app_audit_user ON app_audit_log(user_id);
    CREATE INDEX IF NOT EXISTS idx_app_audit_org ON app_audit_log(org_id);
    CREATE INDEX IF NOT EXISTS idx_app_audit_created ON app_audit_log(created_at DESC);
    """
    with get_connection(db_path) as conn:
        conn.executescript(schema)
        # Non-destructive migrations for existing database instances:
        columns_to_add = [
            ("sha256_hash", "TEXT"),
            ("speaker_id", "TEXT"),
            ("speaker_display_name", "TEXT"),
            ("speaker_verify_decision", "TEXT"),
            ("speaker_similarity", "REAL"),
            ("speaker_threshold", "REAL"),
            ("speaker_verify_request_id", "TEXT"),
            ("audio_integrity_json", "TEXT"),
            ("record_type", "TEXT"),
            ("org_id", "TEXT"),        # RBAC: organisation isolation for cases
            ("created_by_user_id", "TEXT"),  # RBAC: track who created the case
        ]
        for col_name, col_type in columns_to_add:
            try:
                conn.execute(f"ALTER TABLE cases ADD COLUMN {col_name} {col_type};")
            except sqlite3.OperationalError:
                pass  # column already exists
        # Backfill record_type = 'forensic' for any rows where it is NULL
        conn.execute("UPDATE cases SET record_type = 'forensic' WHERE record_type IS NULL;")
        conn.commit()


def generate_case_id() -> str:
    """Generates a human-readable unique case ID for EchoCheck AI."""
    now_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:6].upper()
    return f"EC-{now_str}-{short_uuid}"


def log_case_audit_event(
    case_id: str,
    event_type: str,
    event_description: str,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Appends an immutable audit event to the case history."""
    init_db(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    sql = """
    INSERT INTO case_audit_events (case_id, event_type, event_description, created_at)
    VALUES (?, ?, ?, ?);
    """
    with get_connection(db_path) as conn:
        conn.execute(sql, (case_id, event_type, event_description, now_iso))
        conn.commit()


def get_case_audit_events(case_id: str, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves all chronological audit events for a specific case."""
    init_db(db_path)
    sql = "SELECT * FROM case_audit_events WHERE case_id = ? ORDER BY created_at ASC;"
    with get_connection(db_path) as conn:
        cursor = conn.execute(sql, (case_id,))
        return [dict(r) for r in cursor.fetchall()]


# Legacy financial-era category labels that pre-date the forensics conversion.
# Records with these account_action values are relabeled 'legacy_demo' so they
# are distinguishable from genuine forensic evidence submissions.
_LEGACY_FINANCIAL_CATEGORIES = [
    "Password & 2FA Reset Combined",
    "Password Reset",
    "2FA Reset",
    "Account Unlock",
    "Beneficiary Change",
    "Large Transfer Approval",
    "Account Closure",
]


def mark_legacy_records(db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Non-destructively marks pre-forensics (financial-era) records as 'legacy_demo'.

    Identifies rows whose account_action still contains original financial-fraud
    category labels that existed before the application was converted to forensic use.
    These records are preserved in full; only their record_type is updated.

    Returns the number of rows updated.
    """
    init_db(db_path)
    placeholders = ", ".join("?" for _ in _LEGACY_FINANCIAL_CATEGORIES)
    sql = f"""
        UPDATE cases
        SET record_type = 'legacy_demo', updated_at = ?
        WHERE account_action IN ({placeholders})
          AND record_type != 'legacy_demo';
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.execute(sql, [now_iso] + _LEGACY_FINANCIAL_CATEGORIES)
        conn.commit()
        return cursor.rowcount


def create_case(case_data: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> str:
    """
    Inserts a newly evaluated case record into the database.
    Returns the assigned case_id.
    """
    init_db(db_path)
    case_id = case_data.get("case_id") or generate_case_id()
    now_iso = datetime.now(timezone.utc).isoformat()

    raw_response = case_data.get("raw_response_json")
    if isinstance(raw_response, dict):
        raw_response_json = json.dumps(raw_response)
    elif isinstance(raw_response, str):
        raw_response_json = raw_response
    else:
        raw_response_json = None

    sql = """
    INSERT INTO cases (
        case_id,
        case_reference,
        account_action,
        customer_identifier,
        description,
        audio_filename,
        audio_filesize_bytes,
        audio_duration_seconds,
        api_status,
        api_error_message,
        request_id,
        operation_id,
        verdict,
        label,
        is_synthetic,
        score,
        threshold,
        risk_level,
        recommendation_code,
        recommendation_text,
        analyst_decision,
        analyst_notes,
        case_status,
        raw_response_json,
        sha256_hash,
        speaker_id,
        speaker_display_name,
        speaker_verify_decision,
        speaker_similarity,
        speaker_threshold,
        speaker_verify_request_id,
        audio_integrity_json,
        record_type,
        created_at,
        updated_at
    ) VALUES (
        :case_id,
        :case_reference,
        :account_action,
        :customer_identifier,
        :description,
        :audio_filename,
        :audio_filesize_bytes,
        :audio_duration_seconds,
        :api_status,
        :api_error_message,
        :request_id,
        :operation_id,
        :verdict,
        :label,
        :is_synthetic,
        :score,
        :threshold,
        :risk_level,
        :recommendation_code,
        :recommendation_text,
        :analyst_decision,
        :analyst_notes,
        :case_status,
        :raw_response_json,
        :sha256_hash,
        :speaker_id,
        :speaker_display_name,
        :speaker_verify_decision,
        :speaker_similarity,
        :speaker_threshold,
        :speaker_verify_request_id,
        :audio_integrity_json,
        :record_type,
        :created_at,
        :updated_at
    );
    """

    integrity_raw = case_data.get("audio_integrity_json")
    if isinstance(integrity_raw, dict):
        audio_integrity_json = json.dumps(integrity_raw)
    elif isinstance(integrity_raw, str):
        audio_integrity_json = integrity_raw
    else:
        audio_integrity_json = None

    params = {
        "case_id": case_id,
        "case_reference": case_data.get("case_reference", "").strip() or "UNSPECIFIED",
        "account_action": case_data.get("account_action", "General Disputed Audio Evidence"),
        "customer_identifier": case_data.get("customer_identifier", ""),
        "description": case_data.get("description", ""),
        "audio_filename": case_data.get("audio_filename", "audio_sample.wav"),
        "audio_filesize_bytes": case_data.get("audio_filesize_bytes", 0),
        "audio_duration_seconds": case_data.get("audio_duration_seconds"),
        "api_status": case_data.get("api_status", "UNKNOWN"),
        "api_error_message": case_data.get("api_error_message"),
        "request_id": case_data.get("request_id"),
        "operation_id": case_data.get("operation_id"),
        "verdict": case_data.get("verdict", "unknown"),
        "label": case_data.get("label"),
        "is_synthetic": 1 if case_data.get("is_synthetic") is True else (0 if case_data.get("is_synthetic") is False else None),
        "score": case_data.get("score"),
        "threshold": case_data.get("threshold"),
        "risk_level": case_data.get("risk_level", "UNKNOWN"),
        "recommendation_code": case_data.get("recommendation_code", "REVIEW_REQUIRED"),
        "recommendation_text": case_data.get("recommendation_text", ""),
        "analyst_decision": case_data.get("analyst_decision"),
        "analyst_notes": case_data.get("analyst_notes"),
        "case_status": case_data.get("case_status", "NEW"),
        "raw_response_json": raw_response_json,
        "sha256_hash": case_data.get("sha256_hash"),
        "speaker_id": case_data.get("speaker_id"),
        "speaker_display_name": case_data.get("speaker_display_name"),
        "speaker_verify_decision": case_data.get("speaker_verify_decision"),
        "speaker_similarity": case_data.get("speaker_similarity"),
        "speaker_threshold": case_data.get("speaker_threshold"),
        "speaker_verify_request_id": case_data.get("speaker_verify_request_id"),
        "audio_integrity_json": audio_integrity_json,
        "record_type": case_data.get("record_type", "forensic"),
        "created_at": case_data.get("created_at") or now_iso,
        "updated_at": now_iso,
    }

    with get_connection(db_path) as conn:
        conn.execute(sql, params)
        conn.commit()

    return case_id


def get_case(case_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieves a single case record by case_id."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
    return None


def list_cases(
    db_path: str = DEFAULT_DB_PATH,
    verdict: Optional[str] = None,
    status: Optional[str] = None,
    search_query: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Lists cases with optional filtering, sorted newest first."""
    init_db(db_path)
    query_parts = ["SELECT * FROM cases WHERE 1=1"]
    params: List[Any] = []

    if verdict and verdict != "ALL":
        query_parts.append("AND verdict = ?")
        params.append(verdict)

    if status and status != "ALL":
        query_parts.append("AND case_status = ?")
        params.append(status)

    if search_query and search_query.strip():
        q = f"%{search_query.strip()}%"
        query_parts.append(
            "AND (case_id LIKE ? OR case_reference LIKE ? OR customer_identifier LIKE ? OR description LIKE ?)"
        )
        params.extend([q, q, q, q])

    query_parts.append("ORDER BY created_at DESC LIMIT ?")
    params.append(limit)

    final_query = " ".join(query_parts)
    with get_connection(db_path) as conn:
        cursor = conn.execute(final_query, params)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def update_case_analyst_review(
    case_id: str,
    case_status: str,
    analyst_decision: str,
    analyst_notes: str,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Updates analyst notes, decision, and status for a case."""
    init_db(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    sql = """
    UPDATE cases
    SET case_status = ?,
        analyst_decision = ?,
        analyst_notes = ?,
        updated_at = ?
    WHERE case_id = ?;
    """
    with get_connection(db_path) as conn:
        cursor = conn.execute(sql, (case_status, analyst_decision, analyst_notes, now_iso, case_id))
        conn.commit()
        return cursor.rowcount > 0


def update_case(
    case_id: str,
    updates: Dict[str, Any],
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Updates arbitrary fields for a case."""
    if not updates:
        return True
    init_db(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    fields = []
    values = []
    for k, v in updates.items():
        fields.append(f"{k} = ?")
        if isinstance(v, (dict, list)):
            values.append(json.dumps(v))
        else:
            values.append(v)
    fields.append("updated_at = ?")
    values.append(now_iso)
    values.append(case_id)

    sql = f"UPDATE cases SET {', '.join(fields)} WHERE case_id = ?;"
    with get_connection(db_path) as conn:
        cursor = conn.execute(sql, values)
        conn.commit()
        return cursor.rowcount > 0


def get_dashboard_stats(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Computes summary statistics for the dashboard."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        fake_count = conn.execute("SELECT COUNT(*) FROM cases WHERE verdict = 'fake'").fetchone()[0]
        real_count = conn.execute("SELECT COUNT(*) FROM cases WHERE verdict = 'real'").fetchone()[0]
        not_sure_count = conn.execute("SELECT COUNT(*) FROM cases WHERE verdict = 'not_sure'").fetchone()[0]
        failed_count = conn.execute("SELECT COUNT(*) FROM cases WHERE api_status != 'SUCCESS'").fetchone()[0]
        escalated_count = conn.execute("SELECT COUNT(*) FROM cases WHERE case_status = 'ESCALATED'").fetchone()[0]
        pending_count = conn.execute("SELECT COUNT(*) FROM cases WHERE case_status IN ('NEW', 'PENDING_INVESTIGATION')").fetchone()[0]

    return {
        "total_cases": total,
        "fake_detected": fake_count,
        "real_verified": real_count,
        "inconclusive": not_sure_count,
        "failed_analysis": failed_count,
        "escalated_cases": escalated_count,
        "pending_cases": pending_count,
    }


def save_enrolled_speaker(
    speaker_id: str,
    display_name: Optional[str] = None,
    case_reference: Optional[str] = None,
    provenance_notes: Optional[str] = None,
    samples_count: int = 1,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Inserts or updates an enrolled reference speaker record."""
    init_db(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    clean_spk_id = speaker_id.strip()

    sql = """
    INSERT INTO speakers (speaker_id, display_name, case_reference, provenance_notes, samples_count, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(speaker_id) DO UPDATE SET
        display_name = coalesce(excluded.display_name, speakers.display_name),
        case_reference = coalesce(excluded.case_reference, speakers.case_reference),
        provenance_notes = coalesce(excluded.provenance_notes, speakers.provenance_notes),
        samples_count = excluded.samples_count,
        updated_at = excluded.updated_at;
    """
    with get_connection(db_path) as conn:
        conn.execute(sql, (clean_spk_id, display_name, case_reference, provenance_notes, samples_count, now_iso, now_iso))
        conn.commit()
        return True


def list_local_speakers(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Lists all enrolled reference speakers saved in the local repository."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute("SELECT * FROM speakers ORDER BY created_at DESC;")
        return [dict(r) for r in cursor.fetchall()]


def get_local_speaker(speaker_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieves an enrolled reference speaker by speaker_id."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute("SELECT * FROM speakers WHERE speaker_id = ?;", (speaker_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
    return None


def delete_local_speaker(speaker_id: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Deletes an enrolled speaker from local records."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM speakers WHERE speaker_id = ?;", (speaker_id,))
        conn.commit()
        return cursor.rowcount > 0



# ===========================================================================
# RBAC: Organisation functions
# ===========================================================================

def create_organisation(
    name: str,
    org_id: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Creates a new organisation. Returns the org_id."""
    init_db(db_path)
    oid = org_id or str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO organisations (org_id, name, created_at) VALUES (?, ?, ?);",
            (oid, name, now),
        )
        conn.commit()
    return oid


def get_organisation(org_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieves an organisation by org_id."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM organisations WHERE org_id = ?;", (org_id,)).fetchone()
        return dict(row) if row else None


def list_organisations(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Lists all organisations."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM organisations ORDER BY name;").fetchall()
        return [dict(r) for r in rows]


# ===========================================================================
# RBAC: User functions
# ===========================================================================

def create_user(
    username: str,
    password_hash: str,
    role: str,
    org_id: str,
    display_name: Optional[str] = None,
    user_id: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Creates a new user account. Returns the user_id."""
    init_db(db_path)
    uid = user_id or str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO users (user_id, username, password_hash, display_name, role, org_id, is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?);
            """,
            (uid, username.strip().lower(), password_hash, display_name, role, org_id, now, now),
        )
        conn.commit()
    return uid


def get_user_by_username(username: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieves a user by username (case-insensitive)."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?;",
            (username.strip().lower(),),
        ).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieves a user by user_id."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id = ?;", (user_id,)).fetchone()
        return dict(row) if row else None


def list_users(org_id: Optional[str] = None, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Lists users, optionally filtered by org_id."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        if org_id:
            rows = conn.execute(
                "SELECT * FROM users WHERE org_id = ? ORDER BY username;",
                (org_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM users ORDER BY username;").fetchall()
        return [dict(r) for r in rows]


def update_user(
    user_id: str,
    display_name: Optional[str] = None,
    role: Optional[str] = None,
    is_active: Optional[bool] = None,
    password_hash: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Updates mutable user fields. Only provided (non-None) fields are changed."""
    init_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    updates = ["updated_at = ?"]
    params: List[Any] = [now]
    if display_name is not None:
        updates.append("display_name = ?")
        params.append(display_name)
    if role is not None:
        updates.append("role = ?")
        params.append(role)
    if is_active is not None:
        updates.append("is_active = ?")
        params.append(1 if is_active else 0)
    if password_hash is not None:
        updates.append("password_hash = ?")
        params.append(password_hash)
    params.append(user_id)
    sql = f"UPDATE users SET {', '.join(updates)} WHERE user_id = ?;"
    with get_connection(db_path) as conn:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.rowcount > 0


def user_exists(username: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Returns True if a user with the given username already exists."""
    return get_user_by_username(username, db_path) is not None


# ===========================================================================
# RBAC: Case membership
# ===========================================================================

def assign_case_member(
    case_id: str,
    user_id: str,
    assigned_by: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Assigns a user to a case. Idempotent (ignores duplicates)."""
    init_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO case_members (case_id, user_id, assigned_by, assigned_at)
            VALUES (?, ?, ?, ?);
            """,
            (case_id, user_id, assigned_by, now),
        )
        conn.commit()
        return True


def remove_case_member(
    case_id: str,
    user_id: str,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Removes a user's assignment from a case."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "DELETE FROM case_members WHERE case_id = ? AND user_id = ?;",
            (case_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0


def is_case_member(case_id: str, user_id: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Returns True if the user is assigned to the case."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM case_members WHERE case_id = ? AND user_id = ?;",
            (case_id, user_id),
        ).fetchone()
        return row is not None


def get_case_members(case_id: str, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Returns the list of users assigned to a case (with user details)."""
    init_db(db_path)
    sql = """
        SELECT u.user_id, u.username, u.display_name, u.role,
               cm.assigned_at, cm.assigned_by
        FROM case_members cm
        JOIN users u ON u.user_id = cm.user_id
        WHERE cm.case_id = ?
        ORDER BY cm.assigned_at ASC;
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(sql, (case_id,)).fetchall()
        return [dict(r) for r in rows]


def list_cases_for_user(
    user_id: str,
    org_id: str,
    role: str,
    verdict: Optional[str] = None,
    status: Optional[str] = None,
    search_query: Optional[str] = None,
    limit: int = 200,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Returns cases accessible to a given user, based on their role.
    - admin: all cases belonging to their org_id
    - analyst/senior_reviewer: only cases they are assigned to (via case_members),
      filtered to their org_id for isolation
    """
    init_db(db_path)
    if role == "admin":
        # Admin sees all org cases.  Also includes cases with NULL org_id
        # (legacy records created before RBAC was introduced).
        query_parts = ["SELECT * FROM cases WHERE (org_id = ? OR org_id IS NULL)"]
        params: List[Any] = [org_id]
    else:
        query_parts = [
            """
            SELECT c.* FROM cases c
            JOIN case_members cm ON cm.case_id = c.case_id
            WHERE cm.user_id = ?
              AND (c.org_id = ? OR c.org_id IS NULL)
            """
        ]
        params = [user_id, org_id]

    if verdict and verdict != "ALL":
        query_parts.append("AND verdict = ?")
        params.append(verdict)
    if status and status != "ALL":
        query_parts.append("AND case_status = ?")
        params.append(status)
    if search_query and search_query.strip():
        q = f"%{search_query.strip()}%"
        query_parts.append(
            "AND (c.case_id LIKE ? OR c.case_reference LIKE ? OR c.customer_identifier LIKE ? OR c.description LIKE ?)"
            if role != "admin"
            else "AND (case_id LIKE ? OR case_reference LIKE ? OR customer_identifier LIKE ? OR description LIKE ?)"
        )
        params.extend([q, q, q, q])

    query_parts.append("ORDER BY created_at DESC LIMIT ?")
    params.append(limit)

    final_query = " ".join(query_parts)
    with get_connection(db_path) as conn:
        rows = conn.execute(final_query, params).fetchall()
        return [dict(r) for r in rows]


# ===========================================================================
# Reports workflow
# ===========================================================================

def create_report(
    case_id: str,
    org_id: str,
    author_user_id: str,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Creates a new draft report for a case. Returns report_id."""
    init_db(db_path)
    report_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO reports (report_id, case_id, org_id, author_user_id, status,
                                 report_version, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'draft', 1, ?, ?);
            """,
            (report_id, case_id, org_id, author_user_id, now, now),
        )
        conn.commit()
    return report_id


def get_report(report_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieves a report by report_id."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM reports WHERE report_id = ?;", (report_id,)).fetchone()
        return dict(row) if row else None


def get_reports_for_case(case_id: str, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Returns all reports for a case, newest first."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM reports WHERE case_id = ? ORDER BY created_at DESC;",
            (case_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def update_report_status(
    report_id: str,
    new_status: str,
    reviewer_user_id: Optional[str] = None,
    reviewer_comment: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Updates a report's workflow status. Returns True if a row was changed."""
    init_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE reports
            SET status = ?, reviewer_user_id = COALESCE(?, reviewer_user_id),
                reviewer_comment = COALESCE(?, reviewer_comment),
                updated_at = ?
            WHERE report_id = ?;
            """,
            (new_status, reviewer_user_id, reviewer_comment, now, report_id),
        )
        conn.commit()
        return cur.rowcount > 0


def invalidate_report_on_case_update(case_id: str, db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Bumps the report_version and resets status to 'draft' for any approved/submitted
    report on this case (because the underlying case was updated).
    Returns the number of reports invalidated.
    """
    init_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE reports
            SET status = 'draft',
                report_version = report_version + 1,
                reviewer_user_id = NULL,
                reviewer_comment = 'Report invalidated: underlying case record was updated.',
                updated_at = ?
            WHERE case_id = ? AND status IN ('submitted', 'approved');
            """,
            (now, case_id),
        )
        conn.commit()
        return cur.rowcount


# ===========================================================================
# Application-wide audit log
# ===========================================================================

def log_app_audit_event(
    action: str,
    user_id: Optional[str] = None,
    org_id: Optional[str] = None,
    case_id: Optional[str] = None,
    outcome: str = "SUCCESS",
    metadata: Optional[Dict[str, Any]] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """
    Appends an event to the append-only application audit log.
    Sensitive values (passwords, API keys) must NEVER be passed in metadata.
    """
    try:
        init_db(db_path)
        event_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata) if metadata else None
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO app_audit_log
                    (event_id, action, user_id, org_id, case_id, outcome, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (event_id, action, user_id, org_id, case_id, outcome, meta_json, now),
            )
            conn.commit()
    except Exception:
        pass  # Audit log errors must never crash the application


def list_app_audit_events(
    org_id: Optional[str] = None,
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = 200,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """Lists application audit events with optional filters."""
    init_db(db_path)
    parts = ["SELECT * FROM app_audit_log WHERE 1=1"]
    params: List[Any] = []
    if org_id:
        parts.append("AND org_id = ?")
        params.append(org_id)
    if user_id:
        parts.append("AND user_id = ?")
        params.append(user_id)
    if action:
        parts.append("AND action = ?")
        params.append(action)
    parts.append("ORDER BY created_at DESC LIMIT ?")
    params.append(limit)
    with get_connection(db_path) as conn:
        rows = conn.execute(" ".join(parts), params).fetchall()
        return [dict(r) for r in rows]
