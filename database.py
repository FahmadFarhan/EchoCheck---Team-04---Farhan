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

    CREATE INDEX IF NOT EXISTS idx_cases_case_id ON cases(case_id);
    CREATE INDEX IF NOT EXISTS idx_cases_reference ON cases(case_reference);
    CREATE INDEX IF NOT EXISTS idx_cases_verdict ON cases(verdict);
    CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(case_status);
    CREATE INDEX IF NOT EXISTS idx_cases_created ON cases(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_speakers_speaker_id ON speakers(speaker_id);
    CREATE INDEX IF NOT EXISTS idx_audit_case_id ON case_audit_events(case_id);
    CREATE INDEX IF NOT EXISTS idx_audit_created ON case_audit_events(created_at ASC);
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

