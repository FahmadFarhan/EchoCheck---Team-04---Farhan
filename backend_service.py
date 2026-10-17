"""
backend_service.py - Backend Architecture & Service Layer for EchoCheck AI.

Provides a clean, modular backend service layer designed for:
1. The Streamlit workstation application
2. External / separate frontend integration (Dhvani's team)

Defines explicit request/response dataclasses, schema contracts, and pipeline
execution methods for:
- Evidence intake & SHA-256 integrity hashing
- Aletheia synthetic voice detection
- Eidos reference-speaker verification
- Real audio waveform and spectrogram extraction
- Case persistence & audit event logging
- Report generation (PDF & Text)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import audio_integrity
import audio_visualizer
import database as db
import detectifai as api
import pdf_report
import risk_engine as engine


@dataclass
class AudioDiagnosticsSchema:
    """Technical audio signal and container metadata (not proof of tampering)."""

    container_format: str
    filesize_bytes: int
    duration_seconds: float
    sample_rate_hz: Optional[int] = None
    channels: Optional[int] = None
    channel_layout: Optional[str] = None
    bit_depth: Optional[int] = None
    peak_amplitude_dbfs: Optional[float] = None
    rms_level_dbfs: Optional[float] = None
    clipping_ratio_percent: Optional[float] = None
    silence_ratio_percent: Optional[float] = None
    telephony_standard_rate: bool = False
    notes: str = ""
    limitations: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AletheiaResultSchema:
    """DetectifAI Aletheia synthetic speech detection output (No reference required)."""

    label: str = "Aletheia (Synthetic Speech Detection — No Reference Required)"
    success: bool = False
    api_status: str = "UNKNOWN"
    verdict: str = "unavailable"  # "real", "fake", "not_sure", "unavailable"
    score: Optional[float] = None  # 0.0 to 1.0 (higher = more likely synthetic)
    threshold: Optional[float] = None
    duration_seconds: Optional[float] = None
    request_id: Optional[str] = None
    operation_id: Optional[str] = None
    error_message: Optional[str] = None
    recommendation_code: str = ""
    recommendation_text: str = ""
    finding_level: str = "UNKNOWN"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EidosResultSchema:
    """DetectifAI Eidos speaker verification output (Enrolled voice required)."""

    label: str = "Eidos (Speaker Verification — Enrolled Voice Required)"
    performed: bool = False
    success: bool = False
    api_status: str = "NOT_PERFORMED"
    reference_speaker_id: Optional[str] = None
    reference_display_name: Optional[str] = None
    decision: Optional[str] = None  # "match", "no_match", "inconclusive", "unavailable"
    similarity: Optional[float] = None  # -1.0 to +1.0 acoustic similarity index (NOT a percentage)
    threshold: Optional[float] = None
    request_id: Optional[str] = None
    operation_id: Optional[str] = None
    error_message: Optional[str] = None
    disclaimer: str = (
        "SPEAKER VERIFICATION LIMITATION: A biometric speaker match does not prove "
        "recording authenticity. An AI synthetic clone can match an enrolled voiceprint."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceAnalysisResponse:
    """Unified evidence assessment response returned by the EchoCheck AI backend."""

    case_id: str
    case_reference: str
    evidence_category: str
    customer_identifier: str
    sha256_hash: str
    audio_filename: str
    audio_filesize_bytes: int
    audio_duration_seconds: float
    created_at: str
    case_status: str
    aletheia: AletheiaResultSchema
    eidos: EidosResultSchema
    diagnostics: AudioDiagnosticsSchema
    waveform_data: Dict[str, Any]
    audit_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def safe_log_audit_event(case_id: str, event_type: str, event_description: str) -> None:
    """Safely logs an audit event; an audit log failure must NEVER block or fail audio analysis."""
    try:
        if hasattr(db, "log_case_audit_event"):
            db.log_case_audit_event(
                case_id=case_id,
                event_type=event_type,
                event_description=event_description,
            )
    except Exception:
        # Non-blocking: audit log errors must never halt analysis
        pass


def safe_get_audit_events(case_id: str) -> List[Dict[str, Any]]:
    """Safely retrieves audit events; returns empty list if unavailable."""
    try:
        if hasattr(db, "get_case_audit_events"):
            return db.get_case_audit_events(case_id)
    except Exception:
        pass
    return []


def execute_evidence_analysis(
    audio_bytes: bytes,
    filename: str,
    case_reference: str,
    evidence_category: str,
    customer_identifier: Optional[str] = None,
    description: Optional[str] = None,
    speaker_id: Optional[str] = None,
) -> EvidenceAnalysisResponse:
    """
    Executes the complete unified evidence intake and analysis pipeline:
    1. Computes SHA-256 evidence integrity hash
    2. Performs acoustic container and signal inspection
    3. Extracts waveform data points and plots
    4. Executes Aletheia synthetic voice detection
    5. Executes Eidos speaker verification if a reference speaker is selected
    6. Logs granular audit events to SQLite (non-blocking)
    7. Persists the complete case record
    """
    if not audio_bytes or len(audio_bytes) == 0:
        raise ValueError("Audio recording payload is empty.")

    case_id = db.generate_case_id()
    now_iso = datetime.now(timezone.utc).isoformat()
    sha256_hash = hashlib.sha256(audio_bytes).hexdigest()

    # Log initial intake audit event (safely wrapped)
    safe_log_audit_event(
        case_id=case_id,
        event_type="EVIDENCE_REGISTERED",
        event_description=(
            f"Evidence file '{filename}' ({len(audio_bytes):,} bytes) registered for case '{case_reference}'. "
            f"SHA-256: {sha256_hash}"
        ),
    )

    # 1. Technical signal diagnostics
    raw_report = audio_integrity.inspect_audio_file(audio_bytes, filename)
    diagnostics_schema = AudioDiagnosticsSchema(
        container_format=raw_report.container_format,
        filesize_bytes=raw_report.filesize_bytes,
        duration_seconds=raw_report.duration_seconds,
        sample_rate_hz=raw_report.sample_rate_hz,
        channels=raw_report.channels,
        channel_layout=raw_report.channel_layout,
        bit_depth=raw_report.bit_depth,
        peak_amplitude_dbfs=raw_report.peak_amplitude_dbfs,
        rms_level_dbfs=raw_report.rms_level_dbfs,
        clipping_ratio_percent=raw_report.clipping_ratio_percent,
        silence_ratio_percent=raw_report.silence_ratio_percent,
        telephony_standard_rate=raw_report.telephony_standard_rate,
        notes=raw_report.notes,
        limitations=raw_report.limitations,
    )

    safe_log_audit_event(
        case_id=case_id,
        event_type="INTEGRITY_CHECKED",
        event_description=(
            f"Container: {raw_report.container_format}, Sample Rate: {raw_report.sample_rate_hz} Hz, "
            f"Duration: {raw_report.duration_seconds:.2f}s, Clipping: {raw_report.clipping_ratio_percent}%"
        ),
    )

    # 2. Extract structured waveform data for frontend charts
    waveform_data = audio_visualizer.extract_waveform_data_series(audio_bytes, filename, target_points=800)

    # 3. Service A: Aletheia Deepfake Detection
    safe_log_audit_event(
        case_id=case_id,
        event_type="ALETHEIA_STARTED",
        event_description=f"Initiating DetectifAI Aletheia synthetic voice detection on '{filename}'.",
    )

    detection = api.detect_voice_deepfake(audio_bytes=audio_bytes, filename=filename)

    risk_eval = engine.evaluate_case_risk(
        detection=detection,
        account_action=evidence_category,
        customer_identifier=customer_identifier,
    )

    if detection.success:
        safe_log_audit_event(
            case_id=case_id,
            event_type="ALETHEIA_COMPLETED",
            event_description=(
                f"Aletheia analysis succeeded. Verdict: {detection.verdict.upper()}, "
                f"Score: {detection.score}, Threshold: {detection.threshold}, ReqID: {detection.request_id}"
            ),
        )
    else:
        safe_log_audit_event(
            case_id=case_id,
            event_type="ALETHEIA_FAILED",
            event_description=f"Aletheia analysis failed ({detection.api_status}): {detection.error_message}",
        )

    aletheia_schema = AletheiaResultSchema(
        label="Aletheia (Synthetic Speech Detection — No Reference Required)",
        success=detection.success,
        api_status=detection.api_status,
        verdict=risk_eval.verdict,
        score=detection.score,
        threshold=detection.threshold,
        duration_seconds=detection.duration_seconds or raw_report.duration_seconds,
        request_id=detection.request_id,
        operation_id=detection.operation_id,
        error_message=detection.error_message,
        recommendation_code=risk_eval.recommendation_code,
        recommendation_text=risk_eval.recommendation_text,
        finding_level=risk_eval.risk_level,
    )

    # 4. Service B: Eidos Speaker Verification (if speaker selected)
    eidos_schema = EidosResultSchema(
        label="Eidos (Speaker Verification — Enrolled Voice Required)",
        performed=False,
        api_status="NOT_PERFORMED",
    )
    chosen_speaker_display = None

    if speaker_id and speaker_id.strip():
        clean_spk = speaker_id.strip()
        speaker_obj = db.get_local_speaker(clean_spk)
        if speaker_obj:
            chosen_speaker_display = speaker_obj.get("display_name")

        safe_log_audit_event(
            case_id=case_id,
            event_type="EIDOS_STARTED",
            event_description=f"Initiating DetectifAI Eidos verification against reference speaker '{clean_spk}'.",
        )

        verify_res = api.verify_speaker(audio_bytes=audio_bytes, filename=filename, speaker_id=clean_spk)

        decision_str = verify_res.decision if verify_res.success else "unavailable"
        if verify_res.success:
            safe_log_audit_event(
                case_id=case_id,
                event_type="EIDOS_COMPLETED",
                event_description=(
                    f"Eidos verification succeeded. Decision: {decision_str.upper()}, "
                    f"Similarity: {verify_res.similarity}, Threshold: {verify_res.threshold}, ReqID: {verify_res.request_id}"
                ),
            )
        else:
            safe_log_audit_event(
                case_id=case_id,
                event_type="EIDOS_FAILED",
                event_description=f"Eidos verification failed ({verify_res.api_status}): {verify_res.error_message}",
            )

        eidos_schema = EidosResultSchema(
            label="Eidos (Speaker Verification — Enrolled Voice Required)",
            performed=True,
            success=verify_res.success,
            api_status=verify_res.api_status,
            reference_speaker_id=clean_spk,
            reference_display_name=chosen_speaker_display,
            decision=decision_str,
            similarity=verify_res.similarity,
            threshold=verify_res.threshold,
            request_id=verify_res.request_id,
            operation_id=verify_res.operation_id,
            error_message=verify_res.error_message,
        )

    # Determine initial case status
    if risk_eval.verdict == "fake":
        initial_status = "EXAMINATION_RECOMMENDED"
    elif risk_eval.verdict == "real":
        initial_status = "PENDING_EXAMINATION"
    else:
        initial_status = "NEW"

    # Persist case to SQLite
    db.create_case(
        {
            "case_id": case_id,
            "case_reference": case_reference.strip(),
            "account_action": evidence_category,
            "customer_identifier": (customer_identifier or "").strip(),
            "description": (description or "").strip(),
            "audio_filename": filename,
            "audio_filesize_bytes": len(audio_bytes),
            "audio_duration_seconds": detection.duration_seconds or raw_report.duration_seconds,
            "sha256_hash": sha256_hash,
            "speaker_id": eidos_schema.reference_speaker_id,
            "speaker_display_name": chosen_speaker_display,
            "speaker_verify_decision": eidos_schema.decision,
            "speaker_similarity": eidos_schema.similarity,
            "speaker_threshold": eidos_schema.threshold,
            "speaker_verify_request_id": eidos_schema.request_id,
            "audio_integrity_json": raw_report.to_dict(),
            "api_status": detection.api_status,
            "api_error_message": detection.error_message,
            "request_id": detection.request_id,
            "operation_id": detection.operation_id,
            "verdict": risk_eval.verdict,
            "label": detection.label,
            "is_synthetic": detection.is_synthetic,
            "score": detection.score,
            "threshold": detection.threshold,
            "risk_level": risk_eval.risk_level,
            "recommendation_code": risk_eval.recommendation_code,
            "recommendation_text": risk_eval.recommendation_text,
            "case_status": initial_status,
            "raw_response_json": detection.raw_response,
            "record_type": "forensic",
            "created_at": now_iso,
        }
    )

    audit_events = safe_get_audit_events(case_id)

    return EvidenceAnalysisResponse(
        case_id=case_id,
        case_reference=case_reference.strip(),
        evidence_category=evidence_category,
        customer_identifier=(customer_identifier or "").strip(),
        sha256_hash=sha256_hash,
        audio_filename=filename,
        audio_filesize_bytes=len(audio_bytes),
        audio_duration_seconds=detection.duration_seconds or raw_report.duration_seconds,
        created_at=now_iso,
        case_status=initial_status,
        aletheia=aletheia_schema,
        eidos=eidos_schema,
        diagnostics=diagnostics_schema,
        waveform_data=waveform_data,
        audit_events=audit_events,
    )
