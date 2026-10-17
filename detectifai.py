"""
detectifai.py - DetectifAI AI-generated Voice Detection API Client.

Implements the official DetectifAI deepfake detection endpoint:
    POST https://platform.detectif.ai/v1/deepfake/detect

Handles:
- Authentication via Authorization: Bearer <DETECTIF_API_KEY>
- Idempotency via Idempotency-Key: <UUID>
- Multipart form audio uploads
- Explicit error handling: timeouts, HTTP status codes, malformed payloads
- No fabricated results; all failures require explicit out-of-band verification.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv

# Load local environment variables if present
load_dotenv()

logger = logging.getLogger(__name__)

DETECTIF_ENDPOINT = "https://platform.detectif.ai/v1/deepfake/detect"
DEFAULT_TIMEOUT_SECONDS = int(os.getenv("DETECTIF_API_TIMEOUT", "30"))


@dataclass
class DetectionResult:
    """Standardized representation of a DetectifAI detection response or error."""

    success: bool
    api_status: str  # "SUCCESS", "AUTH_ERROR", "TIMEOUT", "HTTP_ERROR", "INVALID_AUDIO", "MALFORMED_RESPONSE", "CONFIG_ERROR"
    verdict: str  # "real", "fake", "not_sure", or "unavailable"
    label: Optional[str] = None
    is_synthetic: Optional[bool] = None
    score: Optional[float] = None  # Higher score = more likely synthetic
    threshold: Optional[float] = None
    duration_seconds: Optional[float] = None
    request_id: Optional[str] = None
    operation_id: Optional[str] = None
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_api_key() -> Optional[str]:
    """Retrieves the DetectifAI API key from environment variables."""
    key = os.getenv("DETECTIF_API_KEY")
    if key and key.strip():
        return key.strip()
    return None


def is_api_configured() -> bool:
    """Checks whether a non-empty API key is configured."""
    return get_api_key() is not None


def get_masked_api_key() -> str:
    """Returns a masked version of the API key for safe frontend display."""
    key = get_api_key()
    if not key:
        return "NOT_CONFIGURED"
    if len(key) <= 8:
        return "****" + key[-2:]
    return f"{key[:4]}...{key[-4:]}"


SPEAKERS_ENROLL_ENDPOINT = "https://platform.detectif.ai/v1/speakers/enroll"
SPEAKERS_VERIFY_ENDPOINT = "https://platform.detectif.ai/v1/speakers/verify"
SPEAKERS_LIST_ENDPOINT = "https://platform.detectif.ai/v1/speakers"


@dataclass
class EnrollmentResult:
    """Standardized representation of a DetectifAI Eidos speaker enrollment response."""

    success: bool
    api_status: str  # "SUCCESS", "AUTH_ERROR", "PERMISSION_ERROR", "INVALID_INPUT", "MAX_SAMPLES_REACHED", "TIMEOUT", "HTTP_ERROR"
    speaker_id: Optional[str] = None
    display_name: Optional[str] = None
    samples_count: Optional[int] = None
    speech_seconds: Optional[float] = None
    duration_seconds: Optional[float] = None
    request_id: Optional[str] = None
    operation_id: Optional[str] = None
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpeakerVerifyResult:
    """Standardized representation of a DetectifAI Eidos speaker verification response."""

    success: bool
    api_status: str  # "SUCCESS", "NOT_FOUND", "AUTH_ERROR", "PERMISSION_ERROR", "INVALID_INPUT", "TIMEOUT", "HTTP_ERROR"
    speaker_id: Optional[str] = None
    decision: Optional[str] = None  # "match", "no_match"
    similarity: Optional[float] = None  # -1.0 to 1.0 (higher = more alike)
    threshold: Optional[float] = None
    duration_seconds: Optional[float] = None
    request_id: Optional[str] = None
    operation_id: Optional[str] = None
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def is_speaker_verification_supported() -> bool:
    """
    Indicates whether biometric speaker verification/comparison is available.
    Eidos Speaker Verification is active and supported on the DetectifAI platform.
    """
    return is_api_configured()


def detect_voice_deepfake(
    audio_bytes: bytes,
    filename: str,
    api_key: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> DetectionResult:
    """
    Submits an audio recording to DetectifAI for synthetic voice / deepfake detection.

    Parameters:
        audio_bytes: Raw bytes of the audio file to analyze.
        filename: Original filename (used for MIME type inference and multipart metadata).
        api_key: Optional explicit API key; falls back to DETECTIF_API_KEY env var.
        timeout: HTTP request timeout in seconds.

    Returns:
        DetectionResult dataclass with API signals or failure diagnostics.
    """
    key = api_key or get_api_key()

    # Rule: If credentials are missing, do not simulate success; clearly report unconfigured status.
    if not key or key == "your_detectif_api_key_here":
        return DetectionResult(
            success=False,
            api_status="CONFIG_ERROR",
            verdict="unavailable",
            error_message=(
                "DetectifAI API key is missing or not configured. "
                "Set the DETECTIF_API_KEY environment variable or specify it in your .env file. "
                "Live analysis has NOT been performed."
            ),
        )

    # Validate audio payload
    if not audio_bytes or len(audio_bytes) == 0:
        return DetectionResult(
            success=False,
            api_status="INVALID_AUDIO",
            verdict="unavailable",
            error_message="Uploaded audio recording is empty or corrupt.",
        )

    # Determine MIME type
    content_type, _ = mimetypes.guess_type(filename)
    if not content_type:
        content_type = "audio/wav" if filename.lower().endswith(".wav") else "application/octet-stream"

    # Prepare idempotent request
    idempotency_key = str(uuid.uuid4())
    headers = {
        "Authorization": f"Bearer {key}",
        "Idempotency-Key": idempotency_key,
        "User-Agent": "SentinelVoice-FintechGuard/1.0",
    }

    files = {
        "file": (filename, audio_bytes, content_type),
    }

    # Idempotent bounded retry for transient connection/DNS glitches (max 1 retry)
    max_attempts = 2
    response = None
    last_conn_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            # Note: files tuple must be fresh for retry
            request_files = {
                "file": (filename, audio_bytes, content_type),
            }
            response = requests.post(
                DETECTIF_ENDPOINT,
                headers=headers,
                files=request_files,
                timeout=timeout,
            )
            break
        except requests.exceptions.Timeout as err:
            if attempt < max_attempts:
                import time
                time.sleep(1.0)
                continue
            return DetectionResult(
                success=False,
                api_status="TIMEOUT",
                verdict="unavailable",
                error_message=(
                    f"DetectifAI API request timed out after {timeout} seconds. "
                    "The voice analysis is unavailable; treat this case as requiring manual verification."
                ),
            )
        except requests.exceptions.ConnectionError as err:
            last_conn_error = err
            if attempt < max_attempts:
                import time
                time.sleep(1.0)
                continue
            err_str = str(err)
            diag = f"Network connection to DetectifAI platform failed: {err_str}."
            if "nodename nor servname provided" in err_str or "NameResolutionError" in err_str:
                diag += " (DNS resolution for 'platform.detectif.ai' failed. Ensure the application process has outbound internet access)."
            return DetectionResult(
                success=False,
                api_status="CONNECTION_ERROR",
                verdict="unavailable",
                error_message=f"{diag} Analysis unavailable.",
            )
        except requests.exceptions.RequestException as err:
            return DetectionResult(
                success=False,
                api_status="HTTP_ERROR",
                verdict="unavailable",
                error_message=f"Request error communicating with DetectifAI: {str(err)}",
            )

    if response is None:
        return DetectionResult(
            success=False,
            api_status="CONNECTION_ERROR",
            verdict="unavailable",
            error_message=f"Network connection failed: {last_conn_error}. Analysis unavailable.",
        )

    # Inspect HTTP status
    status_code = response.status_code

    # Attempt to parse response body as JSON
    try:
        data = response.json()
    except Exception:
        data = None

    # Extract error metadata if present in response
    error_obj = data.get("error") if isinstance(data, dict) else {}
    error_code = error_obj.get("code") if isinstance(error_obj, dict) else None
    error_detail = error_obj.get("message") if isinstance(error_obj, dict) else (data.get("detail") if isinstance(data, dict) else None)
    err_request_id = data.get("request_id") if isinstance(data, dict) else None

    if status_code == 401:
        return DetectionResult(
            success=False,
            api_status="AUTH_ERROR",
            verdict="unavailable",
            status_code=401,
            request_id=err_request_id,
            error_message=f"DetectifAI Authentication failed (HTTP 401): {error_detail or 'Invalid or expired API key.'}",
            raw_response={"status_code": 401, "body": response.text[:500]},
        )
    elif status_code == 403:
        return DetectionResult(
            success=False,
            api_status="AUTH_ERROR",
            verdict="unavailable",
            status_code=403,
            request_id=err_request_id,
            error_message=f"DetectifAI Access Forbidden (HTTP 403): {error_detail or 'Account lacks permission or quota exhausted.'}",
            raw_response={"status_code": 403, "body": response.text[:500]},
        )
    elif status_code == 413:
        return DetectionResult(
            success=False,
            api_status="INVALID_AUDIO",
            verdict="unavailable",
            status_code=413,
            request_id=err_request_id,
            error_message="Audio payload too large for DetectifAI platform (HTTP 413).",
            raw_response={"status_code": 413, "body": response.text[:500]},
        )
    elif status_code == 422:
        if error_code == "audio_too_short":
            msg = f"Audio recording too short: {error_detail or 'Minimum > 4.0 seconds required'}."
        else:
            msg = f"DetectifAI rejected unprocessable audio format/sample (HTTP 422): {error_detail or response.text[:200]}"
        return DetectionResult(
            success=False,
            api_status="INVALID_AUDIO",
            verdict="unavailable",
            status_code=422,
            request_id=err_request_id,
            error_message=msg,
            raw_response=data if isinstance(data, dict) else {"raw": response.text[:500]},
        )
    elif status_code == 429:
        return DetectionResult(
            success=False,
            api_status="RATE_LIMITED",
            verdict="unavailable",
            status_code=429,
            request_id=err_request_id,
            error_message="DetectifAI API rate limit exceeded (HTTP 429). Please retry shortly.",
            raw_response=data if isinstance(data, dict) else {"raw": response.text[:500]},
        )
    elif status_code >= 500:
        return DetectionResult(
            success=False,
            api_status="SERVER_ERROR",
            verdict="unavailable",
            status_code=status_code,
            request_id=err_request_id,
            error_message=f"DetectifAI upstream server error (HTTP {status_code}). Analysis unavailable.",
            raw_response=data if isinstance(data, dict) else {"raw": response.text[:500]},
        )
    elif not (200 <= status_code < 300):
        return DetectionResult(
            success=False,
            api_status="HTTP_ERROR",
            verdict="unavailable",
            status_code=status_code,
            request_id=err_request_id,
            error_message=f"Unexpected DetectifAI response (HTTP {status_code}): {response.text[:300]}",
            raw_response=data if isinstance(data, dict) else {"raw": response.text[:500]},
        )

    # 2xx Success: Parse documented fields
    if not isinstance(data, dict):
        return DetectionResult(
            success=False,
            api_status="MALFORMED_RESPONSE",
            verdict="unavailable",
            status_code=status_code,
            error_message="DetectifAI returned a 200 OK response but response was not valid JSON.",
            raw_response={"raw": response.text[:500]},
        )

    # The documented response contains request_id, operation_id, and result fields:
    # verdict, label, is_synthetic, score, threshold, and duration_seconds.
    request_id = data.get("request_id")
    operation_id = data.get("operation_id")

    # Result fields may be nested under 'result' or at top-level
    result_block = data.get("result")
    if not isinstance(result_block, dict):
        result_block = data

    raw_verdict = result_block.get("verdict") or data.get("verdict")
    verdict_str = str(raw_verdict).strip().lower() if raw_verdict is not None else "not_sure"

    # Normalize verdict to known values: 'real', 'fake', 'not_sure'
    if verdict_str not in ("real", "fake", "not_sure"):
        # If the API returned a variation like 'synthetic' or 'authentic'
        if "fake" in verdict_str or "synth" in verdict_str:
            verdict_str = "fake"
        elif "real" in verdict_str or "auth" in verdict_str:
            verdict_str = "real"
        else:
            verdict_str = "not_sure"

    label = result_block.get("label") or data.get("label")
    is_synthetic = result_block.get("is_synthetic")
    if is_synthetic is None:
        is_synthetic = data.get("is_synthetic")
    if is_synthetic is not None:
        is_synthetic = bool(is_synthetic)

    # Score, threshold, duration
    def _parse_float(val: Any) -> Optional[float]:
        try:
            return float(val) if val is not None else None
        except (ValueError, TypeError):
            return None

    score = _parse_float(result_block.get("score") if result_block.get("score") is not None else data.get("score"))
    threshold = _parse_float(result_block.get("threshold") if result_block.get("threshold") is not None else data.get("threshold"))
    duration_seconds = _parse_float(
        result_block.get("duration_seconds")
        if result_block.get("duration_seconds") is not None
        else data.get("duration_seconds")
    )

    return DetectionResult(
        success=True,
        api_status="SUCCESS",
        verdict=verdict_str,
        label=label,
        is_synthetic=is_synthetic,
        score=score,
        threshold=threshold,
        duration_seconds=duration_seconds,
        request_id=request_id,
        operation_id=operation_id,
        status_code=status_code,
        raw_response=data,
    )


def enroll_speaker(
    audio_bytes: bytes,
    filename: str,
    speaker_id: str,
    display_name: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> EnrollmentResult:
    """
    Enrolls a reference voice sample for a speaker with DetectifAI Eidos.
    POST /v1/speakers/enroll (Requires scope: speakers:enroll).
    Documented requirement: 10-60 seconds recording with at least 10s of clear speech.
    """
    key = api_key or get_api_key()
    if not key or key == "your_detectif_api_key_here":
        return EnrollmentResult(
            success=False,
            api_status="CONFIG_ERROR",
            error_message="DetectifAI API key not configured. Set DETECTIF_API_KEY in .env.",
        )

    clean_spk_id = speaker_id.strip() if speaker_id else ""
    if not clean_spk_id or len(clean_spk_id) > 64:
        return EnrollmentResult(
            success=False,
            api_status="INVALID_INPUT",
            error_message="speaker_id must be 1-64 characters matching [A-Za-z0-9_.:-].",
        )

    if not audio_bytes or len(audio_bytes) == 0:
        return EnrollmentResult(
            success=False,
            api_status="INVALID_INPUT",
            error_message="Uploaded enrollment audio recording is empty.",
        )

    content_type, _ = mimetypes.guess_type(filename)
    if not content_type:
        content_type = "audio/wav" if filename.lower().endswith(".wav") else "application/octet-stream"

    idempotency_key = str(uuid.uuid4())
    headers = {
        "Authorization": f"Bearer {key}",
        "Idempotency-Key": idempotency_key,
        "User-Agent": "SentinelVoice-Forensics/2.0",
    }

    form_data = {
        "speaker_id": clean_spk_id,
    }
    if display_name and display_name.strip():
        form_data["display_name"] = display_name.strip()[:128]

    max_attempts = 2
    response = None
    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            files = {
                "file": (filename, audio_bytes, content_type),
            }
            response = requests.post(
                SPEAKERS_ENROLL_ENDPOINT,
                headers=headers,
                data=form_data,
                files=files,
                timeout=timeout,
            )
            break
        except requests.exceptions.Timeout as err:
            if attempt < max_attempts:
                import time
                time.sleep(1.0)
                continue
            return EnrollmentResult(
                success=False,
                api_status="TIMEOUT",
                error_message=f"Enrollment request timed out after {timeout}s.",
            )
        except requests.exceptions.ConnectionError as err:
            last_err = err
            if attempt < max_attempts:
                import time
                time.sleep(1.0)
                continue
            err_str = str(err)
            diag = f"Network connection to DetectifAI platform failed: {err_str}."
            if "nodename nor servname provided" in err_str or "NameResolutionError" in err_str:
                diag += " (DNS resolution for 'platform.detectif.ai' failed. Ensure the application process has outbound internet access)."
            return EnrollmentResult(
                success=False,
                api_status="CONNECTION_ERROR",
                error_message=diag,
            )
        except requests.exceptions.RequestException as err:
            return EnrollmentResult(
                success=False,
                api_status="HTTP_ERROR",
                error_message=f"Request exception: {err}",
            )

    if response is None:
        return EnrollmentResult(
            success=False,
            api_status="CONNECTION_ERROR",
            error_message=f"Network connection failed: {last_err}",
        )

    status_code = response.status_code
    try:
        data = response.json()
    except Exception:
        data = None

    error_obj = data.get("error", {}) if isinstance(data, dict) else {}
    error_code = error_obj.get("code") if isinstance(error_obj, dict) else None
    error_msg = error_obj.get("message") if isinstance(error_obj, dict) else (data.get("detail") if isinstance(data, dict) else response.text[:200])
    req_id = data.get("request_id") if isinstance(data, dict) else None

    if status_code == 403:
        return EnrollmentResult(
            success=False,
            api_status="PERMISSION_ERROR",
            status_code=403,
            request_id=req_id,
            error_message=f"Insufficient Scope (HTTP 403): Your DetectifAI API key lacks the 'speakers:enroll' scope. Error: {error_msg}",
            raw_response=data,
        )
    elif status_code == 409:
        return EnrollmentResult(
            success=False,
            api_status="MAX_SAMPLES_REACHED",
            status_code=409,
            request_id=req_id,
            error_message=f"Speaker already has the maximum of 5 voice samples ({error_msg}).",
            raw_response=data,
        )
    elif status_code == 422:
        return EnrollmentResult(
            success=False,
            api_status="INVALID_INPUT",
            status_code=422,
            request_id=req_id,
            error_message=f"Enrollment audio rejected (HTTP 422): {error_msg}. (Requirement: 10-60s recording with >= 10s speech).",
            raw_response=data,
        )
    elif status_code == 400:
        return EnrollmentResult(
            success=False,
            api_status="INVALID_INPUT",
            status_code=400,
            request_id=req_id,
            error_message=f"Invalid request parameters (HTTP 400): {error_msg}",
            raw_response=data,
        )
    elif not (200 <= status_code < 300):
        return EnrollmentResult(
            success=False,
            api_status="HTTP_ERROR",
            status_code=status_code,
            request_id=req_id,
            error_message=f"DetectifAI enrollment error (HTTP {status_code}): {error_msg}",
            raw_response=data,
        )

    # 200 Success
    speaker_info = data.get("speaker", {}) if isinstance(data, dict) else {}
    quality_info = data.get("quality", {}) if isinstance(data, dict) else {}

    return EnrollmentResult(
        success=True,
        api_status="SUCCESS",
        speaker_id=speaker_info.get("speaker_id") or clean_spk_id,
        display_name=speaker_info.get("display_name"),
        samples_count=speaker_info.get("samples", 1),
        speech_seconds=quality_info.get("speech_seconds"),
        duration_seconds=quality_info.get("duration_seconds"),
        request_id=data.get("request_id"),
        operation_id=data.get("operation_id"),
        status_code=status_code,
        raw_response=data,
    )


def verify_speaker(
    audio_bytes: bytes,
    filename: str,
    speaker_id: str,
    detect_deepfake: bool = False,
    api_key: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> SpeakerVerifyResult:
    """
    Verifies whether a disputed voice recording matches an enrolled reference speaker.
    POST /v1/speakers/verify (Requires scope: speakers:verify).
    Documented requirement: 3-60 seconds recording with at least 2s of speech.
    """
    key = api_key or get_api_key()
    if not key or key == "your_detectif_api_key_here":
        return SpeakerVerifyResult(
            success=False,
            api_status="CONFIG_ERROR",
            error_message="DetectifAI API key not configured.",
        )

    clean_spk_id = speaker_id.strip() if speaker_id else ""
    if not clean_spk_id:
        return SpeakerVerifyResult(
            success=False,
            api_status="INVALID_INPUT",
            error_message="A valid enrolled speaker_id must be provided for verification.",
        )

    if not audio_bytes or len(audio_bytes) == 0:
        return SpeakerVerifyResult(
            success=False,
            api_status="INVALID_INPUT",
            error_message="Uploaded audio recording is empty.",
        )

    content_type, _ = mimetypes.guess_type(filename)
    if not content_type:
        content_type = "audio/wav" if filename.lower().endswith(".wav") else "application/octet-stream"

    idempotency_key = str(uuid.uuid4())
    headers = {
        "Authorization": f"Bearer {key}",
        "Idempotency-Key": idempotency_key,
        "User-Agent": "SentinelVoice-Forensics/2.0",
    }

    form_data = {
        "speaker_id": clean_spk_id,
        "detect_deepfake": "true" if detect_deepfake else "false",
    }

    max_attempts = 2
    response = None
    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            files = {
                "file": (filename, audio_bytes, content_type),
            }
            response = requests.post(
                SPEAKERS_VERIFY_ENDPOINT,
                headers=headers,
                data=form_data,
                files=files,
                timeout=timeout,
            )
            break
        except requests.exceptions.Timeout:
            if attempt < max_attempts:
                import time
                time.sleep(1.0)
                continue
            return SpeakerVerifyResult(
                success=False,
                api_status="TIMEOUT",
                error_message=f"Speaker verification request timed out after {timeout}s.",
            )
        except requests.exceptions.ConnectionError as err:
            last_err = err
            if attempt < max_attempts:
                import time
                time.sleep(1.0)
                continue
            err_str = str(err)
            diag = f"Network connection to DetectifAI platform failed: {err_str}."
            if "nodename nor servname provided" in err_str or "NameResolutionError" in err_str:
                diag += " (DNS resolution for 'platform.detectif.ai' failed. Ensure the application process has outbound internet access)."
            return SpeakerVerifyResult(
                success=False,
                api_status="CONNECTION_ERROR",
                error_message=diag,
            )
        except requests.exceptions.RequestException as err:
            return SpeakerVerifyResult(
                success=False,
                api_status="HTTP_ERROR",
                error_message=f"Request exception: {err}",
            )

    if response is None:
        return SpeakerVerifyResult(
            success=False,
            api_status="CONNECTION_ERROR",
            error_message=f"Connection failed: {last_err}",
        )

    status_code = response.status_code
    try:
        data = response.json()
    except Exception:
        data = None

    error_obj = data.get("error", {}) if isinstance(data, dict) else {}
    error_code = error_obj.get("code") if isinstance(error_obj, dict) else None
    error_msg = error_obj.get("message") if isinstance(error_obj, dict) else (data.get("detail") if isinstance(data, dict) else response.text[:200])
    req_id = data.get("request_id") if isinstance(data, dict) else None

    if status_code == 404:
        return SpeakerVerifyResult(
            success=False,
            api_status="NOT_FOUND",
            status_code=404,
            request_id=req_id,
            error_message=f"Enrolled speaker not found (HTTP 404): No speaker with ID '{clean_spk_id}' exists in this account. Please enroll the reference sample first.",
            raw_response=data,
        )
    elif status_code == 403:
        return SpeakerVerifyResult(
            success=False,
            api_status="PERMISSION_ERROR",
            status_code=403,
            request_id=req_id,
            error_message=f"Insufficient Scope (HTTP 403): API key lacks the 'speakers:verify' scope. Error: {error_msg}",
            raw_response=data,
        )
    elif status_code == 422:
        return SpeakerVerifyResult(
            success=False,
            api_status="INVALID_INPUT",
            status_code=422,
            request_id=req_id,
            error_message=f"Verification audio rejected (HTTP 422): {error_msg}. (Requirement: 3-60s recording with >= 2s speech).",
            raw_response=data,
        )
    elif not (200 <= status_code < 300):
        return SpeakerVerifyResult(
            success=False,
            api_status="HTTP_ERROR",
            status_code=status_code,
            request_id=req_id,
            error_message=f"DetectifAI verification error (HTTP {status_code}): {error_msg}",
            raw_response=data,
        )

    # 200 OK
    result_obj = data.get("result", {}) if isinstance(data, dict) else {}
    decision = result_obj.get("decision")  # "match" or "no_match"
    similarity = result_obj.get("similarity")
    threshold = result_obj.get("threshold")
    duration = result_obj.get("duration_seconds")

    return SpeakerVerifyResult(
        success=True,
        api_status="SUCCESS",
        speaker_id=data.get("speaker_id") or clean_spk_id,
        decision=str(decision).lower() if decision else None,
        similarity=float(similarity) if similarity is not None else None,
        threshold=float(threshold) if threshold is not None else None,
        duration_seconds=float(duration) if duration is not None else None,
        request_id=data.get("request_id"),
        operation_id=data.get("operation_id"),
        status_code=status_code,
        raw_response=data,
    )


def list_enrolled_speakers(api_key: Optional[str] = None) -> Dict[str, Any]:
    """Retrieves list of enrolled speakers from DetectifAI Eidos."""
    key = api_key or get_api_key()
    if not key:
        return {"success": False, "data": [], "error": "API key not configured"}

    headers = {"Authorization": f"Bearer {key}"}
    try:
        r = requests.get(SPEAKERS_LIST_ENDPOINT, headers=headers, timeout=10)
        if r.status_code == 200:
            return {"success": True, "data": r.json().get("data", [])}
        else:
            return {"success": False, "data": [], "error": f"HTTP {r.status_code}: {r.text[:200]}"}
    except Exception as e:
        return {"success": False, "data": [], "error": str(e)}


def delete_enrolled_speaker(speaker_id: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Deletes an enrolled speaker and their reference samples from DetectifAI Eidos."""
    key = api_key or get_api_key()
    if not key:
        return {"success": False, "error": "API key not configured"}

    url = f"{SPEAKERS_LIST_ENDPOINT}/{speaker_id}"
    headers = {"Authorization": f"Bearer {key}"}
    try:
        r = requests.delete(url, headers=headers, timeout=10)
        if r.status_code == 200:
            return {"success": True, "data": r.json()}
        else:
            return {"success": False, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

