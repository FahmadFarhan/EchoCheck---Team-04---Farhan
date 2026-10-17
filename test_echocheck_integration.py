"""
test_echocheck_integration.py - Comprehensive Integration & Regression Test Suite for EchoCheck AI.
"""

import hashlib
from pathlib import Path

import audio_visualizer as av
import backend_service as bs
import database as db
import detectifai as api
import pdf_report


def test_branding_and_config():
    print("Testing EchoCheck AI branding...")
    case_id = db.generate_case_id()
    assert case_id.startswith("EC-"), f"Case ID should start with EC-, got {case_id}"

    # Verify report branding
    mock_case = {
        "case_id": case_id,
        "case_reference": "REF-TEST-2026",
        "account_action": "Disputed Telephonic Recording",
        "customer_identifier": "Speaker A",
        "verdict": "real",
        "score": 0.0412,
        "threshold": 0.5000,
        "is_synthetic": False,
        "audio_filename": "test.wav",
        "audio_filesize_bytes": 10000,
        "audio_duration_seconds": 6.5,
        "sha256_hash": "e" * 64,
        "api_status": "SUCCESS",
        "recommendation_text": "Authentic speech markers observed.",
        "case_status": "AUTHENTICATED_PRELIMINARY",
        "created_at": "2026-10-17 10:00:00 UTC",
    }
    pdf_bytes = pdf_report.generate_forensic_pdf(mock_case, [])
    assert pdf_bytes.startswith(b"%PDF"), "Generated PDF must start with %PDF header"
    assert len(pdf_bytes) > 2000, "PDF must contain content"
    print("✓ Branding & PDF generation verified.")


def test_audio_visualizer_multi_sample():
    print("Testing waveform and spectrogram generation across distinct fixtures...")
    samples = [
        ("samples/sample_account_recovery_call.wav", 6.5),
        ("samples/sample_speaker_enrollment_reference.wav", 18.0),
        ("samples/test_silence_fixture.wav", 3.0),
        ("samples/test_chirp_fixture.wav", 4.0),
    ]

    waveforms = []
    spectrograms = []

    for path_str, expected_dur in samples:
        path = Path(path_str)
        assert path.exists(), f"Missing audio fixture: {path}"
        data = path.read_bytes()

        samples_arr, sr, dur = av.decode_audio_samples(data, path.name)
        assert abs(dur - expected_dur) < 0.25, f"{path.name}: expected ~{expected_dur}s, got {dur}s"

        wf_img = av.generate_waveform_plot(data, path.name)
        spec_img = av.generate_spectrogram_plot(data, path.name)

        assert len(wf_img) > 1000, f"{path.name} waveform image too small"
        assert len(spec_img) > 1000, f"{path.name} spectrogram image too small"

        waveforms.append(wf_img)
        spectrograms.append(spec_img)

    # Confirm different recordings produce different waveform and spectrogram outputs
    assert waveforms[0] != waveforms[1], "Waveforms 0 and 1 must differ"
    assert waveforms[1] != waveforms[2], "Waveforms 1 and 2 must differ"
    assert spectrograms[0] != spectrograms[3], "Spectrograms 0 and 3 must differ"
    print("✓ Waveform and Spectrogram generation verified across 4 distinct inputs.")


def test_unified_backend_service():
    print("Testing unified evidence intake pipeline...")
    sample_file = Path("samples/sample_account_recovery_call.wav")
    audio_bytes = sample_file.read_bytes()
    expected_hash = hashlib.sha256(audio_bytes).hexdigest()

    res = bs.execute_evidence_analysis(
        audio_bytes=audio_bytes,
        filename="sample_account_recovery_call.wav",
        case_reference="MATTER-REG-9912",
        evidence_category="Disputed Commercial Negotiation / Oral Contract",
        customer_identifier="Witness X",
        description="Dual analysis test run",
        speaker_id=None,
    )

    assert res.case_id.startswith("EC-")
    assert res.sha256_hash == expected_hash
    assert res.aletheia.label == "Aletheia (Synthetic Speech Detection — No Reference Required)"
    assert res.eidos.label == "Eidos (Speaker Verification — Enrolled Voice Required)"
    assert res.eidos.performed is False, "Eidos should be bypassed when speaker_id is None"
    assert len(res.audit_events) >= 3, "At least 3 audit events should be logged"

    # Verify case record in DB
    stored = db.get_case(res.case_id)
    assert stored is not None
    assert stored["sha256_hash"] == expected_hash
    assert stored["record_type"] == "forensic"
    print("✓ Unified backend pipeline verified.")


def test_database_and_audit_log():
    print("Testing case registry audit trail...")
    cid = db.generate_case_id()
    db.log_case_audit_event(cid, "EVIDENCE_REGISTERED", "Registered master recording")
    db.log_case_audit_event(cid, "ALETHEIA_COMPLETED", "Verdict: REAL")
    db.log_case_audit_event(cid, "REPORT_GENERATED", "Forensic PDF export generated")

    events = db.get_case_audit_events(cid)
    assert len(events) == 3
    assert events[0]["event_type"] == "EVIDENCE_REGISTERED"
    assert events[1]["event_type"] == "ALETHEIA_COMPLETED"
    assert events[2]["event_type"] == "REPORT_GENERATED"
    print("✓ Audit event timeline verified.")


def run_all():
    print("==================================================")
    print("RUNNING ECHOCHECK AI INTEGRATION & REGRESSION TESTS")
    print("==================================================")
    test_branding_and_config()
    test_audio_visualizer_multi_sample()
    test_unified_backend_service()
    test_database_and_audit_log()
    print("==================================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    run_all()
