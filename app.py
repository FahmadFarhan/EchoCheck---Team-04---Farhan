"""
EchoCheck AI - AI-Assisted Voice Evidence Analysis for Legal Cases.

A clean, professional forensic-workstation platform for forensic examiners,
litigation lawyers, investigators, and corporate compliance teams.
Organizes findings from DetectifAI AI-generated voice detection, DetectifAI Eidos
speaker verification, and technical audio integrity inspection into structured,
investigation-ready forensic case records.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import audio_integrity
import audio_visualizer
import backend_service
import database as db
import detectifai as api
import pdf_report
import risk_engine as engine
import auth

# Load environment variables (.env)
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="EchoCheck AI | Audio Evidence Workstation",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Database Schema & Migrations
db.init_db()
# Retroactively tag pre-forensics (financial-era) demo records as 'legacy_demo'
db.mark_legacy_records()

# Professional Forensic Workstation Theme CSS (Light, Clean, Slate-Blue Accents)
CUSTOM_CSS = """
<style>
/* Forensic Light Workstation Palette */
:root {
    --sv-bg: #f8fafc;
    --sv-card: #ffffff;
    --sv-border: #e2e8f0;
    --sv-text: #1e293b;
    --sv-text-muted: #64748b;
    --sv-primary: #2563eb;
    --sv-primary-hover: #1d4ed8;
    --sv-slate: #334155;
    --sv-red: #b91c1c;
    --sv-red-bg: #fef2f2;
    --sv-red-border: #fca5a5;
    --sv-green: #15803d;
    --sv-green-bg: #f0fdf4;
    --sv-green-border: #86efac;
    --sv-amber: #b45309;
    --sv-amber-bg: #fffbeb;
    --sv-amber-border: #fde68a;
}

/* Background overrides */
.stApp {
    background-color: #f8fafc;
    color: #1e293b;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
}

/* Forensic Header Banner */
.sv-header {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #2563eb;
    border-radius: 6px;
    padding: 1.15rem 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.sv-header-title {
    font-size: 1.45rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.sv-header-subtitle {
    color: #475569;
    font-size: 0.88rem;
    margin-top: 0.2rem;
    line-height: 1.4;
}

/* Cards */
.sv-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 1.15rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
}

/* Verdict & Decision Badges */
.badge-fake {
    background-color: #fef2f2;
    border: 1px solid #f87171;
    color: #991b1b;
    padding: 0.35rem 0.8rem;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.95rem;
    display: inline-block;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

.badge-real {
    background-color: #f0fdf4;
    border: 1px solid #4ade80;
    color: #166534;
    padding: 0.35rem 0.8rem;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.95rem;
    display: inline-block;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

.badge-not-sure {
    background-color: #fffbeb;
    border: 1px solid #fbbf24;
    color: #92400e;
    padding: 0.35rem 0.8rem;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.95rem;
    display: inline-block;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

.badge-unavailable {
    background-color: #f1f5f9;
    border: 1px solid #cbd5e1;
    color: #475569;
    padding: 0.35rem 0.8rem;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.95rem;
    display: inline-block;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

.badge-match {
    background-color: #f0fdf4;
    border: 1px solid #4ade80;
    color: #166534;
    padding: 0.35rem 0.8rem;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.95rem;
    display: inline-block;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

.badge-no-match {
    background-color: #fef2f2;
    border: 1px solid #f87171;
    color: #991b1b;
    padding: 0.35rem 0.8rem;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.95rem;
    display: inline-block;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

.badge-inconclusive {
    background-color: #fffbeb;
    border: 1px solid #fbbf24;
    color: #92400e;
    padding: 0.35rem 0.8rem;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.95rem;
    display: inline-block;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

.badge-error {
    background-color: #f1f5f9;
    border: 1px solid #cbd5e1;
    color: #475569;
    padding: 0.35rem 0.8rem;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.95rem;
    display: inline-block;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

/* Metric Pill */
.metric-pill {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.7rem 0.85rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
}

.metric-pill-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    font-weight: 600;
}

.metric-pill-value {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0f172a;
    margin-top: 0.15rem;
}

/* Recommendation Box */
.rec-box {
    background-color: #f8fafc;
    border-radius: 6px;
    padding: 1.1rem;
    margin-top: 0.85rem;
    border: 1px solid #e2e8f0;
}

.rec-box-fake {
    border-left: 4px solid #ef4444;
}

.rec-box-real {
    border-left: 4px solid #22c55e;
}

.rec-box-not-sure {
    border-left: 4px solid #f59e0b;
}

.rec-box-unavailable {
    border-left: 4px solid #94a3b8;
}

/* Forensic Limitation / Notice box */
.sv-disclaimer {
    background: #f8fafc;
    border: 1px dashed #cbd5e1;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 1rem;
    line-height: 1.45;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_sidebar() -> str:
    """Renders the persistent navigation sidebar and forensic evidence metrics."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 1rem;">
                <div style="font-size: 1.25rem; font-weight: 800; color: #0f172a; letter-spacing: -0.01em;">
                    ⚖️ ECHO<span style="color: #2563eb;">CHECK</span> AI
                </div>
                <div style="font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
                    AI-Assisted Audio Evidence Workstation
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "NAVIGATION",
            ["Evidence Assessment", "Eidos Speaker Enrollment", "Forensic Case Registry", "Forensic Systems & API Status"],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # API Configuration Status Indicator
        api_configured = api.is_api_configured()
        if api_configured:
            st.markdown(
                f"""
                <div style="background: #f0fdf4; border: 1px solid #86efac; border-radius: 6px; padding: 0.55rem 0.75rem; margin-bottom: 0.6rem;">
                    <div style="font-size: 0.68rem; color: #166534; font-weight: 700; text-transform: uppercase;">
                        ● DETECTIFAI PLATFORM ACTIVE
                    </div>
                    <div style="font-size: 0.75rem; color: #334155; margin-top: 0.15rem; font-family: monospace;">
                        Key: {api.get_masked_api_key()}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.55rem 0.75rem; margin-bottom: 0.9rem;">
                    <div style="font-size: 0.68rem; color: #2563eb; font-weight: 700; text-transform: uppercase;">
                        ✓ EIDOS SPEAKER VERIFICATION
                    </div>
                    <div style="font-size: 0.73rem; color: #475569; margin-top: 0.15rem;">
                        Enrollment & Matching Enabled
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background: #fef2f2; border: 1px solid #fca5a5; border-radius: 6px; padding: 0.55rem 0.75rem; margin-bottom: 0.9rem;">
                    <div style="font-size: 0.68rem; color: #991b1b; font-weight: 700; text-transform: uppercase;">
                        ⚠ API KEY UNCONFIGURED
                    </div>
                    <div style="font-size: 0.74rem; color: #475569; margin-top: 0.15rem;">
                        Set <code>DETECTIF_API_KEY</code> in <code>.env</code>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Quick Database Snapshot
        stats = db.get_dashboard_stats()
        st.markdown(
            f"""
            <div style="font-size: 0.75rem; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 0.4rem;">
                Evidence Records
            </div>
            <div style="font-size: 0.8rem; color: #334155; line-height: 1.75;">
                • Total Audio Records: <b>{stats['total_cases']}</b><br/>
                • Synthetic Indicators: <b style="color: #b91c1c;">{stats['fake_detected']}</b><br/>
                • Natural Speech Markers: <b style="color: #15803d;">{stats['real_verified']}</b><br/>
                • Inconclusive Samples: <b style="color: #b45309;">{stats['inconclusive']}</b><br/>
                • Under Examination: <b>{stats['pending_cases']}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.7rem; color: #64748b; line-height: 1.45;">
                <b>Forensic Guardrails:</b><br/>
                • Automated decision support only<br/>
                • Not CFSL / NFSU certified<br/>
                • No permanent raw audio storage<br/>
                • Independent examiner confirmation
            </div>
            """,
            unsafe_allow_html=True,
        )

        return page


def page_assessment_station() -> None:
    """Case-entry form, live deepfake detection, Eidos speaker verification, and signal inspection."""
    if not auth.require_permission("create_case"):
        return
    current_user = auth.get_current_user()

    st.markdown(
        """
        <div class="sv-header">
            <div class="sv-header-title">
                <span>⚖️</span> Audio Evidence Assessment & Analysis
            </div>
            <div class="sv-header-subtitle">
                Conduct preliminary acoustic examination of disputed audio recordings for synthetic speech indicators, Eidos reference speaker verification, and technical signal integrity.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not api.is_api_configured():
        st.warning(
            "⚠️ **DetectifAI API Key Not Found**: `DETECTIF_API_KEY` is not set in environment or `.env`. "
            "To enable live detection and speaker verification, provide your key in `.env` or in the Systems & API Status tab.",
            icon="⚠️",
        )

    # Build speaker choices and lookup mapping from local DB
    local_speakers = db.list_local_speakers()
    speaker_choices = ["None (Synthetic Voice Detection Only)"]
    speaker_map = {}
    for s in local_speakers:
        display_label = f"{s['speaker_id']}" + (f" - {s['display_name']}" if s.get('display_name') else "")
        speaker_choices.append(display_label)
        speaker_map[display_label] = s['speaker_id']

    st.markdown("### 📝 Evidence Intake & Case Registration")
    st.markdown(
        "<p style='color: #64748b; font-size: 0.85rem;'>Enter the matter reference, dispute category, optional reference speaker, and upload the disputed audio recording.</p>",
        unsafe_allow_html=True,
    )

    c_in1, c_in2 = st.columns([1, 1], gap="medium")
    with c_in1:
        case_reference = st.text_input(
            "Legal / Matter Case Reference *",
            placeholder="e.g. CRIM-2026-88192, FIR #312, or Matter Ticket #",
            help="Court docket, police FIR, arbitration number, or internal matter reference.",
            key="case_reference_input",
        )
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            account_action = st.selectbox(
                "Evidence / Dispute Category *",
                [
                    "Disputed Telephonic Recording / Extortion Claim",
                    "Disputed Audio Confession / Interrogation",
                    "Disputed Commercial Negotiation / Oral Contract",
                    "Disputed Public Speech / Defamation Claim",
                    "Alleged Voice Clone / Synthetic Impersonation",
                    "Disputed Bribery / Whistleblower Recording",
                    "Disputed Matrimonial / Family Dispute Audio",
                    "Corporate Compliance Investigation",
                    "General Disputed Audio Evidence",
                ],
                index=0,
                key="evidence_category_input",
            )
        with col_a2:
            customer_identifier = st.text_input(
                "Alleged Speaker / Person of Interest",
                placeholder="e.g. Accused Person A, Witness B",
                help="Name, pseudonym, or case identifier of the alleged speaker.",
                key="customer_identifier_input",
            )
        selected_speaker_choice = st.selectbox(
            "Reference Speaker for Eidos Verification (Optional)",
            options=speaker_choices,
            index=0,
            help="If an enrolled reference voice is selected, Eidos will compare the disputed audio against this voiceprint.",
            key="selected_speaker_choice_input",
        )
        description = st.text_area(
            "Case Context & Chain of Custody Notes",
            placeholder="Disputed phone recording submitted by defense counsel. Source device: digital recorder...",
            height=80,
            key="description_input",
        )

    with c_in2:
        uploaded_file = st.file_uploader(
            "Upload Disputed Audio Recording *",
            type=["wav", "mp3", "m4a", "ogg", "flac", "aac"],
            help="Supported audio formats: WAV, MP3, M4A, OGG, FLAC, AAC (minimum > 4.0s for deepfake check; 3-60s for speaker verification).",
            key="uploaded_file_input",
        )
        if uploaded_file is not None:
            st.audio(uploaded_file)
            audio_file_bytes = uploaded_file.getvalue()
            curr_hash = hashlib.sha256(audio_file_bytes).hexdigest()
            st.caption(f"📁 {uploaded_file.name} ({len(audio_file_bytes) / 1024:.1f} KB | SHA-256: {curr_hash[:16]}...)")

            # Generate Real Waveform and Spectrogram directly from decoded PCM samples
            if st.session_state.get("active_audio_hash") != curr_hash:
                try:
                    st.session_state["active_waveform_img"] = audio_visualizer.generate_waveform_plot(
                        audio_file_bytes, uploaded_file.name
                    )
                    st.session_state["active_spectrogram_img"] = audio_visualizer.generate_spectrogram_plot(
                        audio_file_bytes, uploaded_file.name
                    )
                    st.session_state["active_audio_hash"] = curr_hash
                    st.session_state["audio_decode_err"] = None
                except Exception as err:
                    st.session_state["active_waveform_img"] = None
                    st.session_state["active_spectrogram_img"] = None
                    st.session_state["active_audio_hash"] = curr_hash
                    st.session_state["audio_decode_err"] = str(err)

            if st.session_state.get("audio_decode_err"):
                st.error(f"⚠️ Audio decoding error: {st.session_state['audio_decode_err']}")

    submit_btn = st.button(
        "🔬 Run Forensic Audio Analysis",
        use_container_width=True,
        type="primary",
        key="run_forensic_analysis_btn",
    )

    # Handle Analysis Execution
    if submit_btn:
        if not auth.require_permission("upload_audio"):
            return
        if not auth.require_permission("run_deepfake_detection"):
            return

        if not case_reference.strip():
            st.error("Please provide a Legal / Matter Case Reference.", icon="🛑")
            return

        if uploaded_file is None:
            st.error("Please upload an audio recording file to evaluate.", icon="🛑")
            return

        # Extract selected speaker_id if any via robust mapping
        chosen_speaker_id = speaker_map.get(selected_speaker_choice)

        audio_bytes = uploaded_file.getvalue()
        audio_filename = uploaded_file.name

        with st.spinner("Executing acoustic analysis (Aletheia Deepfake Detection, Eidos Verification & Integrity Checks)..."):
            try:
                analysis_res = backend_service.execute_evidence_analysis(
                    audio_bytes=audio_bytes,
                    filename=audio_filename,
                    case_reference=case_reference.strip(),
                    evidence_category=account_action,
                    customer_identifier=customer_identifier.strip() if customer_identifier else None,
                    description=description.strip() if description else None,
                    speaker_id=chosen_speaker_id,
                )
                case_id = analysis_res.case_id
                st.session_state["latest_case_id"] = case_id

                # Tag case with org/creator and auto-assign
                if current_user:
                    try:
                        import sqlite3 as _sqlite3
                        import database as _db
                        _db.get_connection()("UPDATE cases SET org_id = ?, created_by_user_id = ? WHERE case_id = ?")
                    except Exception:
                        pass
                    # Use direct db call for org/creator update
                    try:
                        with db.get_connection() as _conn:
                            _conn.execute(
                                "UPDATE cases SET org_id = ?, created_by_user_id = ? WHERE case_id = ?;",
                                (current_user["org_id"], current_user["user_id"], case_id),
                            )
                            _conn.commit()
                        db.assign_case_member(
                            case_id=case_id,
                            user_id=current_user["user_id"],
                            assigned_by=current_user["user_id"],
                        )
                        db.log_app_audit_event(
                            action="ANALYSIS_RUN",
                            user_id=current_user["user_id"],
                            org_id=current_user["org_id"],
                            case_id=case_id,
                            outcome="SUCCESS",
                            metadata={"filename": audio_filename, "case_reference": case_reference.strip()},
                        )
                    except Exception:
                        pass  # Don't fail analysis if tagging fails

            except Exception as err:
                st.error(f"Acoustic analysis pipeline failed: {err}")

    # Full-width Real Waveform and Spectrogram Section (If active audio available)
    if st.session_state.get("active_waveform_img"):
        st.markdown("---")
        st.markdown("### 📊 Real Acoustic Signal Visualizations")
        st.markdown(
            "<p style='color: #64748b; font-size: 0.85rem;'>High-contrast acoustic signal plots generated directly from authentic decoded PCM samples across full content width.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("##### 📈 Real Acoustic Waveform (Decoded PCM — ~370px Height)")
        st.image(st.session_state["active_waveform_img"], use_container_width=True)

        st.markdown("##### 🌈 Real Acoustic Spectrogram (STFT — ~320px Height)")
        st.image(st.session_state["active_spectrogram_img"], use_container_width=True)
        st.caption("ℹ️ Plots are generated directly from the authentic PCM samples of the uploaded recording.")

    # Unified Results Dashboard or Awaiting Placeholder
    st.markdown("---")
    latest_case_id = st.session_state.get("latest_case_id")
    if latest_case_id:
        active_case = db.get_case(latest_case_id)
        if active_case and auth.assert_case_access(active_case):
            render_result_panel(active_case)
    else:
        st.markdown(
            """
            <div style="background: #ffffff; border: 1.5px dashed #cbd5e1; border-radius: 8px; padding: 3rem 1.5rem; text-align: center; color: #64748b; margin-top: 1rem; margin-bottom: 2rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚖️</div>
                <div style="font-weight: 700; font-size: 1.15rem; color: #1e293b;">Awaiting Disputed Audio Evidence</div>
                <div style="font-size: 0.88rem; margin-top: 0.4rem; max-width: 480px; margin-left: auto; margin-right: auto; line-height: 1.5; color: #64748b;">
                    Upload an audio recording and click <b>Run Forensic Audio Analysis</b> to execute live AI synthetic-voice detection, optional Eidos speaker verification, and technical signal integrity analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def build_text_report(case: dict) -> str:
    """Generates a structured text export of the preliminary forensic report."""
    verdict = (case.get("verdict") or "unavailable").upper()
    sha256 = case.get("sha256_hash") or "Not recorded"
    score_str = f"{case['score']:.4f}" if case.get("score") is not None else "N/A"
    thresh_str = f"{case['threshold']:.4f}" if case.get("threshold") is not None else "N/A"
    dur_str = f"{case['audio_duration_seconds']:.2f}s" if case.get("audio_duration_seconds") is not None else "N/A"

    # Speaker verification details
    spk_id = case.get("speaker_id")
    spk_decision = (case.get("speaker_verify_decision") or "NOT_PERFORMED").upper()
    spk_sim = f"{case['speaker_similarity']:.4f}" if case.get("speaker_similarity") is not None else "N/A"
    spk_thresh = f"{case['speaker_threshold']:.4f}" if case.get("speaker_threshold") is not None else "N/A"

    # Audio integrity details
    integrity_text = "No detailed signal metrics recorded."
    if case.get("audio_integrity_json"):
        try:
            it = json.loads(case["audio_integrity_json"])
            integrity_text = (
                f"Container: {it.get('container_format')}\n"
                f"Sample Rate: {it.get('sample_rate_hz')} Hz | Channels: {it.get('channels')} ({it.get('channel_layout')})\n"
                f"Peak Amplitude: {it.get('peak_amplitude_dbfs')} dBFS | RMS Level: {it.get('rms_level_dbfs')} dBFS\n"
                f"Clipping Ratio: {it.get('clipping_ratio_percent')}% | Silence Ratio: {it.get('silence_ratio_percent')}%\n"
                f"Observations: {it.get('notes')}"
            )
        except Exception:
            pass

    return f"""================================================================================
ECHOCHECK AI — PRELIMINARY AUDIO EVIDENCE ANALYSIS REPORT
================================================================================
Report Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
Case Identifier:  {case.get('case_id')}
Matter Reference: {case.get('case_reference')}
Dispute Category: {case.get('account_action')}
Alleged Speaker:  {case.get('customer_identifier') or 'Not Specified'}
Record Type:      {case.get('record_type', 'forensic').upper()}

EVIDENCE RECORD & INTEGRITY
--------------------------------------------------------------------------------
Audio Filename:   {case.get('audio_filename')}
File Size:        {case.get('audio_filesize_bytes', 0):,} bytes
Audio Duration:   {dur_str}
SHA-256 Hash:     {sha256}
Upload Timestamp: {case.get('created_at')}

1. SYNTHETIC VOICE DETECTION — DETECTIFAI ALETHEIA (NO REFERENCE REQUIRED)
--------------------------------------------------------------------------------
Preliminary Verdict:       {verdict}
Synthetic Probability:     {score_str}
Model Decision Threshold:  {thresh_str}
DetectifAI Request ID:     {case.get('request_id') or 'N/A'}
DetectifAI Operation ID:   {case.get('operation_id') or 'N/A'}
API Status:                {case.get('api_status')}

2. SPEAKER VERIFICATION — DETECTIFAI EIDOS (ENROLLED VOICE REQUIRED)
--------------------------------------------------------------------------------
Reference Speaker ID:      {spk_id or 'None Selected (Bypassed)'}
Verification Decision:     {spk_decision}
Acoustic Similarity Index: {spk_sim} (Scale: -1.0 to +1.0; not a percentage)
Decision Threshold:        {spk_thresh}
Request ID:                {case.get('speaker_verify_request_id') or 'N/A'}
NOTE: A biometric speaker match does NOT prove authenticity. A synthetic recording
can imitate an enrolled speaker. Keep Aletheia and Eidos findings separate.

3. TECHNICAL SIGNAL METRICS (PRELIMINARY INDICATORS ONLY)
--------------------------------------------------------------------------------
NOTE: The metrics below reflect physical file format and acoustic signal
properties only. They do NOT detect audio splicing, AI generation, or editing.
Signal clipping or silent intervals occur naturally in unedited field recordings
and do not constitute evidence of tampering.
{integrity_text}

PRELIMINARY ASSESSMENT GUIDANCE
--------------------------------------------------------------------------------
NOTE: This is automated decision support only. It does not constitute a
certified forensic expert opinion.
"{case.get('recommendation_text')}"

Case Status:      {case.get('case_status')}
Examiner Decision: {case.get('analyst_decision') or 'Pending Review'}
Examiner Notes:   {case.get('analyst_notes') or 'None entered'}

LIMITATIONS & LEGAL DISCLAIMER
--------------------------------------------------------------------------------
1. This automated screening report is provided for preliminary decision support
   and case triage only.
2. It does NOT constitute a certified forensic expert opinion under Section 45 /
   65B of evidentiary rules, nor does it replace formal scientific analysis by a
   Central Forensic Science Laboratory (CFSL), National Forensic Sciences
   University (NFSU), or court-appointed acoustic experts.
3. Audio splicing, analog tape editing, re-recording, and biological speaker
   identity remain unverified.
================================================================================
"""


def render_result_panel(case: dict) -> None:
    """Renders the unified dual-AI result dashboard with Aletheia and Eidos side by side."""
    st.markdown("### 📑 Unified Audio Evidence Results Dashboard")

    case_id = case.get("case_id", "UNKNOWN")
    sha256 = case.get("sha256_hash")
    verdict = (case.get("verdict") or "unavailable").lower()
    risk_level = case.get("risk_level", "UNKNOWN")
    rec_text = case.get("recommendation_text", "")
    api_status = case.get("api_status", "UNKNOWN")
    aletheia_error = case.get("api_error_message")

    # Header Case Reference & Hash Strip
    hash_html = f"<span style='color: #2563eb; font-weight: 700;'>SHA-256:</span> {sha256}" if sha256 else "No SHA-256 hash recorded"
    st.markdown(
        f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.6rem 0.9rem; margin-bottom: 1rem; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 0.5rem;">
            <div style="font-size: 0.88rem; font-weight: 700; color: #0f172a;">
                Case ID: <span style="font-family: monospace; color: #2563eb;">{case_id}</span>
                &nbsp;|&nbsp; Matter Ref: <span style="color: #334155;">{case.get('case_reference', 'N/A')}</span>
            </div>
            <div style="font-size: 0.75rem; color: #475569; font-family: monospace;">
                {hash_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # DUAL-AI SIDE-BY-SIDE RESPONSIVE 2-COLUMN LAYOUT
    col_card_a, col_card_b = st.columns(2, gap="medium")

    # =========================================================================
    # CARD A: ALETHEIA (Synthetic Speech Detection — No Reference Required)
    # =========================================================================
    with col_card_a:
        st.markdown(
            """
            <div style="border-left: 4px solid #2563eb; padding-left: 0.5rem; margin-bottom: 0.5rem;">
                <span style="font-size: 1.1rem; font-weight: 800; color: #0f172a;">Card A: Aletheia</span><br/>
                <span style="font-size: 0.78rem; color: #64748b; font-weight: 600;">(Synthetic Speech Detection — No Reference Required)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Prominent Verdict Badge
        if verdict == "fake":
            badge_html = '<div class="badge-fake" style="display: block; text-align: center; margin-bottom: 0.75rem; font-size: 0.95rem; padding: 0.5rem 0.75rem;">🚨 SYNTHETIC VOICE DETECTED (FAKE)</div>'
            interpretation = "Spectral and acoustic signatures exhibit characteristics typical of algorithmic/AI voice synthesis or deepfake cloning."
        elif verdict == "real":
            badge_html = '<div class="badge-real" style="display: block; text-align: center; margin-bottom: 0.75rem; font-size: 0.95rem; padding: 0.5rem 0.75rem;">🛡️ NATURAL SPEECH MARKERS (REAL)</div>'
            interpretation = "Acoustic signatures, vocal tract resonances, and temporal micro-variations align with authentic biological human speech."
        elif verdict == "not_sure":
            badge_html = '<div class="badge-not-sure" style="display: block; text-align: center; margin-bottom: 0.75rem; font-size: 0.95rem; padding: 0.5rem 0.75rem;">⚠️ INCONCLUSIVE FEATURES (NOT SURE)</div>'
            interpretation = "Spectral features are borderline or ambiguous; neither synthetic manipulation nor pure natural speech could be conclusively proven."
        else:
            badge_html = '<div class="badge-unavailable" style="display: block; text-align: center; margin-bottom: 0.75rem; font-size: 0.95rem; padding: 0.5rem 0.75rem;">🛑 PRELIMINARY ANALYSIS UNAVAILABLE</div>'
            interpretation = "Preliminary synthetic voice evaluation is unavailable for this recording."

        st.markdown(badge_html, unsafe_allow_html=True)

        # Metric Pills Row for Card A
        ma1, ma2 = st.columns(2)
        with ma1:
            score_val = case.get("score")
            score_str = f"{score_val:.4f}" if score_val is not None else "N/A"
            st.markdown(
                f"""
                <div class="metric-pill">
                    <div class="metric-pill-label">Synthetic Score</div>
                    <div class="metric-pill-value">{score_str}</div>
                    <div style="font-size: 0.65rem; color: #64748b;">Scale: 0.00 to 1.00</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with ma2:
            thresh_val = case.get("threshold")
            thresh_str = f"{thresh_val:.4f}" if thresh_val is not None else "N/A"
            st.markdown(
                f"""
                <div class="metric-pill">
                    <div class="metric-pill-label">Decision Threshold</div>
                    <div class="metric-pill-value">{thresh_str}</div>
                    <div style="font-size: 0.65rem; color: #64748b;">Aletheia cut-off</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Plain-language acoustic interpretation
        st.markdown(
            f"""
            <div style="font-size: 0.82rem; color: #334155; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.6rem 0.75rem; margin-bottom: 0.6rem; line-height: 1.45;">
                <b>Acoustic Interpretation:</b> {interpretation}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Duration & API Request ID
        dur_val = case.get("audio_duration_seconds")
        dur_str = f"{dur_val:.2f}s" if dur_val is not None else "N/A"
        req_id = case.get("request_id") or "N/A"
        st.markdown(
            f"""
            <div style="font-size: 0.72rem; color: #64748b; font-family: monospace; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 4px; padding: 0.4rem 0.6rem; margin-bottom: 0.5rem; word-break: break-all;">
                <div><b>Duration Analyzed:</b> {dur_str}</div>
                <div style="margin-top: 2px;"><b>API Request ID:</b> {req_id}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Independent Failure Display for Card A
        if api_status != "SUCCESS" and aletheia_error:
            st.error(f"**Aletheia Error ({api_status})**: {aletheia_error}")

    # =========================================================================
    # CARD B: EIDOS (Speaker Verification — Enrolled Voice Required)
    # =========================================================================
    with col_card_b:
        st.markdown(
            """
            <div style="border-left: 4px solid #7c3aed; padding-left: 0.5rem; margin-bottom: 0.5rem;">
                <span style="font-size: 1.1rem; font-weight: 800; color: #0f172a;">Card B: Eidos</span><br/>
                <span style="font-size: 0.78rem; color: #64748b; font-weight: 600;">(Speaker Verification — Enrolled Voice Required)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        spk_id = case.get("speaker_id")
        spk_display = case.get("speaker_display_name")
        raw_dec = (case.get("speaker_verify_decision") or "").strip().lower()

        if spk_id:
            # Prominent Decision Badge with high contrast
            if raw_dec == "match":
                dec_badge_html = """
                <div style="background-color: #f0fdf4; border: 2px solid #86efac; color: #15803d; padding: 0.5rem 0.75rem; border-radius: 6px; font-weight: 800; font-size: 0.95rem; text-align: center; margin-bottom: 0.75rem; letter-spacing: 0.03em;">
                    ✓ MATCH — VERIFIED BIOMETRIC VOICEPRINT
                </div>
                """
                eidos_interp = "Acoustic voiceprint aligns with the reference speaker within defined similarity threshold."
            elif raw_dec == "no_match":
                dec_badge_html = """
                <div style="background-color: #fef2f2; border: 2px solid #fca5a5; color: #b91c1c; padding: 0.5rem 0.75rem; border-radius: 6px; font-weight: 800; font-size: 0.95rem; text-align: center; margin-bottom: 0.75rem; letter-spacing: 0.03em;">
                    ✕ NO MATCH — VOICEPRINT MISMATCH
                </div>
                """
                eidos_interp = "Acoustic voiceprint deviates significantly from the enrolled reference speaker."
            elif raw_dec == "inconclusive":
                dec_badge_html = """
                <div style="background-color: #fffbeb; border: 2px solid #fde68a; color: #b45309; padding: 0.5rem 0.75rem; border-radius: 6px; font-weight: 800; font-size: 0.95rem; text-align: center; margin-bottom: 0.75rem; letter-spacing: 0.03em;">
                    ⚠ INCONCLUSIVE — INSUFFICIENT BIOMETRIC MARGIN
                </div>
                """
                eidos_interp = "Biometric similarity falls in the indeterminate margin; requires longer reference speech."
            else:
                dec_badge_html = """
                <div style="background-color: #f1f5f9; border: 2px solid #cbd5e1; color: #475569; padding: 0.5rem 0.75rem; border-radius: 6px; font-weight: 800; font-size: 0.95rem; text-align: center; margin-bottom: 0.75rem; letter-spacing: 0.03em;">
                    🛑 ERROR / UNAVAILABLE
                </div>
                """
                eidos_interp = "Speaker verification could not be completed for this recording."

            st.markdown(dec_badge_html, unsafe_allow_html=True)

            # Metric Pills Row for Card B
            mb1, mb2 = st.columns(2)
            with mb1:
                sim_val = case.get("speaker_similarity")
                sim_str = f"{sim_val:.4f}" if sim_val is not None else "N/A"
                st.markdown(
                    f"""
                    <div class="metric-pill">
                        <div class="metric-pill-label">Similarity Index</div>
                        <div class="metric-pill-value">{sim_str}</div>
                        <div style="font-size: 0.65rem; color: #64748b;">Scale: -1.0 to +1.0 (not %)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with mb2:
                spk_thresh_val = case.get("speaker_threshold")
                spk_thresh_str = f"{spk_thresh_val:.4f}" if spk_thresh_val is not None else "N/A"
                st.markdown(
                    f"""
                    <div class="metric-pill">
                        <div class="metric-pill-label">Decision Threshold</div>
                        <div class="metric-pill-value">{spk_thresh_str}</div>
                        <div style="font-size: 0.65rem; color: #64748b;">Eidos cut-off</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Speaker Reference Info
            spk_display_text = f"{spk_display} ({spk_id})" if spk_display else spk_id
            st.markdown(
                f"""
                <div style="font-size: 0.82rem; color: #334155; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.6rem 0.75rem; margin-bottom: 0.6rem; line-height: 1.45;">
                    <b>Target Speaker:</b> {spk_display_text}<br/>
                    <b>Acoustic Finding:</b> {eidos_interp}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Metadata Row
            spk_req_id = case.get("speaker_verify_request_id") or "N/A"
            st.markdown(
                f"""
                <div style="font-size: 0.72rem; color: #64748b; font-family: monospace; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 4px; padding: 0.4rem 0.6rem; margin-bottom: 0.5rem; word-break: break-all;">
                    <div><b>Target Speaker ID:</b> {spk_id}</div>
                    <div style="margin-top: 2px;"><b>Eidos Request ID:</b> {spk_req_id}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Error handling for Card B
            if raw_dec == "unavailable" and case.get("api_error_message"):
                st.error(f"**Eidos Error**: {case.get('api_error_message')}")

            # Prominent Biometric Guardrail Note
            st.markdown(
                """
                <div style="font-size: 0.78rem; color: #b45309; background: #fffbeb; border: 1px solid #fde68a; border-radius: 4px; padding: 0.55rem 0.75rem; line-height: 1.45;">
                    <b>⚠️ Biometric Guardrail:</b> A biometric speaker match does <b>NOT</b> prove the recording is authentic.
                    An AI voice clone can match an enrolled voiceprint. Aletheia and Eidos findings remain independent.
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            # No reference speaker selected
            st.markdown(
                """
                <div style="background: #f8fafc; border: 1.5px dashed #cbd5e1; border-radius: 8px; padding: 2.2rem 1.25rem; text-align: center; color: #475569; margin-bottom: 0.75rem;">
                    <div style="font-size: 1.8rem; margin-bottom: 0.4rem;">👤🎙️</div>
                    <div style="font-weight: 700; font-size: 0.95rem; color: #1e293b;">No Reference Speaker Selected</div>
                    <div style="font-size: 0.82rem; margin-top: 0.35rem; line-height: 1.45; color: #64748b;">
                        No reference speaker selected. Enroll a speaker in the <b>Speaker Enrollment</b> tab to verify speaker identity.
                    </div>
                    <div style="font-size: 0.74rem; margin-top: 0.75rem; color: #2563eb; font-weight: 600;">
                        ✓ Aletheia synthetic voice detection operated independently without a reference voice.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # COMBINED ASSESSMENT SUMMARY (Below the Cards)
    # =========================================================================
    st.markdown("---")
    st.markdown("#### 📋 Combined Forensic Assessment Summary")

    # Parse Audio Integrity diagnostics if present
    it_diag = {}
    if case.get("audio_integrity_json"):
        try:
            it_diag = json.loads(case["audio_integrity_json"])
        except Exception:
            pass

    sum_col1, sum_col2, sum_col3 = st.columns(3, gap="medium")

    # Category 1: Synthetic-Speech Finding
    with sum_col1:
        st.markdown(
            f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-top: 3px solid #2563eb; border-radius: 6px; padding: 0.85rem; height: 100%;">
                <div style="font-size: 0.75rem; text-transform: uppercase; font-weight: 700; color: #2563eb; letter-spacing: 0.05em;">
                    1. Synthetic-Speech Finding
                </div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-top: 0.35rem;">
                    {verdict.upper()}
                </div>
                <div style="font-size: 0.82rem; color: #475569; margin-top: 0.3rem;">
                    • Synthetic Score: <b>{score_str}</b><br/>
                    • Threshold: <b>{thresh_str}</b><br/>
                    • Risk Finding: <b>{risk_level}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Category 2: Reference-Speaker Finding
    with sum_col2:
        dec_display = raw_dec.upper() if spk_id and raw_dec else ("NOT PERFORMED" if not spk_id else "UNAVAILABLE")
        sim_display = f"{case.get('speaker_similarity'):.4f}" if case.get("speaker_similarity") is not None else "N/A"
        st.markdown(
            f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-top: 3px solid #7c3aed; border-radius: 6px; padding: 0.85rem; height: 100%;">
                <div style="font-size: 0.75rem; text-transform: uppercase; font-weight: 700; color: #7c3aed; letter-spacing: 0.05em;">
                    2. Reference-Speaker Finding
                </div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-top: 0.35rem;">
                    {dec_display}
                </div>
                <div style="font-size: 0.82rem; color: #475569; margin-top: 0.3rem;">
                    • Similarity Index: <b>{sim_display}</b> (-1.0 to +1.0)<br/>
                    • Target: <b>{spk_id or 'None Selected'}</b><br/>
                    • Decision Cut-off: <b>{case.get('speaker_threshold') or 'N/A'}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Category 3: Audio Quality & Integrity Diagnostics
    with sum_col3:
        sr_val = it_diag.get("sample_rate_hz")
        sr_str = f"{sr_val:,} Hz" if sr_val else "N/A"
        clip_val = it_diag.get("clipping_ratio_percent")
        clip_str = f"{clip_val:.2f}%" if clip_val is not None else "N/A"
        sil_val = it_diag.get("silence_ratio_percent")
        sil_str = f"{sil_val:.1f}%" if sil_val is not None else "N/A"
        fmt_val = it_diag.get("container_format", "N/A")

        st.markdown(
            f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-top: 3px solid #059669; border-radius: 6px; padding: 0.85rem; height: 100%;">
                <div style="font-size: 0.75rem; text-transform: uppercase; font-weight: 700; color: #059669; letter-spacing: 0.05em;">
                    3. Technical Quality & Integrity
                </div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-top: 0.35rem;">
                    {fmt_val} ({sr_str})
                </div>
                <div style="font-size: 0.82rem; color: #475569; margin-top: 0.3rem;">
                    • Saturation / Clipping: <b>{clip_str}</b><br/>
                    • Silence / Inactivity: <b>{sil_str}</b><br/>
                    • Duration: <b>{dur_str}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Combined Plain-English Guidance & Synthesis Box
    if verdict == "fake" and raw_dec == "match":
        guidance_title = "🚨 CRITICAL FORENSIC ALERT: HIGH PROBABILITY OF AI VOICE CLONE"
        guidance_body = (
            "The disputed recording demonstrates strong acoustic indicators of synthetic/AI speech synthesis, "
            "yet acoustically matches the reference speaker's voiceprint. This pattern strongly suggests an "
            "algorithmic voice clone designed to impersonate the reference speaker."
        )
        box_style = "border-left: 5px solid #dc2626; background: #fef2f2;"
    elif verdict == "fake" and raw_dec == "no_match":
        guidance_title = "🚨 HIGH RISK: UNIDENTIFIED SYNTHETIC VOICE DETECTED"
        guidance_body = (
            "The recording contains significant synthetic voice markers and does not match the enrolled "
            "reference speaker. The audio is likely synthetic speech generated from an un-enrolled source."
        )
        box_style = "border-left: 5px solid #dc2626; background: #fef2f2;"
    elif verdict == "real" and raw_dec == "match":
        guidance_title = "🛡️ CONSISTENT WITH AUTHENTIC REFERENCE SPEAKER"
        guidance_body = (
            "The recording exhibits natural biological speech dynamics and successfully matches the enrolled "
            "reference speaker's voiceprint within statistical confidence thresholds."
        )
        box_style = "border-left: 5px solid #16a34a; background: #f0fdf4;"
    elif verdict == "real" and raw_dec == "no_match":
        guidance_title = "⚠️ THIRD-PARTY NATURAL SPEECH"
        guidance_body = (
            "The recording displays natural human speech characteristics, but does NOT match the enrolled "
            "reference speaker. The speaker is likely an authentic third party rather than the reference individual."
        )
        box_style = "border-left: 5px solid #d97706; background: #fffbeb;"
    elif verdict == "fake":
        guidance_title = "🚨 SYNTHETIC VOICE DETECTED (NO REFERENCE COMPARISON)"
        guidance_body = (
            "Synthetic voice indicators detected. No reference speaker voiceprint was provided to verify identity."
        )
        box_style = "border-left: 5px solid #dc2626; background: #fef2f2;"
    elif verdict == "real":
        guidance_title = "🛡️ NATURAL HUMAN SPEECH (NO REFERENCE COMPARISON)"
        guidance_body = (
            "Natural speech markers observed. Reference speaker identity was not compared for this case."
        )
        box_style = "border-left: 5px solid #16a34a; background: #f0fdf4;"
    else:
        guidance_title = "⚠️ INCONCLUSIVE ACOUSTIC EVALUATION"
        guidance_body = (
            "Preliminary automated assessment is inconclusive. Physical acoustic verification by a certified "
            "forensic examiner (CFSL/NFSU) is recommended."
        )
        box_style = "border-left: 5px solid #64748b; background: #f8fafc;"

    st.markdown(
        f"""
        <div style="{box_style} border-radius: 6px; padding: 1rem 1.15rem; margin-top: 1rem; margin-bottom: 1.25rem;">
            <div style="font-size: 0.92rem; font-weight: 800; color: #0f172a;">{guidance_title}</div>
            <div style="font-size: 0.85rem; color: #334155; margin-top: 0.35rem; line-height: 1.5;">{guidance_body}</div>
            <div style="font-size: 0.74rem; color: #64748b; margin-top: 0.5rem; font-style: italic;">
                System Recommendation: "{rec_text}"
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Forensic Examiner Case Action Station
    st.markdown("#### ✍️ Forensic Examiner Action Station")
    with st.expander("Update Case Status & Examiner Determination", expanded=False):
        with st.form(f"quick_action_{case['case_id']}"):
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                new_status = st.selectbox(
                    "Case Status",
                    ["NEW", "PENDING_EXAMINATION", "EXAMINATION_RECOMMENDED", "AUTHENTICATED_PRELIMINARY", "ARCHIVED"],
                    index=0,
                )
            with col_s2:
                analyst_decision = st.selectbox(
                    "Examiner Assessment",
                    [
                        "Awaiting Formal Laboratory Verification",
                        "Preliminary Natural Speech - Awaiting Speaker Comparison",
                        "Flagged - Suspected Synthetic Voice / AI Clone",
                        "Inconclusive - Inadequate Quality or Duration",
                        "Referred to Expert Forensic Examiner (CFSL/NFSU)",
                    ],
                    index=0,
                )

            analyst_notes = st.text_area(
                "Case Notes & Examination Diary",
                value=case.get("analyst_notes") or "",
                placeholder="Document your acoustic observations, source verification, court submission status...",
                height=100,
            )

            update_submit = st.form_submit_button("Save Case Update", type="primary")
            if update_submit:
                if not auth.require_permission("upload_audio", silent=True):
                    st.error("🚫 Only admins and analysts can update case status.", icon="🚫")
                else:
                    db.update_case_analyst_review(
                        case_id=case["case_id"],
                        case_status=new_status,
                        analyst_decision=analyst_decision,
                        analyst_notes=analyst_notes,
                    )
                    # Invalidate any approved/submitted reports since the case changed
                    db.invalidate_report_on_case_update(case["case_id"])
                    user = auth.get_current_user()
                    db.log_app_audit_event(
                        action="CASE_UPDATED",
                        user_id=user["user_id"] if user else None,
                        org_id=user["org_id"] if user else None,
                        case_id=case["case_id"],
                        metadata={"new_status": new_status, "analyst_decision": analyst_decision},
                    )
                    st.success("Case record updated successfully!")
                    st.rerun()

    # Report Review Workflow
    st.markdown("#### 📋 Report Review Workflow")
    _render_report_workflow(case)

    # Report Download & Audit Timeline
    col_dl1, col_dl2 = st.columns([1, 1])
    with col_dl1:
        try:
            pdf_bytes = pdf_report.generate_forensic_pdf(case, db.get_case_audit_events(case["case_id"]))
            st.download_button(
                "📥 Download Forensic Report (PDF)",
                data=pdf_bytes,
                file_name=f"EchoCheck_Report_{case['case_id']}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as pdf_err:
            st.error(f"PDF generation error: {pdf_err}")

    with col_dl2:
        report_text = build_text_report(case)
        st.download_button(
            "📥 Download Summary Report (TXT)",
            data=report_text,
            file_name=f"EchoCheck_Report_{case['case_id']}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # Chain of Custody & Audit Event Timeline
    events = db.get_case_audit_events(case["case_id"])
    if events:
        with st.expander("🕒 Chain of Custody & Audit Events Timeline", expanded=False):
            for ev in events:
                st.markdown(
                    f"<div style='font-size: 0.8rem; padding: 0.3rem 0; border-bottom: 1px solid #f1f5f9;'>"
                    f"<span style='color: #64748b; font-family: monospace;'>[{ev['created_at'][:19]} UTC]</span> "
                    f"<b style='color: #2563eb;'>{ev['event_type']}</b>: {ev['event_description']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    if case.get("raw_response_json"):
        with st.expander("🔍 Inspect Raw DetectifAI Response Payload"):
            try:
                parsed_json = json.loads(case["raw_response_json"])
                st.json(parsed_json)
            except Exception:
                st.text(case["raw_response_json"])

    # Regulatory & Legal Disclaimer
    st.markdown(
        """
        <div class="sv-disclaimer">
            <b>LEGAL & FORENSIC ADMISSIBILITY DISCLAIMER:</b> EchoCheck AI provides automated preliminary decision support.
            This report does NOT constitute a certified forensic opinion or legal admissibility determination under Section 45 / 65B of the Indian Evidence Act / BSA or relevant jurisdiction.
            It does NOT replace CFSL, NFSU, or certified forensic laboratory acoustic analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_report_workflow(case: dict) -> None:
    """Renders the report draft/submit/approve/reject workflow for a case."""
    current_user = auth.get_current_user()
    if current_user is None:
        return

    case_id = case["case_id"]
    org_id = current_user["org_id"]
    role = current_user["role"]
    user_id = current_user["user_id"]

    reports = db.get_reports_for_case(case_id)

    can_draft = auth.has_permission(role, "generate_draft_report")
    can_approve = auth.has_permission(role, "approve_reject_report")

    # Show existing reports
    if reports:
        latest = reports[0]
        status = latest["status"]
        report_id = latest["report_id"]
        author_id = latest["author_user_id"]
        version = latest.get("report_version", 1)

        status_colors = {
            "draft": "#64748b",
            "submitted": "#2563eb",
            "approved": "#15803d",
            "rejected": "#b91c1c",
        }
        st.markdown(
            f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;
                        padding: 0.65rem 0.9rem; margin-bottom: 0.75rem;">
                <span style="font-size: 0.75rem; text-transform: uppercase; font-weight: 700;
                             color: {status_colors.get(status, '#64748b')};">
                    Report v{version}: {status.upper()}
                </span>
                <span style="font-size: 0.73rem; color: #64748b; margin-left: 0.5rem;">
                    ID: <code>{report_id[:8]}…</code>
                </span>
                {f'<br/><span style="font-size: 0.73rem; color: #475569;">Reviewer note: {latest["reviewer_comment"]}</span>' if latest.get('reviewer_comment') else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Submit for review (author / admin / analyst)
        if status == "draft" and can_draft:
            if st.button("📤 Submit Report for Review", key=f"submit_report_{report_id}"):
                db.update_report_status(report_id, "submitted")
                db.log_app_audit_event(
                    action="REPORT_SUBMITTED",
                    user_id=user_id, org_id=org_id, case_id=case_id,
                    metadata={"report_id": report_id},
                )
                db.log_case_audit_event(case_id, "REPORT_SUBMITTED", f"Report {report_id} submitted for review by {current_user['username']}.")
                st.success("Report submitted for review.")
                st.rerun()

        # Approve / Reject (senior_reviewer only; cannot approve own report)
        if status == "submitted" and can_approve:
            if author_id == user_id:
                st.warning("⚠️ You cannot approve a report you authored.", icon="⚠️")
            else:
                col_ap, col_rj = st.columns(2)
                with col_ap:
                    if st.button("✅ Approve Report", key=f"approve_{report_id}", type="primary"):
                        db.update_report_status(report_id, "approved", reviewer_user_id=user_id)
                        db.log_app_audit_event(
                            action="REPORT_APPROVED",
                            user_id=user_id, org_id=org_id, case_id=case_id,
                            metadata={"report_id": report_id, "reviewer": current_user["username"]},
                        )
                        db.log_case_audit_event(case_id, "REPORT_APPROVED", f"Report {report_id} approved by {current_user['username']}.")
                        st.success("✅ Report approved.")
                        st.rerun()
                with col_rj:
                    with st.expander("Reject with reason"):
                        with st.form(f"reject_form_{report_id}"):
                            reject_reason = st.text_area("Rejection Reason *", height=80)
                            reject_btn = st.form_submit_button("❌ Reject Report", type="secondary")
                            if reject_btn:
                                if not reject_reason.strip():
                                    st.error("A rejection reason is required.", icon="🛑")
                                else:
                                    db.update_report_status(
                                        report_id, "rejected",
                                        reviewer_user_id=user_id,
                                        reviewer_comment=reject_reason.strip(),
                                    )
                                    db.log_app_audit_event(
                                        action="REPORT_REJECTED",
                                        user_id=user_id, org_id=org_id, case_id=case_id,
                                        metadata={"report_id": report_id, "reason": reject_reason.strip()},
                                    )
                                    db.log_case_audit_event(case_id, "REPORT_REJECTED", f"Report {report_id} rejected: {reject_reason.strip()[:80]}")
                                    st.warning("Report rejected.")
                                    st.rerun()
    else:
        # No reports yet — create a draft
        if can_draft:
            if st.button("📝 Create Draft Report", key=f"create_report_{case_id}"):
                report_id = db.create_report(
                    case_id=case_id,
                    org_id=org_id,
                    author_user_id=user_id,
                )
                db.log_app_audit_event(
                    action="REPORT_CREATED",
                    user_id=user_id, org_id=org_id, case_id=case_id,
                    metadata={"report_id": report_id},
                )
                db.log_case_audit_event(case_id, "REPORT_CREATED", f"Draft report {report_id} created by {current_user['username']}.")
                st.success("Draft report created.")
                st.rerun()
        else:
            st.info("No reports exist for this case yet.")


def page_speaker_enrollment() -> None:
    """Eidos Speaker Enrollment Station for reference voiceprints."""
    if not auth.require_permission("enroll_verify_speaker"):
        return
    current_user = auth.get_current_user()
    st.markdown(
        """
        <div class="sv-header">
            <div class="sv-header-title">
                <span>🎙️</span> DetectifAI Eidos — Speaker Enrollment Station
            </div>
            <div class="sv-header-subtitle">
                Enroll reference voice exemplars to create acoustic voiceprints for future speaker verification and court comparison.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.1, 1.2], gap="large")

    with col1:
        st.markdown("### 📋 Enroll New Reference Speaker")
        st.markdown(
            """
            <p style='color: #64748b; font-size: 0.85rem;'>
                <b>Requirements:</b> Recording must be <b>10–60 seconds</b> in duration with at least <b>10 seconds</b> of clear speech from that single individual. Up to 5 samples per speaker ID are supported.
            </p>
            """,
            unsafe_allow_html=True,
        )

        with st.form("speaker_enrollment_form", clear_on_submit=False):
            speaker_id = st.text_input(
                "Speaker Identifier (Unique ID) *",
                placeholder="e.g. SPK-ACCUSED-01 or WITNESS-DOE-24",
                help="1-64 characters matching [A-Za-z0-9_.:-]. Used to look up this voiceprint.",
            )

            display_name = st.text_input(
                "Display Label / Pseudonym (Optional)",
                placeholder="e.g. Accused Person A (Reference Exam)",
                help="Optional human-readable label shown in the registry.",
            )

            case_reference = st.text_input(
                "Associated Case / Docket Reference",
                placeholder="e.g. CRIM-2026-88192 or FIR #312",
            )

            provenance_notes = st.text_area(
                "Provenance & Chain of Custody Notes",
                placeholder="Reference exemplar recorded during formal police inquiry under Section 161 CrPC. Microphones: Sony PCM-D100...",
                height=90,
            )

            enrollment_audio = st.file_uploader(
                "Upload Reference Voice Recording (10-60s) *",
                type=["wav", "mp3", "m4a", "ogg", "flac"],
                help="Must contain 10-60s of audio with >= 10s speech.",
            )

            if enrollment_audio is not None:
                st.audio(enrollment_audio)
                enr_bytes = enrollment_audio.getvalue()
                st.caption(f"📁 {enrollment_audio.name} ({len(enr_bytes) / 1024:.1f} KB)")
                try:
                    _, _, enr_dur = audio_visualizer.decode_audio_samples(enr_bytes, enrollment_audio.name)
                    if enr_dur < 10.0:
                        st.warning(f"⚠️ Recording duration is {enr_dur:.1f}s. Eidos requires at least 10 seconds (10–60s supported).")
                    elif enr_dur > 60.0:
                        st.warning(f"⚠️ Recording duration is {enr_dur:.1f}s. Eidos enrollment supports up to 60 seconds.")
                    else:
                        st.caption(f"✓ Valid Duration: {enr_dur:.1f}s (Within supported 10–60s range)")
                except Exception:
                    pass

            enroll_btn = st.form_submit_button("🎙️ Enroll Speaker with Eidos", type="primary", use_container_width=True)

        if enroll_btn:
            if not speaker_id.strip():
                st.error("Please provide a Speaker Identifier.", icon="🛑")
                return

            if enrollment_audio is None:
                st.error("Please upload an enrollment voice recording file.", icon="🛑")
                return

            audio_bytes = enrollment_audio.getvalue()
            filename = enrollment_audio.name

            # Pre-flight duration check
            try:
                _, _, enr_dur = audio_visualizer.decode_audio_samples(audio_bytes, filename)
                if enr_dur < 10.0:
                    st.error(f"Recording is {enr_dur:.1f}s. Eidos enrollment requires at least 10 seconds (10–60s supported).", icon="🛑")
                    return
                if enr_dur > 60.0:
                    st.error(f"Recording is {enr_dur:.1f}s. Eidos enrollment maximum duration is 60 seconds.", icon="🛑")
                    return
            except Exception:
                pass

            with col2:
                with st.spinner("Submitting reference sample to DetectifAI Eidos Enrollment API..."):
                    res = api.enroll_speaker(
                        audio_bytes=audio_bytes,
                        filename=filename,
                        speaker_id=speaker_id.strip(),
                        display_name=display_name.strip() if display_name else None,
                    )

                    if res.success:
                        db.log_app_audit_event(
                            action="SPEAKER_ENROLLED",
                            user_id=current_user["user_id"] if current_user else None,
                            org_id=current_user["org_id"] if current_user else None,
                            metadata={"speaker_id": speaker_id.strip(), "samples": res.samples_count},
                        )
                        st.success(f"✅ Speaker '{res.speaker_id}' successfully enrolled in DetectifAI Eidos!")
                        # Save in local database
                        db.save_enrolled_speaker(
                            speaker_id=res.speaker_id,
                            display_name=res.display_name or display_name,
                            case_reference=case_reference,
                            provenance_notes=provenance_notes,
                            samples_count=res.samples_count or 1,
                        )
                        speech_str = f"{res.speech_seconds:.1f}s" if res.speech_seconds is not None else "N/A"
                        total_str = f"{res.duration_seconds:.1f}s" if res.duration_seconds is not None else "N/A"
                        req_id_str = res.request_id or "N/A"
                        op_id_str = res.operation_id or "N/A"
                        st.markdown(
                            f"""
                            <div class="sv-card" style="margin-top: 1rem;">
                                <div style="font-weight: 700; color: #15803d; font-size: 1rem; margin-bottom: 0.5rem;">
                                    Enrollment Confirmation
                                </div>
                                <div style="font-size: 0.85rem; color: #334155; line-height: 1.6;">
                                    • <b>Speaker ID:</b> <code>{res.speaker_id}</code><br/>
                                    • <b>Enrolled Samples:</b> {res.samples_count or 1} of 5 max<br/>
                                    • <b>Detected Speech:</b> {speech_str} (Total: {total_str})<br/>
                                    • <b>DetectifAI Request ID:</b> <code>{req_id_str}</code><br/>
                                    • <b>Operation ID:</b> <code>{op_id_str}</code>
                                </div>
                                <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.6rem; border-top: 1px solid #e2e8f0; padding-top: 0.5rem;">
                                    ⓘ This speaker is now enrolled in both the <b>local case registry</b> and <b>DetectifAI's remote voiceprint store</b>.
                                    These are separate records. Deleting locally removes the registry entry but does not automatically delete
                                    the remote voiceprint — use the Delete button to remove both simultaneously.
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error(f"❌ Enrollment Failed ({res.api_status}): {res.error_message}")
                        if res.status_code == 403:
                            st.info("💡 **Scope Notice**: Your DetectifAI API key requires the `speakers:enroll` scope to register new voiceprints.")

    with col2:
        st.markdown("### 🗂️ Currently Enrolled Reference Speakers")

        # Fetch local records
        speakers = db.list_local_speakers()
        if not speakers:
            st.info("No reference speakers are currently registered in this local forensic workstation.")
        else:
            for spk in speakers:
                with st.expander(f"👤 {spk['speaker_id']}" + (f" ({spk['display_name']})" if spk.get('display_name') else "")):
                    st.markdown(
                        f"""
                        <div style="font-size: 0.85rem; line-height: 1.6; color: #334155;">
                            • <b>Associated Case:</b> {spk.get('case_reference') or 'None'}<br/>
                            • <b>Enrolled Samples:</b> {spk.get('samples_count', 1)} sample(s)<br/>
                            • <b>Registered Date:</b> {spk.get('created_at')}<br/>
                            • <b>Provenance Notes:</b> {spk.get('provenance_notes') or 'None entered'}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    col_del1, col_del2 = st.columns([1, 3])
                    with col_del1:
                        if st.button(f"Delete Speaker", key=f"del_{spk['speaker_id']}"):
                            # Attempt API deletion
                            api.delete_enrolled_speaker(spk['speaker_id'])
                            # Delete local record
                            db.delete_local_speaker(spk['speaker_id'])
                            st.success("Speaker deleted.")
                            st.rerun()


def page_case_registry() -> None:
    """Historical forensic case registry with filtering, search, and drilldown."""
    if not auth.require_permission("view_assigned_cases"):
        return
    current_user = auth.get_current_user()
    st.markdown(
        """
        <div class="sv-header">
            <div class="sv-header-title">
                <span>📁</span> Forensic Case Registry
            </div>
            <div class="sv-header-subtitle">
                Comprehensive audit trail of audio evidence assessments, deepfake detections, speaker verification decisions, and examiner case diaries.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats = db.get_dashboard_stats()
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Audio Records", stats["total_cases"])
    with m2:
        st.metric("Synthetic (Fake)", stats["fake_detected"], delta="Flagged", delta_color="inverse")
    with m3:
        st.metric("Natural Speech (Real)", stats["real_verified"], delta="Passed Check")
    with m4:
        st.metric("Inconclusive", stats["inconclusive"])
    with m5:
        st.metric("Under Examination", stats["pending_cases"])

    st.markdown("---")

    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        filter_verdict = st.selectbox(
            "Filter by Deepfake Verdict",
            ["ALL", "fake", "real", "not_sure", "unavailable"],
            index=0,
        )
    with f2:
        filter_status = st.selectbox(
            "Filter by Case Status",
            ["ALL", "NEW", "PENDING_EXAMINATION", "EXAMINATION_RECOMMENDED", "AUTHENTICATED_PRELIMINARY", "ARCHIVED"],
            index=0,
        )
    with f3:
        search_query = st.text_input(
            "Search Evidence Registry",
            placeholder="Search by Case ID, Matter Reference, Speaker, Hash, or Notes...",
        )

    cases = db.list_cases_for_user(
        user_id=current_user["user_id"],
        org_id=current_user["org_id"],
        role=current_user["role"],
        verdict=filter_verdict,
        status=filter_status,
        search_query=search_query,
        limit=200,
    )

    if not cases:
        st.info("No evidence records matching the selected criteria were found in the database.")
        return

    df_cases = pd.DataFrame(cases)

    display_cols = [
        "record_type",
        "case_id",
        "case_reference",
        "account_action",
        "verdict",
        "score",
        "speaker_id",
        "speaker_verify_decision",
        "speaker_similarity",
        "case_status",
        "sha256_hash",
        "created_at",
    ]
    available_cols = [c for c in display_cols if c in df_cases.columns]
    df_display = df_cases[available_cols].copy()
    rename_map = {}
    if "account_action" in df_display.columns:
        rename_map["account_action"] = "evidence_category"
    if "record_type" in df_display.columns:
        rename_map["record_type"] = "type"
    if rename_map:
        df_display = df_display.rename(columns=rename_map)

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
    )

    col_exp1, col_exp2 = st.columns([1, 4])
    with col_exp1:
        csv_data = df_cases.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export Registry to CSV",
            data=csv_data,
            file_name=f"echocheck_ai_case_registry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("### 🔍 Case Evidence Detail Inspector")

    case_options = [c["case_id"] for c in cases]
    selected_case_id = st.selectbox("Select Evidence Case to Inspect", case_options)

    if selected_case_id:
        selected_case = db.get_case(selected_case_id)
        if selected_case and auth.assert_case_access(selected_case):
            render_result_panel(selected_case)


def page_system_health() -> None:
    """Forensic systems, API status, and evidence principles."""
    st.markdown(
        """
        <div class="sv-header">
            <div class="sv-header-title">
                <span>⚙️</span> Forensic Systems & API Status
            </div>
            <div class="sv-header-subtitle">
                Inspect AI detection model endpoints, Eidos speaker verification configuration, authentication scopes, and forensic principles.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### 🔌 API 1: Synthetic Voice Detection (DetectifAI Aletheia)")

        api_configured = api.is_api_configured()
        st.markdown(
            f"""
            <div class="sv-card">
                <div style="font-size: 0.78rem; color: #64748b; text-transform: uppercase; font-weight: 600;">Endpoint</div>
                <div style="font-family: monospace; color: #2563eb; font-size: 0.9rem; margin-top: 0.15rem;">
                    {api.DETECTIF_ENDPOINT}
                </div>
                <hr style="border-color: #e2e8f0; margin: 0.65rem 0;" />
                <div style="font-size: 0.78rem; color: #64748b; text-transform: uppercase; font-weight: 600;">Authentication & Scopes</div>
                <div style="font-size: 0.95rem; font-weight: 600; color: {'#166534' if api_configured else '#991b1b'}; margin-top: 0.15rem;">
                    {'Active & Masked' if api_configured else 'Missing / Unconfigured'}
                </div>
                <div style="font-family: monospace; color: #475569; font-size: 0.85rem; margin-top: 0.15rem;">
                    Bearer Token: {api.get_masked_api_key()}
                </div>
                <hr style="border-color: #e2e8f0; margin: 0.65rem 0;" />
                <div style="font-size: 0.78rem; color: #64748b; text-transform: uppercase; font-weight: 600;">Required Scope</div>
                <div style="font-size: 0.85rem; color: #334155; margin-top: 0.2rem;">
                    • <code>deepfake:detect</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 👤 API 2: Speaker Verification (DetectifAI Eidos)")
        st.markdown(
            f"""
            <div class="sv-card">
                <div style="font-size: 0.78rem; color: #64748b; text-transform: uppercase; font-weight: 600;">Endpoints</div>
                <div style="font-family: monospace; color: #2563eb; font-size: 0.85rem; margin-top: 0.15rem; line-height: 1.6;">
                    • Enrollment: <code>{api.SPEAKERS_ENROLL_ENDPOINT}</code><br/>
                    • Verification: <code>{api.SPEAKERS_VERIFY_ENDPOINT}</code><br/>
                    • Directory: <code>{api.SPEAKERS_LIST_ENDPOINT}</code>
                </div>
                <hr style="border-color: #e2e8f0; margin: 0.65rem 0;" />
                <div style="font-size: 0.78rem; color: #64748b; text-transform: uppercase; font-weight: 600;">Required Scopes</div>
                <div style="font-size: 0.85rem; color: #334155; margin-top: 0.2rem; line-height: 1.5;">
                    • <code>speakers:enroll</code> (Enroll reference voice samples)<br/>
                    • <code>speakers:verify</code> (Verify disputed recordings against voiceprint)<br/>
                    • <code>speakers:read</code> (List enrolled speakers)<br/>
                    • <code>speakers:delete</code> (Delete speaker voiceprints)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # API Key override — admin only
        if auth.has_permission(auth.get_current_user()["role"] if auth.get_current_user() else "analyst", "change_api_config"):
            st.markdown("### 🔑 Live Session API Key Override")
            with st.form("override_key_form"):
                override_key = st.text_input(
                    "DetectifAI API Key",
                    type="password",
                    placeholder="Paste your DetectifAI API key here...",
                )
                override_submit = st.form_submit_button("Apply Key to Session")
                if override_submit and override_key.strip():
                    os.environ["DETECTIF_API_KEY"] = override_key.strip()
                    user = auth.get_current_user()
                    db.log_app_audit_event(
                        action="API_KEY_CHANGED",
                        user_id=user["user_id"] if user else None,
                        org_id=user["org_id"] if user else None,
                        outcome="SUCCESS",
                        metadata={"method": "session_override"},
                    )
                    st.success("API key loaded into process environment!")
                    st.rerun()
        else:
            st.info("🔒 API key configuration is restricted to administrators.", icon="🔒")

    with col2:
        st.markdown("### ⚖️ Legal Forensics Principles & Guardrails")
        st.markdown(
            """
            <div class="sv-card" style="line-height: 1.6; font-size: 0.88rem;">
                <div style="color: #2563eb; font-weight: 700; margin-bottom: 0.35rem;">
                    1. Automated Decision Support (Not CFSL / NFSU Certified)
                </div>
                <p style="color: #475569; font-size: 0.83rem; margin-bottom: 0.85rem;">
                    EchoCheck AI assists litigation teams and examiners with preliminary case triage.
                    It does not replace certified forensic examiners or accredited government laboratories
                    such as the Central Forensic Science Laboratory (CFSL) or National Forensic Sciences University (NFSU).
                </p>

                <div style="color: #2563eb; font-weight: 700; margin-bottom: 0.35rem;">
                    2. Synthetic Voice Detection vs. Speaker Identity
                </div>
                <p style="color: #475569; font-size: 0.83rem; margin-bottom: 0.85rem;">
                    Synthetic voice detection (Aletheia) evaluates whether speech was AI-generated.
                    Speaker verification (Eidos) evaluates acoustic similarity against a reference voiceprint.
                    These two analyses are separate: a high-quality voice clone can match an enrolled voiceprint.
                </p>

                <div style="color: #2563eb; font-weight: 700; margin-bottom: 0.35rem;">
                    3. Evidence Integrity Hashing (SHA-256)
                </div>
                <p style="color: #475569; font-size: 0.83rem; margin-bottom: 0.85rem;">
                    Every submitted recording is hashed with SHA-256 upon intake. The cryptographic hash
                    is recorded in the case diary to assist with chain-of-custody verification.
                </p>

                <div style="color: #2563eb; font-weight: 700; margin-bottom: 0.35rem;">
                    4. Zero Permanent Audio Retention
                </div>
                <p style="color: #475569; font-size: 0.83rem;">
                    Recordings are processed in-memory for model screening and are never stored permanently
                    on the server or database, respecting confidentiality and evidence retention policies.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def page_user_management() -> None:
    """Admin-only user management page."""
    if not auth.require_permission("manage_users"):
        return
    current_user = auth.get_current_user()

    st.markdown(
        """
        <div class="sv-header">
            <div class="sv-header-title"><span>👥</span> User Management</div>
            <div class="sv-header-subtitle">
                Create, deactivate and manage user accounts within your organisation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    users = db.list_users(org_id=current_user["org_id"])

    # --- User table ---
    st.markdown("### Current Users")
    if users:
        df = pd.DataFrame(users)[["username", "display_name", "role", "is_active", "created_at"]].copy()
        df["is_active"] = df["is_active"].map({1: "Active", 0: "Deactivated"})
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No users found for this organisation.")

    st.markdown("---")

    # --- Deactivate / Reactivate ---
    st.markdown("### Manage Existing Account")
    if users:
        usernames = [u["username"] for u in users if u["user_id"] != current_user["user_id"]]
        if usernames:
            selected_username = st.selectbox("Select user to manage", usernames)
            selected_user = next((u for u in users if u["username"] == selected_username), None)
            if selected_user:
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    new_role = st.selectbox(
                        "Change Role",
                        ["admin", "analyst", "senior_reviewer"],
                        index=["admin", "analyst", "senior_reviewer"].index(selected_user["role"]),
                    )
                with col_b:
                    is_active_val = bool(selected_user["is_active"])
                    deactivate_btn = st.button(
                        "Deactivate Account" if is_active_val else "Reactivate Account",
                        type="secondary",
                    )
                with col_c:
                    save_role_btn = st.button("Save Role Change", type="primary")

                if save_role_btn:
                    db.update_user(selected_user["user_id"], role=new_role)
                    db.log_app_audit_event(
                        action="USER_ROLE_CHANGED",
                        user_id=current_user["user_id"],
                        org_id=current_user["org_id"],
                        metadata={"target_user": selected_username, "new_role": new_role},
                    )
                    st.success(f"Role updated to '{new_role}' for {selected_username}.")
                    st.rerun()

                if deactivate_btn:
                    new_status = not is_active_val
                    db.update_user(selected_user["user_id"], is_active=new_status)
                    db.log_app_audit_event(
                        action="USER_DEACTIVATED" if not new_status else "USER_REACTIVATED",
                        user_id=current_user["user_id"],
                        org_id=current_user["org_id"],
                        metadata={"target_user": selected_username},
                    )
                    st.success(f"Account {'deactivated' if not new_status else 'reactivated'} for {selected_username}.")
                    st.rerun()
        else:
            st.info("No other users to manage (you cannot manage your own account here).")

    st.markdown("---")

    # --- Create new user ---
    st.markdown("### Create New User Account")
    with st.form("create_user_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_username = st.text_input("Username *", placeholder="firstname.lastname")
            new_display = st.text_input("Display Name", placeholder="Dr. Jane Smith")
        with col2:
            new_role = st.selectbox("Role *", ["analyst", "senior_reviewer", "admin"])
            new_pass = st.text_input("Initial Password *", type="password", placeholder="Minimum 8 characters")
        create_btn = st.form_submit_button("Create User", type="primary")

        if create_btn:
            if not new_username.strip() or not new_pass.strip():
                st.error("Username and password are required.", icon="🛑")
            elif len(new_pass) < 8:
                st.error("Password must be at least 8 characters.", icon="🛑")
            elif db.user_exists(new_username.strip().lower()):
                st.error(f"Username '{new_username.strip()}' already exists.", icon="🛑")
            else:
                from auth import hash_password as _hp
                pw_hash = _hp(new_pass.strip())
                uid = db.create_user(
                    username=new_username.strip(),
                    password_hash=pw_hash,
                    role=new_role,
                    org_id=current_user["org_id"],
                    display_name=new_display.strip() or None,
                )
                db.log_app_audit_event(
                    action="USER_CREATED",
                    user_id=current_user["user_id"],
                    org_id=current_user["org_id"],
                    metadata={"new_username": new_username.strip(), "role": new_role},
                )
                st.success(f"✅ User '{new_username.strip()}' ({new_role}) created successfully.")
                st.rerun()


def page_report_review() -> None:
    """Senior Reviewer and Admin page to review submitted reports."""
    current_user = auth.get_current_user()
    if current_user is None:
        return
    if not auth.require_permission("approve_reject_report") and current_user["role"] != "admin":
        return

    st.markdown(
        """
        <div class="sv-header">
            <div class="sv-header-title"><span>📋</span> Report Review</div>
            <div class="sv-header-subtitle">
                Review and approve or reject submitted forensic reports.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # List submitted reports for cases accessible to this user
    accessible_cases = db.list_cases_for_user(
        user_id=current_user["user_id"],
        org_id=current_user["org_id"],
        role=current_user["role"],
        limit=500,
    )
    case_ids = [c["case_id"] for c in accessible_cases]

    # Gather all submitted reports
    submitted_reports = []
    for cid in case_ids:
        for rpt in db.get_reports_for_case(cid):
            if rpt["status"] in ("submitted", "approved", "rejected"):
                case = next((c for c in accessible_cases if c["case_id"] == cid), {})
                rpt["case_reference"] = case.get("case_reference", cid)
                submitted_reports.append(rpt)

    if not submitted_reports:
        st.info("No submitted reports awaiting review for your accessible cases.")
        return

    for rpt in submitted_reports:
        status = rpt["status"]
        icon = {"submitted": "⏳", "approved": "✅", "rejected": "❌"}.get(status, "📋")
        with st.expander(f"{icon} Report {rpt['report_id'][:8]}… · Case: {rpt['case_reference']} · Status: {status.upper()}"):
            # Show case details
            case = db.get_case(rpt["case_id"])
            if case and auth.assert_case_access(case):
                render_result_panel(case)


def page_org_audit_log() -> None:
    """Admin-only organisation-wide audit log."""
    if not auth.require_permission("view_org_audit_log"):
        return
    current_user = auth.get_current_user()

    st.markdown(
        """
        <div class="sv-header">
            <div class="sv-header-title"><span>📜</span> Organisation Audit Log</div>
            <div class="sv-header-subtitle">
                Append-only record of all security-sensitive actions. Read-only.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    events = db.list_app_audit_events(org_id=current_user["org_id"], limit=300)
    if not events:
        st.info("No audit events recorded yet.")
        return

    df = pd.DataFrame(events)
    display_cols = [c for c in ["created_at", "action", "user_id", "case_id", "outcome"] if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)


def main() -> None:
    """Main routing function."""
    # ── Authentication gate ──────────────────────────────────────────────────
    if not auth.require_login():
        st.stop()  # Do not render anything else until logged in
        return

    selected_page = render_sidebar()

    if selected_page == "Evidence Assessment":
        page_assessment_station()
    elif selected_page == "Eidos Speaker Enrollment":
        page_speaker_enrollment()
    elif selected_page == "Forensic Case Registry":
        page_case_registry()
    elif selected_page == "Forensic Systems & API Status":
        page_system_health()
    elif selected_page == "User Management":
        page_user_management()
    elif selected_page == "Report Review":
        page_report_review()
    elif selected_page == "Organisation Audit Log":
        page_org_audit_log()


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            # Running directly via python app.py
            import uvicorn
            import os
            port = int(os.getenv("PORT", "8501"))
            print(f"[EchoCheck AI] Starting Forensic Server on port {port}...")
            uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
        else:
            main()
    except Exception:
        main()
