"""
pdf_report.py - Forensic Audio Evidence PDF Report Generator for EchoCheck AI.

Generates an audit-ready, court-structured PDF report utilizing ReportLab.
Contains:
- EchoCheck AI platform branding and case identification
- Exact SHA-256 evidence integrity hash and file metadata
- Aletheia synthetic voice detection findings
- Eidos reference-speaker verification findings
- Technical format and signal diagnostics
- Chronological case audit events timeline
- Preliminary examiner guidance and legal admissibility disclaimers
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def generate_forensic_pdf(
    case: Dict[str, Any],
    audit_events: Optional[List[Dict[str, Any]]] = None,
) -> bytes:
    """
    Generates a structured, printable PDF forensic report for an evidence case.
    Returns the binary PDF bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=10,
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=8,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#334155"),
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=9.5,
        textColor=colors.HexColor("#64748b"),
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("ECHOCHECK AI — FORENSIC AUDIO EVIDENCE REPORT", title_style))
    story.append(
        Paragraph(
            f"Automated Voice Evidence Workstation & Preliminary Decision Support | Report Generated: "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=8))

    # 2. Case Overview Table
    verdict = (case.get("verdict") or "unavailable").upper()
    v_color = "#b91c1c" if verdict == "FAKE" else ("#15803d" if verdict == "REAL" else "#b45309")

    case_overview_data = [
        [
            Paragraph("<b>Case Identifier:</b>", body_style),
            Paragraph(str(case.get("case_id", "N/A")), body_style),
            Paragraph("<b>Matter Reference:</b>", body_style),
            Paragraph(str(case.get("case_reference", "N/A")), body_style),
        ],
        [
            Paragraph("<b>Dispute Category:</b>", body_style),
            Paragraph(str(case.get("account_action", "General Disputed Audio Evidence")), body_style),
            Paragraph("<b>Alleged Speaker:</b>", body_style),
            Paragraph(str(case.get("customer_identifier") or "Not Specified"), body_style),
        ],
        [
            Paragraph("<b>Case Status:</b>", body_style),
            Paragraph(str(case.get("case_status", "NEW")), body_style),
            Paragraph("<b>Preliminary Verdict:</b>", body_style),
            Paragraph(f"<font color='{v_color}'><b>{verdict}</b></font>", body_style),
        ],
    ]

    t_overview = Table(case_overview_data, colWidths=[90, 180, 90, 180])
    t_overview.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t_overview)
    story.append(Spacer(1, 8))

    # 3. Audio Evidence & Integrity
    story.append(Paragraph("EVIDENCE INTEGRITY & CHAIN OF CUSTODY", section_heading))
    dur_str = f"{case.get('audio_duration_seconds', 0.0):.2f}s" if case.get("audio_duration_seconds") is not None else "N/A"
    size_str = f"{case.get('audio_filesize_bytes', 0):,} bytes" if case.get("audio_filesize_bytes") is not None else "N/A"

    evidence_data = [
        [Paragraph("<b>Audio Filename:</b>", body_style), Paragraph(str(case.get("audio_filename", "N/A")), body_style)],
        [Paragraph("<b>File Size / Duration:</b>", body_style), Paragraph(f"{size_str} | {dur_str}", body_style)],
        [
            Paragraph("<b>SHA-256 Hash:</b>", body_style),
            Paragraph(f"<font face='Courier' size=7>{case.get('sha256_hash') or 'Not Recorded'}</font>", body_style),
        ],
        [Paragraph("<b>Intake Timestamp:</b>", body_style), Paragraph(str(case.get("created_at", "N/A")), body_style)],
    ]
    t_evidence = Table(evidence_data, colWidths=[120, 420])
    t_evidence.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(t_evidence)
    story.append(Spacer(1, 8))

    # 4. Service 1: Aletheia Synthetic Voice Detection
    story.append(
        Paragraph("1. SYNTHETIC VOICE DETECTION — DETECTIFAI ALETHEIA (NO REFERENCE REQUIRED)", section_heading)
    )
    score_str = f"{case['score']:.4f}" if case.get("score") is not None else "N/A"
    thresh_str = f"{case['threshold']:.4f}" if case.get("threshold") is not None else "N/A"
    is_synth_str = "YES (Synthetic)" if case.get("is_synthetic") is True else ("NO (Natural)" if case.get("is_synthetic") is False else "Unclassified")

    aletheia_data = [
        [
            Paragraph("<b>Verdict:</b>", body_style),
            Paragraph(f"<font color='{v_color}'><b>{verdict}</b></font>", body_style),
            Paragraph("<b>Synthetic Score:</b>", body_style),
            Paragraph(score_str, body_style),
        ],
        [
            Paragraph("<b>Decision Threshold:</b>", body_style),
            Paragraph(thresh_str, body_style),
            Paragraph("<b>Synthetic Marker:</b>", body_style),
            Paragraph(is_synth_str, body_style),
        ],
        [
            Paragraph("<b>DetectifAI Request ID:</b>", body_style),
            Paragraph(f"<font face='Courier' size=7>{case.get('request_id') or 'N/A'}</font>", body_style),
            Paragraph("<b>API Status:</b>", body_style),
            Paragraph(str(case.get("api_status", "UNKNOWN")), body_style),
        ],
    ]
    t_aletheia = Table(aletheia_data, colWidths=[110, 160, 110, 160])
    t_aletheia.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(t_aletheia)
    story.append(Spacer(1, 8))

    # 5. Service 2: Eidos Speaker Verification
    story.append(Paragraph("2. SPEAKER VERIFICATION — DETECTIFAI EIDOS (ENROLLED VOICE REQUIRED)", section_heading))
    spk_id = case.get("speaker_id")
    spk_dec = (case.get("speaker_verify_decision") or "NOT_PERFORMED").upper()
    dec_color = "#15803d" if spk_dec == "MATCH" else ("#b91c1c" if spk_dec == "NO_MATCH" else "#64748b")
    spk_sim = f"{case['speaker_similarity']:.4f}" if case.get("speaker_similarity") is not None else "N/A"
    spk_thresh = f"{case['speaker_threshold']:.4f}" if case.get("speaker_threshold") is not None else "N/A"

    eidos_data = [
        [
            Paragraph("<b>Reference Speaker:</b>", body_style),
            Paragraph(spk_id or "None Selected (Verification Bypassed)", body_style),
            Paragraph("<b>Verification Decision:</b>", body_style),
            Paragraph(f"<font color='{dec_color}'><b>{spk_dec}</b></font>", body_style),
        ],
        [
            Paragraph("<b>Similarity Index:</b>", body_style),
            Paragraph(f"{spk_sim} (Scale: -1.0 to +1.0; not a percentage)", body_style),
            Paragraph("<b>Decision Threshold:</b>", body_style),
            Paragraph(spk_thresh, body_style),
        ],
        [
            Paragraph("<b>Eidos Request ID:</b>", body_style),
            Paragraph(f"<font face='Courier' size=7>{case.get('speaker_verify_request_id') or 'N/A'}</font>", body_style),
            Paragraph("<b>Finding Note:</b>", body_style),
            Paragraph("A speaker match does NOT prove authenticity (synthetic clones can match).", body_style),
        ],
    ]
    t_eidos = Table(eidos_data, colWidths=[110, 160, 110, 160])
    t_eidos.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(t_eidos)
    story.append(Spacer(1, 8))

    # 6. Technical Signal Metrics
    story.append(Paragraph("3. TECHNICAL AUDIO FORMAT & SIGNAL METRICS", section_heading))
    it_dict = {}
    if case.get("audio_integrity_json"):
        try:
            it_dict = json.loads(case["audio_integrity_json"])
        except Exception:
            pass

    signal_data = [
        [
            Paragraph("<b>Container Format:</b>", body_style),
            Paragraph(str(it_dict.get("container_format", "N/A")), body_style),
            Paragraph("<b>Sampling Rate:</b>", body_style),
            Paragraph(f"{it_dict.get('sample_rate_hz', 'N/A')} Hz", body_style),
        ],
        [
            Paragraph("<b>Channels / Layout:</b>", body_style),
            Paragraph(f"{it_dict.get('channels', 'N/A')} ({it_dict.get('channel_layout', 'N/A')})", body_style),
            Paragraph("<b>Clipping Ratio:</b>", body_style),
            Paragraph(f"{it_dict.get('clipping_ratio_percent', 'N/A')}%", body_style),
        ],
        [
            Paragraph("<b>Silence Ratio:</b>", body_style),
            Paragraph(f"{it_dict.get('silence_ratio_percent', 'N/A')}%", body_style),
            Paragraph("<b>Peak / RMS Level:</b>", body_style),
            Paragraph(f"{it_dict.get('peak_amplitude_dbfs', 'N/A')} dBFS / {it_dict.get('rms_level_dbfs', 'N/A')} dBFS", body_style),
        ],
    ]
    t_signal = Table(signal_data, colWidths=[110, 160, 110, 160])
    t_signal.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(t_signal)
    story.append(Spacer(1, 8))

    # 7. Audit Events Timeline (if available)
    if audit_events:
        story.append(Paragraph("4. CHAIN OF CUSTODY & AUDIT EVENT TIMELINE", section_heading))
        audit_rows = [[Paragraph("<b>Timestamp (UTC)</b>", body_style), Paragraph("<b>Event</b>", body_style), Paragraph("<b>Details</b>", body_style)]]
        for ev in audit_events[:8]:  # show up to 8 events in PDF
            audit_rows.append(
                [
                    Paragraph(f"<font size=6.5>{ev.get('created_at', '')[:19]}</font>", body_style),
                    Paragraph(f"<b>{ev.get('event_type', '')}</b>", body_style),
                    Paragraph(ev.get("event_description", ""), body_style),
                ]
            )
        t_audit = Table(audit_rows, colWidths=[90, 110, 340])
        t_audit.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        )
        story.append(t_audit)
        story.append(Spacer(1, 8))

    # 8. Guidance & Examiner Determination
    story.append(Paragraph("5. PRELIMINARY ASSESSMENT GUIDANCE & EXAMINER NOTES", section_heading))
    guidance_data = [
        [
            Paragraph("<b>Assessment Guidance:</b>", body_style),
            Paragraph(f"<i>\"{case.get('recommendation_text') or 'Pending further review.'}\"</i>", body_style),
        ],
        [
            Paragraph("<b>Examiner Assessment:</b>", body_style),
            Paragraph(case.get("analyst_decision") or "Pending Formal Examination", body_style),
        ],
        [
            Paragraph("<b>Case Notes:</b>", body_style),
            Paragraph(case.get("analyst_notes") or "None documented", body_style),
        ],
    ]
    t_guidance = Table(guidance_data, colWidths=[120, 420])
    t_guidance.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(t_guidance)
    story.append(Spacer(1, 10))

    # 9. Legal & Forensic Admissibility Disclaimer
    disclaimer_text = (
        "<b>LEGAL & FORENSIC ADMISSIBILITY LIMITATIONS (SECTION 45 / 65B NOTICE):</b><br/>"
        "1. This automated screening report is provided by EchoCheck AI for preliminary decision support, investigative triage, and legal discovery screening only.<br/>"
        "2. It does NOT constitute a certified forensic expert opinion under Section 45 / 65B of evidentiary rules, nor does it replace formal scientific analysis by a Central Forensic Science Laboratory (CFSL), National Forensic Sciences University (NFSU), or court-appointed acoustic experts.<br/>"
        "3. Aletheia detects synthetic voice patterns only; Eidos evaluates acoustic similarity against reference voiceprints. Audio splicing, analog tape editing, out-of-context audio insertion, and true biological identity remain unverified."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
