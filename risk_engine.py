"""
risk_engine.py - Forensic Decision Support and Assessment Engine for SentinelVoice Forensics.

Translates DetectifAI AI-generated voice detection signals into structured,
explainable, and legally grounded preliminary evidence assessments.

Forensic & Legal Compliance Guardrails:
- Automated decision support only; does NOT replace CFSL, NFSU, or certified forensic examiners.
- Does NOT claim biometric speaker identity verification (who is speaking).
- Does NOT claim tamper/splicing detection (which requires waveform/spectrogram laboratory examination).
- A detection verdict is a probabilistic model signal, not a definitive legal determination of admissibility or fraud.
- Treat unknown, inconclusive, and failed API runs as requiring further examination; never treat failure as verification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from detectifai import DetectionResult


@dataclass
class RiskEvaluation:
    """Structured, explainable preliminary forensic evaluation package."""

    verdict: str  # "real", "fake", "not_sure", "unavailable"
    risk_level: str  # "LOW", "HIGH", "AMBIGUOUS", "ELEVATED_UNVERIFIED"
    recommendation_code: str
    recommendation_text: str
    headline: str
    rationale: str
    what_api_detected: str
    what_it_indicates: str
    what_remains_unverified: str
    action_items: List[str]
    disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


LEGAL_FORENSIC_DISCLAIMER = (
    "LEGAL & FORENSIC ADMISSIBILITY NOTICE: EchoCheck AI provides an automated preliminary "
    "acoustic screening to assist legal practitioners, forensic investigators, and compliance teams. "
    "This preliminary assessment does NOT constitute a certified forensic expert opinion under Section 45 / 65B "
    "or equivalent evidentiary laws, nor does it replace formal laboratory testing by a Central Forensic Science "
    "Laboratory (CFSL), National Forensic Sciences University (NFSU), or court-appointed acoustic experts. "
    "Speaker identity verification and physical audio splicing/tamper detection were NOT performed."
)


def evaluate_case_risk(
    detection: DetectionResult,
    account_action: str = "General Disputed Audio Evidence",
    customer_identifier: Optional[str] = None,
) -> RiskEvaluation:
    """
    Computes a forensic evidence assessment based on DetectifAI deepfake detection signals
    and disputed evidence context.

    Note: The parameter `account_action` represents the evidence/dispute category,
    preserved for backward-compatible interface and database calls.
    """
    verdict = detection.verdict.lower() if detection.verdict else "unavailable"
    dispute_category = account_action or "General Disputed Audio Evidence"

    # 1. API Failure or Analysis Unavailable
    if not detection.success or verdict == "unavailable":
        error_info = f" ({detection.error_message})" if detection.error_message else ""
        return RiskEvaluation(
            verdict="unavailable",
            risk_level="ELEVATED_UNVERIFIED",
            recommendation_code="ANALYSIS_UNAVAILABLE_FURTHER_EXAMINATION_REQUIRED",
            headline="Preliminary Analysis Unavailable — Further Examination Required",
            recommendation_text="Analysis unavailable; never treat this as a successful verification.",
            rationale=(
                f"Automated acoustic analysis could not be completed by DetectifAI{error_info}. "
                "In forensic evidence handling, an uncompleted analysis must never be construed as evidence "
                "of authenticity or absence of synthetic manipulation."
            ),
            what_api_detected="No acoustic classification could be completed due to API unavailability or invalid payload.",
            what_it_indicates="Technical error, connection restriction, or unsupported audio stream precluded automated analysis.",
            what_remains_unverified="All voice authenticity characteristics, synthetic markers, and speaker identity remain unverified.",
            action_items=[
                "Verify network route and DetectifAI API credentials.",
                "Ensure recording meets minimum duration (> 4.0 seconds) and valid audio format (PCM WAV/MP3).",
                "Log technical exception in forensic case notes with timestamp.",
                "Refer recording directly to a forensic audio laboratory for manual acoustic examination.",
            ],
            disclaimer=LEGAL_FORENSIC_DISCLAIMER,
        )

    # 2. Verdict: FAKE (Synthetic voice / AI clone indicators detected)
    if verdict == "fake":
        score_desc = f" (Synthetic probability score: {detection.score:.4f})" if detection.score is not None else ""
        thresh_desc = f" vs model threshold {detection.threshold:.4f}" if detection.threshold is not None else ""

        rationale = (
            f"The acoustic evaluation model flagged synthetic voice indicators{score_desc}{thresh_desc}. "
            f"Acoustic features exhibit statistical patterns consistent with neural text-to-speech (TTS), "
            f"voice conversion, or AI voice cloning technology in the context of: {dispute_category}."
        )

        return RiskEvaluation(
            verdict="fake",
            risk_level="HIGH",
            recommendation_code="ESCALATE_SYNTHETIC_EXAMINATION",
            headline="Synthetic Voice Indicators Detected — Formal Forensic Review Mandated",
            recommendation_text="Flag suspected synthetic audio and escalate for additional verification.",
            rationale=rationale,
            what_api_detected=f"Acoustic classification identified synthetic/AI-generated voice patterns{score_desc}.",
            what_it_indicates=(
                "Disputed recording demonstrates high probability of artificial voice synthesis or voice cloning. "
                "May support allegations of fabricated or cloned telephonic evidence."
            ),
            what_remains_unverified=(
                "Speaker identity is unverified; exact generative tool/architecture is unconfirmed; "
                "physical carrier medium and analog tampering/splicing remain unexamined."
            ),
            action_items=[
                "Document synthetic score, threshold, and Request ID in forensic case diary.",
                "Maintain strict chain of custody and compute SHA-256 hash of original master file.",
                "Recommend formal laboratory examination by CFSL, NFSU, or a certified acoustic examiner.",
                "Request original acquisition device and raw uncompressed source file for metadata audit.",
                "Flag evidence for legal counsel as contested audio requiring expert testimony.",
            ],
            disclaimer=LEGAL_FORENSIC_DISCLAIMER,
        )

    # 3. Verdict: NOT_SURE (Inconclusive acoustic features)
    if verdict == "not_sure":
        score_info = f"Score: {detection.score:.4f}" if detection.score is not None else "Borderline acoustic metrics"
        dur_info = f", Duration: {detection.duration_seconds:.1f}s" if detection.duration_seconds is not None else ""

        rationale = (
            f"The acoustic analysis yielded an inconclusive result ({score_info}{dur_info}). "
            "Ambient background noise, lossy compression artifacts, short phrase length, or acoustic reverberation "
            "prevented the neural model from reaching a confident synthetic or genuine classification."
        )

        return RiskEvaluation(
            verdict="not_sure",
            risk_level="AMBIGUOUS",
            recommendation_code="REQUEST_SUITABLE_RECORDING_OR_LAB_EXAMINATION",
            headline="Inconclusive Acoustic Features — Original Master Audio Required",
            recommendation_text="Request a suitable recording or independent verification.",
            rationale=rationale,
            what_api_detected="The model found ambiguous acoustic cues that cross the uncertainty threshold.",
            what_it_indicates="The audio cannot be reliably classified as synthetic or natural with current signal quality.",
            what_remains_unverified="Voice authenticity, presence of subtle voice conversion, and speaker identity remain unverified.",
            action_items=[
                "Seek the original, uncompressed master audio recording from the recording device or service provider.",
                "Ensure recording duration exceeds 10 seconds of clear speech for higher analytical confidence.",
                "Check for heavy compression (e.g. repeated WhatsApp/telephony transcodings).",
                "Refer sample to a forensic lab for spectrographic noise filtering and formant analysis.",
            ],
            disclaimer=LEGAL_FORENSIC_DISCLAIMER,
        )

    # 4. Verdict: REAL (Natural human voice characteristics detected)
    if verdict == "real":
        score_info = f" (Synthetic probability score: {detection.score:.4f})" if detection.score is not None else ""

        rationale = (
            f"The voice recording was classified as real/natural speech according to the model's synthetic detection criteria{score_info}. "
            "No automated synthetic voice or neural speech synthesis markers were detected."
        )

        return RiskEvaluation(
            verdict="real",
            risk_level="LOW",
            recommendation_code="NATURAL_SPEECH_FURTHER_EXAMINATION_RECOMMENDED",
            headline="Natural Speech Markers Detected — Further Examination Recommended",
            recommendation_text="Authenticity check passed; continue normal identity verification.",
            rationale=rationale,
            what_api_detected=f"Model classified the recording as natural speech without synthetic speech markers{score_info}.",
            what_it_indicates=(
                "The audio does not display characteristic AI voice-cloning or text-to-speech acoustic signatures. "
                "However, this finding is limited to synthetic voice detection only."
            ),
            what_remains_unverified=(
                "CRITICAL LIMITATION: A 'real' verdict does NOT prove the recording is genuine, unedited, or authentic in whole. "
                "It does NOT verify the speaker's true legal identity, nor does it rule out analog tape editing, "
                "audio splicing, out-of-context audio insertion, or deepfake video lip-sync."
            ),
            action_items=[
                "Record preliminary natural speech finding in case record.",
                "Conduct independent speaker comparison if the dispute centers on speaker identity.",
                "Conduct temporal waveform and background noise continuity checks to rule out audio splicing.",
                "Preserve file hash and transmission chain of custody for court submission.",
            ],
            disclaimer=LEGAL_FORENSIC_DISCLAIMER,
        )

    # Fallback for unexpected verdict values
    return RiskEvaluation(
        verdict=verdict,
        risk_level="AMBIGUOUS",
        recommendation_code="MANUAL_EXAMINATION_REQUIRED",
        headline=f"Unclassified Signal ({verdict}) — Expert Examination Required",
        recommendation_text="Request a suitable recording or independent verification.",
        rationale=f"API returned an unclassified verdict string '{verdict}'. Manual expert review is mandated.",
        what_api_detected=f"Unclassified response verdict '{verdict}'.",
        what_it_indicates="Non-standard classification returned by API.",
        what_remains_unverified="All voice authenticity and identity attributes remain unverified.",
        action_items=[
            "Conduct full manual inspection of the incident.",
            "Verify evidence through independent forensic laboratory channels.",
        ],
        disclaimer=LEGAL_FORENSIC_DISCLAIMER,
    )


@dataclass
class SpeakerEvaluation:
    """Structured, explainable evaluation of Eidos speaker verification."""

    decision: str  # "match", "no_match", "unavailable"
    similarity: Optional[float]
    threshold: Optional[float]
    headline: str
    what_api_detected: str
    what_it_indicates: str
    what_remains_unverified: str
    action_items: List[str]
    disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_speaker_verification(
    verify_result: Any,  # SpeakerVerifyResult
    speaker_id: str,
    speaker_display_name: Optional[str] = None,
) -> SpeakerEvaluation:
    """
    Produces an explainable, forensic interpretation of DetectifAI Eidos speaker verification.
    Keeps speaker similarity separate from synthetic voice detection.
    """
    label_spk = f"'{speaker_display_name}' ({speaker_id})" if speaker_display_name else f"'{speaker_id}'"
    disclaimer = (
        "SPEAKER VERIFICATION LIMITATION: Biometric speaker verification evaluates acoustic alignment "
        "between two voice samples. A 'match' does NOT constitute legal proof of a person's identity, presence, "
        "or authorization. A high-quality AI voice clone or recording replay can match an enrolled voiceprint. "
        "Similarly, a 'no_match' may arise from microphone differences, compression, illness, or background noise. "
        "Similarity is an index from -1.0 to 1.0 (not a percentage)."
    )

    if not getattr(verify_result, "success", False) or not getattr(verify_result, "decision", None):
        err = getattr(verify_result, "error_message", "Unknown verification error")
        return SpeakerEvaluation(
            decision="unavailable",
            similarity=None,
            threshold=None,
            headline="Speaker Verification Unavailable",
            what_api_detected=f"Verification could not be completed against reference speaker {label_spk}. Diagnostics: {err}",
            what_it_indicates="The comparison between disputed audio and enrolled reference was not performed.",
            what_remains_unverified="Speaker identity alignment remains completely unexamined.",
            action_items=[
                "Confirm the reference speaker is enrolled in DetectifAI Eidos with valid audio samples.",
                "Ensure disputed audio meets duration (3-60s) and speech presence (>= 2s) requirements.",
                "Verify API key has 'speakers:verify' scope enabled.",
                "Consult a certified forensic acoustic examiner for manual spectrographic voice comparison.",
            ],
            disclaimer=disclaimer,
        )

    decision = verify_result.decision.lower()
    sim = verify_result.similarity
    thresh = verify_result.threshold
    sim_str = f"{sim:.4f}" if sim is not None else "N/A"
    thresh_str = f"{thresh:.4f}" if thresh is not None else "N/A"

    if decision == "match":
        return SpeakerEvaluation(
            decision="match",
            similarity=sim,
            threshold=thresh,
            headline=f"Acoustic Speaker Match Indication (Similarity: {sim_str} vs Threshold: {thresh_str})",
            what_api_detected=(
                f"Eidos model returned decision 'match' with acoustic similarity index {sim_str} "
                f"exceeding the decision threshold of {thresh_str}."
            ),
            what_it_indicates=(
                f"The disputed voice sample exhibits close vocal tract and prosodic alignment with "
                f"the enrolled reference voiceprint for speaker {label_spk}."
            ),
            what_remains_unverified=(
                "CRITICAL FORENSIC LIMITATION: An acoustic match does NOT establish that the speaker was physically present, "
                "nor does it prove the recording was unedited. If synthetic voice detection flags the audio as 'fake', "
                "the recording may represent an AI-generated clone of this specific enrolled individual."
            ),
            action_items=[
                "Cross-examine finding with the synthetic voice detection result.",
                "Review reference sample provenance to verify how the enrolled audio was obtained.",
                "Submit both recordings to a forensic science laboratory (CFSL/NFSU) for acoustic formant analysis.",
                "Seek corroborating independent evidence (call logs, cell tower data, witness statements).",
            ],
            disclaimer=disclaimer,
        )
    else:  # "no_match"
        return SpeakerEvaluation(
            decision="no_match",
            similarity=sim,
            threshold=thresh,
            headline=f"Acoustic Non-Match Indication (Similarity: {sim_str} vs Threshold: {thresh_str})",
            what_api_detected=(
                f"Eidos model returned decision 'no_match' with acoustic similarity index {sim_str} "
                f"falling below the decision threshold of {thresh_str}."
            ),
            what_it_indicates=(
                f"The disputed recording differs significantly in vocal tract and acoustic parameters "
                f"from the enrolled reference voiceprint for speaker {label_spk}."
            ),
            what_remains_unverified=(
                "A non-match does not definitively exclude the enrolled individual. Channel mismatch "
                "(e.g. low-bitrate cell phone vs studio microphone), severe compression, illness (laryngitis), "
                "extreme emotional arousal, or intentional voice disguise can lower similarity scores."
            ),
            action_items=[
                "Verify whether the reference recording and disputed recording share similar acoustic conditions.",
                "Acquire additional reference exemplars from multiple recording sessions (up to 5 samples supported).",
                "Refer to forensic acoustic expert for spectrographic noise reduction and formant tracking.",
            ],
            disclaimer=disclaimer,
        )

