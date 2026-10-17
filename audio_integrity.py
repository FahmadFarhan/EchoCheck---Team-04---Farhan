"""
audio_integrity.py - Defensible Audio Format & Basic Signal Integrity Inspection.

Performs deterministic, non-destructive acoustic signal inspection using standard Python
and NumPy libraries.

Forensic Guardrails:
- Does NOT claim splice detection, tamper detection, or proof of editing.
- Acoustic properties (clipping, silence, compression) are natural acoustic artifacts
  and do not constitute proof of malicious tampering.
- Keeps results separate from DetectifAI deepfake verdict and Eidos speaker verification.
"""

from __future__ import annotations

import io
import math
import wave
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class AudioIntegrityReport:
    """Structured report of technical audio container and signal metrics."""

    container_format: str
    filesize_bytes: int
    duration_seconds: float
    sample_rate_hz: Optional[int] = None
    channels: Optional[int] = None
    bit_depth: Optional[int] = None
    channel_layout: Optional[str] = None  # "Mono", "Stereo", "Multi-channel"
    peak_amplitude_dbfs: Optional[float] = None
    rms_level_dbfs: Optional[float] = None
    crest_factor_db: Optional[float] = None  # Peak-to-RMS ratio
    clipping_ratio_percent: Optional[float] = None  # % of samples near max amplitude
    silence_ratio_percent: Optional[float] = None  # % of frames < -45 dBFS
    telephony_standard_rate: bool = False  # e.g. 8000 Hz or 16000 Hz
    notes: str = ""
    limitations: str = (
        "INTEGRITY CHECK LIMITATION: These measurements reflect the physical file format "
        "and acoustic signal properties of the submitted audio. They do NOT detect audio "
        "splicing, AI generation, or editing. Signal clipping or silent intervals occur "
        "naturally in unedited field recordings and do not prove tampering."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def inspect_audio_file(audio_bytes: bytes, filename: str) -> AudioIntegrityReport:
    """
    Analyzes an audio file's technical headers and (if PCM WAV) acoustic signal characteristics.
    """
    filesize = len(audio_bytes)
    fn_lower = filename.lower()

    # Detect container format from magic bytes or extension
    container = "UNKNOWN"
    if audio_bytes.startswith(b"RIFF") and len(audio_bytes) > 12 and audio_bytes[8:12] == b"WAVE":
        container = "WAV (RIFF Header)"
    elif audio_bytes.startswith(b"ID3") or audio_bytes.startswith(b"\xff\xfb") or audio_bytes.startswith(b"\xff\xf3"):
        container = "MP3 (MPEG Audio)"
    elif audio_bytes.startswith(b"fLaC"):
        container = "FLAC (Free Lossless Audio Codec)"
    elif audio_bytes.startswith(b"OggS"):
        container = "OGG (Vorbis / Opus Container)"
    elif b"ftypM4A" in audio_bytes[:32] or b"ftypisom" in audio_bytes[:32] or fn_lower.endswith(".m4a"):
        container = "M4A / AAC (MPEG-4 Container)"
    elif fn_lower.endswith(".wav"):
        container = "WAV"
    elif fn_lower.endswith(".mp3"):
        container = "MP3"
    else:
        container = fn_lower.split(".")[-1].upper() if "." in fn_lower else "OCTET-STREAM"

    # If it is a WAV file, we can perform detailed PCM sample analysis using standard wave module
    if "WAV" in container:
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                num_frames = wf.getnframes()
                duration = float(num_frames) / float(sample_rate) if sample_rate > 0 else 0.0
                bit_depth = sample_width * 8

                channel_layout = "Mono" if channels == 1 else ("Stereo" if channels == 2 else f"{channels} Channels")
                is_telephony = sample_rate in (8000, 16000)

                # Read raw samples for signal inspection if 16-bit PCM
                raw_frames = wf.readframes(min(num_frames, sample_rate * 60))  # Inspect up to 60s
                
                peak_dbfs = None
                rms_dbfs = None
                crest_factor = None
                clipping_pct = None
                silence_pct = None
                notes_list = []

                if sample_width == 2 and len(raw_frames) > 0:
                    samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0

                    if len(samples) > 0:
                        max_abs = np.max(np.abs(samples))
                        peak_dbfs = float(20.0 * np.log10(max_abs)) if max_abs > 1e-6 else -100.0

                        rms = np.sqrt(np.mean(samples ** 2))
                        rms_dbfs = float(20.0 * np.log10(rms)) if rms > 1e-6 else -100.0

                        if peak_dbfs is not None and rms_dbfs is not None and peak_dbfs > -90.0:
                            crest_factor = float(peak_dbfs - rms_dbfs)

                        # Clipping: samples reaching >= 0.995 of full scale
                        clipped_count = np.sum(np.abs(samples) >= 0.995)
                        clipping_pct = float((clipped_count / len(samples)) * 100.0)

                        if clipping_pct > 0.5:
                            notes_list.append(f"Acoustic signal contains {clipping_pct:.2f}% saturated/clipped samples (microphone gain or digital limit).")

                        # Silence ratio: frames with RMS < -45 dBFS
                        frame_size = int(sample_rate * 0.05)  # 50ms frames
                        if frame_size > 0 and len(samples) >= frame_size:
                            num_chunks = len(samples) // frame_size
                            chunks = samples[:num_chunks * frame_size].reshape(num_chunks, frame_size)
                            chunk_rms = np.sqrt(np.mean(chunks ** 2, axis=1))
                            silent_chunks = np.sum(chunk_rms < 0.0056)  # Approx -45 dBFS
                            silence_pct = float((silent_chunks / num_chunks) * 100.0)

                if is_telephony:
                    notes_list.append(f"Sampling rate ({sample_rate} Hz) is consistent with telephony/narrowband voice transmission.")
                elif sample_rate >= 44100:
                    notes_list.append(f"Sampling rate ({sample_rate} Hz) indicates high-fidelity or studio audio.")

                notes_str = " ".join(notes_list) if notes_list else "Standard PCM audio format with no severe saturation anomalies."

                return AudioIntegrityReport(
                    container_format=container,
                    filesize_bytes=filesize,
                    duration_seconds=duration,
                    sample_rate_hz=sample_rate,
                    channels=channels,
                    bit_depth=bit_depth,
                    channel_layout=channel_layout,
                    peak_amplitude_dbfs=peak_dbfs,
                    rms_level_dbfs=rms_dbfs,
                    crest_factor_db=crest_factor,
                    clipping_ratio_percent=clipping_pct,
                    silence_ratio_percent=silence_pct,
                    telephony_standard_rate=is_telephony,
                    notes=notes_str,
                )
        except Exception as e:
            # Fall back to generic inspection if wave decoding fails
            pass

    # Generic fallback for non-WAV or compressed files
    approx_duration = max(1.0, float(filesize) / 32000.0)  # rough byte approximation if unavailable
    return AudioIntegrityReport(
        container_format=container,
        filesize_bytes=filesize,
        duration_seconds=approx_duration,
        notes=f"Container detected as {container}. Detailed PCM acoustic waveform inspection requires uncompressed WAV format.",
    )
