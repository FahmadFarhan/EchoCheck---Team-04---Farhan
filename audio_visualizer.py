"""
audio_visualizer.py - Genuine Waveform and Spectrogram Generation for EchoCheck AI.

Provides deterministic, reproducible audio waveform and spectrogram generation
from actual uploaded audio bytes. Decodes real PCM samples, preserves authentic
time and frequency axes, performs intelligent downsampling for rendering
efficiency, and supports both backend image rendering and frontend data schemas.
"""

from __future__ import annotations

import io
import os
import wave
from typing import Any, Dict, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for server execution
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import soundfile as sf


def decode_audio_samples(audio_bytes: bytes, filename: str) -> Tuple[np.ndarray, int, float]:
    """
    Decodes actual audio bytes into a 1D float32 numpy array normalized to [-1.0, 1.0],
    along with sample rate and exact duration in seconds.

    Parameters:
        audio_bytes: Raw bytes of the audio recording.
        filename: Original filename (used for container extension hints).

    Returns:
        (samples, sample_rate, duration_seconds)

    Raises:
        ValueError: If audio cannot be decoded or is empty.
    """
    if not audio_bytes or len(audio_bytes) == 0:
        raise ValueError("Audio recording is empty or corrupt.")

    # 1. Attempt decoding with soundfile (supports WAV, FLAC, OGG, etc.)
    try:
        data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if data is None or len(data) == 0:
            raise ValueError("Decoded audio contains 0 samples.")

        # Convert multi-channel to mono by averaging channels
        if data.ndim == 2:
            mono = np.mean(data, axis=1)
        else:
            mono = data

        duration = float(len(mono)) / float(sample_rate) if sample_rate > 0 else 0.0
        return mono.astype(np.float32), int(sample_rate), duration
    except Exception as sf_err:
        # 2. Fall back to Python standard library wave module for RIFF WAV files
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                num_frames = wf.getnframes()

                if num_frames == 0 or sample_rate == 0:
                    raise ValueError("WAV file has 0 frames or invalid sample rate.")

                raw = wf.readframes(num_frames)
                if sample_width == 1:
                    # 8-bit unsigned PCM
                    raw_arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                    norm = (raw_arr - 128.0) / 128.0
                elif sample_width == 2:
                    # 16-bit signed PCM
                    norm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                elif sample_width == 3:
                    # 24-bit PCM
                    raw_bytes = bytearray(raw)
                    # Pad to 32-bit
                    padded = bytearray()
                    for idx in range(0, len(raw_bytes), 3):
                        padded.extend(b"\x00" + raw_bytes[idx : idx + 3])
                    norm = np.frombuffer(padded, dtype=np.int32).astype(np.float32) / 2147483648.0
                elif sample_width == 4:
                    # 32-bit signed PCM
                    norm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    raise ValueError(f"Unsupported sample width: {sample_width} bytes")

                if channels > 1:
                    norm = norm.reshape(-1, channels)
                    norm = np.mean(norm, axis=1)

                duration = float(len(norm)) / float(sample_rate)
                return norm.astype(np.float32), int(sample_rate), duration
        except Exception as wave_err:
            raise ValueError(f"Unable to decode audio format for '{filename}': {sf_err}; {wave_err}")


def generate_waveform_plot(
    audio_bytes: bytes,
    filename: str,
    target_points: int = 1500,
) -> bytes:
    """
    Generates a genuine waveform plot image (PNG bytes) plotting amplitude vs time (seconds).
    Uses real samples, preserves duration, handles mono/stereo, and performs min/max
    envelope downsampling for long recordings without fabricating data.
    """
    samples, sample_rate, duration = decode_audio_samples(audio_bytes, filename)
    num_samples = len(samples)

    fig, ax = plt.subplots(figsize=(12, 3.7), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    if num_samples <= target_points:
        t = np.linspace(0.0, duration, num_samples)
        ax.plot(t, samples, color="#1d4ed8", linewidth=1.2, alpha=0.95)
        ax.fill_between(t, samples, color="#3b82f6", alpha=0.3)
    else:
        # Min-Max envelope downsampling: groups samples into bins and preserves authentic peaks
        num_bins = target_points
        bin_size = num_samples // num_bins
        trimmed = samples[: num_bins * bin_size].reshape(num_bins, bin_size)
        mins = np.min(trimmed, axis=1)
        maxs = np.max(trimmed, axis=1)
        t_bins = np.linspace(0.0, duration, num_bins)

        ax.fill_between(t_bins, mins, maxs, color="#1d4ed8", alpha=0.75, linewidth=0.8)

    ax.set_xlim(0.0, max(0.1, duration))
    max_amp = float(np.max(np.abs(samples))) if num_samples > 0 else 0.0
    y_lim = max(0.15, min(1.05, max_amp * 1.15))
    ax.set_ylim(-y_lim, y_lim)

    ax.set_title(
        f"Acoustic Waveform — {filename} ({duration:.2f}s | {sample_rate:,} Hz | Decoded PCM)",
        fontsize=11,
        fontweight="bold",
        color="#0f172a",
        pad=10,
    )
    ax.set_xlabel("Time (seconds)", fontsize=9, fontweight="bold", color="#334155")
    ax.set_ylabel("Amplitude", fontsize=9, fontweight="bold", color="#334155")
    ax.tick_params(colors="#334155", labelsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.5, color="#94a3b8")
    ax.axhline(0, color="#64748b", linewidth=1.0, linestyle="-")

    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(1.0)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_spectrogram_plot(
    audio_bytes: bytes,
    filename: str,
    nperseg: int = 512,
    noverlap: int = 256,
) -> bytes:
    """
    Generates a genuine spectrogram plot image (PNG bytes) plotting frequency (Hz)
    vs time (seconds) with power spectral density (dB/Hz) colormap.
    Height: ~310-330px on desktop with high-contrast dark palette.
    """
    samples, sample_rate, duration = decode_audio_samples(audio_bytes, filename)

    # Ensure segment size is appropriate for short recordings
    actual_nperseg = min(nperseg, len(samples))
    actual_noverlap = min(noverlap, actual_nperseg // 2)

    frequencies, times, Sxx = scipy.signal.spectrogram(
        samples,
        fs=sample_rate,
        window="hann",
        nperseg=actual_nperseg,
        noverlap=actual_noverlap,
        scaling="density",
    )

    # Convert power density to decibels
    Sxx_db = 10.0 * np.log10(np.maximum(Sxx, 1e-12))

    fig, ax = plt.subplots(figsize=(12, 3.2), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#0b0f19")

    # Time vs Frequency with high-contrast viridis colormap
    pcm = ax.pcolormesh(
        times,
        frequencies,
        Sxx_db,
        shading="gouraud",
        cmap="viridis",
    )

    ax.set_xlim(0.0, max(0.1, duration))
    # Cap vertical frequency at Nyquist or 8000 Hz for speech clarity
    max_freq = min(sample_rate / 2.0, 8000.0) if sample_rate >= 8000 else sample_rate / 2.0
    ax.set_ylim(0.0, max_freq)

    ax.set_title(
        f"Acoustic Spectrogram (STFT) — {filename} (0–{int(max_freq):,} Hz)",
        fontsize=11,
        fontweight="bold",
        color="#0f172a",
        pad=10,
    )
    ax.set_xlabel("Time (seconds)", fontsize=9, fontweight="bold", color="#334155")
    ax.set_ylabel("Frequency (Hz)", fontsize=9, fontweight="bold", color="#334155")
    ax.tick_params(colors="#334155", labelsize=8.5)

    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label("Power Spectral Density (dB/Hz)", fontsize=8, fontweight="bold", color="#334155")
    cbar.ax.tick_params(labelsize=7.5, colors="#334155")
    cbar.outline.set_edgecolor("#cbd5e1")

    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(1.0)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def extract_waveform_data_series(
    audio_bytes: bytes,
    filename: str,
    target_points: int = 800,
) -> Dict[str, Any]:
    """
    Extracts structured waveform time and amplitude series data for API and frontend consumption.
    Used by Dhvani's external frontend to render client-side interactive charts.
    """
    samples, sample_rate, duration = decode_audio_samples(audio_bytes, filename)
    num_samples = len(samples)

    if num_samples <= target_points:
        times = np.linspace(0.0, duration, num_samples).round(4).tolist()
        amplitudes = np.round(samples, 4).tolist()
    else:
        num_bins = target_points
        bin_size = num_samples // num_bins
        trimmed = samples[: num_bins * bin_size].reshape(num_bins, bin_size)
        mins = np.min(trimmed, axis=1)
        maxs = np.max(trimmed, axis=1)
        # Interleave min and max for faithful waveform envelope
        interleaved = np.empty(num_bins * 2, dtype=np.float32)
        interleaved[0::2] = mins
        interleaved[1::2] = maxs
        times = np.linspace(0.0, duration, num_bins * 2).round(4).tolist()
        amplitudes = np.round(interleaved, 4).tolist()

    return {
        "filename": filename,
        "sample_rate": sample_rate,
        "duration_seconds": round(duration, 3),
        "total_samples": num_samples,
        "data_points": len(times),
        "time_axis_seconds": times,
        "amplitude_values": amplitudes,
    }
