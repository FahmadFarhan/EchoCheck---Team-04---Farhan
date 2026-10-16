"""
create_sample_audio.py - Generate test audio recordings for SentinelVoice evaluation.

Creates sample WAV audio files using Python standard library (wave, math, struct)
with zero external dependencies.
"""

import math
import struct
import wave
from pathlib import Path


def generate_wav(
    filename: str,
    duration_seconds: float = 6.5,
    sample_rate: int = 16000,
    frequency: float = 220.0,
) -> str:
    """Generates a clean PCM WAV audio recording."""
    output_path = Path(filename)
    num_samples = int(duration_seconds * sample_rate)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        frames = bytearray()
        for i in range(num_samples):
            t = float(i) / sample_rate
            envelope = min(1.0, t * 2.0) * min(1.0, (duration_seconds - t) * 2.0)
            # Voice-like fundamental frequency modulation
            f0 = frequency + 20.0 * math.sin(2.0 * math.pi * 2.0 * t)
            value = int(32767.0 * 0.5 * envelope * math.sin(2.0 * math.pi * f0 * t))
            frames.extend(struct.pack("<h", value))

        wav_file.writeframes(frames)

    print(f"Generated sample audio: {output_path.resolve()} ({output_path.stat().st_size} bytes, {duration_seconds}s)")
    return str(output_path.resolve())


if __name__ == "__main__":
    import os

    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)
    # 6.5s sample for deepfake & verification testing
    generate_wav(str(samples_dir / "sample_account_recovery_call.wav"), duration_seconds=6.5, frequency=220.0)
    generate_wav(str(samples_dir / "sample_disputed_call.wav"), duration_seconds=6.5, frequency=220.0)
    # 18.0s sample for Eidos speaker enrollment (requirement: 15-60s)
    generate_wav(str(samples_dir / "sample_speaker_enrollment_reference.wav"), duration_seconds=18.0, frequency=210.0)

