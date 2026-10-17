"""
test_audio_visualizer.py - Automated validation of waveform and spectrogram generation.
Tests 3+ distinct audio fixtures:
1. sample_account_recovery_call.wav (6.5s)
2. sample_speaker_enrollment_reference.wav (18.0s)
3. silence_fixture.wav (3.0s pure silence)
4. modulated_tone.wav (4.0s high frequency chirp)
5. Empty / corrupt file error handling
"""

import math
import struct
import wave
from pathlib import Path

import numpy as np
import audio_visualizer as av


def create_silence_wav(filename: str, duration: float = 3.0, sample_rate: int = 16000) -> str:
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00" * int(duration * sample_rate * 2))
    return filename


def create_chirp_wav(filename: str, duration: float = 4.0, sample_rate: int = 16000) -> str:
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        num_samples = int(duration * sample_rate)
        frames = bytearray()
        for i in range(num_samples):
            t = i / sample_rate
            # Linear chirp from 400Hz to 2400Hz
            freq = 400.0 + (2000.0 * (t / duration))
            val = int(32767.0 * 0.7 * math.sin(2.0 * math.pi * freq * t))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(frames)
    return filename


def run_tests():
    samples_dir = Path("samples")
    f1 = samples_dir / "sample_account_recovery_call.wav"
    f2 = samples_dir / "sample_speaker_enrollment_reference.wav"

    assert f1.exists(), f"Sample 1 missing: {f1}"
    assert f2.exists(), f"Sample 2 missing: {f2}"

    # Create additional distinct test fixtures
    f3_silence = samples_dir / "test_silence_fixture.wav"
    f4_chirp = samples_dir / "test_chirp_fixture.wav"
    create_silence_wav(str(f3_silence), duration=3.0)
    create_chirp_wav(str(f4_chirp), duration=4.0)

    print("--- 1. Testing Sample 1 (sample_account_recovery_call.wav) ---")
    b1 = f1.read_bytes()
    samples1, sr1, dur1 = av.decode_audio_samples(b1, f1.name)
    print(f"Decoded: {len(samples1)} samples, sr={sr1}, duration={dur1:.3f}s")
    assert abs(dur1 - 6.5) < 0.1, f"Expected ~6.5s, got {dur1}"
    assert np.max(np.abs(samples1)) > 0.05, "Sample should contain active signal"
    wf1_img = av.generate_waveform_plot(b1, f1.name)
    spec1_img = av.generate_spectrogram_plot(b1, f1.name)
    assert len(wf1_img) > 1000, "Waveform PNG should be valid image bytes"
    assert len(spec1_img) > 1000, "Spectrogram PNG should be valid image bytes"

    print("--- 2. Testing Sample 2 (sample_speaker_enrollment_reference.wav) ---")
    b2 = f2.read_bytes()
    samples2, sr2, dur2 = av.decode_audio_samples(b2, f2.name)
    print(f"Decoded: {len(samples2)} samples, sr={sr2}, duration={dur2:.3f}s")
    assert abs(dur2 - 18.0) < 0.2, f"Expected ~18.0s, got {dur2}"
    assert dur2 != dur1, "Duration must differ between sample 1 and 2"
    wf2_img = av.generate_waveform_plot(b2, f2.name)
    spec2_img = av.generate_spectrogram_plot(b2, f2.name)
    assert wf1_img != wf2_img, "Waveform image bytes must differ between different files"

    print("--- 3. Testing Sample 3 (Pure Silence Fixture) ---")
    b3 = f3_silence.read_bytes()
    samples3, sr3, dur3 = av.decode_audio_samples(b3, f3_silence.name)
    print(f"Decoded: {len(samples3)} samples, sr={sr3}, duration={dur3:.3f}s")
    assert abs(dur3 - 3.0) < 0.1, f"Expected 3.0s, got {dur3}"
    max_silence = np.max(np.abs(samples3))
    print(f"Max absolute amplitude in silence: {max_silence}")
    assert max_silence == 0.0, "Silence fixture must have exactly near-zero / zero amplitude"
    wf3_img = av.generate_waveform_plot(b3, f3_silence.name)
    assert len(wf3_img) > 1000

    print("--- 4. Testing Sample 4 (Chirp Fixture) ---")
    b4 = f4_chirp.read_bytes()
    samples4, sr4, dur4 = av.decode_audio_samples(b4, f4_chirp.name)
    assert abs(dur4 - 4.0) < 0.1, f"Expected 4.0s, got {dur4}"
    wf4_img = av.generate_waveform_plot(b4, f4_chirp.name)
    spec4_img = av.generate_spectrogram_plot(b4, f4_chirp.name)
    assert wf4_img != wf1_img
    assert spec4_img != spec1_img

    print("--- 5. Testing Structured Waveform Data Series for Frontend ---")
    series_data = av.extract_waveform_data_series(b1, f1.name, target_points=500)
    assert "time_axis_seconds" in series_data
    assert "amplitude_values" in series_data
    assert len(series_data["time_axis_seconds"]) == len(series_data["amplitude_values"])
    assert series_data["duration_seconds"] > 0
    print(f"Series points returned: {series_data['data_points']}")

    print("--- 6. Testing Error Handling for Empty / Malformed Inputs ---")
    try:
        av.decode_audio_samples(b"", "empty.wav")
        assert False, "Should have raised ValueError on empty bytes"
    except ValueError as e:
        print("Empty bytes safely caught:", e)

    try:
        av.decode_audio_samples(b"RIFFcorruptdataheader", "corrupt.wav")
        assert False, "Should have raised ValueError on corrupt bytes"
    except ValueError as e:
        print("Corrupt bytes safely caught:", e)

    print("\nALL AUDIO VISUALIZER VALIDATION CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_tests()
