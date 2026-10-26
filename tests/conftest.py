import os
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: full pipeline with real models (set RUN_E2E=1)")


@pytest.fixture
def tone(tmp_path):
    """Write a sine tone with optional silent padding and return its path."""
    def make(seconds, name="tone.wav", sr=24000, pad=0.0, amplitude=0.3, freq=220):
        t = np.arange(int(seconds * sr)) / sr
        wav = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        silence = np.zeros(int(pad * sr), np.float32)
        path = tmp_path / name
        sf.write(path, np.concatenate([silence, wav, silence]), sr)
        return str(path)
    return make
