import random

import numpy as np
import soundfile as sf
import torch

import utils.separate as sep


class FakeSeparator:
    """Like Demucs with shifts: the result depends on Python's global random."""
    samplerate = 44100
    created = []

    def __init__(self, **options):
        self.options = options
        FakeSeparator.created.append(options)

    def separate_audio_file(self, path):
        offset = random.randint(0, 22050) if self.options["shifts"] else 0
        noise = torch.full((2, 4410), offset / 22050.0 * 0.1)
        return None, {"vocals": noise, "drums": noise * 0.5, "other": noise * 0.25}


def test_same_seed_gives_the_same_separation(monkeypatch, tmp_path):
    monkeypatch.setattr(sep.demucs.api, "Separator", FakeSeparator)
    monkeypatch.setattr(sep.demucs.api, "save_audio",
                        lambda wav, path, samplerate, clip: sf.write(path, wav.T.numpy(), samplerate))

    def vocals(seed, shifts=1, state=0):
        random.seed(state)                      # whatever the process did before must not matter
        path = tmp_path / f"v{seed}.wav"
        sep.separate_vocals("in.wav", str(path), str(tmp_path / "bg.wav"), seed=seed, shifts=shifts)
        return sf.read(path)[0]

    assert np.array_equal(vocals(0, state=1), vocals(0, state=2))
    assert not np.array_equal(vocals(0), vocals(1))
    assert FakeSeparator.created[-1]["shifts"] == 1
    vocals(0, shifts=0)
    assert FakeSeparator.created[-1]["shifts"] == 0
