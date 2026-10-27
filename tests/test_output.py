import logging
import sys
import warnings

import pytest

from utils import output


@pytest.fixture
def run_log(tmp_path):
    path = tmp_path / "run.log"
    restore = []
    yield path, lambda verbose=False: restore.append(output.setup(str(path), verbose=verbose))
    for undo in restore:
        undo()


def test_known_noise_goes_to_the_log_only(run_log, capsys):
    path, setup = run_log
    stdout = sys.stdout
    setup()
    print("▶️ our own line")
    warnings.warn("torchcodec is not installed correctly so built-in audio decoding will fail.", UserWarning)
    warnings.warn("something new and important", UserWarning)
    logging.getLogger("whisperx.asr").info("Detected language: ja (0.93)")
    logging.getLogger("whisperx.asr").warning("Audio is shorter than 30s")
    import lightning.pytorch  # noqa: F401  adds its own console handler on import
    logging.getLogger("lightning.pytorch.utilities.migration.utils").info("Lightning automatically upgraded")
    with output.quiet():
        print("gen_text 0 Hello")
    sys.stderr.write("\r 50%|█████     |")

    console = capsys.readouterr()
    log = path.read_text()
    assert "our own line" in console.out and "our own line" in log
    assert "torchcodec" not in console.err and "torchcodec" in log
    assert "something new and important" in console.err and "something new and important" in log
    assert "Detected language" not in console.out + console.err and "Detected language" in log
    assert "Audio is shorter than 30s" in console.err and "Audio is shorter than 30s" in log
    assert "Lightning automatically upgraded" not in console.err and "Lightning automatically upgraded" in log
    assert "gen_text" not in console.out and "gen_text" in log
    assert "50%" in console.err and "50%" not in log            # progress bars stay out of the log
    assert sys.stdout is not stdout


def test_verbose_shows_everything_and_restore_undoes_setup(run_log, capsys):
    path, setup = run_log
    stdout, showwarning = sys.stdout, warnings.showwarning
    handlers = logging.getLogger("whisperx").handlers[:]
    restore = output.setup(str(path), verbose=True)
    warnings.warn("torchcodec is not installed correctly", UserWarning)
    logging.getLogger("whisperx").info("Performing voice activity detection")
    with output.quiet():
        print("vocab : vocab.txt")
    restore()
    console = capsys.readouterr()
    assert "torchcodec" in console.err and "Performing voice activity detection" in console.err
    assert "vocab" in console.out
    assert sys.stdout is stdout and warnings.showwarning is showwarning
    assert logging.getLogger("whisperx").handlers == handlers
