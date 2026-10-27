import contextlib
import logging
import os
import re
import sys
import time
import traceback
import warnings

# Harmless third-party warnings, by message. Without --verbose they only go to the run log.
KNOWN_WARNINGS = [
    # pyannote, on import: torchcodec's FFmpeg 4-7 libraries don't load next to a newer FFmpeg. Unused here,
    # because all audio is passed to pyannote as in-memory arrays
    r"torchcodec is not installed correctly",
    # transformers, loading the wav2vec2 alignment model
    r"Passing `gradient_checkpointing` to a config initialization is deprecated",
    # torchaudio, called by F5-TTS and pyannote
    r"In 2\.9, this function's implementation will be changed",
]
# Third-party loggers whose info messages only go to the run log; their warnings and errors still show.
# Lightning adds its own console handler to "lightning.pytorch" when imported (after setup), so the logger
# that sends the checkpoint-upgrade message is configured instead, and does not propagate to it
NOISY_LOGGERS = ["whisperx", "lightning.pytorch.utilities.migration.utils",
                 "pytorch_lightning.utilities.migration.utils"]

_state = {"verbose": True, "log": None}


class _Tee:
    """Write to the console and the run log. Progress bar redraws (\\r) only go to the console."""

    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        if "\r" not in text:
            self.log.write(text)
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def __getattr__(self, name):  # isatty, fileno, encoding, ... as the console
        return getattr(self.console, name)


def setup(log_path: str | None = None, verbose: bool = False):
    """
    Route output for one run: the pipeline's own messages, progress bars and real warnings go to the console;
    known harmless warnings, third-party info logs and quiet() output only go to log_path (all of it goes to
    the console with verbose). Returns a function that undoes everything.
    """
    stdout, stderr, showwarning = sys.stdout, sys.stderr, warnings.showwarning
    log = open(log_path, "a", encoding="utf-8", buffering=1) if log_path else None
    if log:
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(sys.argv)}\n")
        sys.stdout, sys.stderr = _Tee(stdout, log), _Tee(stderr, log)
    _state.update(verbose=verbose, log=log)

    known = re.compile("|".join(KNOWN_WARNINGS))

    def show(message, category, filename, lineno, file=None, line=None):
        text = warnings.formatwarning(message, category, filename, lineno, line)
        if verbose or not known.search(str(message)):
            sys.stderr.write(text)
        elif log:
            log.write(text)

    warnings.showwarning = show

    saved_loggers = []
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%H:%M:%S")
    for name in NOISY_LOGGERS:
        logger = logging.getLogger(name)
        saved_loggers.append((logger, logger.handlers[:], logger.level, logger.propagate))
        # adding a handler first also stops WhisperX from installing its own stdout handler
        logger.handlers.clear()
        console = logging.StreamHandler(stderr)
        console.setLevel(logging.INFO if verbose else logging.WARNING)
        console.setFormatter(formatter)
        logger.addHandler(console)
        if log:
            to_log = logging.StreamHandler(log)
            to_log.setFormatter(formatter)
            to_log.setLevel(logging.INFO)
            logger.addHandler(to_log)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    def restore():
        sys.stdout, sys.stderr, warnings.showwarning = stdout, stderr, showwarning
        for logger, handlers, level, propagate in saved_loggers:
            logger.handlers[:] = handlers
            logger.setLevel(level)
            logger.propagate = propagate
        _state.update(verbose=True, log=None)
        if log:
            log.close()

    return restore


@contextlib.contextmanager
def quiet():
    """Send stdout to the run log only (or nowhere), e.g. around libraries that print() their internals."""
    if _state["verbose"]:
        yield
        return
    with contextlib.ExitStack() as stack:
        target = _state["log"] or stack.enter_context(open(os.devnull, "w"))
        stack.enter_context(contextlib.redirect_stdout(target))
        yield


def log_exception():
    """Write the exception being handled to the run log (the console gets it from Python as usual)."""
    if _state["log"]:
        traceback.print_exc(file=_state["log"])
