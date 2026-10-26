import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Stage:
    name: str
    description: str
    run: Callable[[object], None]
    # files/folders the stage writes; it only counts as done if they all exist
    outputs: Callable[[object], list[str]]
    # settings that change the stage's result; changing one re-runs this stage and everything after it
    settings: list[str] = field(default_factory=list)
    # stages from translation on keep separate results per translator
    per_translator: bool = False


class Pipeline:
    """
    Run stages in order and remember each finished stage in <work_dir>/state.json, so a later run skips it.
    A stage is skipped when it finished before with the same settings, on the same upstream result, and its
    outputs still exist. A failed or interrupted stage is never marked finished, so the next run resumes there.
    Re-running a stage gives it a new result id, which makes every later stage run again too.
    """

    def __init__(self, stages: list[Stage], work_dir: str, translator: str):
        self.stages = stages
        self.state_path = os.path.join(work_dir, "state.json")
        self.translator = translator
        os.makedirs(work_dir, exist_ok=True)
        self.state = {}
        if os.path.exists(self.state_path):
            with open(self.state_path, encoding="utf-8") as f:
                self.state = json.load(f)

    def names(self) -> list[str]:
        return [s.name for s in self.stages]

    def run(self, context, config: dict, from_stage: str | None = None, to_stage: str | None = None,
            force: bool = False, inputs_id: str = ""):
        names = self.names()
        for name in (from_stage, to_stage):
            if name and name not in names:
                raise ValueError(f"Unknown stage '{name}'. Stages: {', '.join(names)}")
        start = names.index(from_stage) if from_stage else 0
        stop = names.index(to_stage) if to_stage else len(names) - 1

        upstream = inputs_id   # result id of the previous stage (for the first stage: the input video)
        for i, stage in enumerate(self.stages[:stop + 1]):
            key = self.key(stage)
            fingerprint = self.fingerprint(stage, config, upstream)
            entry = self.state.get(key)
            done = (entry is not None and entry["fingerprint"] == fingerprint
                    and all(os.path.exists(p) for p in stage.outputs(context)))
            requested = force or (from_stage is not None and i >= start)
            if done and not requested:
                print(f"⏭️  [{i + 1}/{len(names)}] {stage.name}: up to date, skipping")
                upstream = entry["result"]
                continue

            print(f"▶️  [{i + 1}/{len(names)}] {stage.name}: {stage.description}")
            started = time.time()
            self.state.pop(key, None)
            self._save()
            stage.run(context)
            upstream = uuid.uuid4().hex
            self.state[key] = {"fingerprint": fingerprint, "result": upstream,
                               "seconds": round(time.time() - started, 1),
                               "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
            self._save()

    def key(self, stage: Stage) -> str:
        return f"{stage.name}:{self.translator}" if stage.per_translator else stage.name

    def fingerprint(self, stage: Stage, config: dict, upstream: str) -> str:
        values = {name: config.get(name) for name in stage.settings}
        payload = json.dumps({"settings": values, "upstream": upstream}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _save(self):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        os.replace(tmp, self.state_path)


def file_id(path: str) -> str:
    """Identify an input file by path, size and modification time, so replacing it re-runs everything."""
    stat = os.stat(path)
    return f"{os.path.abspath(path)}:{stat.st_size}:{int(stat.st_mtime)}"
