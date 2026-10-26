import os

import pytest

from utils.pipeline import Pipeline, Stage


class Context:
    def __init__(self, folder):
        self.folder = folder
        self.ran = []
        self.fail = None


def make_stages(per_translator_from=None):
    def stage(name, per_translator=False):
        def run(ctx):
            if ctx.fail == name:
                raise RuntimeError(f"{name} failed")
            ctx.ran.append(name)
            open(os.path.join(ctx.folder, name), "w").close()
        return Stage(name, name, run, lambda ctx: [os.path.join(ctx.folder, name)], [name.upper()], per_translator)
    names = ["a", "b", "c", "d"]
    return [stage(n, per_translator_from is not None and i >= names.index(per_translator_from))
            for i, n in enumerate(names)]


@pytest.fixture
def ctx(tmp_path):
    return Context(str(tmp_path))


def run(ctx, stages=None, config=None, translator="t", **kw):
    Pipeline(stages or make_stages(), ctx.folder, translator).run(ctx, config or {}, inputs_id="video", **kw)


def test_second_run_skips_everything(ctx):
    run(ctx)
    run(ctx)
    assert ctx.ran == ["a", "b", "c", "d"]


def test_resumes_from_the_failed_stage(ctx):
    ctx.fail = "c"
    with pytest.raises(RuntimeError):
        run(ctx)
    ctx.fail = None
    run(ctx)
    assert ctx.ran == ["a", "b", "c", "d"]


def test_changed_setting_reruns_that_stage_and_later_ones(ctx):
    run(ctx, config={"B": 1})
    run(ctx, config={"B": 2})
    assert ctx.ran == ["a", "b", "c", "d", "b", "c", "d"]


def test_missing_output_reruns_the_stage(ctx):
    run(ctx)
    os.remove(os.path.join(ctx.folder, "c"))
    run(ctx)
    assert ctx.ran[4:] == ["c", "d"]


def test_from_stage_to_stage_and_force(ctx):
    run(ctx, to_stage="b")
    assert ctx.ran == ["a", "b"]
    run(ctx, from_stage="b")
    assert ctx.ran[2:] == ["b", "c", "d"]
    run(ctx, force=True)
    assert ctx.ran[5:] == ["a", "b", "c", "d"]


def test_changed_input_reruns_everything(ctx):
    run(ctx)
    Pipeline(make_stages(), ctx.folder, "t").run(ctx, {}, inputs_id="other video")
    assert ctx.ran[4:] == ["a", "b", "c", "d"]


def test_per_translator_stages_are_kept_separately(ctx):
    stages = make_stages(per_translator_from="c")
    run(ctx, stages, translator="gpt")
    run(ctx, stages, translator="google")
    run(ctx, stages, translator="gpt")
    assert ctx.ran == ["a", "b", "c", "d", "c", "d"]
