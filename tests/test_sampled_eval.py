"""scripts/analysis/sampled_eval.py: a quick evaluation decoded with sampling (R24b, amendment A13)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import sampled_eval as se  # noqa: E402


def test_sampling_seed_depends_on_the_run_seed_and_on_the_prompt_only():
    a = se.sampling_seed(0, [5, 6, 7])
    assert a == se.sampling_seed(0, [5, 6, 7])
    assert a != se.sampling_seed(1, [5, 6, 7])
    assert a != se.sampling_seed(0, [5, 6, 8])
    assert a != se.sampling_seed(0, [56, 7])  # the ids are kept apart
    assert 0 <= a < 2 ** 48


def test_sampling_seed_reads_a_tensor_like_prompt_as_its_list():
    class Tensorish:
        def tolist(self):
            return [5, 6, 7]

    assert se.sampling_seed(3, Tensorish()) == se.sampling_seed(3, [5, 6, 7])


def test_sampled_hands_temperature_top_p_and_one_seed_per_prompt_to_the_generator():
    seen = {}

    def fake(model, prompts, prompt_signals, signal_fns, max_new_tokens, eos_ids, **kw):
        seen.update(kw, prompts=prompts, positional=(model, prompt_signals, signal_fns, max_new_tokens, eos_ids))
        return ["gen"] * len(prompts)

    call = se.sampled(fake, temperature=0.6, top_p=0.95, base=2)
    out = call("m", [[1, 2], [3]], ["s1", "s2"], ["f1", "f2"], [8, 9], {0}, prompt_harness=None)
    assert out == ["gen", "gen"]
    assert seen["temperature"] == 0.6 and seen["top_p"] == 0.95
    assert seen["seed"] == [se.sampling_seed(2, [1, 2]), se.sampling_seed(2, [3])]
    assert seen["prompt_harness"] is None
    assert seen["positional"] == ("m", ["s1", "s2"], ["f1", "f2"], [8, 9], {0})


def test_sampled_gives_the_same_seed_to_the_same_prompt_in_any_batch_position():
    seeds = []

    def fake(model, prompts, prompt_signals, signal_fns, max_new_tokens, eos_ids, **kw):
        seeds.append(dict(zip(map(tuple, prompts), kw["seed"])))
        return [None] * len(prompts)

    call = se.sampled(fake, temperature=0.6, top_p=0.95, base=0)
    call("m", [[1, 2], [3]], [0, 0], [0, 0], 4, set())
    call("m", [[3], [9], [1, 2]], [0, 0, 0], [0, 0, 0], 4, set())
    assert seeds[0][(1, 2)] == seeds[1][(1, 2)] and seeds[0][(3,)] == seeds[1][(3,)]


def test_main_refuses_batch_size_one_because_that_path_does_not_go_through_the_batched_generator(tmp_path):
    with pytest.raises(SystemExit):
        se.main(["--run", str(tmp_path), "--data", str(tmp_path), "--out", str(tmp_path / "o"), "--sampling-seed", "0",
                 "--batch-size", "1"])
