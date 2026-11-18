"""R24b (preregistration amendment A13): a quick evaluation decoded with sampling.

The standard evaluation (`prio.eval.run`: same test episodes, same classifier, same summary files),
channel on only, with the batched generator called at a temperature and a top-p instead of
greedily. Nothing under `src/prio` is changed: for this process the evaluation's reference to
`generate_batch` is replaced by a wrapper that adds `temperature`, `top_p` and one seed per
sequence. The seed of a generation comes from the run's sampling seed and its prompt tokens
(sha256), so it does not depend on the batch order or on a resume; runs that share a forced prefix
(an event run and its quiet twin) draw the same random numbers.

uv run python scripts/analysis/sampled_eval.py --run runs/n6v-wide-levels-s0 --data data/n6v \
    --out results/r24b-varied-t06-s0-b16 --sampling-seed 0
    [--temperature 0.6 --top-p 0.95 --every 4 --batch-size 16]

GPU, about 5 minutes; resumable like every evaluation (`records.partial.jsonl`). `eval_meta.json`
records the decoding. Never mix decodings or batch sizes in one results folder. Reading:
`scripts/analysis/wording_reading.py <folder> <data>` per folder.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path


def sampling_seed(base: int, prompt) -> int:
    """Seed of one generation: from the run's sampling seed and the prompt's token ids."""
    ids = prompt.tolist() if hasattr(prompt, "tolist") else list(prompt)
    text = f"{int(base)}:" + ",".join(str(int(t)) for t in ids)
    return int(hashlib.sha256(text.encode("ascii")).hexdigest()[:12], 16)


def sampled(generate_batch, temperature: float, top_p: float, base: int):
    """`generate_batch` with the decoding fixed: sampling at `temperature` / `top_p`, one seed per prompt."""

    def call(model, prompts, prompt_signals, signal_fns, max_new_tokens, eos_ids, **kw):
        seeds = [sampling_seed(base, p) for p in prompts]
        return generate_batch(model, prompts, prompt_signals, signal_fns, max_new_tokens, eos_ids,
                              temperature=temperature, top_p=top_p, seed=seeds, **kw)

    return call


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sampling-seed", type=int, required=True, help="seed of this draw (A13: 0, 1, 2)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--every", type=int, default=4, help="every n-th test episode (4 = the quick evaluation)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0, help="the evaluation's own seed (trajectories of the twins)")
    parser.add_argument("--model", default=None, help="defaults to the model recorded in the run's train_log.json")
    args = parser.parse_args(argv)
    if args.batch_size < 2:
        parser.error("batch size 1 does not go through the batched generator: use 2 or more")
    if args.temperature <= 0.0:
        parser.error("a temperature of 0 is greedy decoding: use prio.eval.run")

    import prio.eval.run as er
    from prio.datagen.episodes import read_jsonl
    from prio.signals import signal_set
    from prio.train import channel_record, load_run

    model_id = args.model or json.loads((args.run / "train_log.json").read_text(encoding="utf-8"))["model"]
    decoding = {"temperature": args.temperature, "top_p": args.top_p, "sampling_seed": args.sampling_seed,
                "seed_rule": "sha256(sampling seed, prompt token ids)"}
    er.start_note(args.out, {"run": str(args.run), "data": str(args.data), "model": model_id,
                             "channel": channel_record(args.run), "decoding": decoding,
                             "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}},
                  started=time.strftime("%Y-%m-%d %H:%M"))
    episodes = read_jsonl(args.data / "test.jsonl")
    specs = signal_set(episodes[0].signals.shape[1])
    model, tok, ms = load_run(model_id, args.run, len(specs))
    tok.padding_side = "left"
    er.generate_batch = sampled(er.generate_batch, args.temperature, args.top_p, args.sampling_seed)
    records = er.evaluate(model, tok, ms, specs, episodes, ("on",), seed=args.seed,
                          out_path=args.out / "records.partial.jsonl", every=args.every, batch_size=args.batch_size)
    er.write_summary(records, None, args.out)
    er.finish_note(args.out, finished=time.strftime("%Y-%m-%d %H:%M"))
    print((args.out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
