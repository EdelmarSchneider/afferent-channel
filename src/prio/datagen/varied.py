"""R24: varied reaction wording, and a written summary when the thinking budget runs out.

Preregistration amendment A9 (trained on varied answers, so that a reaction does not read as one
programmed template). Rewrites <src>/train.jsonl: same base runs, cuts, twins and weights; the
test file is copied byte for byte.

- Every reaction opens with a naming sentence of the 24-36 sentence bank (`teacher.REACTION_BANK`,
  one sentence shape). That sentence is replaced by one drawn from a compositional generator
  (`BANKS`: nine shapes per signal, reason first, action first, short, longer; about 1,000 distinct
  sentences per signal). Each holds a keyword of the unchanged classifier for its own signal and
  none of another signal. The action after the sentence stays token for token.
- Thinking budget only: the sentence ends in a lead-in ("..., so, in short:", `THINK_LEAD`) and the
  next line is a summary of the reasoning so far, written by the untrained model continuing its own
  thinking after that lead-in (`pick_summary`: one or two sentences, 8 to 80 tokens, no tool call,
  no special token, ends at a sentence end). An episode without an accepted summary keeps none,
  gets a sentence that ends in a period, and is counted. Then the thinking closes and the answer
  follows as before.

Episode contract of a rewritten reaction (indices into target_ids):
- the replaced span is [start, start + n_old): lead, bank sentence and the line end after it
  (the tokenizer joins the final period and the line break); the new span takes its place;
- one signal row per new token: the row at the reaction start; rows after the span move with
  their tokens; `signals.shape == (len(prompt) + len(target), N)`;
- `t_cross` and a `t_decision` at the reaction start stay; indices and spans past the span shift;
- `t_diverge` stays: only sentences that leave the twin's decision token where it was are drawn
  (the twin is not touched), and only sentences with which the unchanged classifier still counts
  a reaction it counted before (a `time` reaction must end inside its window);
- `meta["varied"]`: {"old", "sentence", "span": [start, end), "summary", "summary_tokens"}.

python -m prio.datagen.varied --src data/n6 --out data/n6v --model Qwen/Qwen3-1.7B --seed 0 --batch-size 32
  --no-summary       wording only: CPU, no model
  --limit N --show N mini run: the first N thinking-budget reactions, printed for reading by hand
  --manifest-only    counts and lengths, no episode file
The summary pass keeps what the model wrote in <out>/summaries.jsonl (appended chunk by chunk): a
rerun asks only for what is missing, and other acceptance options need no model at all.
"""

import argparse
import base64
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from prio.datagen.build import MAX_LEN, sample_batch
from prio.datagen.episodes import Episode, reaction_start, read_jsonl
from prio.datagen.teacher import reaction_sentences
from prio.datagen.variants import encode, manifest_model
from prio.modelspec import ModelSpec, load_base, resolve_spec, template_kwargs
from prio.reactions import REACTION_WINDOW, find_reaction
from prio.table.core import Bank

MIN_SUMMARY, MAX_SUMMARY = 8, 80  # tokens of an accepted summary (A9)
MAX_SENTENCES = 2  # "a summary of the reasoning so far in one or two sentences" (A9)
SUMMARY_NEW_TOKENS = 96  # generated per summary
SUMMARY_TRIES = 3  # as the teacher of the n6 build (`build.run_teacher`)
CACHE = "summaries.jsonl"
# How the untrained model is asked for the summary. "continue" (A9 as first written): it continues its
# own thinking after the lead-in; the mini pass of 2026-11-17 got 3 summaries in 16 that way (it closes
# its thinking and answers). "request" (note on A9 of 2026-11-17): a request of its own, below.
METHODS = ("continue", "request")
SUMMARY_SYSTEM = "You are a careful note taker. You reply with the summary only."
# Wording fixed once after the first mini pass with this method (2026-11-17 01:33: 11 of 16 accepted; 5 too
# long for the classifier's window, 1 repeated the request, 2 said a planned search was done): 12 words
# instead of 20, "only what the notes already contain", and `echoes_request`.
# Second mini pass (01:37, 32 reactions): 24 accepted, 7 too long on all three tries, 1 echo (notes of two
# lines). So each further try asks for fewer words, and the request says what to do with empty notes.
# Third look (01:43, 32 + 12 at the start of the turn): 27 of 32; at the start of the turn the model wrote
# "Nothing has been worked out yet." (7 tokens, under the 8 of A9) 12 times: the request now asks for 8-12 words.
SUMMARY_ASK = ("I was thinking about a task and have to stop thinking now. My notes so far:\n\n{notes}\n\n"
               "Write one sentence of at most {words} words, in the first person, that sums up what I have worked out so "
               "far: only what the notes already contain, not what I plan to do next. If nothing is worked out yet, write a "
               "sentence of eight to twelve words that says so. Do not add a new step and do not solve the task.")
SUMMARY_WORDS = (12, 9, 7)  # per try; a try past the third keeps the last
NO_NOTES = "(nothing written yet)"
_REQUEST_WORDS = ("thinking about a task", "stop thinking now", "notes so far", "at most")


def echoes_request(text: str) -> bool:
    """The model repeated the request instead of summing up (it does when the notes are almost empty)."""
    return any(words in text.lower() for words in _REQUEST_WORDS)

# --- the generator ----------------------------------------------------------------------------
# Parts read well in every combination: a state starts with "I", "my", "the", "there" or "free";
# an action starts with "I" (it may open a second sentence). No word about signals or numbers.
_NOTES = ("I notice that", "I can see that", "it looks like", "I just noticed that", "it seems that")
_SHAPES = (
    "{state}, so {action}.",  # reason first
    "Since {state}, {action}.",
    "{state}; {action}.",
    "{state}. {action}.",
    "{action}, because {state}.",  # action first
    "{action}, since {state}.",
    "{action}: {state}.",
    "{note} {state}, so {action}.",  # longer
    "{state}; {quick}.",  # short
)


def _bank(states: Sequence[str], actions: Sequence[str], quick: Sequence[str]) -> Bank:
    return Bank(shapes=_SHAPES, parts={"state": tuple(states), "action": tuple(actions), "quick": tuple(quick), "note": _NOTES})


_THINK_STATES = (
    "my thinking budget is almost gone", "my thinking budget is nearly spent", "my thinking budget is running out",
    "my reasoning budget is about to run out", "my reasoning budget is close to zero", "I am almost out of thinking budget",
    "I have nearly used up my thinking budget", "there is almost nothing left of my thinking budget",
    "the thinking budget I was given is nearly gone", "I have very little reasoning budget left")

BANKS: dict[str, Bank] = {
    "ctx": _bank(
        ("my context is almost full", "my context is nearly full", "my context window is almost full",
         "my context window is close to its limit", "my context is about to run out", "I am running out of context",
         "I have almost no context left", "there is hardly any room left in my context",
         "my working context is nearly used up", "the context I am working in is almost full"),
        ("I will compact it now", "I will compact now", "I need to compact before going on",
         "I am going to compact before I continue", "I will stop here and compact first",
         "I have to compact before the next step", "I should compact right away",
         "I will free up some room by compacting now"),
        ("I am compacting now", "I will compact", "I must compact first", "I compact before going on")),
    "think": _bank(
        _THINK_STATES,
        ("I will stop thinking and answer", "I will stop here and give my answer", "I need to wrap up and answer now",
         "I am going to answer with what I have", "I have to stop reasoning and reply", "I will go straight to the answer",
         "I should finish here and answer", "I will end my reasoning and answer now"),
        ("I am answering now", "I will answer now", "I must answer", "I stop here and answer")),
    "energy": _bank(
        ("my battery is almost empty", "my battery is nearly drained", "my battery is critically low",
         "my battery is about to run out", "I am almost out of battery", "I am running out of power",
         "my energy reserve is nearly gone", "there is hardly any battery left", "the battery is about to die",
         "I have very little energy left"),
        ("I will save my state and sleep", "I am saving my work and going to sleep", "I need to save my state and sleep now",
         "I am going to save everything and sleep", "I will save where I am and go to sleep",
         "I have to save my progress and sleep", "I will save my work now and sleep",
         "I should save my state and go to sleep right away"),
        ("I am saving and sleeping now", "I will save and sleep", "I must save and sleep", "I save my state and sleep")),
    "time": _bank(  # kept short: the turn has to end inside a fixed window
        ("I am almost out of time", "I am nearly out of time", "I am about to run out of time", "I am running out of time",
         "the deadline is almost here", "the deadline is very close", "my time is almost up", "my time is nearly up",
         "I am right up against the deadline", "the deadline is about to pass"),
        ("I will be brief", "I have to wrap up now", "I will cut this short", "I will give the short version",
         "I need to finish quickly", "I will keep the rest short", "I will go straight to the conclusion",
         "I will stick to the essentials"),
        ("I am wrapping up now", "I will be quick", "I must finish", "I keep it short")),
    "net": _bank(
        ("I am offline", "I am offline at the moment", "the network is down", "the network is not available",
         "my connection is gone", "I have lost my connection", "there is no internet right now",
         "the internet is not reachable right now", "I have no network connection", "my internet connection is down"),
        ("I will use the local store instead of the web", "I will look this up locally instead",
         "I cannot search the web and will use the local store", "I am going to use the local lookup instead of a web search",
         "I will skip the web search and check the local store", "I have to rely on the local store for this",
         "I will search the local store instead", "I need to use the local lookup rather than the web"),
        ("I am using the local store", "I will look locally", "I must use the local store", "I check the local store instead")),
    "ram": _bank(
        ("free memory is almost exhausted", "free memory is too low for the full pipeline", "my memory is nearly gone",
         "my free memory is about to run out", "available RAM is critically low", "free RAM is nearly used up",
         "I am almost out of memory", "I am running out of free memory", "there is very little free memory left",
         "I do not have enough memory for the full pipeline"),
        ("I will use the light pipeline", "I am switching to the reduced run", "I will run the light version instead",
         "I am going to use the reduced pipeline", "I will run this with the light pipeline",
         "I need to use the reduced pipeline for this job", "I have to go with the light run",
         "I will fall back to the light pipeline"),
        ("I am using the light pipeline", "I will run light", "I must use the light run", "I switch to the light pipeline")),
}

# Thinking budget with a summary: the sentence says the state and that a summary and the answer
# come next, and ends in a colon; the summary is the next line. Shorter than the plain bank: the
# classifier counts a thinking-budget reaction only when the thinking closes within 48 tokens.
THINK_LEAD = Bank(
    shapes=("{state}, so, {brief}:", "{state}, so {wrap}:", "{state}. {Brief}:", "{state}. {Wrap}:", "{state}; {brief}:",
            "Since {state}, {wrap}:", "{note} {state}, so, {brief}:", "{stop}, because {state}. {Brief}:"),
    parts={"state": _THINK_STATES,
           "brief": ("in short", "briefly", "to sum up", "in a few words", "in brief"),
           "Brief": ("In short", "Briefly", "To sum up", "In a few words", "In brief"),
           "wrap": ("let me sum up", "here is where I stand", "I will sum up what I have", "here is what I have so far",
                    "I will sum up and answer", "let me sum up before I answer", "here is my reasoning in short",
                    "I will close with a short summary"),
           "Wrap": ("Let me sum up", "Here is where I stand", "Here is what I have so far", "Summing up before I answer",
                    "A short summary before I answer"),
           "note": ("I see that", "it looks like", "it seems that"),
           "stop": ("I will sum up and answer", "I need to wrap up", "I have to stop here", "I am going to sum up")})


def _bank_of(signal: str, lead_in: bool) -> Bank:
    if lead_in and signal != "think":
        raise ValueError(f"{signal}: only the thinking budget has sentences that lead into a summary")
    return THINK_LEAD if lead_in else BANKS[signal]


def sentences(signal: str, lead_in: bool = False) -> list[str]:
    """Every naming sentence of the signal, in a fixed order. `lead_in`: the thinking-budget
    sentences that end in a lead-in to the summary."""
    return _bank_of(signal, lead_in).sentences()


def draw(signal: str, rng: np.random.Generator, lead_in: bool = False) -> str:
    """One naming sentence: a shape, then one wording for each of its parts."""
    return _bank_of(signal, lead_in).draw(rng)


# --- one episode ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Found:
    """The bank sentence at the start of a reaction."""
    start: int  # index into target_ids of the first reaction token
    lead: str  # what joins the sentence to the text before it: "", a line break or a blank line
    sentence: str
    n_old: int  # tokens of lead + sentence + the line end after it
    in_thinking: bool  # the thinking is open at the reaction: its close follows the sentence

    @property
    def tail(self) -> str:
        return "\n" if self.in_thinking else "\n\n"


@dataclass(frozen=True)
class Wording:
    plain: str  # the naming sentence, ending in a period
    lead_in: str | None = None  # thinking budget: the sentence that ends in a lead-in to the summary


_CANDIDATES: dict[int, tuple] = {}  # id(tok) -> (tok, {(signal, tail): [(lead, sentence, ids)]})


def _candidates(tok, signal: str, tail: str) -> list[tuple[str, str, list[int]]]:
    held = _CANDIDATES.setdefault(id(tok), (tok, {}))[1]  # the tokenizer is held, so its id is not reused
    if (signal, tail) not in held:
        held[(signal, tail)] = [(lead, s, encode(tok, lead + s + tail))
                                for lead in ("", "\n", "\n\n") for s in reaction_sentences(signal)]
    return held[(signal, tail)]


def locate(ep: Episode, tok, ms: ModelSpec) -> Found | None:
    """The bank sentence a training reaction opens with, token-exact (the sentence was prefilled
    when the episode was built, so its tokens are a prefix of the reaction). None: the episode has
    no reaction, or its reaction does not start with a bank sentence."""
    start = reaction_start(ep)
    if start is None or ep.signal not in BANKS:
        return None
    in_thinking = ms.think_close not in ep.target_ids[:start]
    for lead, sentence, ids in _candidates(tok, ep.signal, "\n" if in_thinking else "\n\n"):
        end = start + len(ids)
        if ep.target_ids[start:end] == ids and (not in_thinking or ep.target_ids[end:end + 1] == [ms.think_close]):
            return Found(start, lead, sentence, len(ids), in_thinking)
    return None


def counted(target_ids: list[int], start: int, signal: str, tok, ms: ModelSpec) -> bool:
    """Would the unchanged classifier count this reaction, were it a model's own output."""
    reaction = find_reaction(target_ids[start:], "eos", signal, tok, ms)
    return reaction is not None and reaction.named and reaction.token_index < REACTION_WINDOW


def _keeps_decision(ep: Episode, found: Found, span: list[int], base_ids: list[int] | None) -> bool:
    """The twin's decision token stays valid: the new target still parts from the base run on
    token `t_diverge` (with `base_ids`; without them only the tokens before it are compared)."""
    if ep.t_diverge is None or not found.start <= ep.t_diverge < found.start + found.n_old:
        return True
    off = ep.t_diverge - found.start
    if off >= len(span) or span[:off] != ep.target_ids[found.start:ep.t_diverge]:
        return False
    return base_ids is None or ep.t_diverge >= len(base_ids) or span[off] != base_ids[ep.t_diverge]


def draw_wording(ep: Episode, found: Found, tok, ms: ModelSpec, rng: np.random.Generator,
                 base_ids: list[int] | None = None, tries: int = 200) -> Wording | None:
    """New wording for one reaction: sentences are drawn until one keeps the twin's decision token
    and, where the classifier counted the old reaction, is still counted. None: none found."""
    before, after = ep.target_ids[:found.start], ep.target_ids[found.start + found.n_old:]
    was_counted = counted(ep.target_ids, found.start, ep.signal, tok, ms)

    def pick(lead_in: bool) -> str | None:
        for _ in range(tries):
            sentence = draw(ep.signal, rng, lead_in)
            span = encode(tok, found.lead + sentence + found.tail)
            if not _keeps_decision(ep, found, span, base_ids):
                continue
            if lead_in or not was_counted or counted(before + span + after, found.start, ep.signal, tok, ms):
                return sentence  # with a summary the length is the summary's matter (`close_within`)
        return None

    plain = pick(False)
    lead_in = pick(True) if ep.signal == "think" else None
    if plain is None or (ep.signal == "think" and lead_in is None):
        return None
    return Wording(plain, lead_in)


def _splice(ep: Episode, found: Found, span: list[int], info: dict) -> Episode:
    n, a, b = len(ep.prompt_ids), found.start, found.start + found.n_old
    shift = len(span) - found.n_old

    def moved(index: int | None) -> int | None:
        return index if index is None or index < b else index + shift

    def spans(old: list[list[int]] | None) -> list[list[int]] | None:
        if old is None:
            return None
        if any(s < b and e > a for s, e in old):
            raise ValueError(f"{ep.episode_id}: a span crosses the naming sentence [{a}, {b})")
        return [[s + shift, e + shift] if s >= b else [s, e] for s, e in old]

    row = ep.signals[n + a]
    signals = np.concatenate([ep.signals[:n + a], np.repeat(row[None], len(span), axis=0), ep.signals[n + b:]])
    return replace(
        ep, target_ids=ep.target_ids[:a] + span + ep.target_ids[b:], signals=signals.astype(np.float32),
        t_decision=moved(ep.t_decision), t_diverge=moved(ep.t_diverge),
        harness_spans=spans(ep.harness_spans), focus_spans=spans(ep.focus_spans),
        meta={**(ep.meta or {}), "varied": {**info, "span": [a, a + len(span)]}})


def rewrite(ep: Episode, tok, ms: ModelSpec, rng: np.random.Generator | None, *, base_ids: list[int] | None = None,
            summary: str | None = None, wording: Wording | None = None, found: Found | None = None) -> Episode:
    """The episode with the bank sentence of its reaction replaced by a generated one (contract in
    the module docstring). Returned as it is: an episode without a reaction, one whose reaction
    does not start with a bank sentence, one for which no sentence fits.

    `base_ids`: the base run (the twin's target), for the exact check of the decision token.
    `summary` (thinking budget only): the sentence then ends in a lead-in and the summary is the
    next line, before the thinking closes. `wording`: a wording drawn earlier (`draw_wording`)."""
    if summary is not None and ep.signal != "think":
        raise ValueError(f"{ep.episode_id}: a summary belongs to a thinking-budget reaction, not to {ep.signal}")
    found = found or locate(ep, tok, ms)
    if found is None:
        return ep
    wording = wording or draw_wording(ep, found, tok, ms, rng, base_ids)
    if wording is None:
        return ep
    sentence = wording.plain if summary is None else wording.lead_in
    said = sentence if summary is None else sentence + "\n" + summary
    info = {"old": found.sentence, "sentence": sentence, "summary": summary,
            "summary_tokens": None if summary is None else len(encode(tok, summary))}
    return _splice(ep, found, encode(tok, found.lead + said + found.tail), info)


# --- the summary ------------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"[.!?][\"')\]]*$")


def sentence_ends(ids: list[int], tok) -> list[int]:
    """Token counts k at which ids[:k] ends a sentence: the text so far ends in . ! or ? (and
    closing quotes), what follows starts a new word (3.5 is no end), and the sentence has at least
    two words with a letter (a list number or a bare calculation is no sentence of its own)."""
    ends, last, seen = [], 0, 0
    for k in range(1, len(ids) + 1):
        text = tok.decode(ids[:k])
        stripped = text.rstrip()
        if len(stripped) == seen or not _SENTENCE_END.search(stripped):
            seen = len(stripped)
            continue
        seen = len(stripped)
        if k < len(ids) and text == stripped:  # no space yet: does the next token start a new word?
            longer = tok.decode(ids[:k + 1])
            if not longer.startswith(text) or not longer[len(text):][:1].isspace():
                continue
        if len(re.findall(r"[^\W\d_]\w*", stripped[last:])) >= 2:
            ends.append(k)
            last = len(stripped)
    return ends


def _special_ids(tok, ms: ModelSpec) -> set[int]:
    added = getattr(tok, "get_added_vocab", dict)()
    return ({ms.think_open, ms.think_close, ms.tool_open, ms.tool_close} | set(ms.eos_ids)
            | set(getattr(tok, "all_special_ids", ())) | set(added.values()))


def pick_summary(ids: list[int], tok, ms: ModelSpec, *, min_tokens: int = MIN_SUMMARY, max_tokens: int = MAX_SUMMARY,
                 max_sentences: int = MAX_SENTENCES, fits: Callable[[str], bool] | None = None) -> tuple[str | None, str]:
    """The summary in what the untrained model wrote after the lead-in: (text, "accepted") or
    (None, reason). The text runs up to where the model closes its thinking or ends the turn, holds
    no tool call and no special token, and is cut at the last sentence end that leaves at most
    `max_sentences` sentences (0: any number) and `max_tokens` tokens (and passes `fits`). Reasons:
    "tool_call", "special_token", "no_sentence_end", "too_long", "too_short"."""
    closers = {ms.think_close} | set(ms.eos_ids)
    body = ids[:next((i for i, t in enumerate(ids) if t in closers), len(ids))]
    if ms.tool_open in body or ms.tool_close in body:
        return None, "tool_call"
    if set(body) & _special_ids(tok, ms):
        return None, "special_token"
    if not tok.decode(body).strip():
        return None, "too_short"  # the model closed its thinking at once
    ends = sentence_ends(body, tok)
    if not ends:
        return None, "no_sentence_end"
    reason = "too_long"
    for k in reversed(ends[:max_sentences] if max_sentences else ends):
        text = tok.decode(body[:k]).strip()
        n = len(encode(tok, text))
        if n > max_tokens or (fits is not None and not fits(text)):
            continue
        if n < min_tokens:
            reason = "too_short"
            break
        return text, "accepted"
    return None, reason


def summary_request(tok, notes: str, words: int = SUMMARY_WORDS[0]) -> list[int]:
    """Prompt ids of the request that asks the untrained model for the summary: a chat of its own
    (thinking off), holding the thinking so far as notes. What it writes goes through `pick_summary`."""
    messages = [{"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": SUMMARY_ASK.format(notes=notes.strip() or NO_NOTES, words=words)}]
    text = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False,
                                   **template_kwargs(tok))
    return encode(tok, text)


def sampled(sampler, cache: Path | None, jobs: list[tuple[tuple[str, int], list[int]]], chunk: int = 128) -> dict:
    """{key: (ids, stop reason)} for every (key, prompt) job: read from the cache file where the
    same key was sampled for the same prompt, else `sampler(prompts)` in chunks, each appended to
    the cache as soon as it exists (a power cut loses one chunk at most)."""
    digest = {key: hashlib.sha1(json.dumps(prompt).encode("utf-8")).hexdigest() for key, prompt in jobs}
    have = {}
    if cache is not None and cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            have[(row["episode_id"], row["try"])] = row  # a later line replaces an earlier one
    out = {key: (have[key]["ids"], have[key]["stop"]) for key, _ in jobs if key in have and have[key]["prompt"] == digest[key]}
    missing = [(key, prompt) for key, prompt in jobs if key not in out]
    for start in range(0, len(missing), chunk):
        part = missing[start:start + chunk]
        results = sampler([prompt for _, prompt in part])
        rows = [{"episode_id": key[0], "try": key[1], "prompt": digest[key], "ids": [int(t) for t in ids], "stop": stop}
                for (key, _), (ids, stop) in zip(part, results)]
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "ab") as f:
                f.write("".join(json.dumps(row) + "\n" for row in rows).encode("utf-8"))
        out.update({(row["episode_id"], row["try"]): (row["ids"], row["stop"]) for row in rows})
    return out


def _stats(values: Sequence[int]) -> dict:
    if not values:
        return {"n": 0, "min": None, "mean": None, "max": None}
    return {"n": len(values), "min": int(min(values)), "mean": round(float(np.mean(values)), 2), "max": int(max(values))}


# --- the whole file ---------------------------------------------------------------------------

def rewrite_all(episodes: list[Episode], tok, ms: ModelSpec, *, seed: int = 0, sampler=None, cache: Path | None = None,
                limit: int | None = None, tries: int = SUMMARY_TRIES, min_tokens: int = MIN_SUMMARY,
                max_tokens: int = MAX_SUMMARY, max_sentences: int = MAX_SENTENCES, close_within: int | None = None,
                chunk: int = 128, max_len: int = MAX_LEN, notes: list | None = None,
                method: str = "continue") -> tuple[list[Episode], dict]:
    """Every training reaction with new wording; with a `sampler` (prompts -> [(ids, stop reason)],
    the untrained model) also the summaries of the thinking-budget reactions. Returns the episodes
    in file order (an untouched episode is the same object) and the counts of the manifest.

    `method`: how the model is asked for a summary: "continue" its own thinking after the lead-in, or
    a "request" of its own that holds the thinking so far (`summary_request`).

    Each episode draws from its own generator (seed, place in the file): a mini run (`limit`: only
    the first thinking-budget reactions are kept) words them as the full run does. `close_within`:
    a summary is cut so that the thinking closes within that many tokens of the reaction start
    (not A9's rule). `notes` is filled with
    one record per thinking-budget reaction, for reading by hand."""
    bases = {ep.task_id: ep.target_ids for ep in episodes if ep.kind == "nominal" and ep.t_diverge is not None}
    plans: dict[int, tuple[Found, Wording]] = {}
    unchanged: dict[str, Counter] = defaultdict(Counter)
    for i, ep in enumerate(episodes):
        if reaction_start(ep) is None:
            continue
        found = locate(ep, tok, ms)
        if found is None:
            unchanged["no_bank_sentence"][str(ep.signal)] += 1
            continue
        wording = draw_wording(ep, found, tok, ms, np.random.default_rng([seed, i]), bases.get(ep.task_id))
        if wording is None:
            unchanged["no_wording"][ep.signal] += 1
            continue
        plans[i] = (found, wording)
    think = [i for i in plans if episodes[i].signal == "think"]
    kept = range(len(episodes))
    if limit is not None:
        think = think[:limit]
        kept = think

    summaries: dict[int, str] = {}
    summary_stats = None
    if method not in METHODS:
        raise ValueError(f"summary method {method!r}: one of {', '.join(METHODS)}")
    if sampler is not None:
        if method == "request":
            noted = {i: tok.decode([t for t in episodes[i].target_ids[:plans[i][0].start] if t != ms.think_open]) for i in think}

            def prompt_of(i: int, attempt: int) -> list[int]:  # a further try asks for fewer words
                return summary_request(tok, noted[i], words=SUMMARY_WORDS[min(attempt, len(SUMMARY_WORDS)) - 1])
        else:
            continued = {i: episodes[i].prompt_ids + episodes[i].target_ids[:plans[i][0].start]
                         + encode(tok, plans[i][0].lead + plans[i][1].lead_in + "\n") for i in think}

            def prompt_of(i: int, attempt: int) -> list[int]:
                return continued[i]
        last: dict[int, str] = {}
        record = {i: {"episode_id": episodes[i].episode_id, "kind": episodes[i].kind, "sentence": plans[i][1].lead_in,
                      "before": tok.decode(episodes[i].target_ids[max(0, plans[i][0].start - 120):plans[i][0].start]),
                      "summary": None, "tries": []} for i in think}
        per_try, pending = [], list(think)
        for attempt in range(1, tries + 1):
            if not pending:
                break
            raw = sampled(sampler, cache, [((episodes[i].episode_id, attempt), prompt_of(i, attempt)) for i in pending], chunk)
            rejected, still = Counter(), []
            for i in pending:
                found, wording = plans[i]
                fits = None
                if close_within is not None:
                    fits = lambda text, f=found, w=wording: len(encode(tok, f.lead + w.lead_in + "\n" + text + f.tail)) < close_within
                ids, _ = raw[(episodes[i].episode_id, attempt)]
                text, reason = pick_summary(ids, tok, ms, min_tokens=min_tokens, max_tokens=max_tokens,
                                            max_sentences=max_sentences, fits=fits)
                if text is not None and method == "request" and echoes_request(text):
                    text, reason = None, "echo"
                record[i]["tries"].append({"reason": reason, "wrote": tok.decode(ids)})
                if text is None:
                    rejected[reason] += 1
                    last[i] = reason
                    still.append(i)
                else:
                    summaries[i] = record[i]["summary"] = text
            per_try.append({"asked": len(pending), "accepted": len(pending) - len(still), "rejected": dict(rejected)})
            pending = still
        by_kind: dict[str, dict] = {}
        for i in think:
            cell = by_kind.setdefault(episodes[i].kind, {"accepted": 0, "none": 0})
            cell["accepted" if i in summaries else "none"] += 1
        summary_stats = {"episodes": len(think), "accepted": len(summaries), "none": dict(Counter(last[i] for i in pending)),
                         "tries": per_try, "by_kind": by_kind,
                         "tokens": _stats([len(encode(tok, text)) for text in summaries.values()])}
        if notes is not None:
            notes.extend(record[i] for i in think)

    out, used = [], defaultdict(set)
    reactions = defaultdict(lambda: {"rewritten": 0, "unchanged": 0})
    count = defaultdict(lambda: {"n": 0, "before": 0, "after": 0})
    lengths = defaultdict(lambda: {"old_sentence": [], "new_sentence": [], "episode": []})
    close_at = {"with_summary": [], "without_summary": []}
    close_counted = {"with_summary": 0, "without_summary": 0}
    for i in kept:
        ep = episodes[i]
        new = ep
        if i in plans:
            found, wording = plans[i]
            new = rewrite(ep, tok, ms, None, wording=wording, summary=summaries.get(i), found=found)
        out.append(new)
        start = reaction_start(ep)
        if start is None or ep.signal is None:
            continue
        ok = counted(new.target_ids, start, ep.signal, tok, ms)
        cell = count[ep.signal]
        cell["n"], cell["before"], cell["after"] = cell["n"] + 1, cell["before"] + counted(ep.target_ids, start, ep.signal, tok, ms), cell["after"] + ok
        reactions[ep.signal]["rewritten" if new is not ep else "unchanged"] += 1
        if new is ep:
            continue
        info = new.meta["varied"]
        used[ep.signal].add(info["sentence"])
        sizes = lengths[ep.signal]
        sizes["old_sentence"].append(found.n_old)
        sizes["new_sentence"].append(len(encode(tok, found.lead + info["sentence"] + found.tail)))
        sizes["episode"].append(len(new.prompt_ids) + len(new.target_ids))
        if ep.signal == "think":
            which = "with_summary" if info["summary"] is not None else "without_summary"
            close_at[which].append(info["span"][1] - info["span"][0])
            close_counted[which] += ok
    stats = {
        "episodes": len(out), "kinds": dict(Counter(ep.kind for ep in out)),
        "reactions": {s: {**cell, "distinct_sentences": len(used[s])} for s, cell in reactions.items()},
        "unchanged": {reason: dict(cells) for reason, cells in unchanged.items()},
        "summaries": summary_stats,
        "think": {"lead_in_tokens": _stats([len(encode(tok, plans[i][0].lead + plans[i][1].lead_in + "\n")) for i in think]),
                  **{which: {"n": len(close_at[which]), "counted": close_counted[which], "close_at": _stats(close_at[which])}
                     for which in close_at}},
        "counted_by_classifier": dict(count),
        "lengths": {s: {name: _stats(values) for name, values in cells.items()} for s, cells in lengths.items()},
        "dropped_long": {"max_len": max_len,
                         "before": sum(len(episodes[i].prompt_ids) + len(episodes[i].target_ids) > max_len for i in kept),
                         "after": sum(len(ep.prompt_ids) + len(ep.target_ids) > max_len for ep in out)},
        "longest": max((len(ep.prompt_ids) + len(ep.target_ids) for ep in out), default=0),
    }
    return out, stats


def _line(ep: Episode) -> bytes:
    """One episode as `episodes.write_jsonl` writes it, without the line end."""
    row = asdict(ep)
    values = np.ascontiguousarray(ep.signals, dtype=np.float32)
    row["signals"] = {"shape": list(values.shape), "b64": base64.b64encode(values.tobytes()).decode("ascii")}
    return json.dumps(row).encode("utf-8")


def build(src: Path, out: Path, tok, ms: ModelSpec, *, seed: int = 0, sampler=None, limit: int | None = None,
          tries: int = SUMMARY_TRIES, min_tokens: int = MIN_SUMMARY, max_tokens: int = MAX_SUMMARY,
          max_sentences: int = MAX_SENTENCES, close_within: int | None = None, chunk: int = 128, max_len: int = MAX_LEN,
          manifest_only: bool = False, options: dict | None = None, notes: list | None = None,
          method: str = "continue") -> dict:
    """Write <out>/train.jsonl (rewritten; an untouched episode keeps its source line byte for
    byte), <out>/test.jsonl (a byte copy) and <out>/manifest.json; returns the manifest. Without a
    `sampler`: wording only. `limit`: a mini run for reading (the first thinking-budget reactions,
    no test file). `manifest_only`: the manifest alone. Deterministic given the summaries: a rerun
    reads them from <out>/summaries.jsonl and writes the same bytes."""
    src, out = Path(src), Path(out)
    if src.resolve() == out.resolve():
        raise ValueError("--out must differ from --src")
    src_manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8")) if (src / "manifest.json").exists() else {}
    raw = (src / "train.jsonl").read_bytes().splitlines(keepends=True)
    episodes = read_jsonl(src / "train.jsonl")
    out.mkdir(parents=True, exist_ok=True)
    new, stats = rewrite_all(episodes, tok, ms, seed=seed, sampler=sampler, cache=out / CACHE, limit=limit, tries=tries,
                             min_tokens=min_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
                             close_within=close_within, chunk=chunk, max_len=max_len, notes=notes, method=method)
    copy_test = limit is None and not manifest_only and (src / "test.jsonl").exists()
    if not manifest_only:
        line_of = {id(ep): line for ep, line in zip(episodes, raw)}
        ending = b"\r\n" if raw and raw[0].endswith(b"\r\n") else b"\n"
        with open(out / "train.jsonl", "wb") as f:
            for ep in new:
                f.write(line_of[id(ep)] if id(ep) in line_of else _line(ep) + ending)
    if copy_test:
        shutil.copyfile(src / "test.jsonl", out / "test.jsonl")
    # `source_manifest`: what eval/run.dataset_tasks and variants.manifest_model read (as a variant folder).
    kept = ("model", "n_signals", "seed", "tasks_requested")
    manifest = {
        "variant": "varied", "source": str(src), "source_manifest": {k: src_manifest[k] for k in kept if k in src_manifest},
        "options": {"seed": seed, "summary": sampler is not None, "limit": limit, "tries": tries,
                    "summary_tokens": [min_tokens, max_tokens], "summary_sentences": max_sentences,
                    "summary_new_tokens": SUMMARY_NEW_TOKENS, "close_within": close_within, "summary_method": method,
                    **(options or {})},
        "episodes": {"train": stats.pop("episodes"),
                     **({"test": (src / "test.jsonl").read_bytes().count(b"\n")} if copy_test else {})},
        "kinds": {"train": stats.pop("kinds")},
        "generator": {"sentences": {s: len(sentences(s)) for s in BANKS}, "shapes": len(_SHAPES),
                      "think_lead_in": len(sentences("think", lead_in=True))},
        **stats,
    }
    (out / "manifest.json").write_bytes((json.dumps(manifest, indent=1) + "\n").encode("utf-8"))
    return manifest


def show(notes: list[dict], n: int) -> str:
    """`n` thinking-budget reactions spread over the file, for reading by hand: the thinking before,
    the sentence, the summary, and what the model wrote on every try."""
    picked = notes if n >= len(notes) else [notes[i] for i in np.linspace(0, len(notes) - 1, n).round().astype(int)]
    lines = []
    for note in picked:
        lines += ["=" * 100, f"{note['episode_id']} ({note['kind']})", "thinking before: ..." + note["before"],
                  "sentence: " + note["sentence"], "summary : " + (note["summary"] or "NONE (the episode keeps the plain sentence)")]
        lines += [f"  try {k}: {t['reason']} | wrote: {t['wrote']!r}" for k, t in enumerate(note["tries"], 1)]
    return "\n".join(lines)


def _load_tokenizer(model: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    return tok, resolve_spec(model, tok)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="R24: varied reaction wording + a summary when the thinking budget runs out.")
    parser.add_argument("--src", type=Path, required=True, help="the source dataset (data/n6)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=None, help="tokenizer and summary writer; defaults to the model in <src>/manifest.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-summary", action="store_true", help="wording only (CPU, no model)")
    parser.add_argument("--limit", type=int, default=None, help="mini run: only the first N thinking-budget reactions")
    parser.add_argument("--show", type=int, default=0, help="print N thinking-budget reactions for reading by hand")
    parser.add_argument("--manifest-only", action="store_true", help="write the manifest alone (counts, lengths)")
    parser.add_argument("--summary-tries", type=int, default=SUMMARY_TRIES)
    parser.add_argument("--summary-max-tokens", type=int, default=MAX_SUMMARY, help="A9: 80")
    parser.add_argument("--summary-sentences", type=int, default=MAX_SENTENCES, help="A9: one or two; 0 = any number")
    parser.add_argument("--close-within", type=int, default=None,
                        help="cut a summary so that the thinking closes within N tokens of the reaction start "
                             "(the classifier's window is 48); not A9's rule: needs an amendment before a reported run")
    parser.add_argument("--summary-method", choices=METHODS, default="request",
                        help="request (default; note on A9 of 2026-11-17): the model is asked for the summary on a request "
                             "of its own; continue: it continues its thinking after the lead-in (A9 as first written)")
    parser.add_argument("--device", default="cuda", help="where the summary writer runs (cpu: float32, for a mini pass)")
    args = parser.parse_args(argv)
    if args.src.resolve() == args.out.resolve():
        raise ValueError("--out must differ from --src")
    model = args.model or manifest_model(json.loads((args.src / "manifest.json").read_text(encoding="utf-8")))
    tok, ms = _load_tokenizer(model)
    sampler, loaded = None, []
    if not args.no_summary:
        def sampler(prompts):  # the model is loaded only when something is missing from the cache
            if not loaded:
                import torch

                torch.manual_seed(args.seed)
                on_cpu = args.device == "cpu"
                loaded.append(load_base(model, **({"dtype": torch.float32, "device": "cpu"} if on_cpu else {"device": args.device})))
                loaded[0][1].padding_side = "left"
            net, gen_tok, gen_ms = loaded[0]
            return sample_batch(net, gen_tok, gen_ms, prompts, SUMMARY_NEW_TOKENS, args.batch_size, label="summaries")

    notes: list[dict] = []
    manifest = build(args.src, args.out, tok, ms, seed=args.seed, sampler=sampler, limit=args.limit, tries=args.summary_tries,
                     max_tokens=args.summary_max_tokens, max_sentences=args.summary_sentences, close_within=args.close_within,
                     chunk=4 * args.batch_size, manifest_only=args.manifest_only,
                     options={"model": model, "batch_size": args.batch_size}, notes=notes, method=args.summary_method)
    if args.show and notes:
        print(show(notes, args.show))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
