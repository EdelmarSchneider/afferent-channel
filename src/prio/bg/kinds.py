"""Kinds of background task: what the user asks to start, what it yields, what to report.

Spec section 6.4: about 8 kinds for training and 4 held out for the test only, differing in verb
and object (not only in wording), so the test measures whether the behavior is generic. Names,
objects and end instructions contain no digits: the closing line and the math answers must keep
the final number of the answer as its last number.
"""

from dataclasses import asdict, dataclass, fields
from typing import Sequence


@dataclass(frozen=True)
class TaskKind:
    name: str
    verb: str  # imperative, capitalized
    noun: str  # in labels: "report download"
    objects: tuple[tuple[str, str], ...]  # (phrase, head noun)
    ends: tuple[tuple[str, str | None], ...]  # (end instruction, detail key or None)
    result: str  # format string over {head} and the detail keys of `ends`


_DONE = ("let me know when it is done", None)

TRAIN_KINDS: tuple[TaskKind, ...] = (
    TaskKind("download", "Download", "download",
             (("the quarterly report", "report"), ("the census spreadsheet", "spreadsheet"),
              ("the training manual", "manual"), ("the conference brochure", "brochure")),
             (("tell me its total", "total"), ("tell me how many pages it has", "pages"), _DONE),
             "{head} downloaded: {pages} pages, total {total}"),
    TaskKind("copy", "Copy", "copy",
             (("the music library", "library"), ("the old invoices", "invoices"),
              ("the design sketches", "sketches"), ("the course materials", "materials")),
             (("tell me how many files it copied", "files"), _DONE),
             "{head} copied: {files} files"),
    TaskKind("upload", "Upload", "upload",
             (("the podcast episode", "podcast"), ("the slide deck", "deck"),
              ("the research poster", "poster"), ("the recipe collection", "recipes")),
             (("tell me how long it took", "minutes"), _DONE),
             "{head} uploaded in {minutes} minutes"),
    TaskKind("build", "Build", "build",
             (("the mobile app", "app"), ("the documentation site", "site"),
              ("the game engine", "engine"), ("the payment service", "service")),
             (("tell me how many warnings it reports", "warnings"), _DONE),
             "{head} built with {warnings} warnings"),
    TaskKind("backup", "Back up", "backup",
             (("the family photos", "photos"), ("the email folders", "folders"),
              ("the tax records", "records"), ("the browser bookmarks", "bookmarks")),
             (("tell me how many files it saved", "files"), _DONE),
             "{head} backed up: {files} files"),
    TaskKind("install", "Install", "installation",
             (("the database server", "server"), ("the photo editor", "editor"),
              ("the language pack", "pack"), ("the printer driver", "driver")),
             (("tell me which version it installed", "version"), _DONE),
             "{head} installed, version {version}"),
    TaskKind("convert", "Convert", "conversion",
             (("the lecture video", "video"), ("the scanned letters", "letters"),
              ("the audio interviews", "interviews"), ("the class grades", "grades")),
             (("tell me the size of the result", "size"), _DONE),
             "{head} converted: {size} MB"),
    TaskKind("sync", "Sync", "sync",
             (("the shared calendar", "calendar"), ("the notes folder", "notes"),
              ("the contact list", "contacts"), ("the team wiki", "wiki")),
             (("tell me how many items changed", "items"), _DONE),
             "{head} synced: {items} items changed"),
)

HELD_OUT_KINDS: tuple[TaskKind, ...] = (
    TaskKind("render", "Render", "render",
             (("the product animation", "animation"), ("the architecture model", "model"),
              ("the title sequence", "sequence"), ("the weather visualization", "visualization")),
             (("tell me how many frames it rendered", "frames"), _DONE),
             "{head} rendered: {frames} frames"),
    TaskKind("export", "Export", "export",
             (("the sales dashboard", "dashboard"), ("the customer database", "database"),
              ("the photo album", "album"), ("the meeting transcripts", "transcripts")),
             (("tell me how many rows it exported", "rows"), _DONE),
             "{head} exported: {rows} rows"),
    TaskKind("compress", "Compress", "compression",
             (("the log files", "logs"), ("the raw footage", "footage"),
              ("the old projects", "projects"), ("the image collection", "images")),
             (("tell me how much space it saved", "saved"), _DONE),
             "{head} compressed: {saved} MB saved"),
    TaskKind("deploy", "Deploy", "deployment",
             (("the web shop", "shop"), ("the chat bot", "bot"),
              ("the booking system", "system"), ("the internal portal", "portal")),
             (("tell me the address it is running at", "address"), _DONE),
             "{head} deployed at {address}"),
)

REASONS = ("connection lost", "disk full", "permission denied", "source not found")


def _value(key: str, head: str, rng) -> str:
    """A plausible value for a detail key (digits allowed here)."""
    grouped = lambda lo, hi: f"{int(rng.integers(lo, hi)):,}"
    plain = lambda lo, hi: str(int(rng.integers(lo, hi)))
    makers = {
        "total": lambda: grouped(1000, 10000), "pages": lambda: plain(12, 480), "files": lambda: grouped(20, 4800),
        "minutes": lambda: plain(2, 45), "warnings": lambda: plain(0, 40), "size": lambda: plain(40, 950),
        "items": lambda: plain(3, 300), "frames": lambda: grouped(120, 7200), "rows": lambda: grouped(100, 90000),
        "saved": lambda: plain(30, 900),
        "version": lambda: f"{int(rng.integers(1, 10))}.{int(rng.integers(0, 21))}.{int(rng.integers(0, 10))}",
        "address": lambda: f"{head}-{int(rng.integers(2, 100))}.example.org",
    }
    return makers[key]()


@dataclass
class BgTask:
    task_id: str
    kind: str
    held_out: bool
    verb: str
    noun: str
    phrase: str
    head: str
    end: str
    detail_key: str | None
    detail_value: str | None
    result: str
    question: str
    reason: str

    def describe(self) -> str:
        return f"{self.verb.lower()} {self.phrase}"

    def label(self) -> str:
        return f"{self.head} {self.noun}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BgTask":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def sample_tasks(n: int, kinds: Sequence[TaskKind], rng, held_out: bool) -> list[BgTask]:
    """n tasks of distinct kinds (heads are unique across kinds), ids t1..tn in order."""
    chosen = [kinds[i] for i in rng.choice(len(kinds), size=n, replace=False)]
    tasks = []
    for i, kind in enumerate(chosen, start=1):
        phrase, head = kind.objects[int(rng.integers(len(kind.objects)))]
        end, key = kind.ends[int(rng.integers(len(kind.ends)))]
        keys = {k for _, k in kind.ends if k}
        values = {k: _value(k, head, rng) for k in sorted(keys)}
        tasks.append(BgTask(
            task_id=f"t{i}", kind=kind.name, held_out=held_out, verb=kind.verb, noun=kind.noun, phrase=phrase,
            head=head, end=end, detail_key=key, detail_value=values.get(key) if key else None,
            result=kind.result.format(head=head.capitalize(), **values),
            question=f"Some files of {phrase} already exist at the destination: replace them? yes / no",
            reason=REASONS[int(rng.integers(len(REASONS)))]))
    return tasks
