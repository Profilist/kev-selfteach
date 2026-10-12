"""Filler text for the synthetic evidence tasks: SQuAD train paragraphs (unrelated prose) and AG News train articles.

Sources (pinned by file hash): rajpurkar/squad @ 7b6d24c, plain_text/train-00000-of-00001.parquet -> data/squad_train.parquet
(CC BY-SA 4.0), and fancyzhx/ag_news @ eb185aa, data/train-00000-of-00001.parquet -> data/ag_news_train.parquet.
No hosted-model output is involved: "no sports" pools use a keyword filter, AG News's own labels, and a drop list from a
sports classifier trained on those labels (baseline/sports_filter.py -> baseline/nonsports_drop.json).
"""
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
DROP = Path(__file__).with_name("nonsports_drop.json")

# Keeps sports content out of "no sports story here" fillers, so their gold label stays valid.
SPORTS_RE = re.compile(
    r"\b(football|soccer|basketball|baseball|hockey|tennis|golf\w*|cricket|rugby|olymp\w*|nfl|nba|nhl|mlb|"
    r"fifa|uefa|world cup|grand slam|tournament\w*|championship\w*|playoff\w*|quarterback\w*|pitcher\w*|"
    r"athlet\w*|stadium\w*|marathon\w*|boxing|boxer\w*|wrestl\w*|nascar|formula one|grand prix|super bowl|"
    r"world series|premier league|sprint\w*|swimm\w*|cyclist\w*|tour de france|ryder cup|davis cup|"
    r"wimbledon|medal\w*|coach\w*|league\w*|striker\w*|goalkeeper\w*|inning\w*|touchdown\w*|racing|jockey\w*|"
    r"\w+ball|sports?|sportsm\w*|player\w*|champion\w*|seed\w*|semifinal\w*|final four|slam)\b",
    re.I,
)


def clean_agnews(text: str) -> str:
    text = text.replace("#39;", "'").replace("quot;", '"').replace("&amp;", "&").replace("\\$", "$")
    text = text.replace("\\", " ").replace("&lt;", "<").replace("&gt;", ">")
    return " ".join(text.split())


@lru_cache
def squad_paragraphs() -> tuple[str, ...]:
    """Distinct SQuAD train paragraphs of at least 300 characters, whitespace-normalised."""
    df = pd.read_parquet(DATA / "squad_train.parquet", columns=["context"])
    seen, out = set(), []
    for c in df.context:
        c = " ".join(c.split())
        if len(c) >= 300 and c not in seen:
            seen.add(c)
            out.append(c)
    return tuple(out)


@lru_cache
def squad_nonsports() -> tuple[str, ...]:
    """SQuAD paragraphs with no sports keyword."""
    return tuple(p for p in squad_paragraphs() if not SPORTS_RE.search(p))


@lru_cache
def squad_nonsports_filtered() -> tuple[str, ...]:
    """squad_nonsports() minus the paragraphs the AG News sports classifier flags (nonsports_drop.json)."""
    drop = set(json.load(open(DROP))["drop_sha256"])
    return tuple(p for p in squad_nonsports() if hashlib.sha256(" ".join(p.split()).encode()).hexdigest() not in drop)


@lru_cache
def agnews_train() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(sports articles, non-sports articles) from a fixed 24,000-article sample of AG News train, by its own labels."""
    df = pd.read_parquet(DATA / "ag_news_train.parquet").sample(n=24000, random_state=0)
    sports = tuple(clean_agnews(t) for t, y in zip(df.text, df.label) if y == 1)
    other = tuple(clean_agnews(t) for t, y in zip(df.text, df.label) if y != 1 and not SPORTS_RE.search(t))
    return sports, other
