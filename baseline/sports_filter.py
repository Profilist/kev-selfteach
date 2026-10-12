"""The "no sports" drop list for the SQuAD filler of categorical ("is there a sports story?") training items.

A sports classifier is trained only on AG News train's own labels (TF-IDF + logistic regression), then applied to the
keyword-filtered SQuAD train paragraphs (fillers.squad_nonsports). Fixed in advance: a paragraph is dropped at
p(sports) >= 0.5, and a whole Wikipedia article when at least a quarter of its paragraphs are. Writes
baseline/nonsports_drop.json (sha256 of every dropped paragraph, and the dropped titles). No hosted-model output is used.

    PYTHONPATH=. data/kev/.venv/bin/python baseline/sports_filter.py      # needs scikit-learn (Kev's environment has it)
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from baseline import fillers

OUT = Path(__file__).with_name("nonsports_drop.json")
P_DROP, ARTICLE_SHARE = 0.5, 0.25


def sha(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def main():
    ag = pd.read_parquet(fillers.DATA / "ag_news_train.parquet")
    vec = TfidfVectorizer(sublinear_tf=True, min_df=3, ngram_range=(1, 2), max_features=200_000)
    X = vec.fit_transform(ag.text)
    clf = LogisticRegression(max_iter=2000, C=4.0).fit(X, (ag.label == 1).astype(int))   # AG News label 1 = Sports

    df = pd.read_parquet(fillers.DATA / "squad_train.parquet", columns=["title", "context"])
    title = {" ".join(c.split()): t for t, c in zip(df.title, df.context)}
    paras = fillers.squad_nonsports()
    p = clf.predict_proba(vec.transform(paras))[:, 1]
    flagged = defaultdict(list)
    for para, prob in zip(paras, p):
        flagged[title.get(" ".join(para.split()), "?")].append(prob >= P_DROP)
    drop_titles = sorted(t for t, f in flagged.items() if sum(f) >= ARTICLE_SHARE * len(f))
    dropped = [para for para, prob in zip(paras, p) if prob >= P_DROP or title.get(" ".join(para.split())) in drop_titles]
    json.dump({"what": __doc__.split("\n\n")[0], "p_drop": P_DROP, "article_share": ARTICLE_SHARE,
               "drop_titles": drop_titles, "drop_sha256": sorted(sha(x) for x in dropped)}, open(OUT, "w"), indent=1)
    print(f"{len(paras)} keyword-filtered paragraphs: dropped {len(dropped)} ({len(drop_titles)} whole articles: {drop_titles[:12]})")


if __name__ == "__main__":
    main()
