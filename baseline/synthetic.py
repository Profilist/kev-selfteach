"""Synthetic long-document evidence tasks for the label-trained baseline: five evidence types x two filler conditions.

Every item has a paired *present* and *absent* state that share the same filler and the same extra sentences at the
same positions; the absent state drops the evidence the gold answer hinges on. Labels are programmatic.

Evidence types
  lexical      exact-phrase lookup (a locker's access code)
  paraphrase   no keyword overlap between question and evidence (landmark -> city)
  categorical  "is there any sports story?": one AG News sports article among other passages
  two_piece    a decision needing two separate facts about the same person
  two_hop      a chain: person -> team, team -> office

Filler conditions
  unrelated    SQuAD Wikipedia paragraphs, no distractors
  hard         four near-miss sentences that reuse the evidence template with other entities; for categorical, other
               AG News articles

Only the training split is here: the entity lists below are cut to their training share, and fillers come from SQuAD
train and AG News train (baseline/fillers.py).
"""
import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from . import fillers


# ---------------------------------------------------------------- haystack: filler built to a length, inserts at depths

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def build_filler(pool, target_chars: int, rng: np.random.Generator, forbidden: re.Pattern | None,
                 paragraph_mode: bool) -> list[list[str]]:
    """Random paragraphs from `pool` (skipping any matching `forbidden`) totalling ~target_chars.

    Returns paragraphs as lists of sentences. The last paragraph is cut at a sentence boundary
    (or, in paragraph mode, whole articles only) so the total lands at or just under the target.
    """
    paras, total = [], 0
    for i in rng.permutation(len(pool)):
        p = pool[i]
        if forbidden is not None and forbidden.search(p):
            continue
        cost = len(p) + 2
        if total + cost <= target_chars:
            paras.append([p] if paragraph_mode else SENT_SPLIT.split(p))
            total += cost
            continue
        if paragraph_mode:
            if target_chars - total < 60:
                break
            continue  # look for a shorter article that fits
        keep = []
        for s in SENT_SPLIT.split(p):
            if total + len(s) + 1 > target_chars:
                break
            keep.append(s)
            total += len(s) + 1
        if keep:
            paras.append(keep)
        break
    return paras


def slots_and_offsets(paras: list[list[str]], paragraph_mode: bool):
    """Insertion slots (before unit k; k == len(units) means the end) and their char offsets."""
    if paragraph_mode:
        units = [(i, " ".join(p)) for i, p in enumerate(paras)]
    else:
        units = [(i, s) for i, p in enumerate(paras) for s in p]
    lens = np.array([len(t) + 1 for _, t in units], dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    return units, cum


def depth_slot(cum: np.ndarray, frac: float) -> int:
    return int(np.argmin(np.abs(cum - frac * cum[-1])))


def assemble(paras, units, inserts: dict[int, list[str]], paragraph_mode: bool) -> str:
    """Rebuild text with `inserts[slot]` placed before unit `slot` (as own paragraphs in paragraph mode)."""
    n = len(units)
    if paragraph_mode:
        out = []
        for k in range(n + 1):
            out.extend(inserts.get(k, []))
            if k < n:
                out.append(units[k][1])
        return "\n\n".join(out)
    out_paras: list[list[str]] = [[] for _ in paras] or [[]]
    for k in range(n + 1):
        if k < n:
            pi, sent = units[k]
            out_paras[pi].extend(inserts.get(k, []))
            out_paras[pi].append(sent)
        else:
            out_paras[-1].extend(inserts.get(k, []))
    return "\n\n".join(" ".join(p) for p in out_paras if p)


# ---------------------------------------------------------------- items

TYPES = ["lexical", "paraphrase", "categorical", "two_piece", "two_hop"]
FILLERS = ["unrelated", "hard"]

NAMES = [
    "Yuki Tanaka", "Mara Voss", "Priya Anand", "Tomas Herrera", "Ingrid Solberg", "Kwame Mensah", "Lena Hoffmann",
    "Arjun Mehta", "Chloe Moreau", "Dmitri Volkov", "Aisha Bello", "Rafael Costa", "Sofia Lindqvist", "Hiroshi Sato",
    "Nadia Karim", "Oliver Brennan", "Mei Lin", "Samuel Okafor", "Elena Petrova", "Jonas Berg", "Fatima Zahra",
    "Lucas Ferreira", "Hana Kowalski", "Mateo Rossi", "Amara Nwosu", "Felix Wagner", "Leila Haddad", "Noah Fischer",
    "Zara Qureshi", "Emil Novak", "Isla McKenzie", "Kofi Asante", "Anya Sokolova", "Diego Ramos", "Freya Larsen",
    "Omar Farouk", "Clara Dubois", "Ravi Iyer", "Talia Cohen", "Marco Bianchi", "Nia Roberts", "Kenji Mori",
    "Ayla Demir", "Viktor Horvat", "Maya Patel", "Rowan Gallagher", "Selin Aydin", "Bruno Silva", "Esme Clarke",
    "Tariq Rahman", "Greta Muller", "Julian Reyes", "Sakura Ito", "Idris Bakr", "Lotte Jansen", "Andrei Popescu",
    "Keira Walsh", "Hamid Nazari", "Paula Mendes", "Soren Dahl", "Ines Carvalho", "Tobias Klein", "Wanjiru Kamau",
    "Luca Moretti", "Astrid Nilsen", "Yara Saleh", "Callum Fraser", "Mina Park", "Otto Lehmann", "Dalia Mansour",
]
COLORS = ["teal", "amber", "crimson", "violet", "olive", "ivory", "cobalt", "scarlet", "bronze", "indigo", "coral",
          "jade", "maroon", "ochre", "slate", "copper"]
PLACES = ["harbor", "depot", "garden", "station", "library", "workshop", "hangar", "cellar", "pavilion", "warehouse",
          "clinic", "studio", "archive", "boathouse"]
# Landmark -> city; no landmark name contains its city's name.
LANDMARKS = [
    ("Semper Opera House", "Dresden"), ("Sagrada Familia", "Barcelona"), ("Christ the Redeemer statue", "Rio de Janeiro"),
    ("Golden Gate Bridge", "San Francisco"), ("Brandenburg Gate", "Berlin"), ("Kinkaku-ji temple", "Kyoto"),
    ("Petronas Towers", "Kuala Lumpur"), ("Charles Bridge", "Prague"), ("Hagia Sophia", "Istanbul"),
    ("CN Tower", "Toronto"), ("Space Needle", "Seattle"), ("Burj Khalifa", "Dubai"), ("Acropolis", "Athens"),
    ("Alhambra", "Granada"), ("Atomium", "Brussels"), ("Little Mermaid statue", "Copenhagen"),
    ("Table Mountain", "Cape Town"), ("Colosseum", "Rome"), ("Rijksmuseum", "Amsterdam"),
    ("Hermitage Museum", "Saint Petersburg"), ("Forbidden City", "Beijing"), ("Taj Mahal", "Agra"),
    ("Eiffel Tower", "Paris"), ("St. Stephen's Cathedral", "Vienna"), ("Cloud Gate sculpture", "Chicago"),
    ("Gateway Arch", "St. Louis"), ("Shwedagon Pagoda", "Yangon"), ("Oriental Pearl Tower", "Shanghai"),
    ("Senso-ji temple", "Tokyo"), ("Rialto Bridge", "Venice"), ("Ponte Vecchio", "Florence"), ("Kremlin", "Moscow"),
    ("Vasa Museum", "Stockholm"), ("Belem Tower", "Lisbon"), ("Prado Museum", "Madrid"),
    ("Ha'penny Bridge", "Dublin"), ("Chapultepec Castle", "Mexico City"), ("Wat Arun temple", "Bangkok"),
]
TEAMS = ["Falcon", "Heron", "Otter", "Lynx", "Badger", "Kestrel", "Marlin", "Osprey", "Puma", "Raven", "Sparrow",
         "Bison", "Cobra", "Dolphin", "Condor", "Jaguar"]
OFFICES = ["Lisbon", "Oslo", "Denver", "Nairobi", "Osaka", "Montreal", "Krakow", "Austin", "Dublin", "Seoul", "Lima",
           "Perth", "Porto", "Tallinn", "Leeds", "Cairo", "Helsinki", "Bogota"]
NEUTRAL = ["recently adopted a rescue dog", "enjoys painting watercolors on weekends", "is learning to play the cello",
           "volunteers at the community garden", "prefers tea over coffee", "collects vintage postcards",
           "started a small sourdough baking hobby", "keeps a detailed bird-watching journal"]
# two_piece: requirement a, requirement b, near-miss requirement c (never stated for the target person).
REQS = [
    dict(a="{n} passed the site safety inspection on the first attempt.", b="The city council approved {n}'s building permit.",
         c="{n} paid the permit filing fee in full.", qa="passed the site safety inspection", qb="had the building permit approved",
         qc="paid the permit filing fee", la="the inspection", lb="the permit"),
    dict(a="{n} completed the mandatory data-privacy training.", b="{n} signed the confidentiality agreement.",
         c="{n} received a security badge.", qa="completed the data-privacy training", qb="signed the confidentiality agreement",
         qc="received a security badge", la="the training", lb="the agreement"),
    dict(a="{n} submitted a valid passport to the consulate.", b="The consulate approved {n}'s visa application.",
         c="{n} booked a return flight.", qa="submitted a valid passport", qb="had the visa application approved",
         qc="booked a return flight", la="the passport", lb="the visa"),
    dict(a="{n} paid the security deposit for the apartment.", b="{n} provided two references from previous landlords.",
         c="{n} signed the apartment lease.", qa="paid the security deposit", qb="provided two landlord references",
         qc="signed the lease", la="the deposit", lb="the references"),
    dict(a="{n} passed the written driving test.", b="{n} passed the on-road driving test.",
         c="{n} completed the vision screening.", qa="passed the written driving test", qb="passed the on-road driving test",
         qc="completed the vision screening", la="the written test", lb="the road test"),
    dict(a="{n} submitted the final thesis draft.", b="The committee reported that {n} passed the oral defense.",
         c="{n} paid the graduation fee.", qa="submitted the final thesis draft", qb="passed the oral defense",
         qc="paid the graduation fee", la="the thesis draft", lb="the defense"),
    dict(a="{n} completed the forklift certification course.", b="{n} passed the warehouse safety quiz.",
         c="{n} was issued steel-toe boots.", qa="completed the forklift certification course", qb="passed the warehouse safety quiz",
         qc="was issued steel-toe boots", la="the course", lb="the quiz"),
    dict(a="{n} returned all borrowed lab equipment.", b="{n} cleared the outstanding library fines.",
         c="{n} turned in the parking permit.", qa="returned all borrowed lab equipment", qb="cleared the library fines",
         qc="turned in the parking permit", la="the equipment", lb="the fines"),
]


# The training share of each entity list (the rest were held out for testing); REQS: 6 templates train, 2 held out.
NAMES, COLORS, PLACES, LANDMARKS = NAMES[:45], COLORS[:10], PLACES[:9], LANDMARKS[:26]
TEAMS, OFFICES, NEUTRAL, REQS = TEAMS[:10], OFFICES[:12], NEUTRAL[:4], REQS[:6]

LEXICAL_NONE = "The text does not give this code"
PARA_NONE = "None of them"
HOP_NONE = "The text does not say"
SPORTS_Q = "Does the text contain a news story about sports?"
SPORTS_CRITERIA = {
    "true": "At least one passage in the text is a news story mainly about a sporting event, team, athlete or competition.",
    "false": "None of the passages in the text is mainly about sports.",
}


def noul(q, criteria=None):
    d = {"type": "noul", "instructions": q}
    if criteria:
        d["criteria"] = criteria
    return d


def choice(q, options):
    return {"type": "choice", "instructions": q, "criteria": {o: None for o in options}}


@dataclass
class Item:
    item_id: str
    type: str
    filler: str
    length: int  # target state tokens
    depth: float
    rep: int
    questions: dict
    golds_present: dict
    golds_absent: dict
    state_present: str = ""
    state_absent: str = ""
    meta: dict = field(default_factory=dict)


def _rng(*key) -> np.random.Generator:
    h = hashlib.sha256("|".join(map(str, key)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _pick(rng, seq, k):
    return [seq[i] for i in rng.choice(len(seq), k, replace=False)]


def _forbid(terms) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.I)


def _second_depth(d: float) -> float:
    return d + 0.5 if d <= 0.5 else d - 0.5


def _spec(itype, filler, rng):
    """Evidence, extras and questions for one item (independent of length/placement)."""
    hard = filler == "hard"
    if itype == "lexical":
        (color, color2, color3), (place, place2, place3) = _pick(rng, COLORS, 3), _pick(rng, PLACES, 3)
        code, c2, c3, c4, c5 = [str(x) for x in rng.choice(np.arange(3000, 10000), 5, replace=False)]
        evidence = [f"The access code for the {color} {place} locker is {code}."]
        near = [f"The access code for the {color2} {place} locker is {c2}.",
                f"The access code for the {color} {place2} locker is {c3}.",
                f"The access code for the {color} {place} cabinet is {c4}.",
                f"The access code for the {color3} {place3} locker is {c5}."]
        opts = [code, c2, c3, c4, LEXICAL_NONE]
        rng.shuffle(opts)
        qs = {"q_pos": noul(f"Does the text state that the access code for the {color} {place} locker is {code}?"),
              "q_mis": noul(f"Does the text state that the access code for the {color} {place} locker is {c2}?"),
              "choice": choice(f"According to the text, what is the access code for the {color} {place} locker?", opts)}
        gp, ga = {"q_pos": 1, "q_mis": 0, "choice": code}, {"q_pos": 0, "q_mis": 0, "choice": LEXICAL_NONE}
        return evidence, (near if hard else []), qs, gp, ga, ["access code", "locker", code, c2, c3, c4, c5]

    if itype == "paraphrase":
        people = _pick(rng, NAMES, 5)
        lms = _pick(rng, LANDMARKS, 5)
        (n, n2, n3, n4, n5), ((l, c), (l2, c2), (l3, c3), (l4, c4), (l5, c5)) = people, lms
        evidence = [f"Actually, {n} lives next to the {l}."]
        acts = _pick(rng, NEUTRAL, 4)
        neutral = [f"{p} {a}." for p, a in zip((n, n2, n3, n4), acts)]
        near = [f"{n2} lives next to the {l2}.", f"{n3} lives next to the {l3}.",
                f"{n4} lives next to the {l4}.", f"{n5} lives next to the {l5}."]
        opts = [n, n2, n3, n4, PARA_NONE]
        rng.shuffle(opts)
        # Wording picked by phase1/probe_paraphrase.py: 19/20 positives and 20/20 mismatches right at 500 tokens.
        qs = {"q_pos": noul(f"Based on the text, does {n} live in {c}?"),
              "q_mis": noul(f"Based on the text, does {n} live in {c2}?"),
              "choice": choice(f"According to the text, which of these people has been to {c}?", opts)}
        gp, ga = {"q_pos": 1, "q_mis": 0, "choice": n}, {"q_pos": 0, "q_mis": 0, "choice": PARA_NONE}
        terms = [x for pair in lms for x in pair] + [p.split()[-1] for p in people] + ["lives next to"]
        return evidence, neutral + (near if hard else []), qs, gp, ga, terms

    if itype == "categorical":
        needles, _ = fillers.agnews_train()
        evidence = [needles[rng.integers(len(needles))]]
        qs = {"q_pos": noul(SPORTS_Q), "q_crit": noul(SPORTS_Q, SPORTS_CRITERIA)}
        return evidence, [], qs, {"q_pos": 1, "q_crit": 1}, {"q_pos": 0, "q_crit": 0}, None

    if itype == "two_piece":
        n, n2, n3 = _pick(rng, NAMES, 3)
        r = REQS[rng.integers(len(REQS))]
        evidence = [r["a"].format(n=n), r["b"].format(n=n)]
        near = [r["a"].format(n=n2), r["b"].format(n=n3), r["c"].format(n=n2), r["a"].format(n=n3)]
        opts = [f"Both {r['la']} and {r['lb']}", f"Only {r['la']}", f"Only {r['lb']}", "Neither"]
        shuffled = list(opts)
        rng.shuffle(shuffled)
        qs = {"q_pos": noul(f"Does the text show that {n} both {r['qa']} and {r['qb']}?"),
              "q_mis": noul(f"Does the text show that {n} both {r['qa']} and {r['qc']}?"),
              "choice": choice(f"According to the text, which of these requirements has {n} met?", shuffled)}
        gp, ga = {"q_pos": 1, "q_mis": 0, "choice": opts[0]}, {"q_pos": 0, "q_mis": 0, "choice": opts[1]}
        return evidence, (near if hard else []), qs, gp, ga, [p.split()[-1] for p in (n, n2, n3)]

    if itype == "two_hop":
        n, n2, n3 = _pick(rng, NAMES, 3)
        t, t2, t3 = _pick(rng, TEAMS, 3)
        c, c2, c3, c4 = _pick(rng, OFFICES, 4)
        evidence = [f"{n} was transferred to the {t} team last month.", f"The {t} team works out of the {c} office."]
        near = [f"The {t2} team works out of the {c2} office.", f"The {t3} team works out of the {c3} office.",
                f"{n2} was transferred to the {t2} team last month.", f"{n3} was transferred to the {t3} team last month."]
        opts = [c, c2, c3, c4, HOP_NONE]
        rng.shuffle(opts)
        qs = {"q_pos": noul(f"According to the text, does {n} work out of the {c} office?"),
              "q_mis": noul(f"According to the text, does {n} work out of the {c2} office?"),
              "choice": choice(f"According to the text, which office does {n} work out of?", opts)}
        gp, ga = {"q_pos": 1, "q_mis": 0, "choice": c}, {"q_pos": 0, "q_mis": 0, "choice": HOP_NONE}
        return evidence, (near if hard else []), qs, gp, ga, [p.split()[-1] for p in (n, n2, n3)] + [c, c2, c3, c4]

    raise ValueError(itype)


def make_item(itype: str, filler: str, length: int, depth: float, rep: int, chars_per_token: dict, pools: dict) -> Item:
    """One item; `pools[(itype, filler)]` is its filler pool."""
    rng = _rng("phase1-v1", itype, filler, length, depth, rep)   # seed text from the generator's first version, kept so the data reproduces
    evidence, extras, qs, gp, ga, terms = _spec(itype, filler, rng)
    paragraph_mode = itype == "categorical"
    corpus = "agnews" if (itype, filler) == ("categorical", "hard") else "squad"
    pool = pools[(itype, filler)]
    forbidden = _forbid(terms) if terms else None

    inserted_chars = sum(len(s) + 2 for s in evidence + extras)
    target_chars = int(length * chars_per_token[corpus]) - inserted_chars
    paras = build_filler(pool, max(target_chars, 200), rng, forbidden, paragraph_mode)
    units, cum = slots_and_offsets(paras, paragraph_mode)

    depths = [depth] if len(evidence) == 1 else [depth, _second_depth(depth)]
    ev_slots = [depth_slot(cum, d) for d in depths]
    free = [k for k in range(len(cum)) if k not in ev_slots]
    extra_slots = list(rng.choice(free, len(extras), replace=len(extras) > len(free))) if extras else []

    def build(with_all_evidence: bool) -> str:
        inserts: dict[int, list[str]] = {}
        for k, s in zip(extra_slots, extras):
            inserts.setdefault(int(k), []).append(s)
        for i, (k, s) in enumerate(zip(ev_slots, evidence)):
            # the absent state drops the single evidence sentence, or the second fact for two-fact types
            if not with_all_evidence and (len(evidence) == 1 or i == 1):
                continue
            inserts.setdefault(k, []).append(s)
        return assemble(paras, units, inserts, paragraph_mode)

    item = Item(item_id=f"{itype}|{filler}|{length}|{depth}|{rep}", type=itype, filler=filler, length=length,
                depth=depth, rep=rep, questions=qs, golds_present=gp, golds_absent=ga,
                meta={"corpus": corpus, "evidence": evidence, "extras": extras})
    item.state_present, item.state_absent = build(True), build(False)
    return item


CPT = {"squad": 4.4, "agnews": 4.5}   # nominal characters per token, to size the filler


@lru_cache
def filler_pools() -> dict:
    squad = fillers.squad_paragraphs()
    pools = {(t, f): squad for t in TYPES for f in FILLERS}
    pools[("categorical", "unrelated")] = fillers.squad_nonsports_filtered()
    pools[("categorical", "hard")] = fillers.agnews_train()[1]
    return pools


def evidence_records(itype: str, filler: str, length: int, depth: float, rep: int) -> list[dict]:
    """The paired present/absent training records of one item."""
    it = make_item(itype, filler, length, depth, rep + 100_000, dict(CPT), filler_pools())
    out = []
    for variant, state, golds in (("present", it.state_present, it.golds_present),
                                  ("absent", it.state_absent, it.golds_absent)):
        qs = {}
        for qid, q in it.questions.items():
            g = golds[qid]
            qs[qid] = {**q, "label": bool(g) if q["type"] == "noul" else g}
        out.append({"state": state, "questions": qs,
                    "_meta": {"family": "evidence", "type": itype, "filler": filler, "length": length, "depth": depth,
                              "rep": rep, "variant": variant, "split": "train",
                              "evidence": [] if itype == "categorical" else it.meta["evidence"]}})
    return out
