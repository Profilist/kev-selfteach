"""Figures for the README and the write-up, drawn from the tracked summaries in results/*.json.

No titles inside the images: captions carry them. Each figure is a PNG, except 4_method.svg (animated, CSS only, so it
plays inside an <img>); the numbers behind each one go to figures/data.json.

    uv run python results/figures.py      # -> results/figures/{1_hero,...,9_sizes}.png, 4_method.svg, data.json
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
RES = HERE
OUT = HERE / "figures"
INK, INK2, MUTED, GRID, RULE, SURFACE, CELL = "#11151b", "#424955", "#737c8a", "#e9edf2", "#d9dee6", "#ffffff", "#dfe4eb"
# Forest (self-taught, the result) and plum (released Kev); every other model is one neutral grey.
OURS, KEV, LABELS, FLASH, CLEF, JEV = "#2f6b4f", "#9b5a8a", "#8e9691", "#d08b00", "#2b3139", "#8e9691"
ROUNDS = ["#a9c9b8", "#5f9a7d", "#2f6b4f"]
plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 11,
                     "axes.edgecolor": RULE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE, "svg.fonttype": "none"})

dense = json.load(open(RES / "complaints.json"))
contracts = json.load(open(RES / "contracts.json"))
long = json.load(open(RES / "complaints_long.json"))
st = json.load(open(RES / "selfteach.json"))
K_KEV, K_LAB0, K_LAB1 = "Kev-0.8B 1.0", "Kev-0.8B 1.0 trained on labels, labels-32k (ours)", "Kev-0.8B 1.0 trained on labels, labels-32k-seed1 (ours)"
K_ST0, K_ST1 = "Kev-0.8B 1.0 self-taught, fresh-32k (ours)", "Kev-0.8B 1.0 self-taught, fresh-32k-seed1 (ours)"
K_R = ["Self-taught, fresh-32k", "Self-taught, fresh-64k", "Self-taught, fresh-128k"]
K_SIZES = [("Kev-0.8B", K_KEV), ("Kev-4B", "Kev-4B 1.0"), ("Kev-9B", "Kev-9B 1.0"), ("Kev-27B", "Kev-27B 1.0")]   # released Kevs
LONG_B = ["alone", "32k", "64k", "128k", "256k"]
PER_READ = ["1\n(alone)", "~123\n32k", "~246\n64k", "~490\n128k", "~990\n256k"]
DATA = {}


def mean(*xs):
    return sum(xs) / len(xs)


def acc(report, key, b):
    return report[key][b]["acc"]


def clean(ax, grid="y"):
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis=grid, color=GRID, lw=1)
    ax.set_axisbelow(True)


def pale(color):
    """A light, opaque tint of a colour (opaque so gridlines don't show through), for read-alone tracks."""
    return tuple(0.22 * v + 0.78 for v in matplotlib.colors.to_rgb(color))


def center_horizontally(fig, ax):
    """Shift the axes so the whole chart, tick labels included, has equal white space left and right."""
    fig.canvas.draw()
    box = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    shift = ((1 - box.x1) - box.x0) / 2
    pos = ax.get_position()
    ax.set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def end_label(ax, x, y, text, color=INK2, dy=0.0, bold=False):
    ax.text(x + 0.12, y + dy, text, va="center", fontsize=10.5, color=color, fontweight="bold" if bold else "normal")


def hero():
    """Released Kev vs the last self-taught round, from one complaint per read to ~990; hosted models stop at their window."""
    r3, kev = long[K_R[2]], long[K_KEV]
    ours = [r3[b]["acc"] for b in LONG_B]
    released = [kev[b]["acc"] for b in LONG_B]
    jev = [acc(dense, "Jev 1.13 (hosted; 64k split into two 32k halves)", b) for b in ("alone", "32k")]
    clef = [acc(dense, "Clef 27B (Workers AI)", b) for b in ("alone", "32k", "64k")]
    DATA["1_hero"] = {"x_complaints_per_read": [1, 123, 246, 490, 990], "x_length": LONG_B, "self_taught": ours,
                      "released_kev": released, "jev_to_32k_window": jev, "clef27b_to_64k_window": clef, "chance": 1 / 9}
    fig, ax = plt.subplots(figsize=(9, 5.2))
    xs = range(5)
    ax.axvspan(2, 5.1, color="#f4f5f2", lw=0, zorder=0)
    ax.text(2.08, 0.965, "past Jev's and Clef's limits", fontsize=9.5, color=MUTED, va="top")
    ax.plot(xs, released, color=KEV, lw=2.2, marker="o", ms=7, mec=SURFACE, mew=1.5, zorder=4)
    ax.plot(xs, ours, color=OURS, lw=3.2, marker="o", ms=8, mec=SURFACE, mew=1.5, zorder=5)
    end_label(ax, 4, ours[-1], f"Self-taught  {100 * ours[-1]:.0f}%", INK, bold=True)
    end_label(ax, 4, released[-1], f"Kev-0.8B  {100 * released[-1]:.0f}%", INK2)
    ax.set_xlim(-0.2, 5.1); ax.set_ylim(0, 1)
    ax.set_xticks(list(xs)); ax.set_xticklabels(["", "32k", "64k", "128k", "256k"], fontsize=10)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticklabels(["0", "20%", "40%", "60%", "80%", "100%"])
    ax.set_xlabel("context window", fontsize=10.5, labelpad=8)
    ax.set_ylabel("accuracy", fontsize=10.5)
    clean(ax)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.96, bottom=0.14)
    save(fig, "1_hero")


def test_strip():
    """One 64k document from the test: 241 complaints back to back; question 208 points at one of them."""
    n, cols, target = 241, 25, 207   # packed-complaints-v1/64k/0007; complaint 6292578 is at position 208
    rows = -(-n // cols)
    DATA["2_test"] = {"document": "packed-complaints-v1/64k/0007", "complaints": n, "state_tokens_approx": 64000,
                      "question": "Which kind of financial product is complaint 6292578 about?", "target_position": target + 1,
                      "first_complaint_label": "mortgage"}
    fig = plt.figure(figsize=(9, 4.4))
    ax = fig.add_axes([0.02, 0.06, 0.54, 0.86]); ax.axis("off")
    ax.set_xlim(-0.5, cols); ax.set_ylim(rows + 0.3, -0.3)
    for i in range(n):
        r, c = divmod(i, cols)
        color = INK2 if i == 0 else OURS if i == target else CELL   # complaint 1 in the teacher's dark grey
        ax.add_patch(Rectangle((c, r), 0.82, 0.78, color=color, lw=0))
    tr, tc = divmod(target, cols)
    ax.annotate("", xy=(tc + 0.85, tr + 0.4), xytext=(cols + 0.4, tr + 0.4), annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=OURS, lw=1.6, mutation_scale=14))
    tx = fig.add_axes([0.58, 0.06, 0.4, 0.86]); tx.axis("off"); tx.set_xlim(0, 1); tx.set_ylim(0, 1)

    def row_y(r):
        """Vertical centre of grid row r, in the text axes' coordinates, so the legend and box line up with the tiles."""
        return tx.transAxes.inverted().transform(ax.transData.transform((0, r + 0.39)))[1]

    box_h = 0.33
    box_top = row_y(tr) + box_h / 2   # centred on the arrow's row
    tx.add_patch(FancyBboxPatch((0.02, box_top - box_h), 0.94, box_h, boxstyle="round,pad=0,rounding_size=0.03", fc="#edf3ef", ec=OURS, lw=1.3))
    tx.text(0.07, box_top - 0.06, "Question 208 of 241", fontsize=10, color=OURS, fontweight="bold", va="top")
    tx.text(0.07, box_top - 0.16, f"Which kind of financial product is\ncomplaint {target + 1} about?", fontsize=11, color=INK,
            va="top", linespacing=1.4)
    for row, color, text in ((0, INK2, "complaint 1: a mortgage"), (1, OURS, f"complaint {target + 1}: a checking account")):
        y = row_y(row)
        tx.add_patch(Rectangle((0.04, y - 0.02), 0.04, 0.04, color=color, lw=0))
        tx.text(0.11, y, text, fontsize=9.5, color=INK2, va="center")
    save(fig, "2_test")


def example():
    """The same complaint (6292578, a closed PNC checking account): alone, packed for released Kev, packed self-taught."""
    options = ["Credit report", "Debt collection", "Mortgage", "Checking or savings", "Credit card", "Student loan",
               "Money transfer", "Vehicle loan", "Personal loan"]
    # rows.json of kev-1.0 and fresh-128k on packed-complaints-v1 (local data/kevbench/results), question c6292578
    panels = [("Alone", [0.0, 0.0, 0.0, 0.994, 0.006, 0.0, 0.0, 0.0, 0.0]),
              ("Packed", [0.001, 0.001, 0.816, 0.032, 0.017, 0.002, 0.064, 0.029, 0.038]),
              ("Packed, self-taught", [0.0, 0.0, 0.0, 0.998, 0.001, 0.0, 0.0, 0.0, 0.0])]
    DATA["3_example"] = {"complaint": "6292578", "correct": "Checking or savings", "options": options,
                         "panels": {t: p for t, p in panels}}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    ys = list(range(len(options)))[::-1]
    for ax, (title, p) in zip(axes, panels):
        top = max(range(9), key=p.__getitem__)
        colors = [OURS if i == 3 and top == 3 else KEV if i == top else RULE for i in range(9)]
        ax.barh(ys, p, height=0.66, color=colors)
        ax.text(p[top] + 0.04 if p[top] < 0.8 else p[top] - 0.03, ys[top], f"{100 * p[top]:.0f}%", va="center",
                ha="left" if p[top] < 0.8 else "right", fontsize=10, color=SURFACE if p[top] >= 0.8 else INK, fontweight="bold")
        ax.set_title(title, fontsize=11, color=INK, loc="center", pad=8)
        ax.set_xlim(0, 1); ax.set_xticks([])
        for name, sp in ax.spines.items():  # left and right edges are 0 and 100%, so bar lengths read against them
            sp.set_visible(name in ("left", "right"))
            sp.set_color(RULE)
        ax.tick_params(length=0)
    axes[0].set_yticks(ys); axes[0].set_yticklabels(options, fontsize=10, color=INK2)
    right = axes[0].get_yticklabels()[3]  # the right answer, checking or savings
    right.set_color(INK); right.set_fontweight("bold")
    fig.subplots_adjust(left=0.15, right=0.98, top=0.88, bottom=0.03, wspace=0.12)
    save(fig, "3_example")


def mini_bars(ax, x, y, w, h, p, highlight):
    bw = w / len(p)
    for i, v in enumerate(p):
        ax.add_patch(Rectangle((x + i * bw + bw * 0.12, y), bw * 0.76, max(h * v, 0.004), color=highlight if v == max(p) else RULE, lw=0))
    ax.plot([x, x + w], [y, y], color=RULE, lw=1)


def method():
    """Teacher over student, as an animated SVG (CSS only, so it plays inside an <img>).

    The student's highlight steps through packed complaints; for each, the teacher's answer for that complaint read
    alone becomes the target, and the student's answer moves from its packed guess (some copy the first complaint,
    a mortgage; some are spread thin) to match it. Reduced motion shows the first frame, already matched.
    """
    # Nine products, in the order of figure 3. Illustrative answers, shaped like the real ones.
    teacher = [
        [0.00, 0.00, 0.02, 0.96, 0.01, 0.00, 0.00, 0.00, 0.01],   # checking or savings
        [0.90, 0.03, 0.02, 0.00, 0.03, 0.00, 0.00, 0.00, 0.02],   # credit report
        [0.05, 0.46, 0.06, 0.03, 0.10, 0.02, 0.22, 0.02, 0.04],   # debt collection, weighing a few options
        [0.00, 0.00, 0.02, 0.04, 0.88, 0.00, 0.04, 0.00, 0.02],   # credit card
        [0.02, 0.03, 0.00, 0.01, 0.02, 0.90, 0.00, 0.01, 0.01],   # student loan
        [0.02, 0.06, 0.04, 0.14, 0.05, 0.01, 0.44, 0.03, 0.21],   # money transfer, spread over three
    ]
    # The student's packed answer before it matches: some copy complaint 1 (a mortgage), some are spread thin.
    packed_guess = [
        [0.03, 0.04, 0.70, 0.06, 0.03, 0.02, 0.06, 0.03, 0.03],
        [0.12, 0.10, 0.22, 0.08, 0.14, 0.06, 0.12, 0.07, 0.09],
        [0.05, 0.08, 0.55, 0.10, 0.04, 0.03, 0.08, 0.03, 0.04],
        [0.08, 0.15, 0.18, 0.12, 0.10, 0.09, 0.11, 0.08, 0.09],
        [0.04, 0.05, 0.62, 0.05, 0.06, 0.04, 0.06, 0.04, 0.04],
        [0.10, 0.13, 0.20, 0.11, 0.07, 0.10, 0.12, 0.08, 0.09],
    ]
    visits = [26, 5, 19, 33, 10, 29]   # which packed complaint is being read at each step (12 x 3 grid)
    DATA["4_method"] = {"teacher_examples": teacher, "packed_guess": packed_guess, "illustrative": True,
                        "training": {"steps": 598, "minutes": 46, "gpu": "one H100 NVL"}}

    W, H_, T, S = 620, 240, 76, 180           # canvas, then the teacher and student rows' centre lines
    BX, PITCH, BW, BH = 420, 14, 10, 56       # answer column: nine bars
    COLS, ROWS, TW, TH, GAP = 12, 3, 10, 11, 3
    STEP, N = 2.6, len(teacher)
    TOTAL = STEP * N
    FAST, HOLD, SETTLE = 0.45, 0.35, 0.5 * STEP  # seconds: the target switching, the student's guess held, then converging
    FADE = 0.6 * STEP                             # seconds: read complaints fading back to grey before the loop restarts
    PALE = "#b3cfbf"
    pct = lambda sec: f"{100 * sec / TOTAL:.2f}%"
    # Start each loop mid-way through step 1, already matched, so a browser that hasn't started the animation yet
    # (or a still preview) shows a complete frame rather than the instant before the first switch.
    START = f"animation-delay:-{FAST + HOLD + SETTLE:.2f}s"
    floor = lambda v: max(v, 0.05)            # a short stub for options at ~0, like the other charts
    css, body = [], []

    def keyframes(name, frames, prop):
        css.append(f"@keyframes {name}{{" + "".join(f"{pct(t)}{{{prop(v)}}}" for t, v in frames) + "}")

    def bar_frames(values_at, settle):
        frames = [(0, values_at(N - 1)[1])]
        for k in range(N):
            t0 = k * STEP
            prev, now = values_at(k - 1)[1], values_at(k)
            frames += [(t0, prev)] + ([(t0 + FAST, now[0]), (t0 + FAST + HOLD, now[0]), (t0 + FAST + HOLD + SETTLE, now[1])] if settle else [(t0 + FAST, now[1])])
        return sorted(frames, key=lambda f: f[0]) + [(TOTAL, values_at(N - 1)[1])]

    def bars(cy, color, cls, values_at, settle):
        base = cy + BH / 2
        for j in range(9):
            name = f"{cls}{j}"
            keyframes(name, bar_frames(lambda k: (packed_guess[k][j], teacher[k][j]), settle) if settle
                      else bar_frames(lambda k: (None, teacher[k][j]), settle),
                      lambda v: f"transform:scaleY({floor(v):.3f})")
            body.append(f'<rect class="a" x="{BX + j * PITCH}" y="{base - BH}" width="{BW}" height="{BH}" fill="{color}" '
                        f'style="transform:scaleY({floor(teacher[0][j]):.3f});animation:{name} {TOTAL}s infinite;{START}"/>')
        body.append(f'<line x1="{BX - 4}" y1="{base}" x2="{BX + 8 * PITCH + BW + 4}" y2="{base}" stroke="{CELL}"/>')

    def text(x, y, s, size=13, color=INK2, bold=False, middle=False):
        body.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}"' + (' font-weight="700"' if bold else "")
                    + (' dominant-baseline="middle"' if middle else "") + f">{s}</text>")

    def arrow(x1, y1, x2, y2):
        body.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{MUTED}" stroke-width="1.5" marker-end="url(#ah)"/>')

    # Teacher row: one complaint, read alone.
    text(24, T - 3, "Teacher", 13.5, INK, bold=True); text(24, T + 15, "read alone", 12, MUTED)
    body.append(f'<rect x="150" y="{T - 13}" width="20" height="26" fill="{INK2}"/>')
    arrow(182, T, BX - 20, T)
    bars(T, INK2, "t", None, settle=False)

    # The teacher's answer becomes the student's target.
    cx = BX + (8 * PITCH + BW) / 2
    arrow(cx, T + BH / 2 + 8, cx, S - BH / 2 - 6)
    text(cx + 12, (T + S) / 2, "target", 12, MUTED, middle=True)

    # Student row: the same complaints, packed. The one being read is blue; ones already read stay pale.
    text(24, S - 3, "Student", 13.5, INK, bold=True); text(24, S + 15, "read packed", 12, MUTED)
    top = S - (ROWS * TH + (ROWS - 1) * GAP) / 2
    for i in range(COLS * ROWS):
        x, y = 150 + (i % COLS) * (TW + GAP), top + (i // COLS) * (TH + GAP)
        attrs = ""
        if i in visits:
            k = visits.index(i)
            on, off = k * STEP, (k + 1) * STEP
            if k < N - 1:   # grey -> blue while it's read -> pale once read -> back to grey as the loop ends
                frames = [(0, CELL), (on, CELL), (on + FAST, OURS), (off, OURS), (off + FAST, PALE),
                          (TOTAL - FADE, PALE), (TOTAL, CELL)]
            else:           # the last one is still blue at the wrap, and fades out as the first lights up
                frames = [(0, OURS), (FAST, CELL), (on, CELL), (on + FAST, OURS), (TOTAL, OURS)]
            keyframes(f"c{i}", frames, lambda v: f"fill:{v}")
            attrs = f' class="a" style="fill:{OURS if k == 0 else CELL};animation:c{i} {TOTAL}s infinite;{START}"'
        body.append(f'<rect x="{x}" y="{y}" width="{TW}" height="{TH}" fill="{CELL}"{attrs}/>')
    arrow(150 + COLS * (TW + GAP) + 10, S, BX - 20, S)
    bars(S, OURS, "s", None, settle=True)

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H_}" width="{W * 3}" height="{H_ * 3}" '
           f'font-family="\'Helvetica Neue\', Helvetica, Arial, sans-serif" role="img">'
           "<title>Teacher and student are the same model</title>"
           "<desc>The teacher reads each complaint alone and its answer becomes the target. The student reads the same "
           "complaints packed among others; for each one, its answer moves from a guess that copies the first complaint "
           "to match the teacher's.</desc>"
           f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
           f'<path d="M0 0L10 5L0 10z" fill="{MUTED}"/></marker></defs>'
           "<style>.a{transform-box:fill-box;transform-origin:50% 100%;animation-timing-function:cubic-bezier(.45,0,.25,1)}"
           + "".join(css)
           + "@media (prefers-reduced-motion:reduce){.a{animation:none!important}}</style>"
           + f'<rect width="{W}" height="{H_}" fill="{SURFACE}"/>' + "".join(body) + "</svg>")
    (OUT / "4_method.svg").write_text(svg)
    (OUT / "4_method.png").unlink(missing_ok=True)   # the animated SVG replaces the old still


def at_64k():
    rows = [("Kev-27B", acc(long, "Kev-27B 1.0", "64k"), acc(long, "Kev-27B 1.0", "alone"), JEV),
            ("Jev 1.13*", acc(dense, "Jev 1.13 (hosted; 64k split into two 32k halves)", "64k"),
             acc(dense, "Jev 1.13 (hosted; 64k split into two 32k halves)", "alone"), JEV),
            ("Clef 27B", acc(dense, "Clef 27B (Workers AI)", "64k"), acc(dense, "Clef 27B (Workers AI)", "alone"), JEV),
            ("Self-taught Kev-0.8B", mean(acc(dense, K_ST0, "64k"), acc(dense, K_ST1, "64k")),
             mean(acc(dense, K_ST0, "alone"), acc(dense, K_ST1, "alone")), OURS),
            ("Kev-0.8B w/ labels", mean(acc(dense, K_LAB0, "64k"), acc(dense, K_LAB1, "64k")),
             mean(acc(dense, K_LAB0, "alone"), acc(dense, K_LAB1, "alone")), JEV),
            ("Clef-flash 9B", acc(dense, "Clef-flash 9B (Workers AI)", "64k"), acc(dense, "Clef-flash 9B (Workers AI)", "alone"), JEV),
            ("Kev-9B", acc(long, "Kev-9B 1.0", "64k"), acc(long, "Kev-9B 1.0", "alone"), JEV),
            ("Kev-0.8B", acc(dense, K_KEV, "64k"), acc(dense, K_KEV, "alone"), KEV)]
    DATA["5_at_64k"] = {"note": "accuracy with ~246 complaints per read; 'alone' = the same model reading each complaint by "
                                "itself; Kev-0.8B rows of ours are the mean of two seeds; *Jev reads 64k as two 32k halves",
                        "rows": [{"model": m, "packed_64k": a, "alone": al} for m, a, al, _ in rows]}
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ys = list(range(len(rows)))[::-1]
    for y, (m, a, al, c) in zip(ys, rows):
        ax.barh(y, al, height=0.6, color=pale(c))  # pale track: the same model reading each complaint alone
        ax.barh(y, a, height=0.6, color=c)  # solid: packed at 64k
        ax.text(a - 0.015 if a > 0.3 else a + 0.015, y, f"{100 * a:.0f}%", va="center", ha="right" if a > 0.3 else "left",
                fontsize=10.5, color=SURFACE if a > 0.3 else INK, fontweight="bold")
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=10.5, color=INK2)
    for lab, r in zip(ax.get_yticklabels(), rows):
        if r[3] == OURS:
            lab.set_color(INK); lab.set_fontweight("bold")
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_xticklabels(["0", "20%", "40%", "60%", "80%", "100%"])
    clean(ax, "x")
    fig.subplots_adjust(left=0.2, right=0.97, top=0.97, bottom=0.09)
    center_horizontally(fig, ax)
    save(fig, "5_at_64k")


def sizes():
    """Released Kev at every size, alone vs packed at 64k (same construction as figure 5)."""
    rows = [(name, acc(long, key, "64k"), acc(long, key, "alone"), KEV) for name, key in K_SIZES]
    DATA["9_sizes"] = {"note": "accuracy with ~246 complaints per read (64k) and reading each complaint alone; released Kev-4B, "
                               "9B and 27B scored 2026-10-09 with the chunked scorer; Kev-27B is the only one trained on long states",
                       "rows": [{"model": m, "packed_64k": a, "alone": al} for m, a, al, _ in rows]}
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ys = [3, 2, 1, 0]
    for y, (m, a, al, c) in zip(ys, rows):
        ax.barh(y, al, height=0.66, color=pale(c))  # pale track: the same model reading each complaint alone
        ax.barh(y, a, height=0.66, color=c)  # solid: packed at 64k
        ax.text(a - 0.015, y, f"{100 * a:.0f}%", va="center", ha="right", fontsize=10.5, color=SURFACE, fontweight="bold")
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=10.5, color=INK2)
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, 3.6)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_xticklabels(["0", "20%", "40%", "60%", "80%", "100%"])
    clean(ax, "x")
    fig.subplots_adjust(left=0.2, right=0.97, top=0.97, bottom=0.1)
    center_horizontally(fig, ax)
    save(fig, "9_sizes")


def lookup():
    s, e = st["complaints"], st["complaints_edge_errors"]
    same = [("Kev-0.8B", s[K_KEV]["64k"]["agree_own_alone"], KEV),
            ("Trained on labels", mean(s[K_LAB0]["64k"]["agree_own_alone"], s[K_LAB1]["64k"]["agree_own_alone"]), LABELS),
            ("Self-taught", mean(s[K_ST0]["64k"]["agree_own_alone"], s[K_ST1]["64k"]["agree_own_alone"]), OURS)]
    edge = [("Kev-0.8B", e[K_KEV]["64k"]["edge"] / e[K_KEV]["64k"]["wrong"], KEV),
            ("Trained on labels", sum(e[k]["64k"]["edge"] for k in (K_LAB0, K_LAB1)) / sum(e[k]["64k"]["wrong"] for k in (K_LAB0, K_LAB1)), LABELS),
            ("Self-taught", sum(e[k]["64k"]["edge"] for k in (K_ST0, K_ST1)) / sum(e[k]["64k"]["wrong"] for k in (K_ST0, K_ST1)), OURS)]
    DATA["6_lookup"] = {"length": "64k (~246 complaints per read)", "same_answer_as_alone": {m: v for m, v, _ in same},
                        "wrong_answers_naming_first_or_last_complaint": {m: v for m, v, _ in edge}}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, data, title in ((axes[0], same, "Gives the answer it gives\nthe complaint alone"),
                            (axes[1], edge, "Of its wrong answers, share naming\nthe first or last complaint's product")):
        ys = list(range(3))[::-1]
        for y, (m, v, c) in zip(ys, data):
            ax.barh(y, v, height=0.58, color=c)
            ax.text(v + 0.02, y, f"{100 * v:.0f}%", va="center", fontsize=10.5, color=INK, fontweight="bold")
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8, linespacing=1.3)
        ax.set_xlim(0, 1.12); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "50%", "100%"], fontsize=9)
        clean(ax, "x")
    axes[0].set_yticks(list(range(3))[::-1]); axes[0].set_yticklabels([m for m, _, _ in same], fontsize=10.5, color=INK2)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.8, bottom=0.1, wspace=0.12)
    save(fig, "6_lookup")


def rounds():
    series = [(K_KEV, "Kev-0.8B", KEV, None)] + [(k, f"Round {i + 1}: trained to {t}", ROUNDS[i], t)
                                                     for i, (k, t) in enumerate(zip(K_R, ("32k", "64k", "128k")))]
    DATA["7_rounds"] = {"x_length": LONG_B, "x_complaints_per_read": [1, 123, 246, 490, 990],
                        "series": {lab: {"acc": [long[k][b]["acc"] for b in LONG_B], "longest_training_doc": t} for k, lab, _, t in series}}
    fig, ax = plt.subplots(figsize=(9, 5.2))  # same format as the hero (figure 1)
    for k, lab, c, t in series:
        ys = [long[k][b]["acc"] for b in LONG_B]
        ax.plot(range(5), ys, color=c, lw=3.2 if t == "128k" else 2.2, marker="o", ms=6.5, mec=SURFACE, mew=1.3, zorder=4)
        if t:
            ax.plot([LONG_B.index(t)], [long[k][t]["acc"]], marker="D", ms=10, mfc=SURFACE, mec=c, mew=2.2, zorder=5)
        name = lab.split(":")[0]
        end_label(ax, 4, ys[-1], f"{name}  {100 * ys[-1]:.0f}%", INK if t == "128k" else INK2, bold=t == "128k")
    ax.set_xlim(-0.2, 5.1); ax.set_ylim(0, 1)
    ax.set_xticks(range(5)); ax.set_xticklabels(["", "32k", "64k", "128k", "256k"], fontsize=10)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticklabels(["0", "20%", "40%", "60%", "80%", "100%"])
    ax.set_xlabel("context window", fontsize=10.5, labelpad=8)
    ax.set_ylabel("accuracy", fontsize=10.5)
    clean(ax)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.96, bottom=0.14)
    save(fig, "7_rounds")


def ceiling():
    groups = [("Complaints\n(~246 per 64k input)", acc(dense, K_KEV, "alone"), acc(dense, K_KEV, "64k"),
               mean(acc(dense, K_ST0, "64k"), acc(dense, K_ST1, "64k"))),
              ("Contracts\n(~14 per 64k input)", acc(contracts, K_KEV, "alone"), acc(contracts, K_KEV, "64k"),
               mean(acc(contracts, K_ST0, "64k"), acc(contracts, K_ST1, "64k")))]
    DATA["8_ceiling"] = {"length": "64k", "groups": [{"group": g.replace("\n", " "), "released_alone": a, "released_packed": k,
                                                      "self_taught_packed": s} for g, a, k, s in groups]}
    # Same construction as figure 5: each packed bar sits on a pale track running to the read-alone score.
    fig, ax = plt.subplots(figsize=(9, 3.8))
    label_x = matplotlib.transforms.blended_transform_factory(fig.transFigure, ax.transData)
    ys, names = [], []
    for top, (g, alone, kev, ours) in zip((3.6, 1.0), groups):
        name, sub = g.strip(")").split("\n(")
        ax.text(0.03, top, name, transform=label_x, fontsize=10.5, color=INK, fontweight="bold", va="center")
        ax.text(0.03, top - 0.32, sub.replace("the kind it ", ""), transform=label_x, fontsize=9.5, color=MUTED, va="center")
        for y, (who, v, c) in zip((top, top - 1), (("Kev-0.8B", kev, KEV), ("Self-taught", ours, OURS))):
            ax.barh(y, alone, height=0.66, color=pale(c))
            ax.barh(y, v, height=0.66, color=c)
            ax.text(v - 0.015 if v > 0.3 else v + 0.015, y, f"{100 * v:.0f}%", va="center", ha="right" if v > 0.3 else "left",
                    fontsize=10.5, color=SURFACE if v > 0.3 else INK, fontweight="bold")
            ys.append(y); names.append(who)
    ax.set_yticks(ys); ax.set_yticklabels(names, fontsize=10.5, color=INK2)
    ax.set_xlim(0, 1); ax.set_ylim(-0.55, 4.15)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_xticklabels(["0", "20%", "40%", "60%", "80%", "100%"])
    clean(ax, "x")
    fig.subplots_adjust(left=0.3, right=0.96, top=0.97, bottom=0.1)
    save(fig, "8_ceiling")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    hero(); test_strip(); example(); method(); at_64k(); sizes(); lookup(); rounds(); ceiling()
    json.dump(DATA, open(OUT / "data.json", "w"), indent=1)
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.png"))), "+ 4_method.svg (animated)")
