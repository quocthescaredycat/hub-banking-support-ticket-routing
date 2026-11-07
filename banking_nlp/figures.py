"""Static report figures (matplotlib, imported lazily)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3de"
SERIES = {"lr": "#2a78d6", "svm": "#eb6834"}  # categorical slots 1-2 (validated order)
NAMES = {"lr": "Logistic Regression", "svm": "Linear SVM"}

STEP_LABELS = {
    "word:L0_sklearn_default": "L0  sklearn defaults (raw text)",
    "word:L1_normalize_tokenize": "L1  normalize + regex tokens",
    "word:L2_contractions": "L2  + negation/contractions",
    "word:L3_numbers": "L3  + number placeholder",
    "word:L4_lemma": "L4  L3 + lemmatization",
    "word:L5_stop_protected": "L5  L3 + protected stop words",
    "word:L6_stop_sklearn": "L6  L3 + generic stop words",
    "char:L3_numbers": "char_wb 2-5 on L3",
    "word+char:L3_numbers": "word 1-2 + char_wb 2-5 on L3",
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def ablation_dotplot(ablation: pd.DataFrame, path: str | Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = list(STEP_LABELS)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    offset = {"lr": -0.17, "svm": 0.17}
    for clf in ("lr", "svm"):
        sub = ablation[ablation.classifier == clf].set_index("step").reindex(steps)
        ys = [i + offset[clf] for i in range(len(steps))]
        lo = [c[0] for c in sub.macro_f1_ci]
        hi = [c[1] for c in sub.macro_f1_ci]
        ax.hlines(ys, lo, hi, color=SERIES[clf], linewidth=2, alpha=0.55)
        ax.scatter(sub.macro_f1, ys, s=46, color=SERIES[clf], edgecolor=SURFACE, linewidth=1.5,
                   zorder=3, label=NAMES[clf])
        best = sub.macro_f1.idxmax()
        i = steps.index(best)
        ax.annotate(f"{sub.macro_f1[best]:.3f}", (hi[i], ys[i]), xytext=(5, 0),
                    textcoords="offset points", fontsize=8, color=TEXT_2, va="center")
    ax.set_yticks(range(len(steps)), [STEP_LABELS[s] for s in steps], color=TEXT, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Validation macro-F1 (dot) with 95% bootstrap interval (line)", color=TEXT_2, fontsize=9)
    ax.set_title(title, loc="left", color=TEXT, fontsize=11, pad=26)
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9,
                    handletextpad=0.3, borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def confusion_bars(conf: pd.DataFrame, path: str | Path, title: str, k: int = 12) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conf = conf.head(k)
    labels = [f"{t}  →  {p}" for t, p in zip(conf["true"], conf["predicted"])]
    fig, ax = plt.subplots(figsize=(8.2, 0.32 * len(conf) + 1.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    from matplotlib.ticker import MaxNLocator

    ax.barh(range(len(conf)), conf["count"], color=SERIES["lr"], height=0.6)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for i, c in enumerate(conf["count"]):
        ax.text(c + 0.1, i, str(c), va="center", fontsize=8, color=TEXT_2)
    ax.set_yticks(range(len(conf)), labels, fontsize=8, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlabel("Validation requests (true intent → predicted intent)", color=TEXT_2, fontsize=9)
    ax.set_title(title, loc="left", color=TEXT, fontsize=11, pad=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path
