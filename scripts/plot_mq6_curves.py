"""mq6 plots: faithful Zoology-Figure-2 recipe in OUR repo, softmax vs GDN(noconv).

Visualizes ALL LR curves (per request -- not the max). Reads slurm/mq6/*.log lines
of the form: `step N: val/loss X val/acc Y`.

Outputs (slurm/plots/):
  mq6_all_lr_curves.png   -- grid [seqlen x mixer], one val-acc curve per LR
  mq6_final_vs_seqlen.png -- final val acc vs seqlen, every LR shown (scatter) + best line

Usage: python scripts/plot_mq6_curves.py
"""
import os, re, glob
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(ROOT, "slurm", "mq6")
OUTDIR = os.path.join(ROOT, "slurm", "plots"); os.makedirs(OUTDIR, exist_ok=True)

SEQLENS = [64, 128, 256, 512]
KV = {64: 4, 128: 8, 256: 16, 512: 64}
MIXERS = ["softmax", "gdn"]
MLABEL = {"softmax": "softmax attn (learned-abs, 1 head)", "gdn": "GDN (short-conv OFF)"}
STEP = re.compile(r"step (\d+): val/loss [\d.]+ val/acc ([\d.]+)")
FNAME = re.compile(r"mq6_(softmax|gdn)_s(\d+)_lr([0-9.eE+-]+)\.log$")
# color per LR (sorted)
LR_COLORS = plt.cm.viridis


def parse(path):
    with open(path, errors="ignore") as f:
        pts = [(int(s), float(a)) for s, a in STEP.findall(f.read())]
    pts.sort()
    return pts


def load():
    data = defaultdict(lambda: defaultdict(dict))  # data[mixer][seqlen][lr] = [(step,acc)]
    for path in glob.glob(os.path.join(LOGDIR, "*.log")):
        m = FNAME.search(os.path.basename(path))
        if not m:
            continue
        mixer, seqlen, lr = m.group(1), int(m.group(2)), float(m.group(3))
        pts = parse(path)
        if pts:
            data[mixer][seqlen][lr] = pts
    return data


def all_lrs(data):
    s = set()
    for mixer in data:
        for seqlen in data[mixer]:
            s |= set(data[mixer][seqlen])
    return sorted(s)


def plot_curves(data):
    lrs = all_lrs(data)
    cmap = {lr: LR_COLORS(i / max(1, len(lrs) - 1)) for i, lr in enumerate(lrs)}
    fig, axes = plt.subplots(len(SEQLENS), len(MIXERS), figsize=(12, 14), sharex=False, sharey=True)
    for r, seqlen in enumerate(SEQLENS):
        for c, mixer in enumerate(MIXERS):
            ax = axes[r, c]
            cells = data[mixer].get(seqlen, {})
            for lr in sorted(cells):
                pts = cells[lr]
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "-", color=cmap[lr], lw=1.8, label=f"lr={lr:.1e}")
            ax.axhline(0.99, color="gray", ls="--", lw=1, alpha=0.5)
            ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(MLABEL[mixer], fontsize=11)
            if c == 0:
                ax.set_ylabel(f"seqlen={seqlen} (kv={KV[seqlen]})\nval accuracy", fontsize=10)
            ax.set_xlabel("step")
            if cells:
                ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("mq6: Zoology Fig-2 recipe in our repo (100k ex, ~64 ep, cosine LR->0, wd=0.1, d=64)\n"
                 "ALL LR curves shown", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(OUTDIR, "mq6_all_lr_curves.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


def plot_final(data):
    lrs = all_lrs(data)
    cmap = {lr: LR_COLORS(i / max(1, len(lrs) - 1)) for i, lr in enumerate(lrs)}
    plt.figure(figsize=(7, 5))
    marker = {"softmax": "o", "gdn": "s"}
    for mixer in MIXERS:
        best_x, best_y = [], []
        for seqlen in SEQLENS:
            cells = data[mixer].get(seqlen, {})
            if not cells:
                continue
            finals = {lr: pts[-1][1] for lr, pts in cells.items()}
            for lr, fa in finals.items():
                plt.scatter([seqlen], [fa], color=cmap[lr], marker=marker[mixer],
                            s=45, alpha=0.8, edgecolor="k", linewidth=0.3, zorder=3)
            best_x.append(seqlen); best_y.append(max(finals.values()))
        if best_x:
            plt.plot(best_x, best_y, "-", color="k" if mixer == "softmax" else "tab:red",
                     lw=1.5, alpha=0.6, label=f"{MLABEL[mixer]} (best LR)")
    plt.axhline(0.99, color="gray", ls="--", lw=1, alpha=0.6)
    plt.xscale("log", base=2)
    plt.xticks(SEQLENS, [f"{s}\n(kv={KV[s]})" for s in SEQLENS])
    plt.xlabel("input sequence length"); plt.ylabel("final val accuracy")
    plt.ylim(-0.02, 1.02)
    plt.title("mq6 final val acc vs seqlen (every LR shown; circle=softmax, square=GDN)")
    # LR legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap[lr],
                      markeredgecolor="k", markersize=8, label=f"lr={lr:.1e}") for lr in lrs]
    leg1 = plt.legend(handles=handles, fontsize=8, loc="lower left", title="LR")
    plt.gca().add_artist(leg1)
    plt.legend(fontsize=8, loc="upper right")
    plt.grid(alpha=0.3); plt.tight_layout()
    out = os.path.join(OUTDIR, "mq6_final_vs_seqlen.png")
    plt.savefig(out, dpi=130); plt.close()
    return out


def summary(data):
    print("\n=== mq6 final val acc (per LR) ===")
    for mixer in MIXERS:
        print(f"\n[{mixer}]")
        for seqlen in SEQLENS:
            cells = data[mixer].get(seqlen, {})
            if not cells:
                print(f"  seqlen={seqlen}: (no logs)"); continue
            parts = [f"lr{lr:.1e}={pts[-1][1]:.3f}" for lr, pts in sorted(cells.items())]
            print(f"  seqlen={seqlen} (kv={KV[seqlen]}): " + "  ".join(parts))


if __name__ == "__main__":
    data = load()
    if not data:
        print("no parseable logs in", LOGDIR); raise SystemExit
    summary(data)
    print("\nwrote:", plot_curves(data))
    print("wrote:", plot_final(data))
