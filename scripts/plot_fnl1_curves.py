"""fnl1 plots: compact K/Q MQAR, softmax(RoPE) vs GDN(noconv), sweep K x LR.

Reads slurm/fnl1/*.log lines `step N: val/loss X val/acc Y`.
Outputs (slurm/plots/):
  fnl1_acc_vs_K.png       -- best-over-LR final val acc vs K (the capacity money plot)
  fnl1_all_lr_curves.png  -- grid [K x mixer], every LR's val-acc curve

Usage: python scripts/plot_fnl1_curves.py
"""
import os, re, glob
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(ROOT, "slurm", "fnl1")
OUTDIR = os.path.join(ROOT, "slurm", "plots"); os.makedirs(OUTDIR, exist_ok=True)

KS = [int(k) for k in os.environ.get("FNL1_KS", "16 32 64 128 256 512").split()]
MIXERS = ["softmax", "gdn"]
MLABEL = {"softmax": "softmax attn (RoPE)", "gdn": "GDN (short-conv OFF)"}
COLOR = {"softmax": "tab:blue", "gdn": "tab:red"}
STEP = re.compile(r"step (\d+): val/loss [\d.]+ val/acc ([\d.]+)")
FNAME = re.compile(r"fnl1_(softmax|gdn)_k(\d+)_lr([0-9.eE+-]+)\.log$")
LRC = plt.cm.viridis


def parse(path):
    with open(path, errors="ignore") as f:
        pts = [(int(s), float(a)) for s, a in STEP.findall(f.read())]
    pts.sort(); return pts


def load():
    data = defaultdict(lambda: defaultdict(dict))  # [mixer][K][lr] = [(step,acc)]
    for path in glob.glob(os.path.join(LOGDIR, "*.log")):
        m = FNAME.search(os.path.basename(path))
        if not m:
            continue
        mixer, K, lr = m.group(1), int(m.group(2)), float(m.group(3))
        pts = parse(path)
        if pts:
            data[mixer][K][lr] = pts
    return data


def all_lrs(data):
    s = set()
    for mx in data:
        for K in data[mx]:
            s |= set(data[mx][K])
    return sorted(s)


def plot_acc_vs_K(data):
    plt.figure(figsize=(6.8, 4.8))
    for mixer in MIXERS:
        xs, ys = [], []
        for K in KS:
            cells = data[mixer].get(K, {})
            if not cells:
                continue
            xs.append(K); ys.append(max(pts[-1][1] for pts in cells.values()))
        if xs:
            plt.plot(xs, ys, "o-", color=COLOR[mixer], lw=2, ms=7, label=MLABEL[mixer])
    plt.axhline(0.99, color="gray", ls="--", lw=1, alpha=0.6, label="solved (0.99)")
    plt.xscale("log", base=2); plt.xticks(KS, [str(k) for k in KS])
    plt.xlabel("K = #KV pairs (= #queries, seqlen=4K)")
    plt.ylabel("best final val acc (over LR)")
    plt.ylim(-0.02, 1.02)
    plt.title("fnl1: compact K/Q MQAR — softmax(RoPE) vs GDN(noconv)\n2L d=64, 400k ex, 16 ep, batch 256, 25k steps")
    plt.legend(loc="lower left", fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
    out = os.path.join(OUTDIR, "fnl1_acc_vs_K.png"); plt.savefig(out, dpi=130); plt.close()
    return out


def plot_curves(data):
    lrs = all_lrs(data)
    cmap = {lr: LRC(i / max(1, len(lrs) - 1)) for i, lr in enumerate(lrs)}
    fig, axes = plt.subplots(len(KS), len(MIXERS), figsize=(12, 14), sharey=True)
    for r, K in enumerate(KS):
        for c, mixer in enumerate(MIXERS):
            ax = axes[r, c]; cells = data[mixer].get(K, {})
            for lr in sorted(cells):
                xs, ys = zip(*cells[lr])
                ax.plot(xs, ys, color=cmap[lr], lw=1.8, label=f"lr={lr:.1e}")
            ax.axhline(0.99, color="gray", ls="--", lw=1, alpha=0.5)
            ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(MLABEL[mixer], fontsize=11)
            if c == 0:
                ax.set_ylabel(f"K={K} (seqlen={4*K})\nval acc", fontsize=10)
            ax.set_xlabel("step")
            if cells:
                ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("fnl1 all-LR curves — softmax(RoPE) vs GDN(noconv), compact K/Q MQAR", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(OUTDIR, "fnl1_all_lr_curves.png"); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def summary(data):
    print("\n=== fnl1 best final val acc (over LR) ===")
    print(f"{'K':>5} {'seqlen':>7} {'softmax(RoPE)':>15} {'gdn(noconv)':>13}")
    for K in KS:
        row = [f"{K:>5}", f"{4*K:>7}"]
        for mixer in MIXERS:
            cells = data[mixer].get(K, {})
            row.append(f"{max(p[-1][1] for p in cells.values()):>15.4f}" if cells else f"{'--':>15}")
        print(" ".join(row))


if __name__ == "__main__":
    data = load()
    if not data:
        print("no parseable logs in", LOGDIR); raise SystemExit
    summary(data)
    print("\nwrote:", plot_acc_vs_K(data))
    print("wrote:", plot_curves(data))
