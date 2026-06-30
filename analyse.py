import argparse, csv, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np

csv.field_size_limit(min(sys.maxsize, 2147483647))

PLATFORM_COLORS = {
    "gpu":     "#76b900",
    "cloud":   "#4285F4",
    "pi5":     "#C51A4A",
    "opi_npu": "#FF6F00",
    "opi_cpu": "#FFB300",
}

def color_for(label: str) -> str:
    l = label.lower()
    if "gemini" in l or "cloud" in l: return PLATFORM_COLORS["cloud"]
    if "4070" in l or "gpu" in l:     return PLATFORM_COLORS["gpu"]
    if "pi5" in l and "opi" not in l: return PLATFORM_COLORS["pi5"]
    if "npu" in l:                    return PLATFORM_COLORS["opi_npu"]
    if "opi" in l or "orange" in l:   return PLATFORM_COLORS["opi_cpu"]
    return "#888"

def setup_style():
    plt.rcParams.update({
        "font.family":"DejaVu Sans","font.size":11,"axes.titlesize":12,
        "axes.labelsize":11,"axes.spines.top":False,"axes.spines.right":False,
        "axes.grid":True,"grid.alpha":0.3,"grid.linestyle":"--",
        "figure.dpi":130,"savefig.dpi":200,"savefig.bbox":"tight",
    })

def load(path):
    rows = []
    with open(path,"r",encoding="utf-8") as f:
        for r in csv.DictReader(f): rows.append(r)
    return rows

def floatify(rows, fields):
    out = []
    for r in rows:
        rr = dict(r)
        for f in fields:
            try: rr[f] = float(r[f]) if r[f] not in ("",None) else 0.0
            except (ValueError, TypeError): rr[f] = 0.0
        out.append(rr)
    return out

def boolify(rows, fields):
    out = []
    for r in rows:
        rr = dict(r)
        for f in fields:
            v = r.get(f,"")
            if isinstance(v, bool): rr[f] = v
            elif str(v).lower() in ("true","1"): rr[f] = True
            elif str(v).lower() in ("false","0"): rr[f] = False
            else: rr[f] = None
        out.append(rr)
    return out

def by(rows, key):
    g = defaultdict(list)
    for r in rows: g[r[key]].append(r)
    return g


def fig_decode_tps(rows, out_dir: Path):
    g = by(rows,"label")
    labels = sorted(g.keys())
    means = [mean([r["decode_tps"] for r in g[l] if r["decode_tps"]>0] or [0]) for l in labels]
    stds = [stdev([r["decode_tps"] for r in g[l] if r["decode_tps"]>0]) if len([r for r in g[l] if r["decode_tps"]>0])>1 else 0 for l in labels]
    fig, ax = plt.subplots(figsize=(10,5))
    bars = ax.bar(labels, means, yerr=stds, color=[color_for(l) for l in labels],
                  capsize=4, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Decode tokens/sec")
    ax.set_title("Decode throughput across platforms")
    ax.axhline(5, color="grey", linestyle=":", linewidth=1)
    ax.text(len(labels)-0.5, 5.3,
            "Sogemeier et al. (2024) conversational threshold",
            fontsize=8, color="grey", ha="right")
    for b, m in zip(bars, means):
        ax.text(b.get_x()+b.get_width()/2, m, f"{m:.1f}",
                ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    fig.savefig(out_dir/"fig_decode_tps.png"); plt.close(fig)


def fig_ttft(rows, out_dir: Path):
    g = by(rows,"label")
    labels = sorted(g.keys())
    means = [mean([r["ttft_s"] for r in g[l] if r["ttft_s"]>0] or [0]) for l in labels]
    fig, ax = plt.subplots(figsize=(10,5))
    bars = ax.bar(labels, means, color=[color_for(l) for l in labels],
                  edgecolor="black", linewidth=0.5)
    ax.set_ylabel("TTFT (s)")
    ax.set_title("Time-To-First-Token across platforms")
    ax.axhline(5, color="red", linestyle=":", linewidth=1)
    ax.text(len(labels)-0.5, 5.2, "5s voice-assistant threshold",
            fontsize=8, color="red", ha="right")
    for b, m in zip(bars, means):
        ax.text(b.get_x()+b.get_width()/2, m, f"{m:.2f}",
                ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    fig.savefig(out_dir/"fig_ttft.png"); plt.close(fig)


def fig_energy_per_token(rows, out_dir: Path):
    g = by(rows,"label")
    labels = sorted([l for l in g if any(r["energy_per_token_j"]>0 for r in g[l])])
    if not labels: return
    means = [mean([r["energy_per_token_j"] for r in g[l] if r["energy_per_token_j"]>0]) for l in labels]
    fig, ax = plt.subplots(figsize=(10,5))
    bars = ax.bar(labels, means, color=[color_for(l) for l in labels],
                  edgecolor="black", linewidth=0.5)
    ax.set_ylabel("J / token")
    ax.set_title("Energy per token across platforms")
    for b, m in zip(bars, means):
        ax.text(b.get_x()+b.get_width()/2, m, f"{m:.2f}",
                ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    fig.savefig(out_dir/"fig_energy_per_token.png"); plt.close(fig)


def fig_ifeval(rows, out_dir: Path):
    g = by(rows,"label")
    labels = sorted(g.keys())
    strict, loose = [], []
    for l in labels:
        graded = [r for r in g[l] if r.get("ifeval_strict_pass") is not None]
        if not graded: strict.append(0); loose.append(0); continue
        strict.append(100*sum(1 for r in graded if r["ifeval_strict_pass"])/len(graded))
        loose.append(100*sum(1 for r in graded if r["ifeval_loose_pass"])/len(graded))
    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(10,5))
    b1 = ax.bar(x-w/2, strict, w, label="Strict", color="#444")
    b2 = ax.bar(x+w/2, loose, w, label="Loose", color="#aaa")
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("IFEval pass rates across platforms")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(); ax.set_ylim(0,105)
    for bs, vs in [(b1,strict),(b2,loose)]:
        for b,v in zip(bs,vs):
            ax.text(b.get_x()+b.get_width()/2, v, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=9)
    fig.savefig(out_dir/"fig_ifeval.png"); plt.close(fig)


def fig_efficiency_scatter(rows, out_dir: Path):
    g = by(rows,"label")
    fig, ax = plt.subplots(figsize=(8,6))
    for l, items in g.items():
        graded = [r for r in items if r.get("ifeval_strict_pass") is not None and r["energy_per_token_j"]>0]
        passes = [r for r in graded if r["ifeval_strict_pass"]]
        if not passes: continue
        x = mean([r["decode_tps"] for r in passes])
        y = mean([r["energy_per_token_j"] for r in passes])
        ax.scatter(x, y, s=180, color=color_for(l), edgecolor="black",
                   linewidth=0.7, zorder=3)
        ax.annotate(l, (x,y), textcoords="offset points",
                    xytext=(8,5), fontsize=9)
    ax.set_xlabel("Decode TPS (higher better →)")
    ax.set_ylabel("J/token (lower better ↓)")
    ax.set_title("Speed vs energy efficiency (correct responses only)")
    ax.invert_yaxis()
    fig.savefig(out_dir/"fig_efficiency_scatter.png"); plt.close(fig)


def fig_rag_vs_raw(rows, out_dir: Path):
    """Only generates if there are both raw and rag mode rows."""
    by_label_mode = defaultdict(list)
    for r in rows:
        by_label_mode[(r["label"], r.get("mode","raw"))].append(r)
    labels = sorted({l for (l,_) in by_label_mode.keys()})
    has_rag = any(m == "rag" for (_,m) in by_label_mode.keys())
    if not has_rag: return

    raw_strict, rag_strict = [], []
    raw_tps, rag_tps = [], []
    for l in labels:
        raw_rows = by_label_mode.get((l,"raw"), [])
        rag_rows = by_label_mode.get((l,"rag"), [])
        def pass_rate(rs):
            g = [r for r in rs if r.get("ifeval_strict_pass") is not None]
            return 100*sum(1 for r in g if r["ifeval_strict_pass"])/max(len(g),1) if g else 0
        def avg_tps(rs):
            v = [r["decode_tps"] for r in rs if r["decode_tps"]>0]
            return mean(v) if v else 0
        raw_strict.append(pass_rate(raw_rows)); rag_strict.append(pass_rate(rag_rows))
        raw_tps.append(avg_tps(raw_rows)); rag_tps.append(avg_tps(rag_rows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(labels)); w = 0.35

    ax1.bar(x-w/2, raw_strict, w, label="Raw", color="#666")
    ax1.bar(x+w/2, rag_strict, w, label="RAG", color="#3a7ca5")
    ax1.set_ylabel("Strict pass rate (%)")
    ax1.set_title("RAG vs Raw - accuracy")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.legend(); ax1.set_ylim(0,105)

    ax2.bar(x-w/2, raw_tps, w, label="Raw", color="#666")
    ax2.bar(x+w/2, rag_tps, w, label="RAG", color="#3a7ca5")
    ax2.set_ylabel("Decode TPS")
    ax2.set_title("RAG vs Raw - throughput")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.legend()

    fig.savefig(out_dir/"fig_rag_vs_raw.png"); plt.close(fig)


def summary_table(rows) -> str:
    g = by(rows,"label")
    out = ["| Label | n | TTFT (s) | Decode TPS | J/token | Power (W) | Strict % | Loose % |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for l in sorted(g.keys()):
        items = g[l]
        ttft = mean([r["ttft_s"] for r in items if r["ttft_s"]>0] or [0])
        tps = mean([r["decode_tps"] for r in items if r["decode_tps"]>0] or [0])
        jpt = mean([r["energy_per_token_j"] for r in items if r["energy_per_token_j"]>0] or [0])
        pw = mean([r["avg_power_w"] for r in items if r["avg_power_w"]>0] or [0])
        graded = [r for r in items if r.get("ifeval_strict_pass") is not None]
        if graded:
            s = 100*sum(1 for r in graded if r["ifeval_strict_pass"])/len(graded)
            ll = 100*sum(1 for r in graded if r["ifeval_loose_pass"])/len(graded)
        else:
            s = ll = 0
        out.append(f"| {l} | {len(items)} | {ttft:.2f} | {tps:.1f} | "
                   f"{jpt:.2f} | {pw:.1f} | {s:.1f} | {ll:.1f} |")
    return "\n".join(out)


def per_mode_table(rows) -> str:
    by_lm = defaultdict(list)
    for r in rows: by_lm[(r["label"], r.get("mode","raw"))].append(r)
    out = ["| Label | Mode | n | Decode TPS | J/token | Strict % |",
           "|---|---|---:|---:|---:|---:|"]
    for (l,m), items in sorted(by_lm.items()):
        tps = mean([r["decode_tps"] for r in items if r["decode_tps"]>0] or [0])
        jpt = mean([r["energy_per_token_j"] for r in items if r["energy_per_token_j"]>0] or [0])
        graded = [r for r in items if r.get("ifeval_strict_pass") is not None]
        s = 100*sum(1 for r in graded if r["ifeval_strict_pass"])/len(graded) if graded else 0
        out.append(f"| {l} | {m} | {len(items)} | {tps:.1f} | {jpt:.2f} | {s:.1f} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    rows = load(args.inp)
    rows = floatify(rows, ["ttft_s","total_time_s","prefill_tps","decode_tps",
                            "total_energy_j","avg_power_w","peak_power_w",
                            "energy_per_token_j","input_tokens","output_tokens"])
    rows = boolify(rows, ["ifeval_strict_pass","ifeval_loose_pass"])

    fig_decode_tps(rows, out_dir)
    fig_ttft(rows, out_dir)
    fig_energy_per_token(rows, out_dir)
    fig_ifeval(rows, out_dir)
    fig_efficiency_scatter(rows, out_dir)
    fig_rag_vs_raw(rows, out_dir)

    t1 = summary_table(rows)
    t2 = per_mode_table(rows)
    print("\n=== Summary by label ===\n"); print(t1)
    print("\n=== Summary by (label, mode) ===\n"); print(t2)
    Path("data").mkdir(exist_ok=True)
    Path("data/tables.md").write_text(
        "# Summary by label\n\n" + t1 + "\n\n# Summary by (label, mode)\n\n" + t2 + "\n")
    print(f"\nFigures: {out_dir}/")
    print("Tables: data/tables.md")


if __name__ == "__main__":
    main()
