"""Generate the paper figures from outputs/results/*.json (matplotlib).

  fig_e41_poisoning.pdf : NDCG@10 vs corruption p, SUGAR vs A3 (the paper's
                          signature figure)
  fig_e42_lag.pdf       : NDCG@10 vs commit lag Lambda
  fig_e3_costquality.pdf: trigger cost vs NDCG across tau grid
  fig_e9_case.pdf       : sim and rho over checkpoints for the case-study user
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("outputs/results")
REPORT = Path("outputs/report")
REPORT.mkdir(parents=True, exist_ok=True)


def fig_e41(dataset: str = "beauty"):
    ps = [0.2, 0.4, 0.6, 0.8]
    curves = {}
    for model, label in (("sugar", "SUGAR"), ("a3_fixed_gate", "A3 (fixed gate)")):
        xs, ys = [], []
        for p in ps:
            f = RESULTS / f"e41_{model}_{dataset}_p{p}.json"
            if f.exists():
                d = json.loads(f.read_text())
                xs.append(p)
                ys.append(d["test"]["NDCG@10"])
        if xs:
            curves[label] = (xs, ys)
    if not curves:
        return
    plt.figure(figsize=(5, 3.5))
    for label, (xs, ys) in curves.items():
        plt.plot(xs, ys, marker="o", label=label)
    plt.xlabel("Profile corruption probability p")
    plt.ylabel("NDCG@10")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT / "fig_e41_poisoning.pdf")
    plt.close()
    print("wrote fig_e41_poisoning.pdf")


def fig_e42(dataset: str = "beauty"):
    f = RESULTS / f"e42_{dataset}.json"
    if not f.exists():
        return
    rows = json.loads(f.read_text())["rows"]
    lags = [r["lag"] for r in rows]
    ndcg = [r["test"]["NDCG@10"] for r in rows]
    plt.figure(figsize=(5, 3.5))
    plt.plot(lags, ndcg, marker="s")
    plt.xlabel("Commit lag $\\Lambda$ (interactions)")
    plt.ylabel("NDCG@10")
    plt.xscale("symlog")
    plt.tight_layout()
    plt.savefig(REPORT / "fig_e42_lag.pdf")
    plt.close()
    print("wrote fig_e42_lag.pdf")


def fig_e9(dataset: str = "beauty"):
    f = RESULTS / f"e9_{dataset}.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    series = d["sim_rho_series"]
    t = [s["t_chk"] for s in series]
    plt.figure(figsize=(6, 3.5))
    plt.plot(t, [s["sim"] for s in series], marker=".", label="sim (drift score)")
    plt.plot(t, [s["rho"] for s in series], marker=".", label="$\\rho$ (reliability)")
    plt.xlabel("Checkpoint position $t_{chk}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT / "fig_e9_case.pdf")
    plt.close()
    print("wrote fig_e9_case.pdf")


if __name__ == "__main__":
    fig_e41()
    fig_e42()
    fig_e9()