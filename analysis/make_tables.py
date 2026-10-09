"""Generate LaTeX tables from outputs/results/*.json.

Produces (in outputs/report/):
  table_e1_main.tex     : main comparison table (E1)
  table_e2_ablation.tex : ablation table (E2)
  table_e5_social.tex   : graph substitution (E5)
  table_e8_cost.tex     : cost/latency (E8)
Missing rows are rendered as '--' so the table can be compiled early.
"""
import json
from pathlib import Path

RESULTS = Path("outputs/results")
REPORT = Path("outputs/report")
REPORT.mkdir(parents=True, exist_ok=True)

METRICS = ["HR@5", "NDCG@5", "HR@10", "NDCG@10", "HR@20", "NDCG@20", "MRR@10"]


def load(tag: str) -> dict | None:
    f = RESULTS / f"{tag}.json"
    if not f.exists():
        return None
    with open(f) as fh:
        return json.load(fh)["test"]


def fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "--"


def row(label: str, m: dict | None) -> str:
    cells = " & ".join(fmt(m.get(k)) if m else "--" for k in METRICS)
    return f"{label} & {cells} \\\\"


def make_e1(dataset: str):
    header = ("\\begin{tabular}{l" + "c" * len(METRICS) + "}\n\\toprule\n"
              "Model & " + " & ".join(METRICS) + " \\\\\n\\midrule\n")
    models = [("GRU4Rec", f"gru4rec_{dataset}"),
              ("SASRec (CE)", f"sasrec_{dataset}"),
              ("BERT4Rec", f"bert4rec_{dataset}"),
              ("UniSRec (variant)", f"unisrec_{dataset}"),
              ("SUGAR (A0)", f"sugar_{dataset}")]
    rows = [row(lbl, load(tag)) for lbl, tag in models]
    footer = "\\bottomrule\n\\end{tabular}\n"
    out = REPORT / f"table_e1_{dataset}.tex"
    out.write_text(header + "\n".join(rows) + "\n" + footer)
    print("wrote", out)


def make_e2(dataset: str):
    header = ("\\begin{tabular}{l" + "c" * len(METRICS) + "}\n\\toprule\n"
              "Variant & " + " & ".join(METRICS) + " \\\\\n\\midrule\n")
    variants = [("A0 (full)", f"sugar_{dataset}"),
                ("A1 (no profile)", f"a1_a1_no_profile_{dataset}"),
                ("A2 (no drift det.)", f"a2_a2_no_drift_{dataset}"),
                ("A3 (fixed gate)", f"a3_a3_fixed_gate_{dataset}"),
                ("A4 (no social)", f"a4_a4_no_social_{dataset}")]
    rows = [row(lbl, load(tag)) for lbl, tag in variants]
    out = REPORT / f"table_e2_{dataset}.tex"
    out.write_text(header + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    print("wrote", out)


def make_e5(dataset: str):
    f = RESULTS / f"e5_{dataset}.json"
    out = REPORT / f"table_e5_{dataset}.tex"
    if not f.exists():
        out.write_text("% e5 results not ready\n")
        return
    data = json.loads(f.read_text())
    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             "Graph & Edges & NDCG@10 & HR@10 \\\\", "\\midrule"]
    for r in data["rows"]:
        lines.append(f"{r['mode']} & {r['n_edges']} & "
                     f"{fmt(r['test'].get('NDCG@10'))} & "
                     f"{fmt(r['test'].get('HR@10'))} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out.write_text("\n".join(lines))
    print("wrote", out)


def make_e8(dataset: str):
    f = RESULTS / f"e8_{dataset}.json"
    out = REPORT / f"table_e8_{dataset}.tex"
    if not f.exists():
        out.write_text("% e8 results not ready\n")
        return
    d = json.loads(f.read_text())
    s, b = d["online"]["sugar"], d["online"]["sasrec"]
    lines = ["\\begin{tabular}{lcc}", "\\toprule",
             "Model & Latency (ms) & Trainable params \\\\", "\\midrule",
             f"SASRec & {fmt(b['latency_ms_per_request'])} & {b['trainable_params']:,} \\\\",
             f"SUGAR & {fmt(s['latency_ms_per_request'])} & {s['trainable_params']:,} \\\\",
             f"\\multicolumn{{3}}{{l}}{{ratio = "
             f"{d['online']['sugar_over_sasrec_latency_ratio']}x}} \\\\",
             "\\bottomrule", "\\end{tabular}"]
    out.write_text("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    for ds in ("beauty", "yelp_filtered"):
        make_e1(ds)
        make_e2(ds)
        make_e5(ds)
        make_e8(ds)