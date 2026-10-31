"""Summarise sim_results.csv (output of simulation.py) into the tables reported in the paper."""
import sys
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
r = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "sim_results.csv")
iv, de = r[r.exp == "interval"], r[r.exp == "design"]
print("replicates per design:", iv.groupby(["scen", "n"]).rep.nunique().to_dict())

# Table A: coverage and relative width
a = iv.groupby(["scen", "n", "method", "level"]).agg(coverage=("coverage", "mean"), cov_sd=("coverage", "std"),
                                                    rel_width=("rel_width", "mean"), width=("width", "mean"))
a[["coverage", "cov_sd"]] *= 100; a["rel_width"] *= 100
print("\nCOVERAGE (% of new cars inside the interval; mean and SD over replicates) and median width (% of prediction, lakh)")
print(a.round(2).to_string())

# Table B: coverage by predicted-price band, CV+ 90%
bands = [c for c in iv.columns if c.startswith("cov ")]
b = iv[(iv.method == "CV+") & (iv.level == "90%")].groupby(["scen", "n"])[bands].mean() * 100
print("\nCV+ 90% COVERAGE BY PREDICTED PRICE BAND (%)")
print(b.round(1).to_string())
b2 = iv[(iv.level == "90%") & (iv.n == 5993)].groupby(["scen", "method"])[bands].mean() * 100
print(b2.round(1).to_string())

# Table C: evaluation pipelines
if len(de):
    out = []
    for m, g in de.groupby("method"):
        row = {"pipeline": m, "reps": len(g), "cars used": g.cars_used.mean()}
        for k in ["R2", "MAPE", "Within10"]:
            row[f"est {k}"] = g[f"est_{k}"].mean(); row[f"true {k}"] = g[f"true_{k}"].mean()
            row[f"bias {k}"] = (g[f"est_{k}"] - g[f"true_{k}"]).mean(); row[f"bias sd {k}"] = (g[f"est_{k}"] - g[f"true_{k}"]).std()
        row["share est R2 > true"] = (g.est_R2 > g.true_R2).mean()
        out.append(row)
    print("\nEVALUATION PIPELINES: estimate vs true accuracy on new cars (means over replicates)")
    print(pd.DataFrame(out).set_index("pipeline").round(3).T.to_string())
