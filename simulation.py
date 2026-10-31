"""Semi-synthetic simulation study for the JAS application note.

Data-generating process (built from the real listings):
  * covariates: rows resampled with replacement from the 5,993 real cars (all features, including
    missing new prices), so the simulated data have exactly the same structure;
  * true log price f(x): XGBoost (paper settings) fitted to the real log prices;
  * noise: out-of-fold residuals of the real data, resampled.
      'het'  - heteroskedastic: standardised residuals rescaled by the robust residual SD of the
               car's price band (< 3, 3-5, 5-10, 10-20, > 20 lakh), as estimated on the real data;
      'hom'  - homoskedastic: raw residuals resampled irrespective of price.

Each replicate draws a training sample of n cars and an independent test sample of 5,000 cars.
Experiment 1 (all designs): 80/90/95% intervals from CV+ (K = 5), split conformal (75/25) and a naive
  interval (prediction +/- quantile of in-sample residuals); coverage and width on the test cars.
Experiment 2 (het, n = 5,993): two evaluation pipelines, each compared with its own true accuracy.
  (a) trim the whole sample with the outlier rules of the earlier literature, then 5-fold CV on the
      retained cars; true accuracy = model trained on the retained cars, scored on all 5,000 new cars;
  (b) 5-fold CV with in-fold preprocessing on all cars; true accuracy = model trained on all n cars,
      scored on the same 5,000 new cars.

Usage: python simulation.py path/to/dataset211_final.csv [reps]
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

DATA = sys.argv[1] if len(sys.argv) > 1 else "dataset211_final.csv"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
NUM = ["Year", "Kilometers_Driven", "Mileage", "Power", "Engine", "Seats", "New_Price"]
CAT = ["Brand", "Family", "Location", "Fuel_Type", "Transmission", "Owner_Type"]
EDGES = np.array([3, 5, 10, 20])                      # price bands (lakh)
BANDS = ["< 3", "3-5", "5-10", "10-20", "> 20"]
LEVELS = [0.20, 0.10, 0.05]

# ---------------------------------------------------------------- data (same preparation as the paper)
d = pd.read_csv(DATA)
d["Name"] = d.Name.str.strip().str.replace(r"\s+", " ", regex=True)
w = d.Name.str.split()
d["Brand"] = w.str[0]; d["Family"] = w.str[:2].str.join(" ").str.lower()
land = d.Brand == "Land"; d.loc[land, "Brand"] = "Land Rover"; d.loc[land, "Family"] = w[land].str[:3].str.join(" ").str.lower()
d = d.reset_index(drop=True)
X, y = d[NUM + CAT], d.Price.values


def preprocessor():
    log_cols = ["Kilometers_Driven", "New_Price"]
    steps = [("log", ColumnTransformer([("l", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), log_cols)],
                                       remainder="passthrough", verbose_feature_names_out=False)),
             ("impute", IterativeImputer(max_iter=10, random_state=0, add_indicator=True))]
    onehot = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False)
    return ColumnTransformer([("num", Pipeline(steps), NUM), ("cat", onehot, CAT)], verbose_feature_names_out=False)


def paper_xgb(seed=0, n_jobs=-1):   # the model used in the paper (also the truth generator)
    return make_pipeline(preprocessor(), XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=6, subsample=0.8,
                                                      colsample_bytree=0.7, min_child_weight=2, n_jobs=n_jobs,
                                                      random_state=seed))


def sim_xgb(seed=0):                # lighter XGBoost used inside the simulation
    return make_pipeline(preprocessor(), XGBRegressor(n_estimators=300, learning_rate=0.08, max_depth=6, subsample=0.8,
                                                      colsample_bytree=0.7, min_child_weight=2, n_jobs=1,
                                                      tree_method="hist", random_state=seed))


class CVPlus:                        # identical to the paper's implementation
    def __init__(self, make_model, K=5, seed=0): self.make_model, self.K, self.seed = make_model, K, seed
    def fit(self, X, ly):
        kf = KFold(self.K, shuffle=True, random_state=self.seed)
        self.m, self.r, self.fo = [], np.zeros(len(ly)), np.zeros(len(ly), int)
        for k, (tr, va) in enumerate(kf.split(X)):
            mm = self.make_model().fit(X.iloc[tr], ly[tr]); self.r[va] = np.abs(ly[va] - mm.predict(X.iloc[va])); self.fo[va] = k
            self.m.append(mm)
        return self
    def predict_all(self, X):
        self.mu = np.stack([mm.predict(X) for mm in self.m]); return self
    def interval(self, a):
        M = self.mu[self.fo]; lo, hi = M - self.r[:, None], M + self.r[:, None]; n = len(self.r)
        kl, kh = int(np.floor(a * (n + 1))) - 1, int(np.ceil((1 - a) * (n + 1))) - 1
        return np.exp(self.mu.mean(0)), np.exp(np.partition(lo, kl, axis=0)[kl]), np.exp(np.partition(hi, kh, axis=0)[kh])


def trim_like_paper(df):
    c = df[df.Price < df.Price.quantile(.99)]
    for col, q, op in [("Year", .01, "gt"), ("Kilometers_Driven", .99, "lt"), ("Mileage", .99, "lt"),
                       ("Power", .99, "lt"), ("Engine", .99, "lt"), ("New_Price", .99, "lt")]:
        thr = c[col].quantile(q)
        c = c[c[col] > thr] if op == "gt" else c[c[col] < thr]
    return c


def score(yt, p):
    e = yt - p
    return dict(R2=r2_score(yt, p), MAPE=100 * np.mean(np.abs(e) / yt), Within10=100 * np.mean(np.abs(e) / yt <= .10))


# ---------------------------------------------------------------- data-generating process
t0 = time.time()
LY = np.log(y)
F_TRUE = paper_xgb(0).fit(X, LY).predict(X)
oof = np.zeros(len(y))
for tr, te in KFold(5, shuffle=True, random_state=11).split(X):
    oof[te] = paper_xgb(0).fit(X.iloc[tr], LY[tr]).predict(X.iloc[te])
RES = LY - oof
b_real = np.digitize(np.exp(oof), EDGES)
SD_BAND = np.array([1.4826 * np.median(np.abs(RES[b_real == k] - np.median(RES[b_real == k]))) for k in range(5)])
Z = RES / SD_BAND[b_real]; Z = Z - Z.mean()
R0 = RES - RES.mean()
print("generator ready in %.0fs" % (time.time() - t0))
print("residual SD (log scale) overall %.3f; robust SD by band:" % RES.std(), dict(zip(BANDS, SD_BAND.round(3))))


def draw(rng, n, scen):
    idx = rng.integers(0, len(X), n)
    f = F_TRUE[idx]
    eps = SD_BAND[np.digitize(np.exp(f), EDGES)] * rng.choice(Z, n) if scen == "het" else rng.choice(R0, n)
    return X.iloc[idx].reset_index(drop=True), f + eps


def one_rep(rep, scen, n, exp2):
    rng = np.random.default_rng(10_000 * (scen == "hom") + 1000 * (n // 1000) + rep)
    seed = int(rng.integers(1e6))
    Xtr, ltr = draw(rng, n, scen)
    Xte, lte = draw(rng, 5000, scen)
    yte = np.exp(lte)
    rows = []
    # --- intervals
    cp = CVPlus(lambda: sim_xgb(seed), K=5, seed=seed).fit(Xtr, ltr).predict_all(Xte)
    full = sim_xgb(seed).fit(Xtr, ltr); pf = full.predict(Xte); ins = np.abs(ltr - full.predict(Xtr))
    perm = np.random.default_rng(seed).permutation(n); ncal = n // 4; cal, trn = perm[:ncal], perm[ncal:]
    sm = sim_xgb(seed).fit(Xtr.iloc[trn], ltr[trn]); rc = np.sort(np.abs(ltr[cal] - sm.predict(Xtr.iloc[cal]))); ps = sm.predict(Xte)
    for a in LEVELS:
        qs = rc[min(int(np.ceil((ncal + 1) * (1 - a))) - 1, ncal - 1)]
        qn = np.quantile(ins, 1 - a)
        meth = {"CV+": cp.interval(a),
                "Split conformal": (np.exp(ps), np.exp(ps - qs), np.exp(ps + qs)),
                "Naive": (np.exp(pf), np.exp(pf - qn), np.exp(pf + qn))}
        for m, (pt, lo, hi) in meth.items():
            cov = (yte >= lo) & (yte <= hi); band = np.digitize(pt, EDGES)
            r = dict(scen=scen, n=n, rep=rep, exp="interval", method=m, level=f"{100 - 100 * a:.0f}%",
                     coverage=cov.mean(), width=np.median(hi - lo), rel_width=np.median((hi - lo) / pt))
            for k, bn in enumerate(BANDS):
                r[f"cov {bn}"] = cov[band == k].mean() if (band == k).any() else np.nan
            rows.append(r)
    # --- evaluation designs
    if exp2:
        truth = score(yte, np.exp(pf))
        def cv(Xs, ls):
            p = np.zeros(len(ls))
            for tr, te in KFold(5, shuffle=True, random_state=seed).split(Xs):
                p[te] = sim_xgb(seed).fit(Xs.iloc[tr], ls[tr]).predict(Xs.iloc[te])
            return score(np.exp(ls), np.exp(p))
        df = Xtr.copy(); df["Price"] = np.exp(ltr); df["New_Price"] = df.New_Price.fillna(df.New_Price.mean())
        keep = trim_like_paper(df).index.values
        Xk, lk = Xtr.iloc[keep].reset_index(drop=True), ltr[keep]
        truth_trim = score(yte, np.exp(sim_xgb(seed).fit(Xk, lk).predict(Xte)))
        for design, est, tru, m in [("Trim the whole sample, then CV", cv(Xk, lk), truth_trim, len(keep)),
                                    ("In-fold preprocessing, all cars", cv(Xtr, ltr), truth, n)]:
            rows.append(dict(scen=scen, n=n, rep=rep, exp="design", method=design, cars_used=m,
                             **{f"est_{k}": v for k, v in est.items()}, **{f"true_{k}": v for k, v in tru.items()}))
    return rows


if __name__ == "__main__":
    jobs = [(r, "het", 5993, True) for r in range(REPS)] + [(r, "hom", 5993, False) for r in range(REPS)] + \
           [(r, "het", 3000, False) for r in range(REPS)] + [(r, "het", 1500, False) for r in range(REPS)]
    out, t1 = [], time.time()
    for i, rows in enumerate(Parallel(n_jobs=2, return_as="generator")(delayed(one_rep)(*j) for j in jobs)):
        out.extend(rows)
        if (i + 1) % 20 == 0:
            pd.DataFrame(out).to_csv("sim_results.csv", index=False)
            print(f"{i + 1}/{len(jobs)} replicates done ({time.time() - t1:.0f}s)", flush=True)
    pd.DataFrame(out).to_csv("sim_results.csv", index=False)
    print("finished in %.0fs" % (time.time() - t0))
