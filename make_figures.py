"""Journal-style figures (no embedded titles, vector PDF) from the final analysis.
Uses exactly the same data preparation, models and seeds as car_price_final_analysis.ipynb."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 9, "font.family": "serif", "pdf.fonttype": 42})
OUT = "figures"
NUM = ["Year", "Kilometers_Driven", "Mileage", "Power", "Engine", "Seats", "New_Price"]
CAT = ["Brand", "Family", "Location", "Fuel_Type", "Transmission", "Owner_Type"]

raw = pd.read_csv("dataset211_final.csv")
d = raw.copy()
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


def xgb(seed=0):
    return make_pipeline(preprocessor(), XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=6, subsample=0.8,
                                                      colsample_bytree=0.7, min_child_weight=2, n_jobs=-1, random_state=seed))


# ---------------- Figure 1: MAPE by model (from the saved fold results)
cv = pd.read_csv("paper_outputs/cv_results_by_fold.csv")
s = cv[cv.subset == "All cars"].groupby("model").MAPE.agg(["mean", "std"]).sort_values("mean", ascending=False)
names = {"Hedonic OLS": "Hedonic regression (OLS)", "ANN": "Neural network (ANN)", "KNN": "k-nearest neighbours",
         "Random Forest": "Random forest", "Gradient Boosting": "Gradient boosting"}
fig, ax = plt.subplots(figsize=(6.0, 3.6))
cols = ["0.35" if m == "Hedonic OLS" else "0.65" for m in s.index]
ax.barh([names.get(m, m) for m in s.index], s["mean"], xerr=s["std"], color=cols, capsize=2.5, error_kw={"lw": 0.8})
for i, v in enumerate(s["mean"]):
    ax.text(v + s["std"].iloc[i] + 0.15, i, f"{v:.1f}", va="center", fontsize=8.5)
ax.set_xlabel("Mean absolute percentage error (%)")
ax.set_xlim(0, s["mean"].max() + s["std"].max() + 1.8)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_mape.pdf"); plt.close()

# ---------------- Figure 2: conformal intervals (same folds and sample as the notebook)
class CVPlus:
    def __init__(self, make_model, K=5, seed=0): self.make_model, self.K, self.seed = make_model, K, seed
    def fit(self, X, ly):
        kf = KFold(self.K, shuffle=True, random_state=self.seed)
        self.m, self.r, self.fo = [], np.zeros(len(ly)), np.zeros(len(ly), int)
        for k, (tr, va) in enumerate(kf.split(X)):
            mm = self.make_model().fit(X.iloc[tr], ly[tr]); self.r[va] = np.abs(ly[va] - mm.predict(X.iloc[va])); self.fo[va] = k
            self.m.append(mm)
        return self
    def predict(self, X, a):
        mu = np.stack([mm.predict(X) for mm in self.m]); M = mu[self.fo]
        lo, hi = M - self.r[:, None], M + self.r[:, None]; n = len(self.r)
        kl, kh = int(np.floor(a * (n + 1))) - 1, int(np.ceil((1 - a) * (n + 1))) - 1
        return np.exp(mu.mean(0)), np.exp(np.partition(lo, kl, axis=0)[kl]), np.exp(np.partition(hi, kh, axis=0)[kh])

recs = []
for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=7).split(X)):
    cp = CVPlus(lambda: xgb(f), seed=f).fit(X.iloc[tr], np.log(y[tr]))
    for a in [0.20, 0.10, 0.05]:
        pt, lo, hi = cp.predict(X.iloc[te], a)
        recs.append(pd.DataFrame(dict(level=f"{100 - 100 * a:.0f}%", y=y[te], pred=pt, lo=lo, hi=hi)))
iv = pd.concat(recs); iv["covered"] = (iv.y >= iv.lo) & (iv.y <= iv.hi)
print("coverage check:", iv.groupby("level").covered.mean().round(4).to_dict())
ex = iv[iv.level == "90%"].sample(60, random_state=3).sort_values("pred").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(6.0, 3.4))
ax.vlines(ex.index, ex.lo, ex.hi, color="0.75", lw=4, label="90% interval")
ax.scatter(ex.index, ex.pred, color="0.2", s=10, zorder=3, label="Prediction")
inside = ex.covered
ax.scatter(ex.index[inside], ex.y[inside], marker="x", color="0.0", s=18, lw=0.9, zorder=4, label="Listed price (inside)")
ax.scatter(ex.index[~inside], ex.y[~inside], marker="^", facecolors="none", edgecolors="0.0", s=26, lw=0.9, zorder=4,
           label="Listed price (outside)")
ax.set_yscale("log"); ax.set_ylabel("Price (lakh rupees, log scale)"); ax.set_xlabel("Held-out cars, sorted by predicted price")
ax.legend(frameon=False, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_intervals.pdf"); plt.close()

# ---------------- Figure 3: SHAP importance; Figure S1: year effect
pipe = xgb(0).fit(X, np.log(y)); pre, model = pipe[0], pipe[-1]
Xt, nm = pre.transform(X), pre.get_feature_names_out()
sv = shap.TreeExplainer(model).shap_values(Xt)
def group(n):
    for c in CAT:
        if n.startswith(c + "_"): return c
    return n.replace("missingindicator_", "Missing flag: ")
G = pd.DataFrame(sv, columns=nm).T.groupby([group(n) for n in nm]).sum().T
imp = G.abs().mean().sort_values(); imp = imp[imp > 0.002]
lab = {"Kilometers_Driven": "Kilometres driven", "New_Price": "New price", "Family": "Model family", "Fuel_Type": "Fuel type",
       "Owner_Type": "Owner type", "Location": "City", "Year": "Registration year", "Missing flag: New_Price": "New price missing"}
print("SHAP importance:", imp.sort_values(ascending=False).round(3).to_dict())
fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.barh([lab.get(i, i) for i in imp.index], imp.values, color="0.55")
for i, v in enumerate(imp.values):
    ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8.5)
ax.set_xlim(0, imp.max() * 1.15); ax.set_xlabel("Mean |SHAP value| (log-price scale)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_shap.pdf"); plt.close()

yi = list(nm).index("Year")
fig, ax = plt.subplots(figsize=(6.0, 3.4))
jit = np.random.default_rng(0).uniform(-.25, .25, len(d))
ax.scatter(d.Year + jit, (np.exp(sv[:, yi]) - 1) * 100, s=3, alpha=.25, color="0.45", rasterized=True)
me = pd.Series(sv[:, yi]).groupby(d.Year.values).mean()
ax.plot(me.index, (np.exp(me) - 1) * 100, color="0.0", lw=1.6, label="Average by year")
ax.axhline(0, color="0.6", lw=0.8)
ax.set_xlabel("Registration year"); ax.set_ylabel("Effect on predicted price (%)")
ax.legend(frameon=False, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/figS1_shap_year.pdf", dpi=300); plt.close()
print("figures written")
