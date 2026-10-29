"""UCI Adult dataset inspection + quick attribute-inference baseline.
Usage: python3 -I inspect_adult.py <data_dir> <out_dir>
"""
import sys, json, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, brier_score_loss)
from sklearn.metrics import mutual_info_score

data_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

COLS = ["age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]

tr = pd.read_csv(os.path.join(data_dir, "adult.data"), header=None, names=COLS,
                 skipinitialspace=True, na_values="?")
te = pd.read_csv(os.path.join(data_dir, "adult.test"), header=None, names=COLS,
                 skipinitialspace=True, na_values="?", skiprows=1)
tr = tr.dropna(how="all"); te = te.dropna(how="all")
te["income"] = te["income"].str.rstrip(".")
tr["split"], te["split"] = "train", "test"
df = pd.concat([tr, te], ignore_index=True)

R = {}
R["rows"] = {"train": len(tr), "test": len(te), "total": len(df)}
R["columns"] = len(COLS) - 1
R["dtypes"] = {c: str(df[c].dtype) for c in COLS}
R["missing"] = {c: int(df[c].isna().sum()) for c in COLS if df[c].isna().sum()}
R["rows_with_any_missing"] = int(df[COLS].isna().any(axis=1).sum())
R["duplicate_rows"] = int(df[COLS].duplicated().sum())
R["income_dist"] = df["income"].value_counts(normalize=True).round(4).to_dict()

num = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
cat = [c for c in COLS if c not in num and c != "income"]
R["numeric_summary"] = df[num].describe().round(2).to_dict()
R["pct_zero"] = {c: round(float((df[c] == 0).mean()), 4) for c in ["capital_gain", "capital_loss"]}
R["categorical_cardinality"] = {c: int(df[c].nunique()) for c in cat}
R["top_values"] = {c: df[c].value_counts(normalize=True).head(5).round(4).to_dict() for c in cat}

# education vs education_num redundancy
R["education_num_redundant"] = bool(df.groupby("education")["education_num"].nunique().max() == 1)

# Pairwise mutual information between categorical attributes (normalized), top pairs
binned = df.copy()
binned["age_bin"] = pd.cut(df["age"], [0, 25, 35, 45, 55, 65, 120]).astype(str)
binned["hours_bin"] = pd.cut(df["hours_per_week"], [0, 30, 40, 50, 100]).astype(str)
mi_cols = cat + ["age_bin", "hours_bin", "income"]
def nmi(a, b):
    a = a.fillna("NA").astype(str); b = b.fillna("NA").astype(str)
    mi = mutual_info_score(a, b)
    ha = mutual_info_score(a, a); hb = mutual_info_score(b, b)
    return mi / np.sqrt(ha * hb) if ha > 0 and hb > 0 else 0.0
pairs = []
for i, a in enumerate(mi_cols):
    for b in mi_cols[i + 1:]:
        if {a, b} == {"education", "education_num"}:
            continue
        pairs.append((a, b, round(nmi(binned[a], binned[b]), 3)))
pairs.sort(key=lambda x: -x[2])
R["top_attribute_dependencies_nmi"] = pairs[:12]

# Quasi-identifier uniqueness (relevant to identity resolution / re-identification risk)
qi_sets = {
    "age+sex+race": ["age", "sex", "race"],
    "age+sex+race+native_country": ["age", "sex", "race", "native_country"],
    "age+sex+race+marital+education": ["age", "sex", "race", "marital_status", "education"],
    "age+sex+race+marital+education+occupation": ["age", "sex", "race", "marital_status", "education", "occupation"],
}
qi = {}
for name, cols in qi_sets.items():
    sizes = df.groupby(cols, dropna=False).size()
    k = df.merge(sizes.rename("k").reset_index(), on=cols, how="left")["k"]
    qi[name] = {"pct_unique_k1": round(float((k == 1).mean()), 4),
                "pct_k_lt_5": round(float((k < 5).mean()), 4),
                "num_groups": int(len(sizes))}
R["quasi_identifier_uniqueness"] = qi

# ---------------- Baseline attribute inference ----------------
# Adversary sees "public-profile-like" attributes, tries to infer a hidden one.
def run_task(target, drop_extra=()):
    d = df.dropna(subset=[target]).copy()
    feats = [c for c in COLS if c not in (target, "fnlwgt", "education_num") + tuple(drop_extra)]
    X, y = d[feats], d[target]
    nf = [c for c in feats if c in num]; cf = [c for c in feats if c not in num]
    pre = ColumnTransformer([
        ("num", StandardScaler(), nf),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cf)])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    for c in cf:
        Xtr[c] = Xtr[c].fillna("Missing"); Xte[c] = Xte[c].fillna("Missing")
    out = {"n": len(d), "n_classes": int(y.nunique()), "features": feats}
    models = {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline([("p", pre), ("m", LogisticRegression(max_iter=2000))]),
        "gradient_boosting": Pipeline([("p", ColumnTransformer([
            ("num", "passthrough", nf),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), cf)])),
            ("m", HistGradientBoostingClassifier(random_state=42))]),
    }
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        res = {"accuracy": round(accuracy_score(yte, p), 4),
               "precision_macro": round(precision_score(yte, p, average="macro", zero_division=0), 4),
               "recall_macro": round(recall_score(yte, p, average="macro", zero_division=0), 4),
               "f1_macro": round(f1_score(yte, p, average="macro", zero_division=0), 4)}
        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(Xte)
            conf = proba.max(axis=1)
            correct = (p == yte.values)
            # Expected calibration error (10 bins) as an "analyst confidence" measure
            bins = np.linspace(0, 1, 11); ece = 0.0
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (conf > lo) & (conf <= hi)
                if mask.any():
                    ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
            res["mean_confidence"] = round(float(conf.mean()), 4)
            res["ece"] = round(float(ece), 4)
            res["accuracy_when_conf_ge_0.9"] = (round(float(correct[conf >= 0.9].mean()), 4)
                                               if (conf >= 0.9).any() else None)
            res["pct_preds_conf_ge_0.9"] = round(float((conf >= 0.9).mean()), 4)
        out[name] = res
    return out

R["baseline"] = {
    "income": run_task("income"),
    "marital_status": run_task("marital_status", drop_extra=("relationship",)),
    "occupation": run_task("occupation"),
    "sex": run_task("sex", drop_extra=("relationship",)),
}

with open(os.path.join(out_dir, "adult_inspection.json"), "w") as f:
    json.dump(R, f, indent=2, default=str)
print(json.dumps(R, indent=1, default=str))
