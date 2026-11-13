"""P25: UAVDT site-level analysis from SAVED test predictions (no new test pass).
Site = first 3 chars of the UAVDT sequence name (M0206 -> M02).
Inputs: 2026-10-23_TestPredictions_v1.csv(.gz), 2026-10-19_ClassCrops_v1.csv(.gz)
"""
import sys, numpy as np, pandas as pd
from sklearn.metrics import f1_score, accuracy_score

pred_path, crops_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
SEEDS, LAB = [42, 43, 44], ["car", "truck", "bus"]

p = pd.read_csv(pred_path)
c = pd.read_csv(crops_path, low_memory=False)
u = p[p.dataset == "UAVDT"].copy()
u["site"] = u.source.str[:3]

cu = c[c.dataset == "UAVDT"].assign(site=lambda d: d.source.str[:3])
site_split = pd.crosstab(cu.site, cu.split)
train_sites = set(site_split.index[site_split.get("train", 0) > 0])
u["site_status"] = np.where(u.site.isin(train_sites), "seen_in_train", "unseen")

def mf1(df, col):  # macro-F1 over the 3 UAVDT classes (labels present in GT)
    labs = [l for l in LAB if (df.category == l).any()]
    return f1_score(df.category, df[col], labels=labs, average="macro", zero_division=0)

def summarize(df):
    r = {"n_crops": len(df), "n_seq": df.source.nunique(),
         **{f"n_{l}": int((df.category == l).sum()) for l in LAB}}
    f = [mf1(df, f"fullFT_s{s}_raw") for s in SEEDS]
    a = [accuracy_score(df.category, df[f"fullFT_s{s}_raw"]) for s in SEEDS]
    r.update({f"mF1_s{s}": round(v, 4) for s, v in zip(SEEDS, f)})
    r.update(mF1_mean=round(np.mean(f), 4), mF1_std=round(np.std(f, ddof=1), 4),
             acc_mean=round(np.mean(a), 4))
    for l in LAB:  # per-class F1, seed mean
        if (df.category == l).any():
            r[f"F1_{l}"] = round(np.mean([f1_score(df.category, df[f"fullFT_s{s}_raw"],
                               labels=[l], average="macro", zero_division=0) for s in SEEDS]), 4)
    return r

rows = [{"group": "UAVDT_all", **summarize(u)}]
for st, g in u.groupby("site_status"):
    rows.append({"group": st, **summarize(g)})
for site, g in u.groupby("site"):
    rows.append({"group": f"site_{site}", "status": g.site_status.iloc[0], **summarize(g)})
res = pd.DataFrame(rows)

# sequence-level bootstrap CI on the seen group (resample sequences)
rng = np.random.default_rng(0)
def seq_boot(df, B=2000):
    seqs = df.source.unique(); out = []
    for _ in range(B):
        pick = rng.choice(seqs, len(seqs), replace=True)
        d = pd.concat([df[df.source == s] for s in pick])
        out.append(np.mean([mf1(d, f"fullFT_s{s}_raw") for s in SEEDS]))
    return np.percentile(out, [2.5, 97.5]).round(4)
seen = u[u.site_status == "seen_in_train"]
ci_seen, ci_all = seq_boot(seen), seq_boot(u)

site_split.to_csv(f"{out_dir}/P25_SiteSplitTable.csv")
res.to_csv(f"{out_dir}/P25_UAVDT_SiteMetrics.csv", index=False)
pd.set_option("display.width", 250)
print(site_split); print()
print(res.drop(columns=[c for c in res.columns if c.startswith("mF1_s")]).to_string(index=False))
print("\nseq-bootstrap 95% CI  all:", ci_all, " seen:", ci_seen,
      " unseen: n/a (1 site, 2 sequences)")
