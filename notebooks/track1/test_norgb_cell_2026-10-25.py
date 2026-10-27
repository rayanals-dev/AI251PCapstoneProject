# Test without SDS group 'rgb' (25 Oct). Reads ONLY 2026-10-23_TestPredictions_v1.csv.gz: no checkpoint, no crops, no model.
# Same group bootstrap as step8 section 6 (2,000 draws, seed 42, SDS videos and UAVDT sequences resampled within dataset,
# metric = mean over seeds, classes absent from a draw skipped). Writes only 02_Classification/2026-10-25_TestNoRGB_v1.csv.
import re, time, numpy as np, pandas as pd
from pathlib import Path
OUT_DIR = Path('/content/drive/MyDrive/AI251P_Capstone_CrossDomainAerial/02_Classification')
PRED_FILE = OUT_DIR / '2026-10-23_TestPredictions_v1.csv.gz'
OUT_FILE = OUT_DIR / '2026-10-25_TestNoRGB_v1.csv'
N_BOOT, SEED = 2000, 42                   # must equal N_BOOT, SEED in step8_test_eval_2026-10-23_DONOTRERUN.ipynb
SELECTED = 'fullFT'                        # column prefix of 'full FT' (lr 0.002) in the predictions file
EXPECTED_FULL_CI = (0.570, 0.837)          # step8: selected model, raw macro-F1, 95% group-bootstrap CI
EXCLUDE = ('SeaDronesSee', 'rgb')
CLASSES = ['swimmer', 'boat', 'jetski', 'life_saving_appliances', 'buoy', 'car', 'truck', 'bus']   # aerial_crops order
K = len(CLASSES); SEA, URBAN = list(range(5)), list(range(5, 8)); CID = {c: i for i, c in enumerate(CLASSES)}
assert not OUT_FILE.exists(), f'{OUT_FILE.name} exists already; bump _v1 instead of overwriting'
 
P = pd.read_csv(PRED_FILE, keep_default_na=False)
assert set(P.category) <= set(CLASSES), set(P.category) - set(CLASSES)
COLS = {}                                  # (model, kind) -> {seed: column}
for c in P.columns:
    if (m := re.fullmatch(r'(.+)_s(\d+)_(raw|adjusted)', c)): COLS.setdefault((m[1], m[3]), {})[int(m[2])] = c
MODELS = sorted({m for m, _ in COLS}); print('models in file:', MODELS, '| kinds:', sorted({k for _, k in COLS}))
assert SELECTED in MODELS, f'{SELECTED} not in {MODELS}'
Y_ALL = P.category.map(CID).to_numpy()
PR_ALL = {(m, k, s): P[col].map(CID).to_numpy() for (m, k), d in COLS.items() for s, col in d.items()}
 
def metrics(Y, pred):                      # step8 summary_metrics(present_only=True): absent class -> NaN, macro skips it
    cm = np.bincount(Y * K + pred, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float); p = tp / np.maximum(cm.sum(0), 1); r = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    f1 = np.where(cm.sum(1) == 0, np.nan, f1)
    return {'acc': np.trace(cm) / cm.sum(), 'macro_f1': np.nanmean(f1), 'macro_f1_sea': np.nanmean(f1[SEA]),
            'macro_f1_urban': np.nanmean(f1[URBAN]), **{f'f1/{c}': v for c, v in zip(CLASSES, f1)}}
 
def run_case(keep):
    D = P[keep].reset_index(drop=True); Y = Y_ALL[keep]
    grp = (D.dataset + '|' + D.source).to_numpy()
    members = {g: np.flatnonzero(grp == g) for g in np.unique(grp)}
    by_ds = {d: [g for g in members if g.startswith(d + '|')] for d in ('SeaDronesSee', 'UAVDT')}
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([members[g] for d in by_ds for g in rng.choice(by_ds[d], len(by_ds[d]), replace=True)])
             for _ in range(N_BOOT)]
    res = {}
    for (m, kind), seeds in COLS.items():
        preds = [PR_ALL[(m, kind, s)][keep] for s in sorted(seeds)]
        per_seed = pd.DataFrame([metrics(Y, p) for p in preds])
        B = pd.DataFrame([np.mean([list(metrics(Y[i], p[i]).values()) for p in preds], 0) for i in draws], columns=per_seed.columns)
        with np.errstate(all='ignore'):
            res[(m, kind)] = pd.DataFrame({'mean': per_seed.mean(), 'std': per_seed.std(), 'n_seeds': len(preds),
                                           'ci_low': B.quantile(0.025), 'ci_high': B.quantile(0.975),
                                           'draws_without_class': B.isna().mean().round(3)})
    comp = D.groupby('category').agg(crops=('crop_id', 'size'), groups=('source', 'nunique')).reindex(CLASSES).fillna(0).astype(int)
    return res, comp, {d: len(v) for d, v in by_ds.items()}
 
t0 = time.time()
is_rgb = ((P.dataset == EXCLUDE[0]) & (P.source == EXCLUDE[1])).to_numpy()
CASES = {'full test': ~is_rgb | is_rgb, 'without rgb': ~is_rgb}
OUT = {name: run_case(keep) for name, keep in CASES.items()}
print(f'rgb crops removed: {is_rgb.sum()} of {len(P)} | bootstrap done in {time.time() - t0:.0f}s')
 
# 1) reproduction check: the full case must give step8's CI
lo, hi = OUT['full test'][0][(SELECTED, 'raw')].loc['macro_f1', ['ci_low', 'ci_high']]
print(f'full test, {SELECTED} raw macro-F1 CI: [{lo:.3f}, {hi:.3f}] (step8: {EXPECTED_FULL_CI})')
assert (round(lo, 3), round(hi, 3)) == EXPECTED_FULL_CI, 'does not reproduce step8: check N_BOOT, SEED, SELECTED and the file'
 
# 2) models: mean ± std over seeds, raw and adjusted, both cases, with group-bootstrap CI
rows = []
for case, (res, comp, ngroups) in OUT.items():
    for (m, kind), t in res.items():
        for met in ('macro_f1', 'macro_f1_sea', 'macro_f1_urban', 'acc'):
            rows.append({'table': 'models', 'case': case, 'model': m, 'kind': kind, 'metric': met, **t.loc[met].to_dict(),
                         'sds_groups': ngroups['SeaDronesSee'], 'uavdt_groups': ngroups['UAVDT']})
M = pd.DataFrame(rows)
show = M[M.metric == 'macro_f1'].copy()
show['value'] = [f'{a:.3f} ± {b:.3f}  [{c:.3f}, {d:.3f}]' for a, b, c, d in zip(show['mean'], show['std'], show.ci_low, show.ci_high)]
print('\nmacro-F1, mean ± std over seeds [95% group-bootstrap CI of the seed mean]:')
print(show.pivot_table(index=['model', 'kind'], columns='case', values='value', aggfunc='first').to_string())
 
# 3) per class, selected model: crops and groups with and without rgb, F1 in both cases
rows = []
for c in CLASSES:
    for case, (res, comp, _) in OUT.items():
        n_g = int(comp.loc[c, 'groups'])
        for kind in ('raw', 'adjusted'):
            if (SELECTED, kind) not in res: continue
            t = res[(SELECTED, kind)].loc[f'f1/{c}']
            rows.append({'table': 'per_class', 'case': case, 'model': SELECTED, 'kind': kind, 'class': c,
                         'crops': int(comp.loc[c, 'crops']), 'groups': n_g,
                         'flag': 'NO GROUP' if n_g == 0 else 'SINGLE GROUP' if n_g == 1 else '',
                         'metric': 'f1', **t.to_dict()})
C = pd.DataFrame(rows)
v = C[C.kind == 'raw'].copy()
v['F1 raw'] = [('n/a' if np.isnan(a) else f'{a:.3f} ± {b:.3f} [{lo:.2f}, {hi:.2f}]') for a, b, lo, hi in zip(v['mean'], v['std'], v.ci_low, v.ci_high)]
tab = v.pivot_table(index='class', columns='case', values=['crops', 'groups', 'F1 raw', 'flag'], aggfunc='first').reindex(CLASSES)
print(f'\nper class, {SELECTED}, raw (adjusted is in the CSV):'); print(tab.to_string())
flag = C[(C.case == 'without rgb') & (C.flag != '')].drop_duplicates('class')
print('\nclasses with one group or none without rgb:', dict(zip(flag['class'], flag.flag)) or 'none')
 
pd.concat([M, C], ignore_index=True).to_csv(OUT_FILE, index=False); print('saved', OUT_FILE)
 