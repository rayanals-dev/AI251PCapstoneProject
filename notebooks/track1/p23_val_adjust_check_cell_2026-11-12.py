# P23 (12 Nov 2026): raw vs logit-adjusted, decided on VALIDATION ONLY. Test crops are never loaded; TestLog.csv is read, not written.
# Rule (fixed before running): use adjusted scores only if, on val, ALL hold for the selected model (full FT lr0.002, mean of seeds 42/43/44):
#   (1) cross-fit macro-F1 gain >= 0.01   (bias fitted on one half of the val groups, scored on the other half, both directions)
#   (2) accuracy drop <= 0.005            (stored bias, as published)
#   (3) ECE worsening <= 0.01             (15 bins, stored bias, as published)
# Otherwise raw stays the headline. Requires /content/aerial_crops.py (written by cell 3 of step8b_topk_report) and Drive mounted.
import re, sys, json, itertools, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, torchvision
from pathlib import Path
from sklearn.metrics import f1_score
sys.path.insert(0, '/content'); import aerial_crops as ac
R = Path('/content/drive/MyDrive/AI251P_Capstone_CrossDomainAerial'); OUT, CROP = R / '02_Classification', R / '01_Data/Processed/Crops'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'; K = 8; SEEDS = (42, 43, 44); LABELS = list(range(K))
LOG = pd.read_csv(OUT / 'TestLog.csv', keep_default_na=False)
names = {int(re.search(r'-s(\d+)-lr0\.002_v', n)[1]): n for n in LOG.checkpoints.iloc[0].split(';') if re.search(r'cls-full-64-s\d+-lr0\.002_v', n)}
assert sorted(names) == list(SEEDS), names
CK = {s: torch.load(OUT / 'Checkpoints' / n, map_location='cpu', weights_only=False) for s, n in names.items()}
used = {json.dumps(c['crops_used'], sort_keys=True) for c in CK.values()}; assert len(used) == 1; U = json.loads(used.pop())
idx = pd.read_csv(CROP / U['index'], keep_default_na=False); VAL = idx[idx.split == 'val'].sort_values('row').reset_index(drop=True)
XV = np.load(CROP / U['val']); assert len(XV) == len(VAL); YV = VAL.category.map(ac.CLASS_TO_ID).to_numpy()
GRP = (VAL.dataset + '/' + VAL.source).to_numpy(); G = sorted(set(GRP)); print(len(VAL), 'val crops,', len(G), 'groups')

class Net(nn.Module):
    def __init__(self, drop):
        super().__init__(); n = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(n.children())[:-1]); self.head = nn.Sequential(nn.Flatten(), nn.Dropout(drop), nn.Linear(512, K))
    def forward(self, x): return self.head(self.backbone(x))
MEAN, STD = torch.tensor(ac.MEAN).view(1, 3, 1, 1).to(DEVICE), torch.tensor(ac.STD).view(1, 3, 1, 1).to(DEVICE)
@torch.no_grad()
def val_logits(ck):
    m = Net(ck['config']['model']['dropout']); m.load_state_dict(ck['model']); m.to(DEVICE).eval(); img = ck['img_size']; out = []
    for i in range(0, len(XV), 512):
        x = torch.from_numpy(XV[i:i + 512]).to(DEVICE).permute(0, 3, 1, 2).float() / 255
        if x.shape[-1] != img: x = F.interpolate(x, size=(img, img), mode='bilinear', antialias=True, align_corners=False)
        with torch.autocast('cuda', enabled=DEVICE == 'cuda'): out.append(m((x - MEAN) / STD).float().cpu())
    return torch.cat(out).numpy()
def mf1(y, p): return float(f1_score(y, p, labels=LABELS, average='macro', zero_division=0))
def ece(P, y, bins=15):
    c, p = P.max(1), P.argmax(1); e = 0.0; ed = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(ed[:-1], ed[1:]):
        m = (c > lo) & (c <= hi); e += m.mean() * abs((p[m] == y[m]).mean() - c[m].mean()) if m.any() else 0
    return float(e)
def softmax(L): e = np.exp(L - L.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def fit_bias(L, y, grid, passes):                     # same coordinate search as train_cls (fit_logit_bias)
    lo, hi, st = grid; g = np.round(np.arange(lo, hi + st / 2, st), 4); b = np.zeros(K); best = mf1(y, L.argmax(1))
    for _ in range(passes):
        ch = False
        for k in LABELS:
            for v in g:
                b2 = b.copy(); b2[k] = v; s = mf1(y, (L + b2).argmax(1))
                if s > best + 1e-9: best, b, ch = s, b2, True
        if not ch: break
    return b - b.mean()

rng = np.random.default_rng(42); rows = []
for s, ck in CK.items():
    L = val_logits(ck); b = np.array(ck['logit_bias']); cfg = ck['config']['logit_adjust']
    gains = []
    for rep in range(10):                                # 10 random 50/50 splits of the val groups, both directions
        perm = rng.permutation(len(G)); A = set(np.array(G)[perm[:len(G) // 2]]); inA = np.array([g in A for g in GRP])
        for fit, ev in ((inA, ~inA), (~inA, inA)):
            bb = fit_bias(L[fit], YV[fit], cfg['grid'], cfg['passes'])
            gains.append(mf1(YV[ev], (L[ev] + bb).argmax(1)) - mf1(YV[ev], L[ev].argmax(1)))
    Pr, Pa = softmax(L), softmax(L + b)
    rows.append({'seed': s, 'f1_raw': mf1(YV, L.argmax(1)), 'f1_adj_insample': mf1(YV, (L + b).argmax(1)), 'crossfit_gain': float(np.mean(gains)),
                 'acc_raw': float((L.argmax(1) == YV).mean()), 'acc_adj': float(((L + b).argmax(1) == YV).mean()),
                 'ece_raw': ece(Pr, YV), 'ece_adj': ece(Pa, YV)}); print(rows[-1])
D = pd.DataFrame(rows); M = D.mean(numeric_only=True)
print(D.round(4).to_string(index=False)); print('\nmean over seeds:'); print(M.round(4).to_string())
c1, c2, c3 = M.crossfit_gain >= 0.01, (M.acc_raw - M.acc_adj) <= 0.005, (M.ece_adj - M.ece_raw) <= 0.01
print(f'\n(1) cross-fit gain {M.crossfit_gain:+.4f} >= 0.01: {c1}\n(2) acc drop {M.acc_raw - M.acc_adj:+.4f} <= 0.005: {c2}\n(3) ECE change {M.ece_adj - M.ece_raw:+.4f} <= 0.01: {c3}')
print('DECISION:', 'adjusted' if (c1 and c2 and c3) else 'raw')
D.to_csv(OUT / '2026-11-12_P23_ValAdjustCheck_v1.csv', index=False) if not (OUT / '2026-11-12_P23_ValAdjustCheck_v1.csv').exists() else print('output exists, not overwritten')
