from google.colab import drive; drive.mount('/content/drive')
# Al-Hada held-out evaluation v2 (26 Oct): CLIP01 complete (t0001-t0164). Same protocol as v1, fixed before results (25 Oct):
# same checkpoints, same crop as Step 4 (square, pad 0.15, 96 px stored, 64 px input), primary metric = macro-F1 over car/truck/bus,
# any prediction of a sea class counts as an error, rotated CVAT boxes -> enclosing axis-aligned box, every run appended to AlHadaLog.csv.
import zipfile, json, glob, os, re, math, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, torchvision
from PIL import Image
R = '/content/drive/MyDrive/AI251P_Capstone_CrossDomainAerial'
FR, AN, OUT = f'{R}/01_Data/Team_AlHada/Frames', f'{R}/01_Data/Team_AlHada/Annotations', f'{R}/02_Classification'
FILES = ['Annotations_CLIP02_t0075-0148_Ziyad_V2.zip', 'Annotations_CLIP02_t0001-00074_Ziyad_V2.zip', 'task_2685723_annotations_CLIP01-T0001-T0164_coco 1.0RAYAN.zip']
CLASSES = ['swimmer','boat','jetski','life_saving_appliances','buoy','car','truck','bus']; K = 8; URBAN = [5, 6, 7]
MAP = {'Car': 'car', 'Truck': 'truck', 'Bus': 'bus'}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'; print('device', DEVICE)

# 1) boxes
rows = []
for f in FILES:
    z = zipfile.ZipFile(f'{AN}/{f}'); c = json.loads(z.read([n for n in z.namelist() if n.endswith('.json')][0]))
    cat = {x['id']: MAP[x['name']] for x in c['categories']}; img = {i['id']: i['file_name'] for i in c['images']}
    for a in c['annotations']:
        x, y, w, h = a['bbox']; t = math.radians((a.get('attributes') or {}).get('rotation', 0) or 0)
        W, H = abs(w*math.cos(t)) + abs(h*math.sin(t)), abs(w*math.sin(t)) + abs(h*math.cos(t)); cx, cy = x + w/2, y + h/2
        rows.append({'file': f, 'image': img[a['image_id']], 'category': cat[a['category_id']], 'x': cx-W/2, 'y': cy-H/2, 'w': W, 'h': H,
                     'rotated': t != 0})
B = pd.DataFrame(rows); B['clip'] = B.image.str.extract(r'(CLIP0\d)')[0]; B['box_side'] = B[['w','h']].max(axis=1)
assert not B.duplicated(['image','x','y','w','h']).any()
assert B.groupby('clip').image.nunique().to_dict() == {'CLIP01': 157, 'CLIP02': 143}, B.groupby('clip').image.nunique().to_dict()
assert len(B) == 9484, len(B)
print(B.groupby('clip').agg(frames=('image','nunique'), boxes=('image','size'), rotated=('rotated','sum')).to_string())
print(B.groupby(['clip','category']).size().unstack(fill_value=0).to_string())

# 2) crops, exactly as Step 4
def square(img, x, y, w, h, pad=0.15, out=96):
    cx, cy, s = x + w/2, y + h/2, max(w, h) * (1 + 2*pad)
    return img.crop((round(cx-s/2), round(cy-s/2), round(cx+s/2), round(cy+s/2))).resize((out, out), Image.BICUBIC)
X = np.zeros((len(B), 96, 96, 3), np.uint8)
for name, g in B.groupby('image'):
    with Image.open(f'{FR}/{name}') as im:
        im = im.convert('RGB')
        for i, r in g.iterrows(): X[i] = np.asarray(square(im, r.x, r.y, r.w, r.h))
Y = B.category.map({c: i for i, c in enumerate(CLASSES)}).to_numpy(); print('crops', X.shape)

# 3) models (same code as Step 8)
class SmallCNN(nn.Module):
    def __init__(self, n_classes=8, width=32, dropout=0.3):
        super().__init__(); layers, c_in = [], 3
        for i in range(4):
            c = width * 2**i
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.MaxPool2d(2)]; c_in = c
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, n_classes))
    def forward(self, x): return self.head(self.backbone(x))
class Pretrained(nn.Module):
    def __init__(self, n_classes, mode, dropout):
        super().__init__(); net = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(net.fc.in_features, n_classes))
    def forward(self, x): return self.head(self.backbone(x))
def build(cfg):
    M = cfg['model']; return SmallCNN(K, M['width'], M['dropout']) if M['name'] == 'SmallCNN' else Pretrained(K, M.get('mode'), M['dropout'])
MEAN = torch.tensor((0.485,0.456,0.406)).view(1,3,1,1).to(DEVICE); STD = torch.tensor((0.229,0.224,0.225)).view(1,3,1,1).to(DEVICE)
@torch.no_grad()
def logits_on(ck, X):
    m = build(ck['config']); m.load_state_dict(ck['model']); m.to(DEVICE).eval(); img = ck['img_size']; out = []
    for i in range(0, len(X), 512):
        x = torch.from_numpy(X[i:i+512]).to(DEVICE).permute(0,3,1,2).float() / 255
        if x.shape[-1] != img: x = F.interpolate(x, size=(img, img), mode='bilinear', antialias=True, align_corners=False)
        with torch.autocast('cuda', enabled=DEVICE == 'cuda'): out.append(m((x - MEAN) / STD).float().cpu())
    return torch.cat(out).numpy()
CK_DIR = f'{OUT}/Checkpoints'
EVAL = {'full FT': 'cls-full-64-s{s}-lr0.002', 'full FT lr0.001': 'cls-full-64-s{s}', 'scratch': 'cls-scratch-64-s{s}', 'frozen': 'cls-frozen-64-s{s}'}
def find(run):
    hits = sorted(f for f in os.listdir(CK_DIR) if re.match(rf'\d{{4}}-\d{{2}}-\d{{2}}_{re.escape(run)}_v\d+\.pt$', f)); return f'{CK_DIR}/{hits[-1]}'

# 4) metrics
def softmax(L): e = np.exp(L - L.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def ece(P, Y, bins=15):
    conf, pred = P.max(1), P.argmax(1); e = 0.0; edges = np.linspace(0, 1, bins+1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs((pred[m] == Y[m]).mean() - conf[m].mean())
    return float(e)
def metrics(Y, p):
    out = {'acc': float((p == Y).mean()), 'pred_sea_%': float((p < 5).mean() * 100)}
    f1s = []
    for c in URBAN:
        tp = ((p == c) & (Y == c)).sum(); pr = tp / max((p == c).sum(), 1); rc = tp / max((Y == c).sum(), 1)
        f1 = 2*pr*rc / (pr + rc) if pr + rc else 0.0; f1s.append(f1)
        out.update({f'P/{CLASSES[c]}': pr, f'R/{CLASSES[c]}': rc, f'F1/{CLASSES[c]}': f1})
    out['macro_f1'] = float(np.mean(f1s)); return out
rows, PRED = [], {}
for name, pat in EVAL.items():
    for s in (42, 43, 44):
        p = find(pat.format(s=s)); ck = torch.load(p, map_location='cpu', weights_only=False)
        L = logits_on(ck, X); bias = np.array(ck['logit_bias'])
        for kind, LL in (('raw', L), ('adjusted', L + bias)):
            pred = LL.argmax(1); PRED[(name, s, kind)] = pred
            rows.append({'model': name, 'seed': s, 'kind': kind, 'checkpoint': os.path.basename(p), **metrics(Y, pred), 'ece': ece(softmax(LL), Y)})
Rr = pd.DataFrame(rows)
main = ['macro_f1', 'acc', 'F1/car', 'F1/truck', 'F1/bus', 'R/truck', 'R/bus', 'pred_sea_%', 'ece']
print(Rr.groupby(['model','kind'])[main].agg(['mean','std']).round(3).to_string())

# 5) selected model: confusion, recall by size, per clip, frame-bootstrap CI
sel = np.stack([PRED[('full FT', s, 'raw')] for s in (42, 43, 44)])
cm = sum(pd.crosstab(pd.Categorical(B.category, ['car','truck','bus']), pd.Categorical([CLASSES[k] for k in p], CLASSES), dropna=False) for p in sel)
print('\nconfusion, full FT raw, 3 seeds pooled (rows true, cols predicted):'); print(cm.to_string())
B['size'] = pd.cut(B.box_side, [0, 16, 32, 64, 1e9], labels=['<16','16-32','32-64','>64'], right=False)
B['ok'] = (sel == Y).mean(0)
print('\nrecall by size:'); print(B.pivot_table(index='category', columns='size', values='ok', aggfunc='mean', observed=False).round(2).to_string())
print(B.pivot_table(index='category', columns='size', values='ok', aggfunc='size', observed=False).fillna(0).astype(int).to_string())
print('\nmacro-F1 per clip:', {c: round(np.mean([metrics(Y[B.clip == c], p[B.clip == c])['macro_f1'] for p in sel]), 3) for c in ['CLIP01','CLIP02']})
rng = np.random.default_rng(42); frames = B.groupby('clip').image.unique(); idx = B.groupby('image').indices; boot = []
for _ in range(1000):
    pick = np.concatenate([idx[f] for c in frames.index for f in rng.choice(frames[c], len(frames[c]))])
    boot.append(np.mean([metrics(Y[pick], p[pick])['macro_f1'] for p in sel]))
print('macro-F1 95% CI (frame bootstrap within clip; optimistic, frames are correlated):', np.percentile(boot, [2.5, 97.5]).round(3))

# 6) save + log
T = time.strftime('%Y-%m-%d'); v = 1 + len([f for f in os.listdir(OUT) if re.search(r'_AlHadaResults_v\d+\.csv$', f)])   # v1 = 25 Oct partial run
assert v == 2 and not os.path.exists(f'{OUT}/{T}_AlHadaResults_v{v}.csv'), v
Rr.to_csv(f'{OUT}/{T}_AlHadaResults_v{v}.csv', index=False)
pd.concat([B.drop(columns='ok')] + [pd.Series([CLASSES[k] for k in p], name=f'{m.replace(" ","")}_s{s}_{k}') for (m, s, k), p in PRED.items()], axis=1) \
  .to_csv(f'{OUT}/{T}_AlHadaPredictions_v{v}.csv.gz', index=False)
log = f'{OUT}/AlHadaLog.csv'; e = pd.DataFrame([{'when': time.strftime('%Y-%m-%d %H:%M'), 'version': v, 'files': ';'.join(FILES), 'crops': len(B)}])
(pd.concat([pd.read_csv(log), e]) if os.path.exists(log) else e).to_csv(log, index=False); print('saved v', v)