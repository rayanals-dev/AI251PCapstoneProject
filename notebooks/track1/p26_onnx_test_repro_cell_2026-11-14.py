# P26: reproduce the logged test predictions from the exported ONNX (report-only, nothing tuned).
# Opens the test crops ONCE. The opening is written to TestLog.csv BEFORE the test file is read.
import csv, hashlib, datetime, numpy as np, pandas as pd, torch, torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import f1_score

RUN_BY = 'Abdulmohsen Alharthi'   # name of the person running the cell. It is written to TestLog.csv.
R = Path('/content/drive/MyDrive/AI251P_Capstone_CrossDomainAerial')   # project folder on Drive

ONNX   = R/'02_Classification/Export/2026-10-31_ClsResNet18_s43_v1/model.onnx'
SHA    = '7a82f8742aad3db7848bfc9740735262fac2c7b0c533d16689d67e1493a42b8b'   # export_report.json
NPY    = R/'01_Data/Processed/Crops/2026-10-20_CropsTest_v1.npy'
INDEX  = R/'01_Data/Processed/Crops/2026-10-20_CropIndex_v1.csv.gz'
PREDS  = R/'02_Classification/2026-10-23_TestPredictions_v1.csv.gz'
LOG    = R/'02_Classification/TestLog.csv'
OUT    = R/'02_Classification/2026-11-14_P26_OnnxTestRepro_v1.csv'
FLIPS  = R/'02_Classification/2026-11-14_P26_OnnxTestFlips_v1.csv'
COL    = 'fullFT_s43_raw'
CLASSES = ['swimmer', 'boat', 'jetski', 'life_saving_appliances', 'buoy', 'car', 'truck', 'bus']
MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1); STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

def gpu_eval(x, out):   # copied verbatim from classify.py / train_cls_2026-10-25.ipynb
    x = x.permute(0, 3, 1, 2).float() / 255
    if x.shape[-1] != out: x = F.interpolate(x, size=(out, out), mode='bilinear', antialias=True, align_corners=False)
    return (x - MEAN.to(x.device)) / STD.to(x.device)

# ---- checks that do NOT touch the test crops ----
assert RUN_BY.strip(), 'Set RUN_BY to your name first.'
assert not OUT.exists(), f'{OUT.name} already exists: P26 was already run. Do not re-open test.'
for f in (ONNX, NPY, INDEX, PREDS, LOG): assert f.exists(), f'missing: {f}'
sha = hashlib.sha256(ONNX.read_bytes()).hexdigest(); assert sha == SHA, f'model.onnx sha256 mismatch: {sha}'
import onnxruntime as ort
prov = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in ort.get_available_providers()]
sess = ort.InferenceSession(str(ONNX), providers=prov)
idx = pd.read_csv(INDEX, low_memory=False); idx = idx[idx.split == 'test'].sort_values('row').reset_index(drop=True)
saved = pd.read_csv(PREDS)
assert len(idx) == len(saved) == 13551 and (idx.row.values == np.arange(len(idx))).all()
assert (idx.crop_id.values == saved.crop_id.values).all() and (idx.category.values == saved.category.values).all()

# ---- log the opening, then open test ----
when = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M')
with open(LOG, 'a', newline='') as fh:
    csv.writer(fh).writerow([when, 'p26_onnx_test_repro_2026-11-14',
        '2026-10-31_ClsResNet18_s43_v1/model.onnx (export of 2026-10-23_cls-full-64-s43-lr0.002_v4.pt)',
        f'P26 report-only: ONNX reproduction of the logged test predictions ({COL}), raw logits, no tuning; run by {RUN_BY.strip()}'])
print('TestLog.csv row written:', when)

X = np.load(NPY, mmap_mode='r'); assert X.shape == (13551, 96, 96, 3) and X.dtype == np.uint8, X.shape
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; logits = []
with torch.no_grad():
    for i in range(0, len(X), 512):
        x = gpu_eval(torch.from_numpy(np.ascontiguousarray(X[i:i+512])).to(dev), 64).cpu().numpy().astype(np.float32)
        logits.append(sess.run(['logits'], {'input': x})[0])
L = np.concatenate(logits); pred = np.array(CLASSES)[L.argmax(1)]

# ---- compare with the logged predictions ----
y, old = saved.category.values, saved[COL].values
flip = pred != old; top2 = np.sort(L, 1)[:, -2:]; margin = top2[:, 1] - top2[:, 0]
norgb = (saved.source != 'rgb').values
mf1 = lambda m, p: f1_score(y[m], p[m], average='macro')
rows = []
for name, m in (('all test', np.ones(len(y), bool)), ('without rgb', norgb)):
    a, b = mf1(m, old), mf1(m, pred)
    rows.append(dict(case=name, n_crops=int(m.sum()), macro_f1_logged=round(a, 6), macro_f1_onnx=round(b, 6),
                     abs_diff=round(abs(a - b), 6), argmax_flips=int(flip[m].sum())))
res = pd.DataFrame(rows)
res['pass'] = (res.argmax_flips == 0) & (res.abs_diff <= 0.01)
res['run_by'] = RUN_BY.strip(); res['when_utc'] = when; res['providers'] = '+'.join(prov); res['preprocess_device'] = dev
res['onnxruntime'] = ort.__version__; res['torch'] = torch.__version__
res.to_csv(OUT, index=False)
if flip.any():
    pd.DataFrame(dict(crop_id=saved.crop_id[flip], category=y[flip], logged=old[flip], onnx=pred[flip],
                      onnx_top2_margin=margin[flip].round(6))).to_csv(FLIPS, index=False)
print(res[['case', 'n_crops', 'macro_f1_logged', 'macro_f1_onnx', 'abs_diff', 'argmax_flips', 'pass']].to_string(index=False))
print('\nP26:', 'PASS (0 flips, diff <= 0.01)' if res['pass'].all() else f'FAIL - {int(flip.sum())} flips, see {FLIPS.name}. Change nothing; report it.')
