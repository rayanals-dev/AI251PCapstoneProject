# Phase 1 report figures (26 Oct). Reads only saved files: History csv, TestPredictions_v1, TestProbs_v1, test crops (for the gallery).
# No model is run and nothing is tuned. Writes PNG + PDF to 02_Classification/figures/. CPU is enough.
from google.colab import drive; drive.mount('/content/drive')
import glob, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
R = '/content/drive/MyDrive/AI251P_Capstone_CrossDomainAerial'
OUT = f'{R}/02_Classification'; FIG = f'{OUT}/figures'; os.makedirs(FIG, exist_ok=True)
CROP_DIR = f'{R}/01_Data/Processed/Crops'
C = ['swimmer', 'boat', 'jetski', 'life_saving_appliances', 'buoy', 'car', 'truck', 'bus']
SHORT = ['swimmer', 'boat', 'jetski', 'LSA', 'buoy', 'car', 'truck', 'bus']
SEEDS = (42, 43, 44)
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
def save(fig, name):
    fig.savefig(f'{FIG}/{name}.png', bbox_inches='tight'); fig.savefig(f'{FIG}/{name}.pdf', bbox_inches='tight'); plt.show(); print('saved', name)

# 1) learning curves: scratch vs full fine-tune (lr 0.002), one line per seed
RUNS = {'SmallCNN from scratch': 'cls-scratch-64-s{s}', 'ResNet18 full fine-tune (lr 0.002)': 'cls-full-64-s{s}-lr0.002'}
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for (name, pat), col in zip(RUNS.items(), ['tab:orange', 'tab:blue']):
    for s in SEEDS:
        f = sorted(glob.glob(f'{OUT}/*_History_{pat.format(s=s)}_v*.csv'))
        assert f, pat.format(s=s); H = pd.read_csv(f[-1])
        lab = name if s == 42 else None
        ax[0].plot(H.epoch, H['train/loss'], '--', color=col, alpha=.5)
        ax[0].plot(H.epoch, H['val/loss'], '-', color=col, alpha=.8, label=lab)
        ax[1].plot(H.epoch, H['val/macro_f1'], '-', color=col, alpha=.8, label=lab)
        b = H['val/macro_f1'].idxmax(); ax[1].plot(H.epoch[b], H['val/macro_f1'][b], 'o', color=col, ms=4)
ax[0].set(xlabel='epoch', ylabel='loss', title='Loss (dashed: train, solid: val)'); ax[0].legend(fontsize=8)
ax[1].set(xlabel='epoch', ylabel='val macro-F1', title='Validation macro-F1 (dot: selected epoch)'); ax[1].legend(fontsize=8)
for a in ax: a.grid(alpha=.3)
fig.suptitle('Learning curves, 3 seeds each (42, 43, 44)'); save(fig, 'fig_learning_curves')

# 2) confusion matrix: selected model, raw, 3 seeds pooled, test without rgb, row-normalised (recall)
P = pd.read_csv(f'{OUT}/2026-10-23_TestPredictions_v1.csv.gz', keep_default_na=False)
keep = ~((P.dataset == 'SeaDronesSee') & (P.source == 'rgb'))
cm = np.zeros((8, 8))
for s in SEEDS:
    cm += pd.crosstab(pd.Categorical(P.category[keep], C), pd.Categorical(P[f'fullFT_s{s}_raw'][keep], C), dropna=False).to_numpy()
cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.2, 5.4)); im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
for i in range(8):
    for j in range(8):
        if cm[i, j]: ax.text(j, i, f'{cmn[i,j]:.2f}', ha='center', va='center', fontsize=7, color='white' if cmn[i, j] > .5 else 'black')
ax.set_xticks(range(8), SHORT, rotation=45, ha='right'); ax.set_yticks(range(8), SHORT)
ax.set(xlabel='predicted', ylabel='true', title=f'Confusion matrix, full fine-tune (raw)\ntest without rgb, {int(keep.sum())} crops x 3 seeds, row-normalised')
ax.axhline(4.5, color='k', lw=.8); ax.axvline(4.5, color='k', lw=.8); fig.colorbar(im, fraction=.046); save(fig, 'fig_confusion_matrix')

# 3) reliability diagram: raw probabilities, test without rgb, 3 seeds pooled, 15 bins (same bins as the ECE)
Q = pd.read_csv(f'{OUT}/2026-10-25_TestProbs_v1.csv.gz', keep_default_na=False)
kq = (~((Q.dataset == 'SeaDronesSee') & (Q.source == 'rgb'))).to_numpy(); Y = Q.category.map(C.index).to_numpy()[kq]
fig, ax = plt.subplots(figsize=(5.2, 5)); edges = np.linspace(0, 1, 16)
for m, lab, col in [('fullFT', 'full fine-tune', 'tab:blue'), ('scratch', 'from scratch', 'tab:orange'), ('frozen', 'frozen', 'tab:green')]:
    conf, corr = [], []
    for s in SEEDS:
        pr = Q[[f'{m}_s{s}_raw_p_{c}' for c in C]].to_numpy()[kq]; conf.append(pr.max(1)); corr.append(pr.argmax(1) == Y)
    conf, corr = np.concatenate(conf), np.concatenate(corr); xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (conf > lo) & (conf <= hi)
        if b.sum() >= 30: xs.append(conf[b].mean()); ys.append(corr[b].mean())
    ax.plot(xs, ys, 'o-', color=col, ms=4, label=lab)
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='perfect calibration')
ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='confidence (max softmax)', ylabel='accuracy',
       title='Reliability diagram, raw, test without rgb\n3 seeds pooled, 15 bins (bins with >= 30 crops)')
ax.grid(alpha=.3); ax.legend(fontsize=8); save(fig, 'fig_reliability')

# 4) error gallery: 24 most confident errors of the selected model (seed 42, raw), test without rgb
import json, torch
ck = torch.load(sorted(glob.glob(f'{OUT}/Checkpoints/2026-10-23_cls-full-64-s42-lr0.002_v*.pt'))[-1], map_location='cpu', weights_only=False)
idx = pd.read_csv(f'{CROP_DIR}/{ck["crops_used"]["index"]}', keep_default_na=False)
T = idx[idx.split == 'test'].sort_values('row').reset_index(drop=True); assert T.crop_id.tolist() == Q.crop_id.tolist()
X = np.load(f'{CROP_DIR}/{ck["crops_used"]["test"]}', mmap_mode='r')
pr = Q[[f'fullFT_s42_raw_p_{c}' for c in C]].to_numpy(); pred = pr.argmax(1); y = Q.category.map(C.index).to_numpy()
err = np.flatnonzero((pred != y) & kq); err = err[np.argsort(-pr[err, pred[err]])][:24]
fig, axs = plt.subplots(3, 8, figsize=(14, 6.2))
for a, i in zip(axs.ravel(), err):
    a.imshow(X[i], interpolation='nearest'); a.axis('off')
    a.set_title(f'true {SHORT[y[i]]}\npred {SHORT[pred[i]]} ({pr[i, pred[i]]:.2f})\n{int(T.box_side[i]) if "box_side" in T else ""}', fontsize=7)
fig.suptitle('Most confident errors, full fine-tune seed 42 (raw), test without rgb. Title: true / predicted (confidence)', fontsize=10)
save(fig, 'fig_error_gallery')
print('all figures in', FIG)
