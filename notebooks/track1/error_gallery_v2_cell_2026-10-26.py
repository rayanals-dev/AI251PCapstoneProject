# Error gallery v2: one crop per object (dedupe by track, or by image for SeaDronesSee), 16 distinct objects.
# Run in the same notebook right after the figures cell (it reuses Q, T, X, pr, pred, y, kq, SHORT, save).
src = T.source.astype(str).to_numpy()
trk = T['track'].astype(str).to_numpy() if 'track' in T else np.array([''] * len(T))
key = np.where(trk != '', trk, src + '|' + T.image.astype(str).to_numpy() if 'image' in T else src)
err = np.flatnonzero((pred != y) & kq); err = err[np.argsort(-pr[err, pred[err]])]
seen, pick = set(), []
for i in err:
    if key[i] in seen: continue
    seen.add(key[i]); pick.append(i)
    if len(pick) == 16: break
fig, axs = plt.subplots(2, 8, figsize=(14, 4.4))
for a, i in zip(axs.ravel(), pick):
    a.imshow(X[i], interpolation='nearest'); a.axis('off')
    side = f'\n{int(T.box_side[i])} px' if 'box_side' in T else ''
    a.set_title(f'true {SHORT[y[i]]}\npred {SHORT[pred[i]]} ({pr[i, pred[i]]:.2f}){side}', fontsize=7)
fig.suptitle('16 most confident errors, one crop per object. Full fine-tune seed 42 (raw), test without rgb', fontsize=10)
save(fig, 'fig_error_gallery')
print('distinct objects shown:', len(pick))
