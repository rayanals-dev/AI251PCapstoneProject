# Rebuild the finished-run files of cls-full s42 and s44 (lr 0.002) from what training saved on 23 Oct. No training, no
# inference, no test. Run in train_cls.ipynb AFTER sections 0-6 with CONFIG_NAME = 'cls_full', VERIFY_ONLY = True, BEFORE section 7.
# Sources per run: the final checkpoint (epoch, val macro-F1, logit_bias, config), the History csv (lr the optimizer used at the
# end of warm-up, epochs run) and the ValPredictions csv (val logits of the best epoch). The History / ValPredictions file of a
# run is accepted only if it matches the checkpoint (same best-epoch val F1, same predictions with the stored logit_bias).
# Writes: Checkpoints/resume/<run>.pt for each run; the existing 134 MB partial resume file of s42 is renamed, not deleted.
REBUILD = {42: '2026-10-23_cls-full-64-s42-lr0.002_v3.pt',      # final checkpoints, the ones test was opened on
           44: '2026-10-23_cls-full-64-s44-lr0.002_v4.pt'}
LR_EXPECTED = 0.002
assert CONFIG_NAME == 'cls_full' and VERIFY_ONLY, "set CONFIG_NAME = 'cls_full' and VERIFY_ONLY = True, run sections 0-6 first"
BP = CFG['data']['balance_power']; RUN_PREFIX = f"{RUN}-{CFG['data']['img_sizes'][0]}"
import datetime
STAMP = datetime.date.today().isoformat()
 
def pick(pattern, ok):
    """the one file matching pattern for which ok(file) holds; stops if none or several match"""
    good = [f for f in sorted(OUT_DIR.glob(pattern)) if ok(f)]
    assert len(good) == 1, f'{pattern}: {len(good)} files match the checkpoint ({[f.name for f in good]})'
    return good[0]
 
for seed, ck_name in REBUILD.items():
    name = f'{RUN_PREFIX}-s{seed}-lr{LR_EXPECTED:g}'; rfile = RESUME_DIR / f'{name}.pt'
    ck_path = CKPT_DIR / ck_name; ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    assert ck['seed'] == seed and ck['img_size'] == 64 and 'logit_bias' in ck, f'{ck_name}: seed/size/logit_bias do not fit'
    ccfg = ck['config']
    print(f"\n{name}: checkpoint {ck_name} (modified {datetime.datetime.fromtimestamp(ck_path.stat().st_mtime):%Y-%m-%d %H:%M}), "
          f"best epoch {ck['epoch']}, val macro-F1 {ck['val_macro_f1']:.6f}")
 
    # 1) an existing resume file: keep it under another name if unfinished, never delete it
    if rfile.exists():
        old = torch.load(rfile, map_location='cpu', weights_only=False)
        assert not old.get('done'), f'{rfile.name} is already a finished-run file; nothing to rebuild'
        o = old['state']
        print(f"  existing resume file: unfinished, after epoch {o['epoch']}, best {o['best']:.6f} at epoch {o['best_epoch']}, "
              f"its checkpoint path {Path(o['ckpt']).name}")
        if Path(o['ckpt']).name == ck_name:
            print('  WARNING: this unfinished run writes to the same checkpoint file test was opened on; '
                  'compare the checkpoint time above with TestLog before going on')
        keep = RESUME_DIR / f'{name}.partial-{STAMP}.pt.bak'
        assert not keep.exists(), keep
        rfile.rename(keep); print(f'  renamed to {keep.name} ({keep.stat().st_size / 1e6:.0f} MB)')
 
    # 2) History of this run: best-epoch val F1 must equal the checkpoint's
    def hist_ok(f):
        H = pd.read_csv(f); r = H[H.epoch == ck['epoch']]
        return len(r) == 1 and abs(float(r['val/macro_f1'].iloc[0]) - ck['val_macro_f1']) < 1e-9
    hf = pick(f'*_History_{name}_v*.csv', hist_ok); H = pd.read_csv(hf)
    w = ccfg['train']['warmup_epochs']
    lr = float(H.loc[H.epoch == w, 'lr'].iloc[0])          # head lr after warm-up = the base lr given to the optimizer
    cfg_lr = ccfg['train']['lr']
    assert abs(lr - LR_EXPECTED) < 1e-9, f'{hf.name}: optimizer lr after warm-up {lr}, expected {LR_EXPECTED}'
    assert (abs(cfg_lr - lr) < 1e-9) if not isinstance(cfg_lr, list) else any(abs(v - lr) < 1e-9 for v in cfg_lr), (cfg_lr, lr)
    lr = LR_EXPECTED
    print(f'  lr {lr:g}: optimizer lr after warm-up (epoch {w}) in {hf.name}; checkpoint config lr {cfg_lr}')
 
    # 3) ValPredictions of this run: raw and adjusted predictions must equal the checkpoint's
    bias = np.array(ck['logit_bias'])
    def load_vp(f):
        vp = pd.read_csv(f, keep_default_na=False)
        L = vp[[f'logit_{c}' for c in ac.CLASSES]].to_numpy(float); Yv = vp.category.map(ac.CLASS_TO_ID).to_numpy()
        return vp, L, Yv
    def vp_ok(f):
        vp, L, Yv = load_vp(f)
        same_raw = (np.array(ac.CLASSES)[L.argmax(1)] == vp.pred.to_numpy()).mean()
        same_adj = (np.array(ac.CLASSES)[(L + bias).argmax(1)] == vp.pred_adj.to_numpy()).mean()
        return same_adj == 1.0 and same_raw == 1.0 and abs(macro_f1_fast(Yv, L.argmax(1)) - ck['val_macro_f1']) < 1e-6
    vf = pick(f'*_ValPredictions_{name}_v*.csv.gz', vp_ok); vp, L, Yv = load_vp(vf)
    assert vp.crop_id.tolist() == IDX['val'].crop_id.tolist(), f'{vf.name}: val crops differ from the current crop files'
    raw = metrics(L, Yv); adj = metrics(L + bias, Yv, prefix='val_adj')
    print(f"  {vf.name}: val macro-F1 raw {raw['val/macro_f1']:.6f} (checkpoint {ck['val_macro_f1']:.6f}), "
          f"adjusted {adj['val_adj/macro_f1']:.6f}  [logits stored at 4 decimals; val loss not stored]")
 
    boost = {k: v for k, v in (ccfg['train'].get('class_loss_boost') or {}).items() if v != 1.0}
    result = {'run': name, 'model': ccfg['model']['name'], 'mode': ccfg['model']['mode'], 'img_size': ck['img_size'], 'seed': seed,
              'lr': lr, 'lr_sweep': isinstance(cfg_lr, list), 'lr_in_name': True, 'balance_power': ccfg['data']['balance_power'],
              'loss_boost': boost, 'gpu_aug': ccfg['data']['gpu_aug'], 'wandb_url': None, 'best_epoch': int(ck['epoch']),
              'epochs_run': int(H.epoch.max()), 'mean_epoch_s': float(H.epoch_time_s.mean()), 'checkpoint': ck_name,
              'val_raw': raw, 'val_adjusted': adj, 'logit_bias': dict(zip(ac.CLASSES, bias.tolist())),
              'rebuilt': {'date': STAMP, 'from': [ck_name, hf.name, vf.name],
                          'note': 'finished-run file rebuilt from saved outputs, no training or inference'}}
    assert result['balance_power'] == BP and result['gpu_aug'] == CFG['data']['gpu_aug'] and not boost, \
        'checkpoint settings differ from cls_full: the run name would not match'
    state = {'epoch': int(H.epoch.max()), 'best': float(ck['val_macro_f1']), 'best_epoch': int(ck['epoch']), 'bad': None,
             'history': H.to_dict('records'), 'wandb_id': None, 'ckpt': str(ck_path)}
    assert not rfile.exists()
    torch.save({'done': True, 'result': result, 'state': state}, rfile); print(f'  wrote {rfile.name}')
print('\nnow run section 7 (VERIFY_ONLY) and the cells after it')
 