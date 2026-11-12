"""P22: check that group counts and headline numbers quoted in the Track 1 report and dataset card match the files.
Run from the repo root: python3 -I scripts/check_report_numbers.py. Exits 1 on any mismatch."""
import json, re, sys, collections, csv
fails = []
def check(name, ok, detail=''):
    print(('OK   ' if ok else 'FAIL ') + name + (f'  {detail}' if detail else '')); ok or fails.append(name)
sp = json.load(open('data/splits/2026-10-19_Splits_v1.json')); s, u = sp['seadronessee'], sp['uavdt']
g = s['group_to_split']; stills = [k for k in g if k in ('multispectral', 'rgb')]; vids = [k for k in g if k not in stills]
cs = collections.Counter(g.values()); cu = collections.Counter(u['group_to_split'].values())
check('SDS split groups = 39 (37 videos + 2 stills)', len(g) == 39 and len(vids) == 37 and len(stills) == 2, f'{len(g)} groups, {len(vids)} videos')
check('SDS groups train/val/test = 30/5/4', (cs['train'], cs['val'], cs['test']) == (30, 5, 4), dict(cs))
check('UAVDT sequences train/val/test = 34/7/9', (cu['train'], cu['val'], cu['test']) == (34, 7, 9), dict(cu))
check('video_to_split and group_to_split agree', s['video_to_split'] == g)
card, rd = open('data/DATASET_CARD.md').read(), open('reports/track1/README.md').read()
check('card says 37 videos plus 2 still groups', '37 videos (flights) plus 2 still-image groups' in card)
check('card has no "39 video groups"', '39 video groups' not in card)
check('no text says "4 SeaDronesSee videos"', '4 SeaDronesSee videos' not in card + rd)
check('card split table says 30 / 5 / 4', '30 / 5 / 4' in card)
rows = list(csv.DictReader(open('reports/track1/2026-10-23_Track1Comparison_v2.csv')))
for r in rows:
    if r['model'] == 'full_lr0.002' and r['split'] == 'val':
        check('README val macro-F1 raw (selected) matches CSV', r['raw_macro_f1'].split(' ')[0] in rd, r['raw_macro_f1'])
        check('README val macro-F1 adjusted (selected) matches CSV', r['adj_macro_f1'].split(' ')[0] in rd, r['adj_macro_f1'])
sys.exit(1 if fails else 0)
