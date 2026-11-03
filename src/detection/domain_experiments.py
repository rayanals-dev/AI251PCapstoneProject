"""Optional research question: class-agnostic sea/urban domain transfer.

The 8 semantic labels do not overlap across sea vs urban, so transfer tests
of an 8-class model cannot be interpreted as 'boat vs car' classification.
A separate 1-class target detector addresses generic localisation transfer.
Never use held-out test to select an experiment or hyperparameters.
"""
from pathlib import Path
import argparse, json, os, random, shutil, csv
import yaml


def build_oneclass(work, destination):
    work=Path(work);dest=Path(destination)
    manifest=json.loads((work/'manifest.json').read_text())
    for r in manifest:
        split=r['split'];domain=r['domain'];source=Path(r['image_path'])
        folder=dest/'images'/split;labels=dest/'labels'/split
        folder.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
        image=folder/source.name
        if not image.exists() and not image.is_symlink():image.symlink_to(source.resolve())
        lines=[]
        for line in Path(r['label_path']).read_text().splitlines():
            fields=line.split()
            if len(fields)==5:lines.append('0 '+' '.join(fields[1:]))
        (labels/(source.stem+'.txt')).write_text('\n'.join(lines)+'\n' if lines else '')
    for split in ('train','val','test'):
        for domain in ('sea','urban','joint'):
            items=[str(dest/'images'/split/Path(r['image_path']).name) for r in manifest
                   if r['split']==split and (domain=='joint' or r['domain']==domain)]
            (dest/f'{domain}_{split}.txt').write_text('\n'.join(items)+'\n')
    for regime in ('sea','urban','joint'):
        d={'path':str(dest),'train':f'{regime}_train.txt','val':'joint_val.txt',
           'test':'joint_test.txt','names':{0:'target'},'nc':1}
        (dest/f'{regime}.yaml').write_text(yaml.safe_dump(d,sort_keys=False))
    for eval_domain in ('sea','urban'):
        # same trained weights evaluated on each target domain's own heldout sequences
        d={'path':str(dest),'train':'joint_train.txt',
           'val':f'{eval_domain}_val.txt','test':f'{eval_domain}_test.txt','names':{0:'target'},'nc':1}
        (dest/f'eval_{eval_domain}.yaml').write_text(yaml.safe_dump(d,sort_keys=False))
    print('Single-class (target) data prepared:',dest,'Images:',len(manifest))


def run(work, out, epochs=20, imgsz=640, batch=16, include_appearance=False):
    from ultralytics import YOLO
    import torch, numpy as np
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    work=Path(work);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    one=Path('/content/AI251P_M4_oneclass')
    build_oneclass(work,one)
    rows=[]
    regimes=['sea','urban','joint']+(['joint_color_aug'] if include_appearance else [])
    for regime in regimes:
        src='joint' if regime=='joint_color_aug' else regime
        args={'data':str(one/f'{src}.yaml'),'epochs':epochs,'imgsz':imgsz,'batch':batch,
              'device':0 if torch.cuda.is_available() else 'cpu','seed':42,
              'project':str(out/'domain_runs'),'name':regime,'exist_ok':True,
              'patience':10,'workers':2,'plots':True,
              'hsv_h':0.03 if regime=='joint_color_aug' else .015,
              'hsv_s':0.65 if regime=='joint_color_aug' else .35,
              'hsv_v':0.5 if regime=='joint_color_aug' else .25}
        model=YOLO('yolo11n.pt');model.train(**args)
        best=Path(model.trainer.save_dir)/'weights'/'best.pt'
        y=YOLO(str(best))
        for domain in ('sea','urban'):
            # Domain selection and comparative development ONLY use val.
            m=y.val(data=str(one/f'eval_{domain}.yaml'),split='val',
                    imgsz=imgsz,conf=.001,iou=.7,plots=False)
            rows.append({'training_regime':regime,'evaluation_domain':domain,'split':'val',
                         'mAP50':float(m.box.map50),'mAP50_95':float(m.box.map),
                         'best_checkpoint':str(best),'imgsz':imgsz})
    p=out/'class_agnostic_domain_VAL.csv'
    with p.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'domain_experiment_note.txt').write_text(
        'All source images and labels remain in official splits. A separate one-class target label was used.\n'
        'Results are validation only for choosing the research model; do not claim test results.\n'
        'joint_color_aug is HSV-only, not texture transfer/copy-paste and must not be described as such.\n'
        'Classes do not overlap across sea and land, so only object-localisation transfer is evaluated.\n')
    print('Validation domain comparison saved:',p)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--work',default='/content/AI251P_M4_Work')
    p.add_argument('--out',required=True)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--include-appearance',action='store_true')
    a=p.parse_args();run(a.work,a.out,a.epochs,a.imgsz,a.batch,a.include_appearance)
