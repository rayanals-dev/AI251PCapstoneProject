"""Train and evaluate YOLO for AI251P Object Detection (M4).
Run in Colab AFTER verified preparation; stores reports only in 03_Detection.
"""
from __future__ import annotations
import argparse
import collections
import csv
import json
import os
import random
import shutil
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np

CLASS_NAMES=['swimmer','boat','jetski','life_saving_appliance','buoy','car','truck','bus']

def dump(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')

def csv_write(rows,path,columns=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if columns is None:columns=list(rows[0]) if rows else ['note']
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)

def txt_records(work,split=None,domain=None):
    rows=json.loads((Path(work)/'manifest.json').read_text(encoding='utf-8'))
    return [r for r in rows if (split is None or r['split']==split) and (domain is None or r['domain']==domain)]

def seed_all(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def _log_wandb(run_name, config, history_file, summary, project, entity=None):
    try:
        import wandb
        kw=dict(project=project,name=run_name,config=config,reinit=True,job_type='detection')
        if entity:kw['entity']=entity
        with wandb.init(**kw) as run:
            if Path(history_file).exists():
                import pandas as pd
                df=pd.read_csv(history_file);df.columns=df.columns.str.strip()
                for _,row in df.iterrows():
                    clean={str(k):float(v) for k,v in row.items() if isinstance(v,(int,float,np.number)) and np.isfinite(v)}
                    if clean:run.log(clean)
            run.summary.update(summary)
        return True
    except Exception as exc:
        print('W&B logging not finished. Do not claim reproducibility credit until fixed:',str(exc))
        return False

def train(args):
    from ultralytics import YOLO
    import torch
    seed_all(args.seed)
    work=Path(args.work);report=Path(args.out);report.mkdir(parents=True,exist_ok=True)
    data=work/'data.yaml'
    if not data.exists():raise FileNotFoundError('Run prepare before train: '+str(data))
    counts=collections.Counter((r['domain'],r['split']) for r in txt_records(work))
    print('Source/split counts:',counts)
    # Fine-tune pre-trained YOLO model (rather than using off-the-shelf inference only).
    run_name=f'{datetime.now():%Y-%m-%d}_M4_joint_v1'
    model=YOLO(args.weights)
    device=0 if torch.cuda.is_available() else 'cpu'
    if device=='cpu':print('WARNING: GPU not enabled. Colab > Runtime > Change runtime type > T4 GPU.')
    trained=model.train(data=str(data),epochs=args.epochs,imgsz=args.imgsz,
                        batch=args.batch,device=device,seed=args.seed,
                        project=str(report/'runs'),name=run_name,exist_ok=True,
                        cache=False,workers=2,patience=12,
                        fliplr=0.5,flipud=0.0, degrees=5.0,
                        hsv_h=0.015,hsv_s=0.4,hsv_v=0.3, plots=True,
                        verbose=True)
    run_dir=Path(model.trainer.save_dir)
    best=run_dir/'weights'/'best.pt'
    if not best.exists():raise FileNotFoundError('Ultralytics did not write best checkpoint: '+str(best))
    trained_metrics=model.val(data=str(data),split='val',imgsz=args.imgsz,conf=.001,iou=.7,plots=True,device=device)
    summary={'best_checkpoint':str(best),'val_mAP50':float(trained_metrics.box.map50),
             'val_mAP50_95':float(trained_metrics.box.map),'image_size':args.imgsz,
             'validation_conf_for_AP':0.001,'validation_nms_iou_for_AP':0.7}
    dump(summary,report/'training_summary.json')
    spec={'model':args.weights,'epochs':args.epochs,'batch':args.batch,'imgsz':args.imgsz,
          'seed':args.seed,'device':str(device),'dataset_yaml':str(data),
          'ultralytics_version':__import__('ultralytics').__version__,
          'torch_version':torch.__version__}
    dump(spec,report/'train_config.json')
    (report/'best_model_path.txt').write_text(str(best),encoding='utf-8')
    tracked=_log_wandb(run_name,spec,run_dir/'results.csv',summary,args.wandb_project,args.wandb_entity)
    dump({'logged_to_wandb':tracked,'project':args.wandb_project,'entity':args.wandb_entity}, report/'wandb_status.json')
    print('Training complete:',summary,'W&B logged:',tracked)

def yolo_labels(path,imgwidth,imgheight):
    p=Path(path)
    if not p.exists():return []
    out=[]
    for line in p.read_text().splitlines():
        vals=line.split()
        if len(vals)!=5:continue
        c=int(vals[0]);cx,cy,w,h=map(float,vals[1:])
        out.append({'class':c,'xyxy':np.array([(cx-w/2)*imgwidth,(cy-h/2)*imgheight,(cx+w/2)*imgwidth,(cy+h/2)*imgheight]),
                    'size':(w*imgwidth)*(h*imgheight)})
    return out

def bbox_iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    overlap=max(0,x2-x1)*max(0,y2-y1)
    sa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);sb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return overlap/(sa+sb-overlap+1e-9)

def match_boxes(gts,preds,iou_thr=.5,class_agnostic=False):
    matched=set();tp=[];fps=[]
    for pred in sorted(preds,key=lambda x:-x['conf']):
        best_score=iou_thr;best=None
        for k,gt in enumerate(gts):
            if k in matched or (not class_agnostic and pred['class']!=gt['class']):continue
            score=bbox_iou(pred['xyxy'],gt['xyxy'])
            if score>=best_score:best_score=score;best=k
        if best is not None:matched.add(best);tp.append((pred,best))
        else:fps.append(pred)
    fns=[(gt,j) for j,gt in enumerate(gts) if j not in matched]
    return tp,fps,fns

def predictions(model,image_path,conf,nms_iou,imgsz):
    response=model.predict(source=str(image_path),conf=conf,iou=nms_iou,imgsz=imgsz,
                           verbose=False,max_det=300)[0]
    output=[]
    if response.boxes is not None:
        for box in response.boxes:
            output.append({'class':int(box.cls.item()),'conf':float(box.conf.item()),'xyxy':box.xyxy[0].cpu().numpy().astype(float)})
    return output

def score_records(model, records, thresholds, nms_values, imgsz, gallery=None, max_gallery=4):
    import cv2
    accum=[];cases=[];size_stats=collections.defaultdict(lambda:collections.Counter())
    galleries=collections.Counter()
    for ni,nms in enumerate(nms_values):
        print(f'NMS {nms}, images {len(records)}')
        cached=[]
        for j,r in enumerate(records):
            im=cv2.imread(r['image_path'])
            if im is None:raise FileNotFoundError(f'Unopenable source image: {r["image_path"]}')
            ih,iw=im.shape[:2]
            gt=yolo_labels(r['label_path'],iw,ih)
            det=predictions(model,r['image_path'],min(min(thresholds),.005),nms,imgsz)
            # Do not cache decoded image arrays: long UAV sequences can exhaust Colab RAM.
            cached.append((r,gt,det))
            if (j+1)%100==0:print(' processed',j+1,'/',len(records))
        for threshold in thresholds:
            tp_count=fp_count=fn_count=0
            per_domain=collections.defaultdict(lambda:collections.Counter())
            for r,gt,det in cached:
                pred=[p for p in det if p['conf']>=threshold]
                tp,fp,fn=match_boxes(gt,pred)
                n_tp,n_fp,n_fn=len(tp),len(fp),len(fn)
                tp_count+=n_tp;fp_count+=n_fp;fn_count+=n_fn
                per_domain[r['domain']].update({'tp':n_tp,'fp':n_fp,'fn':n_fn})
                if gallery is not None and threshold==thresholds[0] and nms==nms_values[0]:
                    for p,_ in tp:
                        area=gt[_]['size'];size='small' if area<1024 else ('medium' if area<9216 else 'large')
                        size_stats[(r['domain'],size)]['tp']+=1
                    for g,_ in fn:
                        area=g['size'];size='small' if area<1024 else ('medium' if area<9216 else 'large')
                        size_stats[(r['domain'],size)]['fn']+=1
                    if (fp or fn) and galleries[(r['domain'],'FP' if fp else 'FN')]<max_gallery:
                        typ='FP' if fp else 'FN'
                        galleries[(r['domain'],typ)]+=1
                        # Load only gallery candidates (never keep whole test set in RAM).
                        im_gallery=cv2.imread(r['image_path'])
                        if im_gallery is None:continue
                        canvas=im_gallery.copy()
                        # Ground truth = green; predictions = red. False-positive boxes outlined thicker.
                        for g in gt:
                            x1,y1,x2,y2=map(int,g['xyxy']);cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,180,0),2)
                        for d in pred:
                            x1,y1,x2,y2=map(int,d['xyxy']);cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,0,230),2)
                        gallery=Path(gallery);gallery.mkdir(parents=True,exist_ok=True)
                        fnm=gallery/f"{r['domain']}_{typ}_{galleries[(r['domain'],typ)]:02d}.jpg"
                        cv2.imwrite(str(fnm),canvas)
                        cases.append({'domain':r['domain'],'source':r['source_id'],'kind':typ,'gt':len(gt),'tp':n_tp,'fp':n_fp,'fn':n_fn,'image':str(fnm)})
            p=tp_count/max(1,tp_count+fp_count);rec=tp_count/max(1,tp_count+fn_count)
            f1=2*p*rec/max(1e-9,p+rec)
            row={'conf':threshold,'nms_iou':nms,'tp':tp_count,'fp':fp_count,'fn':fn_count,
                 'precision':round(p,5),'recall':round(rec,5),'f1':round(f1,5),'images':len(records)}
            for domain in ('sea','urban'):
                c=per_domain[domain]
                row[f'{domain}_recall']=round(c['tp']/max(1,c['tp']+c['fn']),5)
                row[f'{domain}_precision']=round(c['tp']/max(1,c['tp']+c['fp']),5)
            accum.append(row)
    size_rows=[{'domain':d,'size':s,'tp':v['tp'],'fn':v['fn'],
                'recall':round(v['tp']/max(1,v['tp']+v['fn']),5)} for (d,s),v in size_stats.items()]
    return accum,cases,size_rows

def choose_op(rows,required_recall=.85):
    # choose operating point ONLY on validation; prefer satisfying both domains.
    eligible=[x for x in rows if x['sea_recall']>=required_recall and x['urban_recall']>=required_recall]
    if eligible:
        winner=max(eligible,key=lambda x:(x['f1'],min(x['sea_recall'],x['urban_recall'])))
        reason='Both validation-domain recall targets met; highest pooled validation F1 among eligible candidates.'
    else:
        winner=max(rows,key=lambda x:(min(x['sea_recall'],x['urban_recall']),x['f1']))
        reason='No candidate met both domain recall targets; maximize worst-domain validation recall, then F1. Flag unmet targets.'
    return winner,reason

def do_sweep(args):
    from ultralytics import YOLO
    report=Path(args.out);report.mkdir(parents=True,exist_ok=True)
    work=Path(args.work)
    model=YOLO(args.model)
    allval=txt_records(work,split='val');random.Random(42).shuffle(allval)
    # Fixed deterministic, domain-balanced subset for threshold selection.
    half=max(1,args.max_images//2)
    sea=[r for r in allval if r['domain']=='sea'][:half]
    urban=[r for r in allval if r['domain']=='urban'][:half]
    val=sea+urban
    if not val:raise RuntimeError('No validation images found')
    thresholds=[float(x) for x in args.thresholds.split(',')]
    nms=[float(x) for x in args.nms.split(',')]
    rows,_,_=score_records(model,val,thresholds,nms,args.imgsz)
    chosen,reason=choose_op(rows,required_recall=args.target_recall)
    csv_write(rows,report/'threshold_sweep_VAL.csv')
    dump({'source_split':'val','selected':chosen,'selection_reason':reason,'sampling':'deterministic balanced per domain',
          'evaluated_sea_images':len(sea),'evaluated_urban_images':len(urban),
          'matching_iou':0.5,'mAP': 'Not measured here; AP is measured independently using Ultralytics validation at low confidence.',
          'unseen_test_used_for_selection':False},report/'operating_point_VAL.json')
    print('VAL operating point:',chosen,'\nSelection:',reason)

def test(args):
    from ultralytics import YOLO
    import ultralytics
    work=Path(args.work);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    op_path=out/'operating_point_VAL.json'
    if not op_path.exists():raise FileNotFoundError('First run sweep on validation (test set is never for threshold selection)')
    op=json.loads(op_path.read_text())['selected']
    model=YOLO(args.model)
    mAP_rows=[]
    per_class_rows=[]
    for domain,filename in [('all','data.yaml'),('sea','sea.yaml'),('urban','urban.yaml')]:
        result=model.val(data=str(work/filename),split='test',imgsz=args.imgsz,
                         conf=.001,iou=.7,plots=True,verbose=False)
        metrics={'domain':domain,'split':'test','imgsz':args.imgsz,'AP_conf':0.001,'AP_nms_iou':0.7,
                 'mAP50':round(float(result.box.map50),5),'mAP50_95':round(float(result.box.map),5)}
        # Report per-class AP only when that class occurs in the evaluation labels.
        # Metric.ap_class_index aligns with Metric.ap50 and Metric.ap arrays.
        cls_ids = list(getattr(result.box, 'ap_class_index', []))
        ap50_values = list(getattr(result.box, 'ap50', []))
        ap_values = list(getattr(result.box, 'ap', []))
        for cid, ap50, apmean in zip(cls_ids, ap50_values, ap_values):
            cls_id=int(cid)
            if not 0<=cls_id<len(CLASS_NAMES):continue
            row = {'domain':domain,'split':'test','class_id':cls_id,
                   'class_name':CLASS_NAMES[cls_id], 'AP50':round(float(ap50),5),
                   'AP50_95':round(float(apmean),5),'imgsz':args.imgsz,
                   'AP_conf':0.001,'AP_nms_iou':0.7}
            per_class_rows.append(row)
        mAP_rows.append(metrics)
        print('TEST (after VAL threshold locked)',metrics)
    csv_write(mAP_rows,out/'mAP_TEST.csv')
    csv_write(per_class_rows,out/'per_class_AP_TEST.csv')
    alltest=txt_records(work,split='test')
    # The entire test set is scored with the single locked operating point.
    scored,cases,size=score_records(model,alltest,[op['conf']],[op['nms_iou']],args.imgsz,
                                   gallery=out/'error_gallery',max_gallery=8)
    csv_write(scored,out/'operating_point_TEST.csv')
    csv_write(size,out/'recall_by_object_size_TEST.csv')
    csv_write(cases,out/'FP_FN_cases_TEST.csv')
    caveats=['UAVDT official splits group by sequence rather than common prefix; some similar sites may occur in different splits.',
             'Truck class is estimated by data-team audit to contain ~13% annotation errors; verify examples before attributing all truck errors to model.',
             'Sea and urban class sets are disjoint. Within-domain per-class AP is valid, but cross-domain class-aware recall is not an interpretable same-class transfer metric.',
             'Threshold sweep and model choice use validation; test evaluated once after the choice.']
    report={'model':str(args.model),'test_image_count':len(alltest),'chosen_on_validation':op,
            'test_operating_point_counts':scored[0], 'metrics_mAP':mAP_rows,
            'selection_set':'val','test_set':'test','input_size':args.imgsz,'matching_iou':0.5,
            'limitation_notes':caveats,'ultralytics_version':ultralytics.__version__}
    dump(report,out/'detection_report_TEST.json')
    print('TEST completed. Reports & galleries saved in',out)

def video(args):
    import cv2
    from ultralytics import YOLO
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    op=json.loads((out/'operating_point_VAL.json').read_text())['selected']
    recs=txt_records(args.work,split='test')
    # Use held-out image sequence; synthetic MP4 demonstrates actual video inference interface.
    grouped=collections.defaultdict(list)
    for r in recs:grouped[(r['domain'],r['group'])].append(r)
    selected=max(grouped.values(),key=len)
    selected=sorted(selected,key=lambda x:x['source_id'])[:min(args.max_video_frames, len(selected))]
    if not selected:raise RuntimeError('No held-out frames to compose video from')
    first=cv2.imread(selected[0]['image_path'])
    if first is None:raise RuntimeError('Cannot read sequence images')
    h,w=first.shape[:2]
    raw_path=out/'heldout_sequence_input.mp4'
    sink=cv2.VideoWriter(str(raw_path),cv2.VideoWriter_fourcc(*'mp4v'),5,(w,h))
    for r in selected:
        frame=cv2.imread(r['image_path'])
        if frame is not None:sink.write(cv2.resize(frame,(w,h)))
    sink.release()
    model=YOLO(args.model)
    output_path=out/'heldout_sequence_detected.mp4'
    sink=cv2.VideoWriter(str(output_path),cv2.VideoWriter_fourcc(*'mp4v'),5,(w,h))
    count=0
    for result in model.predict(source=str(raw_path),stream=True,imgsz=args.imgsz,
                                conf=op['conf'],iou=op['nms_iou'],verbose=False):
        vis=result.plot()
        sink.write(cv2.resize(vis,(w,h)));count+=1
    sink.release()
    dump({'input_video':str(raw_path),'annotated_video':str(output_path),'frames':count,
          'source':'reassembled held-out images from one sequence','fps':5,
          'warning':'Frame order is based on source IDs; inspect sequence timestamps before claiming this represents source timing.'}, out/'video_demo_manifest.json')
    print('Video inference:',output_path,'frames:',count)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','sweep','test','video'])
    p.add_argument('--work',default='/content/AI251P_M4_Work')
    p.add_argument('--out',required=True)
    p.add_argument('--weights',default='yolo11n.pt')
    p.add_argument('--model',default='')
    p.add_argument('--epochs',type=int,default=35)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-images',type=int,default=240)
    p.add_argument('--thresholds',default='0.05,0.10,0.20,0.35,0.50')
    p.add_argument('--nms',default='0.45,0.60,0.70')
    p.add_argument('--target-recall',type=float,default=.85)
    p.add_argument('--max-video-frames',type=int,default=75)
    p.add_argument('--wandb-project',default='AI251P_Detection')
    p.add_argument('--wandb-entity',default='')
    args=p.parse_args()
    if args.command=='train':train(args)
    else:
        if not args.model:raise ValueError('Provide --model path to trained best.pt (not original pretrained checkpoint)')
        if args.command=='sweep':do_sweep(args)
        elif args.command=='test':test(args)
        elif args.command=='video':video(args)

if __name__=='__main__':main()
