"""AI251P M4 detection pipeline. Data are not bundled. Run from Google Colab.

This module never edits 01_Data, the team splits, or other workstreams.
Project-specific schemas must be verified during the inspect stage.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

CLASS_NAMES = ["swimmer", "boat", "jetski", "life_saving_appliance", "buoy", "car", "truck", "bus"]
EXPECTED_COCO_IDS = {i+1: k for i,k in enumerate(CLASS_NAMES)}
SPLIT_NAMES = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def json_dump(obj, path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def normalize_name(s):
    s=re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip('_')
    aliases={"jet_ski":"jetski","jet_skiing":"jetski","jet_ski_rider":"jetski", "life_saving_appliances":"life_saving_appliance", "life_saving_device":"life_saving_appliance", "lifesaving_appliance":"life_saving_appliance", "lifesaving_appliances":"life_saving_appliance", "people":"swimmer", "swimming_person":"swimmer", "boats":"boat", "cars":"car", "trucks":"truck", "buses":"bus", "buoys":"buoy"}
    return aliases.get(s,s)


def get_any(d, names, default=None):
    if not isinstance(d,dict):return default
    for name in names:
        if name in d and d[name] is not None and str(d[name]).strip()!='':return d[name]
    # Allow case variants
    low={str(k).lower():k for k in d}
    for name in names:
        if name.lower() in low:return d[low[name.lower()]]
    return default


def sample_structure(obj):
    if isinstance(obj,dict):
        out={}
        for k,v in obj.items():
            if isinstance(v,dict):out[k]={"type":"dict", "size":len(v),"first_entry":str(next(iter(v.items()), ''))[:400]}
            elif isinstance(v,list):out[k]={"type":"list","size":len(v),"first_entry":str(v[:1])[:400]}
            else:out[k]=str(v)[:300]
        return out
    return {"type":type(obj).__name__, "sample":str(obj)[:400]}


def get_paths(root):
    root=Path(root)
    proc=root/'01_Data'/'Processed'
    return dict(root=root,processed=proc,raw=root/'01_Data'/'Raw',
                splits=proc/'Splits'/'2026-10-19_Splits_v1.json',
                sds=proc/'SeaDronesSee'/'2026-10-18_SDSClean_v1.json',
                frames=proc/'UAVDT'/'2026-10-18_UAVDTFrames_v1.csv',
                boxes=proc/'UAVDT'/'2026-10-18_UAVDTBoxes_v1.csv.gz')


def validate_paths(root):
    paths=get_paths(root)
    missing=[str(p) for k,p in paths.items() if k not in ('root','processed','raw') and not p.is_file()]
    if missing:raise FileNotFoundError('Missing processed inputs (ask data owner; do not recreate splits):\n'+'\n'.join(missing))
    if not paths['raw'].exists():raise FileNotFoundError('Raw folder missing: '+str(paths['raw']))
    return paths


def load_splits(path):
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    for d in ('seadronessee','uavdt'):
        if d not in obj:raise ValueError('Missing dataset in official splits: '+d)
    return obj


def index_images(raw, extra_root=None):
    # Scan only image paths; do not inspect/read or modify image pixels.
    roots=[Path(raw)]
    if extra_root and Path(extra_root).exists():roots.append(Path(extra_root))
    by_basename=collections.defaultdict(list); paths=[]; archives=[]
    for top in roots:
        if not top.exists():continue
        for cur,dirs,files in os.walk(top):
            dirs[:]=[x for x in dirs if not x.startswith('.') and x!='__MACOSX']
            for fn in files:
                path=Path(cur)/fn
                if path.suffix.lower() in IMAGE_EXTS:
                    paths.append(path);by_basename[path.name.lower()].append(path)
                elif fn.lower().endswith(('.zip','.tar','.tar.gz','.tgz','.7z')):archives.append(path)
    return paths,by_basename,archives


def resolve_image(name, by_basename, hint=None):
    if name is None:return None
    name=str(name).replace('\\','/').strip()
    if not name:return None
    p=Path(name)
    if p.is_file():return p
    cands=by_basename.get(p.name.lower(),[])
    if not cands:return None
    if len(cands)==1:return cands[0]
    # Prefer suffix path match; never return ambiguous frames from another sequence.
    parts=[s.lower() for s in p.parts]
    if len(parts)>1:
        exact=[x for x in cands if str(x).replace('\\','/').lower().endswith('/'.join(parts))]
        if len(exact)==1:return exact[0]
    if hint:
        hints=[str(hint).lower(),Path(str(hint)).stem.lower()]
        filtered=[x for x in cands if any(h in [part.lower() for part in x.parts] for h in hints)]
        if len(filtered)==1:return filtered[0]
    return None


def _rows(container):
    if isinstance(container,list):return container
    if isinstance(container,dict):
        result=[]
        for k,v in container.items():
            if isinstance(v,dict):
                d=dict(v);d.setdefault('id',k);result.append(d)
        return result
    return []


def find_coco(obj):
    if not isinstance(obj,dict):return None
    if all(k in obj for k in ('images','annotations','categories')):return obj
    for value in obj.values():
        if isinstance(value,dict) and all(k in value for k in ('images','annotations','categories')):return value
    return None


def dict_split_lookup(mapping,key):
    if not isinstance(mapping,dict) or key is None:return None
    for candidate in [key,str(key)]:
        val=mapping.get(candidate)
        if isinstance(val,str) and val in SPLIT_NAMES:return val
        if isinstance(val,dict) and val.get('split') in SPLIT_NAMES:return val['split']
    # alternative split -> [ids] representation
    for split in SPLIT_NAMES:
        val=mapping.get(split)
        if isinstance(val,(list,tuple,set)) and (key in val or str(key) in {str(x) for x in val}):return split
    return None


def extract_split(ds,record,id_keys,group_keys,group_map_key,record_map_key):
    """Resolve from official, immutable split assignments, never random-resplit."""
    imgid=get_any(record,id_keys)
    group=get_any(record,group_keys)
    src=record.get('source') if isinstance(record.get('source'),dict) else {}
    if group is None: group=get_any(src,group_keys)
    result = dict_split_lookup(ds.get(record_map_key,{}),imgid)
    if result:return result
    # Often `images` or `frames` list entries contain IDs and split labels
    rm=ds.get(record_map_key)
    if isinstance(rm,list):
        for rec in rm:
            if isinstance(rec,dict) and str(get_any(rec,id_keys))==str(imgid) and rec.get('split') in SPLIT_NAMES:return rec['split']
    for key in (group_map_key,'group_to_split'):
        result=dict_split_lookup(ds.get(key,{}),group)
        if result:return result
    return None


def xywh_from_annotation(ann,width,height):
    if 'bbox' in ann:
        box=ann['bbox']
        if isinstance(box,dict):
            box=[get_any(box,['x','left','xmin']),get_any(box,['y','top','ymin']),get_any(box,['width','w']),get_any(box,['height','h'])]
        if not isinstance(box,(list,tuple)) or len(box)<4:raise ValueError('Unrecognized COCO bbox: '+str(box))
        x,y,w,h=map(float,box[:4])
    else:
        x=float(get_any(ann,['x','left','xmin']))
        y=float(get_any(ann,['y','top','ymin']))
        w=float(get_any(ann,['width','w']))
        h=float(get_any(ann,['height','h']))
    # Actual COCO format is absolute x,y,width,height; normalized COCO is nonstandard.
    return x,y,w,h


def xywh_yolo(x,y,w,h,iw,ih):
    if min(w,h)<=0 or min(iw,ih)<=0:return None
    x1=max(0,min(float(iw),x)); y1=max(0,min(float(ih),y))
    x2=max(0,min(float(iw),x+w));y2=max(0,min(float(ih),y+h))
    if x2<=x1 or y2<=y1:return None
    return ((x1+x2)/(2*iw),(y1+y2)/(2*ih),(x2-x1)/iw,(y2-y1)/ih)


def _sds_category_map(coco):
    cats=_rows(coco['categories'])
    mp={}
    for c in cats:
        cat_id=int(get_any(c,['id','category_id']))
        name=normalize_name(get_any(c,['name','category','label'],''))
        if name == 'ignored':
            continue
        if name not in CLASS_NAMES:raise ValueError(f'Unknown SeaDronesSee class {name!r} for category_id {cat_id}; verify COCO mapping before training.')
        mp[cat_id]=CLASS_NAMES.index(name)
    if not mp:raise ValueError('No categories found in SDS cleaned annotations')
    return mp


def make_sds_records(paths,splits,by_basename,limit_report=12):
    coco=find_coco(json.loads(paths['sds'].read_text(encoding='utf-8')))
    if coco is None:raise ValueError('SeaDronesSee cleaned JSON not standard COCO. Inspect schema and adjust adapter: '+str(paths['sds']))
    categories=_sds_category_map(coco)
    images=_rows(coco['images']);annotations=_rows(coco['annotations'])
    ann_map=collections.defaultdict(list)
    for a in annotations:ann_map[str(get_any(a,['image_id','imageId']))].append(a)
    ds=splits['seadronessee'];records=[];bad=[]
    for im in images:
        image_id=get_any(im,['id','image_id','imageId']);name=get_any(im,['file_name','filename','name','path','image_path'])
        src=im.get('source') if isinstance(im.get('source'),dict) else {}
        video=get_any(im,['video','video_id','video_name','sequence','seq','group']) or get_any(src,['video','video_id','video_name','sequence','seq','group'])
        split=extract_split(ds,im,['id','image_id','imageId'],['video','video_id','video_name','sequence','seq','group'],'video_to_split','images')
        width=get_any(im,['width','image_width']);height=get_any(im,['height','image_height'])
        path=resolve_image(name,by_basename,video)
        if not split or path is None:
            bad.append({'image_id':image_id,'name':name,'video':video,'split':split,'file_found':path is not None})
            continue
        if not width or not height:
            import cv2
            pic=cv2.imread(str(path));height,width=pic.shape[:2] if pic is not None else (None,None)
        if not width or not height:
            bad.append({'image_id':image_id,'name':name,'problem':'missing dimensions'});continue
        labels=[]
        for ann in ann_map.get(str(image_id),[]):
            cid=get_any(ann,['category_id','categoryId'])
            if cid is not None and str(cid).strip() == '0':
                continue
            if cid is None or int(cid) not in categories:
                bad.append({'image_id':image_id,'problem':'unknown category','cid':cid});continue
            try:
                box=xywh_yolo(*xywh_from_annotation(ann,int(width),int(height)),int(width),int(height))
            except Exception as e: bad.append({'image_id':image_id,'problem':str(e)});continue
            if box is not None:labels.append([categories[int(cid)],*box])
        records.append({'dataset':'seadronessee','domain':'sea','group':str(video),'split':split,
                        'source_id':str(image_id),'path':str(path),'width':int(width),'height':int(height),'labels':labels})
    return records,bad, {'images':len(images),'annotations':len(annotations),'category_map_COCO_to_YOLO':categories}


def _df_col(df,names):
    cols={str(c).lower().strip():c for c in df.columns}
    for candidate in names:
        if candidate in cols:return cols[candidate]
    return None


def _uavdt_category(v,mode='auto'):
    if isinstance(v,str) and not re.fullmatch(r'\s*\d+(?:\.0+)?\s*',v):
        label=normalize_name(v)
        if label in ('car','truck','bus'):return CLASS_NAMES.index(label)
        return None
    try:i=int(float(v))
    except:return None
    if mode=='unified678':mapping={6:5,7:6,8:7}
    elif mode=='uavdt012':mapping={0:5,1:6,2:7}
    else: mapping={1:5,2:6,3:7} # native UAVDT (verify first)
    return mapping.get(i)


def make_uavdt_records(paths,splits,by_basename,label_mode='uavdt123'):
    import pandas as pd
    frames=pd.read_csv(paths['frames']);boxes=pd.read_csv(paths['boxes'])
    if 'keep_5fps' not in frames.columns:
        raise ValueError('Missing keep_5fps column.')
    frames=frames[
        frames['keep_5fps'].astype(str).str.strip().str.lower().isin(
            ['true','1','yes']
        )
    ].copy()
    if frames.empty:
        raise ValueError('No selected UAVDT frames found.')
    seq_candidates=['sequence','seq','seq_name','sequence_name','video','video_id','folder','group']
    frm_candidates=['frame','frame_idx','frame_id','frame_num','frame_number','frame_index','img_idx','img_id']
    seq_f=_df_col(frames,seq_candidates);frame_f=_df_col(frames,frm_candidates)
    seq_b=_df_col(boxes,seq_candidates);frame_b=_df_col(boxes,frm_candidates)
    if not all([seq_f,frame_f,seq_b,frame_b]):
        raise ValueError('UAVDT frame/box join keys unknown. Frames='+str(list(frames.columns))+'; Boxes='+str(list(boxes.columns)))
    class_b=_df_col(boxes,['class','category','class_id','category_id','class_name','object_type','type','obj_type','label'])
    xcol=_df_col(boxes,['x','left','xmin','bbox_left','x1','bb_left'])
    ycol=_df_col(boxes,['y','top','ymin','bbox_top','y1','bb_top'])
    wcol=_df_col(boxes,['w','width','bbox_width','bb_width'])
    hcol=_df_col(boxes,['h','height','bbox_height','bb_height'])
    x2col=_df_col(boxes,['xmax','x2','right']);y2col=_df_col(boxes,['ymax','y2','bottom'])
    if not class_b or not all([xcol,ycol]) or not (all([wcol,hcol]) or all([x2col,y2col])):
        raise ValueError('UAVDT box columns not recognized. Found: '+str(list(boxes.columns)))
    path_f=_df_col(frames,['file_name','filename','image_path','image_file','path','filepath','relative_path','img_path','frame_path'])
    width_f=_df_col(frames,['width','img_width','image_width','w']);height_f=_df_col(frames,['height','img_height','image_height','h'])
    sds=splits['uavdt'];bad=[]
    boxes['_join_seq']=boxes[seq_b].astype(str).str.strip()
    boxes['_join_frame']=boxes[frame_b].astype(str).str.replace(r'\.0$', '', regex=True).str.strip().str.lstrip('0').replace('', '0')
    groups={k:sub for k,sub in boxes.groupby(['_join_seq','_join_frame'],sort=False)}
    results=[]
    for row in frames.to_dict('records'):
        seq=str(row[seq_f]).strip();fr_raw=str(row[frame_f]);fr=str(fr_raw).removesuffix('.0').lstrip('0') or '0'
        spl=extract_split(sds,{'seq':seq,'frame':fr,'id':get_any(row,['id','frame_id','frame_key'])},
                          ['id','frame_id','frame_key'],['seq'],'seq_to_split','frames')
        # Some datasets use frames map with 'seq/frame' keys
        if not spl:
            spl=dict_split_lookup(sds.get('frames',{}),f'{seq}/{fr}') or dict_split_lookup(sds.get('frames',{}),f'{seq}_{fr}')
        filename=row.get(path_f) if path_f else None
        path=resolve_image(filename,by_basename,seq)
        if path is None:
            # UAVDT raw frame commonly {seq}/img1/{frame:06d}.jpg
            variants=[]
            try:
                fi=int(float(fr_raw))
                for pad in (6,5,4,1):
                    for ext in ('.jpg','.png','.jpeg'):
                        variants.append(f'{fi:0{pad}d}{ext}')
            except:pass
            for name in variants:
                path=resolve_image(name,by_basename,seq)
                if path is not None:break
        if not spl or not path:
            bad.append({'seq':seq,'frame':fr,'name':filename,'split':spl,'file_found':path is not None});continue
        try:width=int(row[width_f]) if width_f else 0;height=int(row[height_f]) if height_f else 0
        except:width=height=0
        if not width or not height:
            import cv2
            img=cv2.imread(str(path))
            if img is None:bad.append({'seq':seq,'frame':fr,'error':'image unreadable'});continue
            height,width=img.shape[:2]
        group=groups.get((seq,fr));labels=[]
        if group is not None:
            for b in group.to_dict('records'):
                c=_uavdt_category(b.get(class_b),label_mode)
                if c is None:
                    bad.append({'seq':seq,'frame':fr,'error':'unmapped class','value':str(b.get(class_b))});continue
                try:
                    x=float(b[xcol]);y=float(b[ycol])
                    w=float(b[wcol]) if wcol else float(b[x2col])-x
                    h=float(b[hcol]) if hcol else float(b[y2col])-y
                    box=xywh_yolo(x,y,w,h,width,height)
                except Exception as e:
                    bad.append({'seq':seq,'frame':fr,'error':'invalid bbox '+str(e)});continue
                if box is not None:labels.append([c,*box])
        results.append({'dataset':'uavdt','domain':'urban','group':seq,'split':spl,'source_id':f'{seq}_{fr}',
                        'path':str(path),'width':width,'height':height,'labels':labels})
    extra={'frames':len(frames),'box_rows':len(boxes),'frame_columns':list(frames.columns),
           'box_columns':list(boxes.columns),'label_mode':label_mode,
           'raw_classes_sample':boxes[class_b].value_counts(dropna=False).head(15).to_dict()}
    return results,bad,extra


def audit_records(records,bad,stats, report_path):
    report={'generated':datetime.now().isoformat(timespec='seconds'),'source_statistics':stats,
            'usable_images':len(records),'unresolved_record_count':len(bad),'unresolved_examples':bad[:40],
            'images_by_split':dict(collections.Counter(r['split'] for r in records)),
            'images_by_domain':dict(collections.Counter(r['domain'] for r in records)),
            'classes_by_domain':{},'groups_per_split':{},'split_leakage':[]}
    for domain in ('sea','urban'):
        report['classes_by_domain'][domain]=dict(collections.Counter(CLASS_NAMES[int(box[0])] for r in records if r['domain']==domain for box in r['labels']))
    for ds in ('seadronessee','uavdt'):
        splits_by_group=collections.defaultdict(set)
        for r in records:
            if r['dataset']==ds:splits_by_group[r['group']].add(r['split'])
        for group, assignments in splits_by_group.items():
            if len(assignments)>1:report['split_leakage'].append([ds,group,sorted(assignments)])
        report['groups_per_split'][ds]=dict(collections.Counter(next(iter(s)) for s in splits_by_group.values() if len(s)==1))
    json_dump(report,report_path)
    return report


def build_dataset(records,dirpath,copy_images=False):
    dirpath=Path(dirpath)
    for split in SPLIT_NAMES:
        (dirpath/'images'/split).mkdir(parents=True,exist_ok=True)
        (dirpath/'labels'/split).mkdir(parents=True,exist_ok=True)
    manifest=[]
    for r in records:
        unique=hashlib.md5((r['dataset']+':'+r['source_id']+':'+r['path']).encode()).hexdigest()[:14]
        fname=f"{r['dataset']}_{unique}{Path(r['path']).suffix.lower()}"
        dst=dirpath/'images'/r['split']/fname
        if not dst.exists() and not dst.is_symlink():
            if copy_images:shutil.copy2(r['path'],dst)
            else:dst.symlink_to(Path(r['path']).resolve())
        lab=dirpath/'labels'/r['split']/(Path(fname).stem+'.txt')
        lab.write_text(''.join(f'{int(b[0])} '+ ' '.join(f'{float(v):.7f}' for v in b[1:])+'\n' for b in r['labels']),encoding='utf-8')
        manifest.append({k:r[k] for k in ('dataset','domain','group','split','source_id','path','width','height')}
                        | {'image_path':str(dst),'label_path':str(lab),'objects':len(r['labels'])})
    with (dirpath/'manifest.json').open('w') as f:json.dump(manifest,f,indent=2)
    import yaml
    yaml_path=dirpath/'data.yaml'
    yaml_path.write_text(yaml.safe_dump({'path':str(dirpath),'train':'images/train','val':'images/val','test':'images/test','names':{i:k for i,k in enumerate(CLASS_NAMES)},'nc':8},sort_keys=False), encoding='utf-8')
    for domain in ('sea','urban'):
        for split in SPLIT_NAMES:
            lines=[m['image_path'] for m in manifest if m['domain']==domain and m['split']==split]
            (dirpath/f'{domain}_{split}.txt').write_text('\n'.join(lines)+'\n' if lines else '',encoding='utf-8')
        d={'path':str(dirpath),'train':f'{domain}_train.txt','val':f'{domain}_val.txt','test':f'{domain}_test.txt',
           'names':{i:k for i,k in enumerate(CLASS_NAMES)},'nc':8}
        (dirpath/f'{domain}.yaml').write_text(yaml.safe_dump(d,sort_keys=False), encoding='utf-8')
    return manifest


def prepare(root,work_dir,output_dir,copy_images=False,label_mode='uavdt123',extra_root=None):
    paths=validate_paths(root);splits=load_splits(paths['splits'])
    paths_found,by_basename,archives=index_images(paths['raw'],extra_root)
    print('Images found in Raw/staged:',len(paths_found))
    print('Archive files:',[str(p) for p in archives[:20]])
    if len(paths_found)==0:raise RuntimeError('No extracted images found. Extract permitted source archives under 01_Data/Raw or /content/ai251p_extracted; do NOT unpack into GitHub. Run inspect first.')
    sds,bad_sds,stats_sds=make_sds_records(paths,splits,by_basename)
    uav,bad_uav,stats_uav=make_uavdt_records(paths,splits,by_basename,label_mode)
    records=sds+uav
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    report=audit_records(records,bad_sds+bad_uav,{'sea':stats_sds,'urban':stats_uav},output_dir/'preparation_audit.json')
    print(json.dumps({k:v for k,v in report.items() if k not in ('unresolved_examples','source_statistics')},indent=2))
    if report['split_leakage']:raise RuntimeError('Official split assignment violation; inspect preparation_audit.json')
    if len(sds)==0 or len(uav)==0:raise RuntimeError('Could not resolve images/official splits in one domain; inspect audit and check paths/CSV schema.')
    # Prevent silently dropping most rows because of wrong group or image matching.
    expected=stats_sds['images']+stats_uav['frames']
    if len(records)<expected*.80:
        raise RuntimeError(f'Only {len(records)}/{expected} source frames resolved. Check preparation_audit.json before accepting this dataset; no training dataset was built.')
    for d in ('sea','urban'):
        for s in SPLIT_NAMES:
            if not any(r['domain']==d and r['split']==s for r in records):raise RuntimeError(f'Missing {s}/{d} samples; verify official split alignment.')
    manifest=build_dataset(records,work_dir,copy_images=copy_images)
    print('PREPARED',len(manifest),'images. YAML:',str(Path(work_dir)/'data.yaml'))
    json_dump({'data_yaml':str(Path(work_dir)/'data.yaml'),'source_splits':str(paths['splits']),
               'source_splits_sha256':hashlib.sha256(paths['splits'].read_bytes()).hexdigest(),
               'record_count':len(manifest),'copy_images':copy_images},output_dir/'manifest_summary.json')
    return manifest


def inspect(root,extra_root=None):
    import pandas as pd
    paths=validate_paths(root);splits=load_splits(paths['splits'])
    print('OFFICIAL SPLITS summary:')
    print(json.dumps({key:sample_structure(splits[key]) for key in ('seadronessee','uavdt')},indent=2,default=str)[:8000])
    obj=json.loads(paths['sds'].read_text(encoding='utf-8'))
    coco=find_coco(obj)
    print('\nSDS cleaned schema:',json.dumps(sample_structure(obj),default=str,indent=2)[:4500])
    if coco:
        print('SDS category map:',[(x.get('id'),x.get('name')) for x in _rows(coco['categories'])]);print('SDS first image:',_rows(coco['images'])[:1])
        print('SDS first annotation:',_rows(coco['annotations'])[:1])
    for label,path in [('UAVDT frames',paths['frames']),('UAVDT boxes',paths['boxes'])]:
        frame=pd.read_csv(path,nrows=4)
        print('\n'+label,'columns:',list(frame.columns));print(frame.to_string(index=False)[:2500])
    img,by_name,archive=index_images(paths['raw'],extra_root)
    print('\nRaw/staged image count',len(img),'sample',[str(x) for x in img[:10]])
    print('Archives count',len(archive),'sample',[str(x) for x in archive[:12]])
    print('\nINSPECT ONLY: no file has been changed.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['inspect','prepare'])
    parser.add_argument('--root',required=True)
    parser.add_argument('--work-dir',default='/content/AI251P_M4_Work')
    parser.add_argument('--output-dir',default='/content/AI251P_M4_reports')
    parser.add_argument('--extra-root',default=None,help='Optional local extracted source directory')
    parser.add_argument('--copy-images',action='store_true')
    parser.add_argument('--uavdt-label-mode',choices=['uavdt123','uavdt012','unified678'],default='uavdt123')
    args=parser.parse_args()
    if args.command=='inspect':inspect(args.root,args.extra_root)
    else:prepare(args.root,args.work_dir,args.output_dir,copy_images=args.copy_images,label_mode=args.uavdt_label_mode,extra_root=args.extra_root)


if __name__=='__main__':main()
