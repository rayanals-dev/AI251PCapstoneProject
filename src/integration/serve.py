# AI251P service launcher: builds the pipeline ONLY from a manifest, checks the model files and one image
# before it starts, then serves Rayan's app (app.create_app). If any check fails, the service does not start.
#
#     app = make_app('manifest.json', root='/content/drive/MyDrive/AI251P_Capstone_CrossDomainAerial',
#                    domain='urban', out_dir='results', check_image='frame.jpg')
#
# The manifest gives, per stage: checkpoint path (relative to root, or absolute), expected sha256, thresholds, input size.
# domain picks the detector threshold chosen on validation (P30): 'sea' or 'urban'.
import hashlib, json, time
from pathlib import Path
import numpy as np, cv2


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _path(root, p):
    p = Path(p)
    return p if p.is_absolute() else Path(root) / p


def load_manifest(manifest_path):
    m = json.loads(Path(manifest_path).read_text())
    for stage in ('detector', 'classifier', 'segmentation'):
        if stage not in m['stages']:
            raise ValueError(f'manifest has no {stage} stage')
        if not m['stages'][stage].get('sha256'):
            raise ValueError(f'manifest has no sha256 for {stage}')
    return m


def verify_models(m, root):
    """sha256 of every model file against the manifest. Returns a list of checks; raises if one fails."""
    checks = []
    for stage in ('detector', 'classifier', 'segmentation'):
        s = m['stages'][stage]
        p = _path(root, s['checkpoint'])
        if not p.is_file():
            raise FileNotFoundError(f'{stage}: {p} not found')
        got = sha256(p)
        checks.append({'check': f'{stage} sha256', 'expected': s['sha256'], 'actual': got, 'pass': got == s['sha256']})
    bad = [c['check'] for c in checks if not c['pass']]
    if bad:
        raise RuntimeError(f'model files differ from the manifest: {bad}')
    return checks


def build_pipeline(m, root, domain):
    import pipeline as pl
    d, c, s = m['stages']['detector'], m['stages']['classifier'], m['stages']['segmentation']
    if domain not in d['conf_by_domain']:
        raise ValueError(f"domain must be one of {list(d['conf_by_domain'])}")
    pipe = pl.Pipeline(_path(root, d['checkpoint']), _path(root, c['checkpoint']),
                       det_conf=d['conf_by_domain'][domain], det_iou=d['nms_iou'], det_imgsz=d['imgsz'],
                       seg_weights=_path(root, s['checkpoint']), seg_backend=s['backend'], seg_conf=s['conf'],
                       seg_imgsz=s['imgsz'], seg_match_iou=s['match_iou'])
    names = [pipe.det_names[i] for i in sorted(pipe.det_names)]
    if names != d['classes']:
        raise RuntimeError(f'detector classes {names} differ from the manifest {d["classes"]}')
    return pipe


def startup_check(pipe, image_path):
    """Runs one image; every object must carry class, confidence, box and size. Raises if not."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f'start-up image {image_path} cannot be read')
    t = time.perf_counter()
    r = pipe.analyze(np.ascontiguousarray(bgr[..., ::-1]))
    ms = (time.perf_counter() - t) * 1e3
    bad = [o for o in r['objects'] if not (o.get('class') and 0 <= o.get('confidence', -1) <= 1
                                            and len(o.get('box_xywh', [])) == 4 and o.get('size'))]
    if bad:
        raise RuntimeError(f'start-up image: {len(bad)} objects without class, confidence, box or size')
    return {'check': 'start-up image', 'objects': len(r['objects']),
            'with_mask': sum(o['mask'] is not None for o in r['objects']), 'ms': round(ms, 1), 'pass': True}


def make_app(manifest_path, root, domain, out_dir, check_image, **app_kw):
    import pipeline as pl, app as svc
    m = load_manifest(manifest_path)
    checks = verify_models(m, root)
    pipe = build_pipeline(m, root, domain)
    checks.append(startup_check(pipe, check_image))
    info = {'manifest': str(manifest_path), 'domain': domain,
            'detector_conf': m['stages']['detector']['conf_by_domain'][domain],
            'detector_conf_source': m['stages']['detector'].get('conf_source'),
            'segmentation_conf': m['stages']['segmentation']['conf'],
            'startup_checks': [{k: v for k, v in c.items() if k in ('check', 'pass', 'objects', 'ms')} for c in checks]}
    return svc.create_app(pipe, pl.draw, out_dir, info=info, **app_kw), pipe, checks
