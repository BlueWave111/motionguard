"""Event-level matching on fully reviewed streams, separate from window metrics."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from .config import LABELS


def read_events(path):
    rows=[json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    for r in rows:
        for k in ('camera_id','person_id','label','start_time_sec','end_time_sec'):
            if k not in r: raise ValueError(f'Missing {k}')
        if r['label'] not in LABELS or r['end_time_sec']<r['start_time_sec']:
            raise ValueError('Invalid event')
    return rows


def temporal_iou(a,b):
    overlap=max(0,min(a['end_time_sec'],b['end_time_sec'])-max(a['start_time_sec'],b['start_time_sec']))
    union=max(a['end_time_sec'],b['end_time_sec'])-min(a['start_time_sec'],b['start_time_sec'])
    return overlap/union if union>0 else 0.


def event_evaluate(ground_truth,predictions,hours,out,iou_threshold=.3):
    if hours<=0 or not 0<iou_threshold<=1: raise ValueError('Invalid duration or IoU')
    gt=read_events(ground_truth); pred=read_events(predictions); report={}
    for label in LABELS:
        g=[r for r in gt if r['label']==label]; p=[r for r in pred if r['label']==label]
        matrix=np.zeros((len(g),len(p)))
        for i,a in enumerate(g):
            for j,b in enumerate(p):
                if (a['camera_id'],str(a['person_id']))==(b['camera_id'],str(b['person_id'])):
                    overlap=temporal_iou(a,b)
                    if overlap>=iou_threshold: matrix[i,j]=1000+overlap
        matched=[]
        if len(g) and len(p):
            ii,jj=linear_sum_assignment(-matrix)
            matched=[(i,j) for i,j in zip(ii,jj) if matrix[i,j]>0]
        tp=len(matched); fp=len(p)-tp; fn=len(g)-tp
        delays=[p[j]['detected_time_sec']-g[i]['start_time_sec'] for i,j in matched if 'detected_time_sec' in p[j]]
        report[label]={'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(tp+fp,1),
                       'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1),
                       'false_alarms_per_camera_hour':fp/hours,
                       'median_detection_delay_sec':float(np.median(delays)) if delays else None}
    result={'matching':'one-to-one temporal IoU with identical camera and mapped person IDs',
            'iou_threshold':iou_threshold,'fully_reviewed_camera_hours':hours,'metrics':report}
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    Path(out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
