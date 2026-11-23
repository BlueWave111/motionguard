import json
import numpy as np
import pytest
from motionguard.config import load_config
from motionguard.features import FeatureExtractor
from motionguard.data import read_manifest,annotate,load_record
from motionguard.events import EventManager


def pose():
    k=np.ones((17,3),dtype=float); k[:,0]=np.arange(17)*3+40; k[:,1]=np.arange(17)*8+20; k[:,2]=.9
    return k


def test_missing_pose_and_reappearance_do_not_make_velocity():
    cfg=load_config(); fe=FeatureExtractor(cfg); box=[0,0,300,400]
    fe.step(pose(),box,0,640,480)
    missing=pose(); missing[:,2]=0
    f,q,ok=fe.step(missing,box,.1,640,480)
    assert np.isfinite(f).all() and not ok and q==0
    # 34 rel + 34 global + 17 confidence + 17 valid + 17 age = velocity starts 119
    assert np.allclose(f[119:153],0)
    moved=pose(); moved[:,:2]+=100
    f,_,_=fe.step(moved,box,.2,640,480)
    assert np.allclose(f[119:153],0)
    with pytest.raises(ValueError): fe.step(moved,box,.2,640,480)


def test_multilabel_hysteresis_and_unknown():
    cfg=load_config(); cfg['confirm_sec']=[.1]*7; cfg['release_sec']=.1
    m=EventManager(cfg); p=np.array([.9,0,0,0,.8,0,0])
    m.update(7,0,p); m.update(7,.1,p)
    assert len(m.active(7))==2
    assert m.update(7,.2,np.zeros(7))==[]
    closed=m.update(7,.3,np.zeros(7))
    assert len(closed)==2 and closed[0]['start_time_sec']==0
    assert closed[0]['detected_time_sec']==.1
    m.update(8,1,p); m.update(8,1.1,p)
    closed=m.update(8,1.2,p,False)
    assert len(closed)==2 and not m.active(8)


def test_split_leakage(tmp_path):
    rows=[{'path':'a.npz','split':'train','subject_id':'same','source_id':'a'},
          {'path':'b.npz','split':'test','subject_id':'same','source_id':'b'}]
    p=tmp_path/'manifest.jsonl'; p.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError,match='leakage'): read_manifest(p)


def test_unknown_labels_not_negatives(tmp_path):
    p=tmp_path/'track.npz'; np.savez(p,features=np.zeros((3,4)),timestamps=np.array([0.,1.,2.]),quality=np.ones(3,bool))
    a=tmp_path/'annotation.json'; a.write_text(json.dumps({'reviewed_labels':['fall'],'events':[{'label':'fall','start':1,'end':2}]}))
    out=tmp_path/'labeled.npz'; annotate(p,a,out); r=load_record(out)
    assert np.array_equal(r['labels'][:,0],[0,1,0])
    assert (r['labels'][:,1:]==-1).all()


def test_windows_reset_after_bad_observation(tmp_path):
    from motionguard.data import windows
    cfg=load_config(); cfg.update(window=8,min_history=3,stride=1)
    path=tmp_path/'r.npz'
    quality=np.ones(12,bool); quality[5]=False
    np.savez(path,features=np.zeros((12,4)),timestamps=np.arange(12)/15,
             quality=quality,labels=np.zeros((12,7)))
    ws=windows([{'path':str(path)}],cfg)
    assert all(not (a<=5<=b) for r,a,b,e in ws)
    assert any(a==6 and b==8 for r,a,b,e in ws)


def test_extraction_config_mismatch_rejected(tmp_path):
    from motionguard.data import windows
    from motionguard.config import FEATURE_CONFIG_KEYS
    cfg=load_config(); stored={k:cfg[k] for k in FEATURE_CONFIG_KEYS}; stored['sample_hz']=30.
    path=tmp_path/'r.npz'
    np.savez(path,features=np.zeros((20,4)),timestamps=np.arange(20)/15,
        quality=np.ones(20,bool),labels=np.zeros((20,7)),feature_config=np.array(json.dumps(stored)))
    with pytest.raises(ValueError,match='config mismatch'): windows([{'path':str(path)}],cfg)
