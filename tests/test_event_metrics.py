import json
from motionguard.event_metrics import event_evaluate

def test_event_matching_is_one_to_one(tmp_path):
    gt=[dict(camera_id='cam01',person_id=1,label='fall',start_time_sec=1.,end_time_sec=3.)]
    pred=[dict(**gt[0],detected_time_sec=1.5),dict(**gt[0],detected_time_sec=1.6)]
    a=tmp_path/'gt.jsonl'; b=tmp_path/'p.jsonl'
    a.write_text('\n'.join(map(json.dumps,gt))); b.write_text('\n'.join(map(json.dumps,pred)))
    r=event_evaluate(a,b,2,tmp_path/'report.json')
    assert r['metrics']['fall']['tp']==1 and r['metrics']['fall']['fp']==1
    assert r['metrics']['fall']['false_alarms_per_camera_hour']==.5
