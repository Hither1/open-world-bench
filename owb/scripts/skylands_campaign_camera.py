"""Validate camera visibility and export camera-only corrections from checked replays."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from .skylands_campaign import replay
from .skylands_reference import ROOT,checked_step,save,sha
from owb.render.skylands_campaign import safe_camera,camera_obscures_traveler


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--task',choices=('a','b'),required=True);args=parser.parse_args()
    task=args.task;cache=ROOT/'.cache/skylands-campaign';source=cache/f'video-{task}'
    full=cache/f'video-camera-{task}';patch=cache/f'camera-only-{task}'
    for directory in (full,patch):
        directory.mkdir(exist_ok=False);(directory/'states').mkdir()
        for name in ('static.npz','ground-cover.npz','grass-prototypes.npz','geometry.json'):
            os.link(source/name,directory/name)
    data=json.loads((source/'frames.json').read_text());entries={e['step']:e for e in data['entries']};changed=[]
    env,manifest,rows=replay(ROOT/f'assets/demos/skylands-campaign/task-{task}-seed-7')
    for row in rows:
        checked_step(env,row)
        if row['step'] not in entries:continue
        entry=entries[row['step']];oldpath=source/entry['scene'];newpath=full/entry['scene']
        assert sha(oldpath)==entry['scene_sha256']
        with np.load(oldpath,allow_pickle=False) as archive:
            values={key:archive[key] for key in archive.files}
        meta=json.loads(str(values['metadata']));oldcamera=meta['camera']
        if camera_obscures_traveler(env,oldcamera):
            before=env.state_digest();meta['camera']=safe_camera(env)
            assert not camera_obscures_traveler(env,meta['camera']),('No clear camera',entry['index'])
            assert before==env.state_digest()==entry['digest']
            values['metadata']=np.array(json.dumps(meta));np.savez_compressed(newpath,**values)
            with np.load(oldpath,allow_pickle=False) as old,np.load(newpath,allow_pickle=False) as new:
                assert old.files==new.files
                for key in old.files:
                    if key!='metadata':assert np.array_equal(old[key],new[key]),key
                oldmeta=json.loads(str(old['metadata']));newmeta=json.loads(str(new['metadata']))
                oldmeta.pop('camera');newmeta.pop('camera');assert oldmeta==newmeta
            entry['original_scene_sha256']=entry['scene_sha256'];entry['scene_sha256']=sha(newpath)
            entry['camera_correction']=dict(reason='Original camera obscured or cropped the traveler',original=oldcamera,corrected=meta['camera'])
            changed.append(entry);os.link(newpath,patch/entry['scene'])
            print('CAMERA CORRECTED',task,entry['index'],entry['milestone'],flush=True)
        else:os.link(oldpath,newpath)
    assert env.success and env.state_digest()==manifest['final_digest']
    source_hash=sha(Path(__file__));data['camera_review']=dict(checked_states=len(entries),corrected_states=len(changed),
        actions_replayed=len(rows),geometry_unchanged=True,lighting_unchanged=True,simulation_unchanged=True,source_sha256=source_hash)
    data['sources']['owb/scripts/skylands_campaign_camera.py']=source_hash
    data['sources']['owb/render/skylands_campaign.py']=sha(ROOT/'owb/render/skylands_campaign.py')
    save(full/'frames.json',data);save(patch/'frames.json',{**data,'entries':changed})
    save(ROOT/f'assets/demos/skylands-campaign/task-{task}-seed-7/camera-verification.json',data['camera_review'])
    print('CAMERA REVIEW COMPLETE',task,len(changed),'/',len(entries),flush=True);env.close()


if __name__=='__main__':main()
