#!/usr/bin/env python3
"""Capture privileged canonical initial object poses from the existing THOR build.

Run once when introducing a new scene/build. Existing snapshots are not overwritten.
"""
import argparse
import json
from pathlib import Path
from spatial_interaction.environments.dynamic_memory.backend import create_controller
from spatial_interaction.environments.dynamic_memory.interventions import checked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes',nargs='+',default=['FloorPlan1','FloorPlan2','FloorPlan3','FloorPlan201','FloorPlan202','FloorPlan203'])
    parser.add_argument('--output',default=str(Path(__file__).parent/'spatial_interaction/environments/dynamic_memory/scene_states'))
    parser.add_argument('--runtime-dir',required=True)
    args=parser.parse_args()
    from ai2thor.build import COMMIT_ID
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    c=create_controller(args.scenes[0],128,args.runtime_dir)
    try:
        for scene in args.scenes:
            target=output/f'{scene}.json'
            if target.exists():
                raise FileExistsError(target)
            c.reset(scene)
            checked(c,'SetRandomSeed',seed=42)
            checked(c,'PausePhysicsAutoSim')
            poses=[{'objectName':o['name'],'position':o['position'],'rotation':o['rotation']}
                   for o in c.last_event.metadata['objects'] if o.get('pickupable') or o.get('moveable')]
            # Validate restoration before accepting a snapshot.
            checked(c,'SetObjectPoses',objectPoses=poses,placeStationary=True)
            record={'scene_id':scene,'build_id':COMMIT_ID,'privileged':True,
                    'semantics':'All movable/pickupable object poses restored kinematically before each episode',
                    'object_poses':poses}
            target.write_text(json.dumps(record,indent=2))
            print(scene,len(poses),'canonical poses captured and restoration verified',flush=True)
    finally:
        c.stop()


if __name__=='__main__':
    main()
