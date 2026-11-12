"""Record actual in-action views and public state without adding agent actions."""
import base64
import json
from pathlib import Path
import subprocess

from cases import ROOT,public_observation


class ProcessRecording:
    def __init__(self,native,interval_ms=500):
        from PyQt5.QtCore import QTimer
        self.native=native;self.folder=native.run/'process';self.folder.mkdir()
        self.samples=[];self.error=None
        self.timer=QTimer();self.timer.setInterval(interval_ms)
        self.timer.timeout.connect(self.capture)
        self.capture();self.timer.start()

    def capture(self):
        try:
            state=self.native._capture()
            snapshot=self.native.harness.observe()
            sample={'time_s':state['sim_time_s'],'action_index':len(self.native.commands),
                    'state':public_observation(self.native.case,state)['state']}
            index=len(self.samples)
            for view,image in enumerate(snapshot['images']):
                if not image:raise RuntimeError('A process recording view is missing')
                (self.folder/f'frame_{index:05d}-{view}.png').write_bytes(base64.b64decode(image.split(',',1)[1]))
            self.samples.append(sample)
            with (self.folder/'samples.jsonl').open('a') as file:file.write(json.dumps(sample)+'\n')
        except Exception as error:
            self.error=str(error);self.timer.stop()
            (self.folder/'capture-error.txt').write_text(self.error)

    def finish(self):
        self.timer.stop()
        if self.error:raise RuntimeError('Process capture failed: '+self.error)
        self.capture()
        if self.error or len(self.samples)<2:raise RuntimeError('Insufficient process recording evidence')
        ffmpeg=ROOT.parent/'conda/envs/spatialagent/bin/ffmpeg'
        streams=[]
        for number,name in enumerate(('Global map','Local map','First-person RGB')):
            lines=[]
            for i,sample in enumerate(self.samples):
                duration=self.samples[i+1]['time_s']-sample['time_s'] if i+1<len(self.samples) else .5
                lines += [f"file 'frame_{i:05d}-{number}.png'",f'duration {max(.01,duration):.6f}']
            lines.append(f"file 'frame_{len(self.samples)-1:05d}-{number}.png'")
            listing=self.folder/f'view-{number}.txt';listing.write_text('\n'.join(lines)+'\n')
            target=self.folder/f'view-{number}.mp4'
            subprocess.run([str(ffmpeg),'-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(listing),
                '-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-fps_mode','vfr','-an','-c:v','libx264','-crf','20','-preset','veryfast',
                '-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)
            streams.append({'name':name,'path':str(target.relative_to(self.native.run))})
        manifest={'schema':'delivery-process/1','kind':'actual rollout samples, not interpolated or replayed',
            'target_interval_ms':self.timer.interval(),'time_offset_s':self.samples[0]['time_s'],
            'sample_count':len(self.samples),'streams':streams,'samples':'samples.jsonl'}
        (self.folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        return manifest
