"""Record the actual exit state of one native task validation allocation."""
import json
from pathlib import Path
import sys
from datetime import datetime,timezone
p=Path(sys.argv[1])/'run_metadata.json';d=json.loads(p.read_text());code=int(sys.argv[2])
d['slurm'].update(state='COMPLETED' if code==0 else 'FAILED',exit_code=code,
    ended_at=datetime.now(timezone.utc).isoformat(),failure_reason='None' if code==0 else 'See failure.log and scheduler output')
p.write_text(json.dumps(d,indent=2)+'\n')
