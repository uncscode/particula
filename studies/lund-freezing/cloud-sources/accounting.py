"""Retain campaign clock, completed setting counts and an exact replay matrix."""
import json
import gzip
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import numpy as np
from campaign import Experiment,key,design
from challenge import sequence
from itertools import chain

folder=Path(sys.argv[1])
f=folder/'runs.jsonl'
opener=open
if not f.exists():f=folder/'runs.jsonl.gz';opener=gzip.open
with opener(f,'rt') as handle:
    rows=[json.loads(line) for line in handle if line.endswith('\n')]
matrix=[Experiment(**r['experiment']) for r in rows]
seen=set()
unique=[]
for e in matrix:
    if key(e) not in seen:
        unique.append(asdict(e));seen.add(key(e))
with gzip.GzipFile(filename=str(folder/'executed-cases.json.gz'),mode='wb',mtime=0) as destination:
    destination.write(json.dumps(unique,separators=(',',':')).encode())
planned={}
for e in chain(sequence(),design(include_extra_replicates=False)):
    planned.setdefault(key(e),e)
remaining=[asdict(e) for k,e in planned.items() if k not in seen]
with gzip.GzipFile(filename=str(folder/'remaining-cases.json.gz'),mode='wb',mtime=0) as destination:
    destination.write(json.dumps(remaining,separators=(',',':')).encode())
es=[r['integrator'] for r in rows]
refcases=Path(__file__).parent/'replay-check.txt'
counts=dict(planned_finite_settings=len(planned),remaining_finite_settings=len(remaining),integrations=len(rows),unique_cases=len(seen),six_hour_runs=sum(e.hours==6 for e in matrix),
            clouds=sorted({e.sky for e in matrix}),n_clouds=len({e.sky for e in matrix}),
            grids=dict(Counter(e.grid for e in matrix)),samples=dict(Counter(e.abundance for e in matrix)),
            mineral_runs=sum(e.donor.startswith('grain_') for e in matrix),simulated_hours=sum(e.hours for e in matrix),
            models=sorted({r['GPU'] for r in es}),records_total=sum(r['records'] for r in es),
            records_peak=max(r['records'] for r in es),integrator_steps=sum(r['steps'] for r in es),
            integrator_seconds=sum(r['seconds'] for r in es),
            case_seconds=sum(z['wall'] for z in rows),max_water_error=max(r['max_water_error'] for r in es),
            maximum_grid=max(e.grid for e in matrix))
if refcases.is_file():
    counts['verification']=refcases.read_text().strip()
if (folder/'finished.json').is_file():
    counts.update(json.loads((folder/'finished.json').read_text()))
counts['budget']=json.loads((folder/'budget.json').read_text())
(folder/'accounting.json').write_text(json.dumps(counts,indent=2)+'\n')
print(counts['integrations'],counts['unique_cases'],counts['grids'],round(counts['integrator_seconds']),counts['max_water_error'])
