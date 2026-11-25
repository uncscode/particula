"""Convert nominal sampled air and counted assay survivors to rinse-pool equivalents."""
from functools import lru_cache
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
IDS={'ContinentalPolluted':6,'Marine':7,'RuralContinental':8,'ContinentalPristine':9,
     'CombustionDominated':10,'MineralDust':11}

@lru_cache(None)
def loadings():
    f=HERE/'lund-table1.xlsx'
    assert hashlib.sha256(f.read_bytes()).hexdigest()=='87cd1af2b3ae2f0d46e4932cbace489afc753ffa759471ae15e0d7195bfa206d'
    with zipfile.ZipFile(f) as wb:
        sheet=ET.fromstring(wb.read('xl/worksheets/sheet1.xml'))
    cells={c.get('r'):c.findtext('{*}v') for c in sheet.findall('.//{*}c')}
    wells=pd.read_csv(HERE.parent/'well_forecasts.csv.gz')
    rows=[]
    for name,row in IDS.items():
        full=float(cells['E'+str(row)])
        litre=full*.25
        w=wells[wells['sample']==name]
        entry=len(w);survivors=int((w.h>2).sum())
        # A plate starts with 100 one-microlitre wells; each source has five plates.
        rows.append(dict(source=name,sampled_m3=full,litres_per_well=litre,entry=entry,
                         h2=survivors,entry_per_litre=(entry/500)/litre,aged_per_litre=(survivors/500)/litre,
                         observed_pool_events_l=(entry-survivors)/(500*litre),
                         poisson_increment_l=np.log(entry/survivors)/litre))
    return pd.DataFrame(rows).set_index('source')
