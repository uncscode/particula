"""Extract the 64 published formation energies, verifying the original archive digest."""
import sys
from pathlib import Path
import pandas as pd
here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parent/'charge-evaporation'))
from evaporation_reference import read_energies
e=read_energies(Path(sys.argv[1]))
pd.DataFrame([(*key,g) for key,g in sorted(e.items())],columns=['sulfates','ammonias','charge','formation_kcal_mol']).to_csv(sys.argv[2],index=False)
