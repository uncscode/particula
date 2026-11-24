"""Independent rates reported in the source note for one transition of each sign."""
import json
from molecular import ammonia_capture,E,STANDARD,RT
import numpy as np
rows=[]
for q,target in [(0,(6.8482e-10,4.8120e4)),(-1,(2.0536e-9,6.6304e11)),(1,(2.0529e-9,5.3151e-6))]:
    beta=ammonia_capture(2,1,q)
    gamma=STANDARD*beta*np.exp((E[2,2,q]-E[2,1,q])/RT)
    np.testing.assert_allclose((beta,gamma),target,rtol=5e-5)
    rows.append(dict(charge=q,beta_cm3_s=beta,release_s=gamma,reference=target))
print(json.dumps(rows,indent=2))
