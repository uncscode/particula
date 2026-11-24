"""Count simulated trajectories from the actual retained fit and recovery records."""
import json,os
from pathlib import Path
import pandas as pd
out=Path(os.environ['INV_OUT']) if 'INV_OUT' in os.environ else Path(os.environ['PACKET'])/'inverse-model'
load=lambda name:json.loads((out/name).read_text())
a=pd.read_csv(out/'blind-assignments.csv').query("replicate==0 and window=='both_lamps'")
nonlinear=sum(len(set([r.truth,r.selected,r.runner_up])) for r in a.itertuples())*2
rows={
    'reference_and_directional_gradient_pilot':6,
    'coarse_primary_training_trajectories_with_AD':load('fit-runtime.json')['direct_ad_evaluations'],
    'fine_primary_training_trajectories_with_AD':load('refine-runtime.json')['calls'],
    'acidic_training_trajectories_with_AD':load('acidic-runtime.json')['calls'],
    'restricted_training_trajectories_with_AD':int(pd.read_csv(out/'restriction-fits.csv').nfev.sum()),
    'full_frozen_training_and_transfer_trajectories':load('transfer-runtime.json')['boxes'],
    'frozen_restriction_transfer_trajectories':load('restriction-runtime.json')['fine_replays'],
    'cluster_cell_refinement_trajectories':sum(load(f'cluster-grid-{v}-runtime.json')['boxes'] for v in [6,12,24]),
    'separate_dispatch_fine_trajectories':len(load('batch-validation.json')['results']),
    'normal_and_instrumented_conservation_trajectories':2*len(load('budget-audit.json')['results']),
    'synthetic_template_trajectories_with_AD':2*load('recovery.json')['truth_classes'],
    'synthetic_generator_train_and_transfer_segments':2*load('recovery.json')['independent_synthetic_records'],
    'synthetic_noise_reference':1,
    'synthetic_nonlinear_subset_replays':nonlinear
}
(out/'run-inventory.json').write_text(json.dumps(dict(device=load('pilot.json')['device'],trajectories_in_one_production_replay=sum(rows.values()),components=rows,
    convention='One physical parameter box and time segment per trajectory. Six tangent directions per AD evaluation are additional computational work. Optimizer trial trajectories are regenerated from seeds. Runtime records include concurrent GPU contention.'),indent=2))
