"""Join dark and illuminated predictions and regenerate the study comparison figure."""
from pathlib import Path
import sys
import pandas as pd

root=Path(sys.argv[1])
frozen=pd.concat([pd.read_csv(root/f).assign(score=kind) for f,kind in
                  [('frozen.csv','absolute'),('shape-frozen.csv','temporal')]])
transfer=pd.read_csv(root/'transfer.csv').rename(columns={'cells':'simulation_cells'})
cols=['eddy','dilution','low_capture']
linked=frozen.merge(transfer,on=cols,how='left',validate='many_to_many')
assert len(linked)==len(frozen)*2 and linked.simulation_cells.notna().all()
linked.to_csv(root/'04-held-transfer.csv',index=False)
score=pd.read_csv(root/'03-growth-confusion.csv')
base=score.query('truth_id<6 and channels=="joint" and flex_growth==False')
avg=base.groupby(['noise','precision','eddy'],as_index=False).correct_wall.mean()
avg.to_csv(root/'classification-summary.csv',index=False)
def select_curves(held, series):
    curves = []
    for kind in ("upper", "joint"):
        chosen = held.query('held == 0 and score == "absolute" and calibration_cells == 288 and simulation_cells == 288 and channels == @kind and flex_growth and precision == 0')
        [fit] = chosen.itertuples(index=False)
        window = series[series.archive_h.between(9.6,11.15)]
        selection = (window.cells == 288) & (window.eddy == fit.eddy) & (window.dilution == fit.dilution) & (window.low_capture == fit.low_capture)
        curves.append(window.loc[selection, ["archive_h","observed","predicted"]].assign(channels=kind))
    return pd.concat(curves)
select_curves(linked,pd.read_csv(root/"transfer-series.csv.gz")).to_csv(root/"plot-series.csv",index=False)
from plot import plot
plot(root, root/"recovery-and-transfer.png")
