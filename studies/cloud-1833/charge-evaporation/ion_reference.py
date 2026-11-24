"""Extract Donahue Figure 3 AIS series on the Figure 6 or native AIS clock."""
import argparse
import hashlib
import io
import os
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARCHIVE = Path(os.environ.get("CLOUD_INPUTS", HERE.parent / "inputs")) / "donahue2026-figure-data.zip"
DIGEST = "5f6f48d653a989a57b633472908ebbddffad10a014db73032dc9f940912d4c85"
# Figure 2a boundary coordinates on the archive clock, read to 0.03--0.04 h.
STARTS = [-6.50,-5.99,-4.83,-3.58,-2.77,-1.77,-.72,-.14,4.29,5.87,
          7.81,9.82,11.14,11.68,13.03,13.53,13.82,15.22,16.20,16.76,
          19.87,21.27,22.77,24.27,25.02,26.54]
FIELD_ON = {2, 3, 5, 9, 11, 14, 19, 22, 26}
CLEANING = {8, 20}


def series(z, name):
    data = z.read(f"archive/fig{name}.dat").decode()
    parts = re.split(r"^% xy for ", data, flags=re.MULTILINE)[1:]
    return [np.loadtxt(io.StringIO("\n".join(s.splitlines()[1:])), comments="%") for s in parts]


def matrix(z, name):
    lines = z.read(f"archive/fig{name}.dat").decode().splitlines()
    return [np.fromstring(s, sep=" ") for s in lines if s.strip() and not s.startswith("%")]


def nearest(source, x):
    right = np.searchsorted(source, x).clip(1, len(source)-1)
    return np.where(abs(x-source[right-1]) <= abs(x-source[right]), right-1, right)


def extract(z, native):
    time = series(z, "03c")[1][:,0]
    x = time if native else matrix(z,"06a")[0]
    i = nearest(time,x)
    stage = np.searchsorted(STARTS,x,side="right").clip(1,26)
    field = np.where(np.isin(stage,sorted(FIELD_ON)), "on",
                     np.where(np.isin(stage, sorted(CLEANING)), "alternating", "off"))
    out = dict(archive_h=x, stage=stage, field=field, ais_native_h=time[i], ais_native_row=i,
               ais_time_offset_s=(time[i]-x)*3600)
    for sign,small,large,panel in [("negative","03c","03b","03a"),("positive","03f","03e","03d")]:
        smooth,raw=series(z,small)
        cpc,particle=series(z,large)
        for reference in [raw,cpc,particle]:
            np.testing.assert_array_equal(reference[:,0],time)
        # Only nearest native AIS observations are carried forward. Retain raw
        # signed values, including negative numbers from instrumental noise.
        out[f"primary_{sign}_raw_cm3"]=raw[i,1]
        smooth_value=np.interp(x,smooth[:,0],smooth[:,1])
        out[f"primary_{sign}_filtered_cm3"]=smooth_value
        for offset,key in [(200,""),(150,"_offset150")]:
            proxy=np.maximum(smooth_value-offset,0)
            # The exact switching clock during cleaning is unavailable.
            out[f"primary_{sign}{key}_cm3"]=np.where(field=="alternating",np.nan,
                                             np.where(field=="on",0.,proxy))
        out[f"ais_{sign}_ge2p2_cm3"]=particle[i,1]
        if sign=="negative":
            out["cpc_ge2p5mob_cm3"]=cpc[i,1]
        else:
            np.testing.assert_allclose(out["cpc_ge2p5mob_cm3"],cpc[i,1])
        mt,centers,*records=matrix(z,panel)
        np.testing.assert_array_equal(mt,time)
        density=np.asarray(records)
        assert density.shape==(28,804) and np.all(np.isfinite(density)) and density.min()==0
        density=np.where(density>0,10.**density,0.)
        mid=(centers[1:]+centers[:-1])/2
        edges=np.r_[2*centers[0]-mid[0],mid,2*centers[-1]-mid[-1]]
        for low,high,label in [(2.2,2.9,"2p2_to_2p9"),(2.9,5,"2p9_to_5"),
                               (5,9.7,"5_to_9p7"),(9.7,40,"9p7_to_40")]:
            width=np.maximum(0,np.minimum(edges[1:],np.log10(high))-np.maximum(edges[:-1],np.log10(low)))
            out[f"ais_{sign}_{label}_cm3"]=(width@density)[i]
    return pd.DataFrame(out)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,default=ARCHIVE)
    parser.add_argument("--output",type=Path,default=HERE/"ion-reference.csv")
    parser.add_argument("--native",action="store_true",help="write all 804 native AIS sample times")
    args=parser.parse_args()
    with args.archive.open("rb") as f:
        if hashlib.file_digest(f,"sha256").hexdigest()!=DIGEST:
            raise ValueError("Figure archive checksum mismatch")
    with ZipFile(args.archive) as z:
        frame=extract(z,args.native)
    frame.to_csv(args.output,index=False,float_format="%.10g")
