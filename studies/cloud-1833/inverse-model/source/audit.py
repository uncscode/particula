from fit import *
def main():
    start=time.monotonic()
    original=os.environ['INVERSE_SOLVER'];audit=os.environ["AUDIT_SOLVER"]
    results=[]
    for entry,two in [(0,0),(0,1),(1,0),(1,1)]:
        p=CENTER.copy();s,q=run(96,1,'test',entry,two,p)
        os.environ['INVERSE_SOLVER']=audit
        ss,qq=run(96,1,'test',entry,two,p)
        os.environ['INVERSE_SOLVER']=original
        err=qq[0,0,-1,:3];diff=float(np.max(abs(s-ss)))
        assert max(err)<2e-9 and diff<1e-6,(entry,two,err,diff)
        results.append(dict(entry=entry,two=two,relative_particle_ledger=err[0],relative_cluster_ledger=err[1],relative_acid_ledger=err[2],audit_spectrum_abs_delta=diff))
    (OUT/'budget-audit.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,results=results),indent=2))
if __name__=='__main__':main()
