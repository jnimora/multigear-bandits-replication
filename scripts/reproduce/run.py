"""Portable, explicit per-case reruns. No production campaign runs by default."""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys,tomllib
ROOT=Path(__file__).resolve().parents[2]
def val(x):
    if isinstance(x,bool):return str(x).lower()
    if isinstance(x,str):return json.dumps(x)
    if isinstance(x,(int,float)):return str(x)
    if isinstance(x,list):return '['+', '.join(val(v) for v in x)+']'
    if isinstance(x,dict):return '{'+', '.join(json.dumps(k)+' = '+val(v) for k,v in x.items())+'}'
    raise TypeError(type(x))
def toml(p,d):p.write_text('\n'.join(json.dumps(k)+' = '+val(v) for k,v in d.items())+'\n')
def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('cohort',choices=['binary','multigear','block','allocation'])
    a.add_argument('--case');a.add_argument('--list',action='store_true')
    a.add_argument('--check-input',action='store_true',help='Generate and hash only; no timing run')
    a.add_argument('--julia',default=os.environ.get('JULIA','julia'))
    a.add_argument('--python',default=sys.executable,help='Pinned benchmark Python for GGK')
    a.add_argument('--output',type=Path,default=ROOT/'results/local/reproduction')
    a.add_argument('--keep-input',action='store_true',help='Keep generated binary matrices; normally deleted after a successful run')
    args=a.parse_args()
    if args.cohort in ('binary','multigear'):
        jobs=json.loads((ROOT/f'configs/{args.cohort}_cases.json').read_text());key='case' if args.cohort=='binary' else 'id'
    elif args.cohort=='allocation':
        jobs=tomllib.loads((ROOT/'configs/heterogeneity_confirmation_native.toml').read_text())['jobs'];key='id'
    else:
        jobs=[dict(r['job'],id=f"N{r['job']['N']}_A{r['job']['A']}_p{r['job']['profile']}") for r in json.loads((ROOT/'paper/block_raw_measurements.json').read_text())];key='id'
    if args.list:
        for j in jobs:print(j[key])
        return
    if args.case is None:a.error('Select --case or --list; no automatic full campaign')
    selected=[j for j in jobs if j[key]==args.case]
    if len(selected)!=1:a.error('Unknown or ambiguous case')
    j=selected[0];out=args.output.resolve()/args.cohort/args.case
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',JULIA_NUM_THREADS='1')
    julia=[args.julia,'--startup-file=no','--threads=1',f'--project={ROOT}']
    def call(command):
        # Prevent idle sleep on macOS for exactly the child's lifetime.
        command=(['caffeinate','-i']+command) if sys.platform=='darwin' and shutil.which('caffeinate') else command
        subprocess.run(command,cwd=ROOT,env=env,check=True)
    job=out/'job.toml';toml(job,j)
    if args.check_input:
        if args.cohort not in ('binary','multigear'):a.error('--check-input is available for dense timing inputs')
        call(julia+[str(ROOT/'scripts/reproduce/input_check.jl'),args.cohort,str(job)]);return
    if args.cohort=='multigear':
        call(julia+[str(ROOT/'scripts/blocked_multigear/public_worker.jl'),str(job),str(out/'run')])
        if tomllib.loads((out/'run/result.toml').read_text())['status']!='accepted':raise RuntimeError('Incomplete or rejected multi-gear timing block; inspect retained records')
    elif args.cohort=='block':
        call(julia+[str(ROOT/'scripts/rayleigh_checked/worker.jl'),str(ROOT/'configs/rayleigh_reference_production.toml'),str(job),str(out/'run')])
    elif args.cohort=='allocation':
        call(julia+[str(ROOT/'scripts/reproduce/allocation.jl'),args.case,str(out/'run')])
    else:
        worker=str(ROOT/'scripts/binary_benchmark/blocked_campaign_worker.jl')
        inp=out/'input';generation=out/'generate.toml';toml(generation,dict(j,kind='generate'))
        call(julia+[worker,str(generation),str(inp)])
        actual=tomllib.loads((inp/'input.toml').read_text())['sha256']
        if actual!=j['input_sha256']:raise RuntimeError('Regenerated primitive bytes differ')
        j=dict(j,input=str(inp),kind='validate');toml(job,j)
        validation=out/'validation';call(julia+[worker,str(job),str(validation)])
        if tomllib.loads((validation/'result.toml').read_text())['status']!='eligible':raise RuntimeError('Failed PCL path audit; no accepted timings')
        j.update(kind='measure',validation=str(validation/'result.toml'))
        for block in j['blocks']:
            jj=dict(j,methods=block['methods']);language=block['language']
            jf=out/(language+'.toml');toml(jf,jj)
            if language=='julia':call(julia+[worker,str(jf),str(out/language)])
            else:
                # The unchanged released GGK implementation is vendored with its license.
                env['PYTHONPATH']=str(ROOT/'vendor')+os.pathsep+env.get('PYTHONPATH','')
                # The unchanged GGK worker accepts the same TOML job schema.
                call([args.python,str(ROOT/'scripts/binary_benchmark/worker.py'),str(jf),str(out/language)])
            result=tomllib.loads((out/language/'result.toml').read_text()) if language=='julia' else json.loads((out/language/'result.json').read_text())
            if result['status']!='complete':raise RuntimeError('Ineligible timing observation; retained for diagnosis, not accepted')
        if not args.keep_input:shutil.rmtree(inp)
if __name__=='__main__':main()
