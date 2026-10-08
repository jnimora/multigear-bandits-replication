"""Unmodified authors' kernel for primary timings; a separately labelled copy
with two clock reads supplies phase diagnostics. Never call the cached model API.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import ast, gc, hashlib, inspect, json, platform, resource, sys, time, tomllib
from pathlib import Path
import numpy as np
import scipy
import numba
from threadpoolctl import threadpool_info, threadpool_limits
from markovianbandit import whittle_computation as ggk

def emit(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=True));temp.replace(path)

def phase_function():
    source=inspect.getsource(ggk.compute_whittle_indices)
    tree=ast.parse(source); fun=tree.body[0]; fun.name='phase_compute'
    loop=next(i for i,node in enumerate(fun.body) if isinstance(node,ast.For))
    fun.body.insert(loop,ast.parse('_phase_marks.append(_phase_clock())').body[0])
    # The final return only; early failures are invalid phase observations.
    fun.body.insert(len(fun.body)-1,ast.parse('_phase_marks.append(_phase_clock())').body[0])
    ns=dict(ggk.__dict__);ns['_phase_clock']=time.perf_counter;ns['_phase_marks']=[]
    exec(compile(ast.fix_missing_locations(tree),'<phase-only instrumentation>','exec'),ns)
    return ns['phase_compute'],ns['_phase_marks']

def agrees(a,b):
    return np.isfinite(a).all() and np.max(np.abs(a-b)/(1+np.abs(b)))<=1e-7

def main(jobpath,outdir):
    job=tomllib.loads(Path(jobpath).read_text());outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    n=job['N']; root=Path(job['input']); meta=tomllib.loads((root/'input.toml').read_text())
    arrays=[]
    for key in ('P0','P1','R0','R1'):
        path=root/(key+'.f64')
        assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==meta['sha256'][path.name]
        a=np.fromfile(path,dtype='<f8')
        # Natural row-major input layout for the authors' C-order kernel;
        # conversion is input preparation, outside all algorithm times.
        if key.startswith('P'):a=np.ascontiguousarray(a.reshape((n,n),order='F'))
        arrays.append(a)
    reference=np.asarray(tomllib.loads(Path(job['validation']).read_text())['reference_indices'])
    beta=job['beta'];phase,marks=phase_function()
    methods=job['methods'];rows=[];outputs={}
    info={'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'numba':numba.__version__,
          'markovianbandit':'0.4','platform':platform.platform(),'threadpools':threadpool_info(),
          'source_sha256':hashlib.sha256(Path(ggk.__file__).read_bytes()).hexdigest()}
    expected=os.environ.get('BINARY_BENCH_GGK_SHA256')
    if expected:assert info['source_sha256']==expected,'Released source changed after snapshot'
    data={'status':'running','job':job,'environment':info,'measurements':rows,'outputs':outputs}
    def kwargs(method,check=False):
        return dict(beta=beta,check_indexability=check,number_of_updates=0 if method=='GGK_cubic' else '2n**0.1',atol=1e-12)
    tiny=(np.full((8,8),.125),np.full((8,8),.125),np.zeros(8),np.arange(8.)/8)
    start=time.perf_counter()
    with threadpool_limits(limits=1):
        for method in methods:
            for _ in range(2):
                ggk.compute_whittle_indices(*tiny,**kwargs(method))
                ggk.compute_whittle_indices(*tiny,**kwargs(method,True))
                marks.clear();phase(*tiny,**kwargs(method))
        assert ggk.update_W.nopython_signatures,'Numba did not compile the numerical update'
        info['threadpools']=threadpool_info()
        assert all(p['num_threads']==1 for p in info['threadpools'])
        for method in methods:
            emit(outdir/'progress.json',{'stage':'indexability_validation','method':method})
            gc.collect(); t0=time.perf_counter()
            flag,index=ggk.compute_whittle_indices(*arrays,**kwargs(method,True))
            outputs[method]={'indexability_flag':int(flag),'validation_seconds':time.perf_counter()-t0,
                'validated_indices':index.tolist(),'agrees_with_fp':bool(agrees(index,reference))}
            emit(outdir/'result.json',data)
        for rep in range(1,job['repetitions']+1):
            for method in (methods if rep%2 else methods[::-1]):
                emit(outdir/'progress.json',{'stage':'timing','method':method,'repetition':rep})
                gc.collect();c0=time.process_time();t0=time.perf_counter()
                flag,index=ggk.compute_whittle_indices(*arrays,**kwargs(method))
                elapsed=time.perf_counter()-t0;cpu=time.process_time()-c0
                valid=agrees(index,reference) and outputs[method]['agrees_with_fp'] and outputs[method]['indexability_flag']>0
                rows.append({'method':method,'phase':'end_to_end','repetition':rep,'seconds':elapsed,
                    'cpu_seconds':cpu,'valid':bool(valid),'indexability_check':False})
                outputs[method]['indices']=index.tolist();emit(outdir/'result.json',data)
        for rep in range(1,job.get('phase_repetitions',1)+1):
            for method in (methods if rep%2 else methods[::-1]):
                emit(outdir/'progress.json',{'stage':'phase_diagnostic','method':method,'repetition':rep})
                gc.collect();marks.clear();t0=time.perf_counter()
                flag,index=phase(*arrays,**kwargs(method));t1=time.perf_counter()
                valid=len(marks)==2 and agrees(index,reference)
                if len(marks)==2:
                    for name,seconds in [('initialization',marks[0]-t0),('index_loop',marks[1]-marks[0]),('output',t1-marks[1])]:
                        rows.append({'method':method,'phase':name,'repetition':rep,'seconds':seconds,'valid':bool(valid),
                            'scope':'separate clock-instrumented diagnostic; original function used for primary timings'})
                else:rows.append({'method':method,'phase':'phase_failure','repetition':rep,'seconds':t1-t0,'valid':False})
                emit(outdir/'result.json',data)
    data['status']='complete' if all(r['valid'] for r in rows) else 'measurement_failed'
    data['elapsed_seconds']=time.perf_counter()-start;data['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    data['numba_nopython_signatures']=[str(s) for s in ggk.update_W.nopython_signatures]
    emit(outdir/'result.json',data)

if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception:
        import traceback
        emit(Path(sys.argv[2])/'error.json',{'error':traceback.format_exc()});raise
