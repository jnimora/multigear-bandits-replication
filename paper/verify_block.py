from pathlib import Path
import json,statistics
import numpy as np
P=Path(__file__).resolve().parent
def linear(X,y,w):
    X=np.array(X,float);y=np.array(y,float);w=np.array(w,float)
    scale=np.maximum(np.max(np.abs(X),axis=0),1e-100);Z=X/scale
    c=np.linalg.lstsq(Z*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]/scale
    pred=X@c
    return c,pred
def sensitivity(data,phase,predictors,levelkey,weight):
    X=[[1]+predictors(r) for r in data];y=[r[phase] for r in data]
    w=np.array([weight(r) for r in data]);w=w/w.max()
    c,p=linear(X,y,w);cv=[];levels=[]
    for level in sorted({r[levelkey] for r in data}):
        tr=[i for i,r in enumerate(data) if r[levelkey]!=level];te=[i for i,r in enumerate(data) if r[levelkey]==level]
        cc,_=linear([X[i] for i in tr],[y[i] for i in tr],w[tr])
        predictions=np.array([X[i] for i in te])@cc
        errors=100*abs(np.array([y[i] for i in te])-predictions)/np.array([y[i] for i in te])
        levels.append({'level':level,'median_APE':float(np.median(errors)),
            'max_APE':float(errors.max()),'negative_predictions':int(sum(predictions<0))})
        cv.extend(errors)
    mean=np.average(y,weights=w)
    return {'coefficients':c.tolist(),'weighted_R2':float(1-np.sum(w*(y-p)**2)/np.sum(w*(y-mean)**2)),
        'median_APE':float(np.median(100*abs(y-p)/y)),'CV_median_APE':float(np.median(cv)),
        'CV_max_APE':float(max(cv)),'holdouts':levels,'n':len(data),
        'negative_operation_coefficients':bool(any(c[1:]<0))}


raw=json.loads((P/'block_raw_measurements.json').read_text())
rows=json.loads((P/'rayleigh_instances.json').read_text())
fits=json.loads((P/'block_sensitivity.json').read_text())
for r in rows:
    record=next(v for v in raw if all(v['job'][k]==r[k] for k in ('N','A','profile','criterion')))
    assert record['status']=='complete' and record['cross_method_validated']
    assert len(record['measurements'])==2
    for phase in ('seconds','initialization_seconds','index_loop_seconds','output_seconds'):
        assert np.isclose(r[phase],statistics.median(x[phase] for x in record['measurements']),rtol=1e-13,atol=1e-15)
for f in fits:
    data=[r for r in rows if f['panel'] in r['panels']]
    predictors=(lambda r:[r['x']]) if f['model']=='bound' else (lambda r:[r['H']*r['A']/1e5,r['H']*r['A']**2/1e6])
    actual=sensitivity(data,'index_loop_seconds',predictors,'A' if f['panel']=='gears' else 'N',lambda r:r['x']**-2)
    for key in ('coefficients','weighted_R2','median_APE','CV_median_APE','CV_max_APE'):
        assert np.allclose(actual[key],f[key],rtol=1e-12,atol=1e-12),(f['panel'],key)
print('Verified 24 BLOCK cases, 48 repetitions, and four fits with holdouts.')
