"""Independent arithmetic/provenance QA for the reporting revision (no timing runs)."""
from pathlib import Path
from fractions import Fraction as F
import collections,json,statistics
import numpy as np
root=Path(__file__).resolve().parent
rows=json.loads((root/'observations.json').read_text());raw=json.loads((root/'raw_scalar_measurements.json').read_text())
fits=json.loads((root/'fits.json').read_text())
lookup={(r['id'],r['method']):r for r in rows}
for r in rows:
 for kind,field,key in [('end_to_end','total','seconds'),('phases','loop','index_loop_seconds'),('phases','init','initialization_seconds')]:
  selected=[v for v in raw if v['case']==r['id'] and v['variant']==r['method'] and v['execution']==kind and v['eligible']][:2]
  assert len(selected)==2 and statistics.median(v[key] for v in selected)==r[field]
  assert all(v['counts']==r['counts'] for v in selected)
def matrix(rr,f):
 values=[]
 for r in rr:
  if f['model']=='kernel_split':v=([r['init_work']/1e9] if f['phase']=='total' else [])+[r['gemm_work']/1e9,r['other_work']/1e9]
  else:v=[(r['work']+(r['init_work'] if f['phase']=='total' else 0))/1e9]+([r['quadratic_work']/1e6] if f['model']=='count_quadratic' else [])
  values.append([1,*v])
 return np.asarray(values)
def qr_prediction(train,test,f):
 X=matrix(train,f);Y=matrix(test,f);y=np.array([r[f['phase']] for r in train])
 sw=np.ones(len(y)) if f['sensitivity']=='ordinary_OLS' else 1/np.array([r['work']+(r['init_work'] if f['phase']=='total' else 0) for r in train])
 sw/=max(sw);Z=X*sw[:,None];scale=np.linalg.norm(Z,axis=0);Q,R=np.linalg.qr(Z/scale)
 b=np.linalg.solve(R,Q.T@(y*sw))/scale
 return Y@b
maximum=0.;nfit=0;nhold=0
for f in fits:
 if not f['identifiable']:continue
 rr=[lookup[cid,f['method']] for cid in f['ids']]
 pred=qr_prediction(rr,rr,f);expected=np.asarray(f['predictions'])
 error=np.max(abs(pred-expected)/np.maximum(1,abs(expected)));maximum=max(maximum,float(error));assert error<2e-7,(f,error);nfit+=1
 axis='A' if f['panel']=='gears' else 'N'
 for level in f['levels']:
  hh=[h for h in f['holdouts'] if h['level']==level and h['identifiable']]
  if not hh:continue
  train=[r for r in rr if r[axis]!=level];test=[lookup[h['id'],f['method']] for h in hh]
  pred=qr_prediction(train,test,f);expected=np.array([h['prediction'] for h in hh]);error=np.max(abs(pred-expected)/np.maximum(1,abs(expected)))
  maximum=max(maximum,float(error));assert error<2e-7,(f['kind'],f['panel'],f['method'],f['model'],error);nhold+=len(hh)
# Exact rational test of the new deferred-update and projection identities.
# Use arbitrary dense nonsymmetric tableaux/reference vectors, both pivot types,
# deletion while corrections are pending, and several flush boundaries.
checks=0
for block in [1,2,4]:
 for trial in range(1,9):
  n=6;X=np.array([[F((trial+2*i+3*j)%7+1,100)+(F(2) if i==j else F(0)) for j in range(n)] for i in range(n)],dtype=object)
  base=X.copy();U=np.empty((n,0),object);V=U.copy();C=np.array([[F(i+j+1,11) for j in range(3)] for i in range(n)],dtype=object);Y=X.T@C
  for step,final in enumerate([False,True,False,False,True,True,False,True,True]):
   m=len(X);j=(step+trial)%m;current=base-U@V.T;assert np.array_equal(current,X)
   v=current[:,j].copy();z=current[j,:].copy() if final else np.array([F((trial+i)%5+1,50) for i in range(m)])@current
   gamma=current[j,j] if final else 1+z[j];new=X-np.outer(v,z)/gamma
   Y=Y-np.outer(z,v@C)/gamma;U=np.column_stack([U,v]);V=np.column_stack([V,z/gamma]);assert np.array_equal(Y,new.T@C)
   if final:
    assert all(x==0 for x in new[j,:]) and all(x==0 for x in new[:,j]);keep=[i for i in range(m) if i!=j]
    base=base[np.ix_(keep,keep)];U=U[keep,:];V=V[keep,:];C=C[keep,:];Y=Y[keep,:];new=new[np.ix_(keep,keep)]
   if U.shape[1]==block:base=base-U@V.T;U=np.empty((len(new),0),object);V=U.copy()
   assert np.array_equal(base-U@V.T,new) and np.array_equal(Y,new.T@C);X=new;checks+=1
result=dict(method_case_medians_verified=len(rows),regression_fits_verified=nfit,holdout_predictions_verified=nhold,maximum_relative_prediction_difference=maximum,exact_rational_blocked_pivots_verified=checks)
(root/'independent_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
