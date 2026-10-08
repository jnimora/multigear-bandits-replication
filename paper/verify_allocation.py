"""Recompute reported statistics from independent replication-level observations."""
from pathlib import Path
import collections,csv,json,math,statistics
import analysis_tools as stats
ROOT=Path(__file__).resolve().parent/'policy_evidence'
def read(name):return list(csv.DictReader((ROOT/name).open()))
raw=read('raw_replications.csv');assert len(raw)==6528
groups=collections.defaultdict(list)
for r in raw:groups[r['group'],r['policy']].append(r)
assert len(groups)==204
for r in groups.values():
    r.sort(key=lambda x:int(x['replication']))
    assert [int(x['replication']) for x in r]==list(range(1,33))
def vals(group,policy,metric):return [float(r[metric]) for r in groups[group,policy]]
checks=0
def same(a,b):
    global checks
    if isinstance(a,dict):
        if isinstance(b,str):b=json.loads(b)
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,list):
        if isinstance(b,str):b=json.loads(b)
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    elif isinstance(a,str):assert a==b
    elif isinstance(a,bool):assert str(a).lower()==str(b).lower()
    else:assert math.isclose(a,float(b),rel_tol=3e-12,abs_tol=3e-12),(a,b)
    checks+=1
for row in read('policy_summary.csv'):
    for key in row:
        if not key.endswith('_mean'):continue
        metric=key[:-5];v=stats.mean_interval(vals(row['group'],row['policy'],metric))
        for field in ('mean','standard_error','lower','upper','zero_sample_variance'):same(v[field],row[metric+'_'+field])
for row in read('scaling.csv'):
    x=vals(row['group'],row['policy'],'residual_gap_per_project')
    b=vals(row['group'],row['policy'],'bound_per_project')[0]
    v=stats.residual_summary(x,b)
    for key in ('mean','standard_error','lower','upper','halfwidth'):same(v[key],row[key])
for row in read('primary_gains.csv'):
    g=row['group'];v=stats.paired_gain(vals(g,'MYOPIC_K','cost_per_project'),vals(g,'WHITTLE_K','cost_per_project'))
    for key,value in v.items():
        if key in row:same(value,row[key])
    for key in ('mean','standard_error','lower','upper'):same(v['absolute_difference'][key],row['absolute_'+key])
for row in read('secondary_48.csv'):
    v=stats.secondary_whittle_contrast(vals(row['group'],row['policy'],'cost_per_project'),vals(row['group'],'WHITTLE_K','cost_per_project'))
    for key in v:
        if key in row:same(v[key],row[key])
for row in read('paired_policies.csv'):
    for metric in ('cost_per_project','residual_gap_per_project','raw_bound_gap_per_project','resource_per_project'):
        x=vals(row['group'],row['policy'],metric);y=vals(row['group'],row['minus_policy'],metric)
        v=stats.mean_interval([a-b for a,b in zip(x,y)])
        for field in ('mean','standard_error','lower','upper','halfwidth'):same(v[field],row[metric+'_'+field])
classes=collections.defaultdict(list)
rawclasses=read('raw_class_replications.csv')
assert len(rawclasses)==13056
for r in rawclasses:classes[r['group'],r['policy'],r['class']].append(r)
for row in read('class_mechanism.csv'):
    cl=classes[row['group'],row['policy'],row['class']]
    assert sorted(int(r['replication']) for r in cl)==list(range(1,33))
    for key in row:
        if key.endswith('_mean'):
            metric=key[:-5];v=stats.mean_interval([float(r[metric]) for r in cl])
            same(v['mean'],row[key]);same(v['standard_error'],row[metric+'_standard_error'])
    amounts=[statistics.fmean(float(r['resource_per_class_project']) for r in classes[row['group'],row['policy'],str(k)]) for k in (1,2)]
    same(amounts[int(row['class'])-1]/sum(amounts),row['used_resource_share_ratio_of_means'])
result=dict(replication_records=len(raw),class_replication_records=len(rawclasses),groups=len(groups),checked_statistics=checks)
(ROOT.parent/'allocation_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
