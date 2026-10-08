"""Review plots from completed fresh timings; numpy and matplotlib required.
Does not edit the manuscript. Ordinary axes, point observations, fitted curves.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import MaxNLocator

METHODS=('FP2020_up','FP2020_down','FP_blocked_down','GGK_cubic','GGK_recompute')
LABELS=('FP up','FP down','FP blocked down','GGK, no recomputation','GGK, default')
STYLES={
    'FP2020_up':('.45','o',':','white'),
    'FP2020_down':('.25','s','--','white'),
    'FP_blocked_down':('0','^','-','0'),
    'GGK_cubic':('.25','D','-.','white'),
    'GGK_recompute':('0','v',(0,(5,2,1,2,1,2)),'0'),
}

def ggk_count(n, recompute):
    planned=int(2*n**.1) if recompute else 0
    frequency=n//max(1,planned);start=0;count=0;rebuilds=0
    while start<n-1:
        size=min(frequency,n-1-start)
        s1=size*(size-1)//2;s2=(size-1)*size*(2*size-1)//6
        count+=(2*n+1-2*start)*s1-s2
        start+=frequency
        if start<n-1:rebuilds+=1
    return count+(8/3)*rebuilds*n**3,(planned,rebuilds)


def work(n,method,phase):
    loop=ggk_count(int(n),method=='GGK_recompute')[0] if method.startswith('GGK') else (n-1)*n*(2*n-1)/3
    return loop+((8/3)*n**3 if phase=='end_to_end' else 0)


def fit_group(rr,method,phase,expanded):
    n=np.array([r['N'] for r in rr],float)
    y=np.array([r[phase] for r in rr],float)
    w=np.array([work(int(z),method,phase)/1e9 for z in n])
    X=np.column_stack([np.ones(len(n)),w]+([n*n/1e6] if expanded else []))
    idx=np.arange(len(n))
    def solve(mask):return np.linalg.lstsq(X[mask]/w[mask,None],y[mask]/w[mask],rcond=None)[0]
    coef=solve(idx);pred=X@coef;err=100*np.abs(pred-y)/y;holds=[];errors=[]
    for level in sorted(set(n)):
        test=idx[n==level];co=solve(idx[n!=level]);pp=X[test]@co;ape=100*np.abs(pp-y[test])/y[test]
        errors.extend(ape)
        holds.append(dict(N=int(level),coefficients=co.tolist(),predictions=pp.tolist(),
                          median_APE=float(np.median(ape)),maximum_APE=float(max(ape)),negative_predictions=int(sum(pp<0))))
    weights=w**-2
    return dict(coefficients=coef.tolist(),n=len(rr),levels=len(set(n)),
                model='leading_count_plus_quadratic' if expanded else 'leading_count',weights='W^-2 scale weights',
                weighted_R2=float(1-sum(weights*(y-pred)**2)/sum(weights*(y-np.average(y,weights=weights))**2)),
                median_APE=float(np.median(err)),maximum_APE=float(max(err)),
                holdout_median_APE=float(np.median(errors)),holdout_maximum_APE=float(max(errors)),
                holdouts=holds,predictions=pred.tolist(),cases=[r['case'] for r in rr])


def predictions(xx,method,phase,coefficients):
    count=np.array([work(int(n),method,phase)/1e9 for n in xx])
    return coefficients[0]+coefficients[1]*count+coefficients[2]*np.asarray(xx,dtype=float)**2/1e6


def main(sessionarg):
    session=Path(sessionarg);analysis=session/'analysis';path=analysis/'binary_instances.json'
    finish=json.loads((session/'finished.json').read_text())
    assert finish['status']=='complete','Only complete campaigns may generate review plots'
    rows=json.loads(path.read_text())
    levels=sorted({r['N'] for r in rows})
    if len(levels)<4:
        print('Smoke grid: skip regression/figure generation');return
    fits=[]
    for family in ('uniform','exponential'):
        for criterion in sorted({r['criterion'] for r in rows}):
            for method in METHODS:
                rr=sorted([r for r in rows if (r['family'],r['criterion'],r['method'])==(family,criterion,method)],key=lambda r:(r['N'],r['draw']))
                for phase in ('end_to_end','index_loop'):
                    for expanded in (False,True):
                        fit=fit_group(rr,method,phase,expanded)
                        fits.append(dict(family=family,criterion=criterion,method=method,phase=phase,**fit))
    result=dict(input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),numpy=np.__version__,matplotlib=matplotlib.__version__,
                scope='Descriptive fixed-block finite-size models. Same model families for all methods, no free exponent. Whole-size holdouts; signed coefficients are not a positive runtime decomposition.',fits=fits)
    (analysis/'binary_count_models.json').write_text(json.dumps(result,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,
                         'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
    dest=session/'figures';dest.mkdir(exist_ok=True)

    def panels(phases,name):
        fig,axs=plt.subplots(len(phases),2,figsize=(7.2,3.0 if len(phases)==1 else 5.7),squeeze=False)
        fig.subplots_adjust(left=.10,right=.99,bottom=.16 if len(phases)==1 else .10,
                            top=.72 if len(phases)==1 else .83,wspace=.27,hspace=.40)
        entries={}
        for row,phase in enumerate(phases):
            for col,family in enumerate(('uniform','exponential')):
                ax=axs[row,col]
                for method in METHODS:
                    rr=[r for r in rows if (r['family'],r['criterion'],r['method'])==(family,'average',method)]
                    fit=next(f for f in fits if (f['family'],f['criterion'],f['method'],f['phase'],f['model'])==(family,'average',method,phase,'leading_count_plus_quadratic'))
                    color,marker,style,fill=STYLES[method]
                    xmin=min(r['N'] for r in rr);xmax=max(r['N'] for r in rr)
                    xx=np.arange(xmin,xmax+1)
                    if method=='GGK_recompute':
                        # Never join different matrix-recomputation schedules.
                        regimes={ggk_count(r['N'],True)[1] for r in rr}
                        segments=[np.array([n for n in xx if ggk_count(int(n),True)[1]==regime]) for regime in sorted(regimes)]
                    else:segments=[xx]
                    for segment in segments:
                        if not len(segment):continue
                        yy=predictions(segment,method,phase,fit['coefficients'])
                        line=ax.plot(segment,yy,color=color,ls=style,lw=1.1,zorder=2)[0]
                    points=ax.plot([r['N'] for r in rr],[r[phase] for r in rr],linestyle='none',color=color,
                                   marker=marker,mfc=fill,mec=color,ms=3.5,mew=.8,zorder=3)[0]
                    entries[method]=(line,points)
                ax.set_title(f'{family.capitalize()} ensemble')
                ax.set_xlabel('State count N');ax.set_ylim(bottom=0)
                ax.yaxis.set_major_locator(MaxNLocator(5));ax.xaxis.set_major_locator(MaxNLocator(5,integer=True))
                ax.grid(axis='y',color='.89',lw=.55)
            axs[row,0].set_ylabel('End-to-end time (s)' if phase=='end_to_end' else 'Index-loop time (s)')
        fig.legend([entries[m] for m in METHODS],LABELS,handler_map={tuple:HandlerTuple(ndivide=1)},
                   loc='upper center',bbox_to_anchor=(.5,.995),ncol=3,frameon=False,handlelength=3.6,columnspacing=1.4)
        fig.savefig(dest/(name+'.png'),bbox_inches='tight')
        fig.savefig(dest/(name+'.svg'),bbox_inches='tight')
        plt.close(fig)

    panels(['end_to_end'],'figure3_end_to_end_review')
    panels(['index_loop'],'figure3_loop_review')
    panels(['end_to_end','index_loop'],'figure3_combined_review')
    (dest/'README.txt').write_text('Review figures only; not installed in the manuscript. Average criterion, ordinary axes, individual-input medians of two primary runs (top) or two separate phase runs (bottom). Lines are count-plus-quadratic regressions, not data-point joins. Both ensembles use the same styles; GGK default schedule regimes are disconnected. Review residuals and whole-size holdouts before publication.\n')
    print('Created separate total/loop review plots and a combined four-panel preview.')


if __name__=='__main__':main(*sys.argv[1:])
