"""Publication figures and tables from the audited frozen multi-gear scalars.

Default location after installation: reference_timing_analysis/ alongside a
blocked_multigear_evidence/ directory. Optional CLI paths support staging.
"""
from pathlib import Path
import argparse, hashlib, json, statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import MaxNLocator

STYLES={'M':('.35','s','--','white'),'FP':('.55','o',':','white'),
        'FP_BLOCKED':('0','^','-','0'),'FP_CACHED':('.18','D','-.','white')}
LABEL={'M':'M','FP':'FP-U','FP_BLOCKED':'FP-B','FP_CACHED':'FP-BC'}
MODEL={'count':'C','count_quadratic':'Q','kernel_split':'K'}
PANELS={'rayleigh':[('states1','A = 1','N'),('states4','A = 4','N'),('gears','N = 2,049','A')],
        'channel':[('states4','A = 4','N'),('states8','A = 8','N'),('gears','N = 1,024','A')]}
def pchip(x,y,z):
    x=np.asarray(x,float);y=np.asarray(y,float);z=np.asarray(z,float)
    assert np.all(np.diff(x)>0) and z.min()>=x[0] and z.max()<=x[-1]
    h=np.diff(x);d=np.diff(y)/h;m=np.zeros(len(x))
    if len(x)==2:m[:]=d[0]
    else:
        for k in range(1,len(x)-1):
            if d[k-1]*d[k]>0:
                w1=2*h[k]+h[k-1];w2=h[k]+2*h[k-1];m[k]=(w1+w2)/(w1/d[k-1]+w2/d[k])
        for j,h0,h1,d0,d1 in ((0,h[0],h[1],d[0],d[1]),(-1,h[-1],h[-2],d[-1],d[-2])):
            s=((2*h0+h1)*d0-h0*d1)/(h0+h1)
            if s*d0<=0:s=0
            elif d0*d1<0 and abs(s)>3*abs(d0):s=3*d0
            m[j]=s
    k=np.clip(np.searchsorted(x,z,side='right')-1,0,len(x)-2);t=(z-x[k])/h[k]
    return (2*t**3-3*t**2+1)*y[k]+(t**3-2*t**2+t)*h[k]*m[k]+(-2*t**3+3*t**2)*y[k+1]+(t**3-t**2)*h[k]*m[k+1]
def features(r,phase,model):
    if model=='kernel_split':return ([r['init_work']/1e9] if phase=='total' else [])+[r['gemm_work']/1e9,r['other_work']/1e9]
    return [(r['work']+(r['init_work'] if phase=='total' else 0))/1e9]+([r['quadratic_work']/1e6] if model=='count_quadratic' else [])
def fmt(x):return f'{x:.3f}' if abs(x)<1 else f'{x:.2f}'
def texwrite(p,lines):p.write_text('\n'.join(lines)+'\n')

def build(evidence,manuscript,binary_evidence=None):
    rows=json.loads((evidence/'observations.json').read_text());fits=json.loads((evidence/'display_fits.json').read_text())
    groups=json.loads((evidence/'summary.json').read_text());audit=json.loads((evidence/'audit.json').read_text())
    figdir=manuscript/'figures';tabdir=manuscript/'tables';figdir.mkdir(exist_ok=True);tabdir.mkdir(exist_ok=True)
    receipt=dict(inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence.glob('*.json') if p.name!='figure_receipt.json'},figures={})
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,
        'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':190})
    def fitfor(kind,panel,method,phase):return next(f for f in fits if (f['kind'],f['panel'],f['method'],f['phase'])==(kind,panel,method,phase))
    def save(fig,name,points,curves,scope):
        assert all(ax.get_xscale()==ax.get_yscale()=='linear' for ax in fig.axes)
        for suffix in ('pdf','png'):fig.savefig(figdir/(name+'.'+suffix),bbox_inches='tight')
        receipt['figures'][name]=dict(points=points,curves=curves,scope=scope,sha256=hashlib.sha256((figdir/(name+'.pdf')).read_bytes()).hexdigest())
        plt.close(fig)
    for kind in ('rayleigh','channel'):
        methods=('M','FP_BLOCKED','FP_CACHED') if kind=='rayleigh' else ('M','FP_BLOCKED')
        fig,axs=plt.subplots(2,3,figsize=(7.2,4.9));fig.subplots_adjust(left=.092,right=.99,bottom=.10,top=.87,wspace=.33,hspace=.49)
        handles={};flaghandle=None;pointcount=0;curvecount=0
        for i,phase in enumerate(('total','loop')):
          for j,(panel,title,axis) in enumerate(PANELS[kind]):
            ax=axs[i,j]
            for method in methods:
                rr=sorted([r for r in rows if r['kind']==kind and r['method']==method and r['criterion']=='average' and r['profile']==1 and panel in r['panels']],key=lambda r:r[axis])
                if not rr:continue
                f=fitfor(kind,panel,method,phase);color,marker,linestyle,fill=STYLES[method]
                # Include observations outside the fitted initialization regime,
                # but never connect a curve across the A=1/A>1 branch change.
                curve_rows=[r for r in rr if not(kind=='rayleigh' and panel=='gears' and phase=='total' and r['A']==1)]
                x=np.array([r[axis] for r in curve_rows],float);xx=np.unique(np.r_[np.linspace(x[0],x[-1],321),x])
                X=np.asarray([features(r,phase,f['model']) for r in curve_rows]);transform=(lambda z:z**3) if axis=='N' else (lambda z:z)
                cov=np.column_stack([pchip(transform(x),X[:,k],transform(xx)) for k in range(X.shape[1])])
                yy=f['coefficients'][0]+cov@np.asarray(f['coefficients'][1:])
                line,=ax.plot(xx,yy,color=color,ls=linestyle,lw=1.1,zorder=2);curvecount+=len(xx)
                for r in curve_rows:
                    at=f['coefficients'][0]+np.dot(f['coefficients'][1:],features(r,phase,f['model']))
                    assert np.isclose(yy[np.where(xx==r[axis])[0][0]],at,rtol=1e-12,atol=1e-12)
                    if r['id'] in f['ids']:assert np.isclose(at,f['predictions'][f['ids'].index(r['id'])],rtol=1e-12,atol=1e-12)
                xobs=[r[axis] for r in rr];y=[r[phase] for r in rr]
                error=np.asarray([[r[phase]-min(r[phase+'_samples']) for r in rr],[max(r[phase+'_samples'])-r[phase] for r in rr]])
                ax.errorbar(xobs,y,yerr=error,fmt='none',ecolor=color,elinewidth=.7,capsize=2,zorder=3)
                markerline,=ax.plot(xobs,y,ls='none',marker=marker,color=color,mfc=fill,mec=color,mew=.85,ms=4,zorder=4)
                handles[method]=(line,markerline);pointcount+=len(rr)
                flagged=[r for r in rr if not r['ordinary']]
                if flagged:flaghandle,=ax.plot([r[axis] for r in flagged],[r[phase] for r in flagged],ls='none',marker='x',color='0',ms=7,mew=.7,zorder=5)
            ax.set_title(title);ax.set_xlabel('State count N' if axis=='N' else 'Positive gear count A')
            ax.set_ylim(bottom=0);ax.yaxis.set_major_locator(MaxNLocator(4));ax.xaxis.set_major_locator(MaxNLocator(4,integer=True));ax.grid(axis='y',color='.89',lw=.55)
            if axis=='A':ax.set_xticks([1,4,8,16] if kind=='rayleigh' else [2,16,32,48,64])
            if j==0:ax.set_ylabel('End-to-end time (s)' if phase=='total' else 'Index-loop time (s)')
        hs=[handles[m] for m in methods];labels=[LABEL[m] for m in methods]
        if flaghandle is not None:hs.append(flaghandle);labels.append('Certified selection')
        fig.legend(hs,labels,handler_map={tuple:HandlerTuple(ndivide=1)},loc='upper center',bbox_to_anchor=(.5,.995),ncol=len(hs),frameon=False,handlelength=3.2,columnspacing=1.3)
        save(fig,'rayleigh_main' if kind=='rayleigh' else 'two_dimensional_main',pointcount,curvecount,
            'Profile 1, average. Points: first two eligible repetitions summarized by median; whiskers: their range. Curves: saved count fits evaluated on interpolated count covariates, not interpolated times. Both profiles fitted. Rayleigh gear total curves start at A=2.')

    fig,axs=plt.subplots(2,3,figsize=(7.2,4.7));fig.subplots_adjust(left=.10,right=.99,bottom=.10,top=.86,wspace=.35,hspace=.52)
    handles={};pointcount=0
    for i,kind in enumerate(('rayleigh','channel')):
      for j,(panel,title,axis) in enumerate(PANELS[kind]):
        ax=axs[i,j]
        for method in ('M','FP','FP_BLOCKED','FP_CACHED'):
            ff=[f for f in fits if (f['kind'],f['panel'],f['method'],f['phase'])==(kind,panel,method,'loop')]
            if not ff:continue
            f=ff[0];cv={v['id']:v for v in f['holdouts']};rr=sorted([r for r in rows if r['kind']==kind and r['method']==method and r['profile']==1 and r['id'] in cv],key=lambda r:r[axis])
            color,marker,style,fill=STYLES[method]
            h,=ax.plot([r[axis] for r in rr],[cv[r['id']]['relative_residual'] for r in rr],ls='none',marker=marker,mfc=fill,mec=color,color=color,ms=4,mew=.8)
            handles[method]=h;pointcount+=len(rr)
        ax.axhline(0,color='.7',lw=.6);ax.set_title(('Rayleigh: ' if i==0 else 'Channel: ')+title)
        ax.set_xlabel('State count N' if axis=='N' else 'Positive gear count A');ax.grid(axis='y',color='.91',lw=.5)
        ax.xaxis.set_major_locator(MaxNLocator(4,integer=True));ax.yaxis.set_major_locator(MaxNLocator(4))
        if j==0:ax.set_ylabel('Holdout residual / time (%)')
    fig.legend([handles[k] for k in handles],[LABEL[k] for k in handles],loc='upper center',bbox_to_anchor=(.5,.995),ncol=4,frameon=False)
    save(fig,'blocked_multigear_diagnostics',pointcount,0,'Whole-size/gear-level loop holdout residuals for the saved display models. Profile 1 shown; both profiles held out together. Points are not joined.')

    for kind in ('rayleigh','channel'):
        methods=('M','FP','FP_BLOCKED','FP_CACHED') if kind=='rayleigh' else ('M','FP','FP_BLOCKED')
        ids=sorted({r['id'] for r in rows if r['kind']==kind},key=lambda cid:next((r['N'],r['A'],r['B'],r['profile'],r['criterion']) for r in rows if r['id']==cid))
        spec='rrrrl'+'r'*len(methods);colcount=5+len(methods)
        title='Rayleigh' if kind=='rayleigh' else 'Observed-channel'
        header=r'$N$ & $A$ & $J$ & Prof. & Crit. & '+ ' & '.join(LABEL[m] for m in methods)+r'\\'
        lines=[r'\begingroup\footnotesize\setlength{\tabcolsep}{3pt}',r'\begin{longtable}{'+spec+'}',
            r'\caption{'+title+r' frozen-input comparison: end-to-end time (index-loop time), seconds. Each entry is a two-execution median; phase times come from separate executions. D/Av: discounted/average. FP-U is the unblocked control.}\label{tab:ec-blocked-'+kind+r'}\\',
            r'\toprule '+header+r'\midrule\endfirsthead',r'\multicolumn{'+str(colcount)+r'}{l}{\emph{Continued from previous page}}\\',r'\toprule '+header+r'\midrule\endhead',r'\midrule\endfoot\bottomrule\endlastfoot']
        for cid in ids:
            rs={r['method']:r for r in rows if r['id']==cid};r=next(iter(rs.values()))
            vals=[str(r['N']),str(r['A']),str(r['B']),str(r['profile']),'Av' if r['criterion']=='average' else 'D']
            vals += [f'{fmt(rs[m]["total"])} ({fmt(rs[m]["loop"])})' if m in rs else '--' for m in methods]
            lines.append(' & '.join(vals)+r' \\')
        if kind=='channel':
            lines.extend([r'\midrule',r'1024 & 64 & 2 & 1 & D & \multicolumn{3}{c}{Previously unresolved; not timed} \\',
                          r'1024 & 64 & 2 & 2 & Av & \multicolumn{3}{c}{Previously unresolved; not timed} \\'])
        lines.extend([r'\end{longtable}\endgroup'])
        texwrite(tabdir/('blocked_'+kind+'_complete.tex'),lines)
    header=r'Cohort & Panel & Method & Total model & Fit / holdout & Loop model & Fit / holdout\\'
    lines=[r'\begingroup\footnotesize\setlength{\tabcolsep}{4pt}',r'\begin{longtable}{lllcrcr}',
        r'\caption{Average-cost count-model diagnostics: median absolute percentage errors in fitting and whole-level holdouts. C: combined count; Q: count plus $An^2$; K: separate kernel counts. Both fixed profiles enter each fit.}\label{tab:ec-blocked-fits}\\',
        r'\toprule '+header+r'\midrule\endfirsthead',r'\toprule '+header+r'\midrule\endhead',r'\midrule\endfoot\bottomrule\endlastfoot']
    for kind in ('rayleigh','channel'):
      for panel,_,_ in PANELS[kind]:
       for m in ('M','FP','FP_BLOCKED','FP_CACHED'):
        ff=[f for f in fits if (f['kind'],f['panel'],f['method'])==(kind,panel,m)]
        if not ff:continue
        total=next(f for f in ff if f['phase']=='total');loop=next(f for f in ff if f['phase']=='loop')
        paneltex={'states1':'$A=1$','states4':'$A=4$','states8':'$A=8$','gears':'Gear sweep'}[panel]
        lines.append(' & '.join(['Rayleigh' if kind=='rayleigh' else 'Channel',paneltex,LABEL[m],MODEL[total['model']],f"{total['MdAPE']:.2f} / {total['CV_MdAPE']:.2f}",MODEL[loop['model']],f"{loop['MdAPE']:.2f} / {loop['CV_MdAPE']:.2f}"])+r' \\')
    lines.extend([r'\end{longtable}\endgroup']);texwrite(tabdir/'blocked_count_fits.tex',lines)

    lines=[r'\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{4pt}',
        r'\caption{Paired average/discounted end-to-end time ratios on identical primitive inputs. Multi-gear entries use the fresh blocked campaign; binary entries retain their separate frozen campaign.}\label{tab:ec-criterion-ratios}',r'\begin{tabular}{llrrrr}',
        r'\toprule Cohort & Method & Pairs & Minimum & Median & Maximum\\\midrule']
    for kind in ('rayleigh','channel'):
      for m in ('M','FP','FP_BLOCKED','FP_CACHED'):
        if kind+':'+m not in groups:continue
        g=groups[kind+':'+m]['average_over_discounted']['total']
        lines.append(' & '.join(['Rayleigh' if kind=='rayleigh' else 'Channel',LABEL[m],str(g['n']),*[f'{g[k]:.4f}' for k in ('minimum','median','maximum')]])+r' \\')
    if binary_evidence:
        bb=json.loads((binary_evidence/'binary_instances.json').read_text());lookup={(r['family'],r['N'],r['draw'],r['criterion'],r['method']):r for r in bb}
        names={'FP_blocked_down':'Blocked FP','FP2020_up':'FP up','FP2020_down':'FP down','GGK_cubic':'GGK no-rec.','GGK_recompute':'GGK default'}
        for m,label in names.items():
            ratios=[lookup[r['family'],r['N'],r['draw'],'average',m]['end_to_end']/r['end_to_end'] for r in bb if r['method']==m and r['criterion']=='discounted']
            lines.append(' & '.join(['Binary',label,str(len(ratios)),f'{min(ratios):.4f}',f'{st.median(ratios):.4f}',f'{max(ratios):.4f}'])+r' \\')
    lines.extend([r'\bottomrule\end{tabular}\end{table}']);texwrite(tabdir/'blocked_criterion_ratios.tex',lines)
    (evidence/'figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,default=here/'blocked_multigear_evidence');p.add_argument('--manuscript',type=Path,default=here.parent);p.add_argument('--binary-evidence',type=Path,default=here/'blocked_binary_evidence')
    a=p.parse_args();build(a.evidence,a.manuscript,a.binary_evidence if a.binary_evidence.exists() else None)
