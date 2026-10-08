"""Regenerate the current figures currently included in the main/companion.

Frozen observations and fitted coefficients are read only. Legends use the
actual plotted artists, including their marker fills and dash patterns.
For path-dependent counts, interpolate the count covariate, never runtimes.
"""
from pathlib import Path
import csv, hashlib, json, math, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parent; FIG=ROOT.parent/'figures'
receipt={'inputs':{},'figures':{}}
def load(name):
 p=ROOT/name;receipt['inputs'][name]=hashlib.sha256(p.read_bytes()).hexdigest()
 return json.loads(p.read_text()) if p.suffix=='.json' else list(csv.DictReader(p.open()))
ray=load('rayleigh_instances.json')
bs=load('block_sensitivity.json')
load('blocked_binary_evidence/binary_instances.json')
load('blocked_binary_evidence/binary_count_models.json')
primary=load('policy_evidence/primary_gains.csv');policy=load('policy_evidence/policy_summary.csv');scaling=load('policy_evidence/scaling.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
 'axes.titlesize':8.5,'legend.fontsize':7.4,'axes.spines.top':False,'axes.spines.right':False,
 'pdf.fonttype':42,'ps.fonttype':42,'legend.handlelength':3.4,'legend.numpoints':1,
 'savefig.dpi':220,'axes.axisbelow':True})
STYLE={'R':('0.48','o',':'),'M':('0.20','s','--'),'FP':('0','^','-'),
 'FP2020_up':('0','o','-'),'FP2020_down':('0.40','s',':'),
 'GGK_cubic':('0.23','^','--'),'GGK_recompute':('0.08','D','-.')}
LABEL={'FP2020_up':'FP2020 up','FP2020_down':'FP2020 down',
 'GGK_cubic':'GGK, no recomputation','GGK_recompute':'GGK, default'}

def pchip(x,y,z):
 """Shape-preserving cubic Hermite interpolation, no extrapolation."""
 x=np.asarray(x,float);y=np.asarray(y,float);z=np.asarray(z,float)
 assert len(x)>=2 and np.all(np.diff(x)>0)
 assert z.min()>=x[0] and z.max()<=x[-1]
 h=np.diff(x);d=np.diff(y)/h;m=np.zeros(len(x))
 if len(x)==2:m[:]=d[0]
 else:
  for k in range(1,len(x)-1):
   if d[k-1]*d[k]>0:
    w1=2*h[k]+h[k-1];w2=h[k]+2*h[k-1]
    m[k]=(w1+w2)/(w1/d[k-1]+w2/d[k])
  for j,h0,h1,d0,d1 in ((0,h[0],h[1],d[0],d[1]),(-1,h[-1],h[-2],d[-1],d[-2])):
   slope=((2*h0+h1)*d0-h0*d1)/(h0+h1)
   if slope*d0<=0:slope=0
   elif d0*d1<0 and abs(slope)>3*abs(d0):slope=3*d0
   m[j]=slope
 k=np.clip(np.searchsorted(x,z,side='right')-1,0,len(x)-2)
 t=(z-x[k])/h[k]
 return (2*t**3-3*t**2+1)*y[k]+(t**3-2*t**2+t)*h[k]*m[k]+(-2*t**3+3*t**2)*y[k+1]+(t**3-t**2)*h[k]*m[k+1]

def count_curve(x,work,intercept,slope,state_axis):
 """Evaluate the original regression on interpolated operation counts.

 Cubic state scaling is preserved by using N^3 as the interpolation coordinate;
 gear sweeps use A. At every observed size the original prediction is exact.
 """
 x=np.asarray(x,float);work=np.asarray(work,float)
 xx=np.unique(np.r_[np.linspace(x.min(),x.max(),321),x])
 transform=(lambda z:z**3) if state_axis else (lambda z:z)
 predicted=intercept+slope*pchip(transform(x),work,transform(xx))
 at_data=intercept+slope*pchip(transform(x),work,transform(x))
 assert np.allclose(at_data,intercept+slope*work,rtol=2e-14,atol=2e-14)
 return xx,predicted

def axes(ax,key):
 ax.set_xlabel('State count N' if key=='N' else 'Positive gear count A')
 ax.xaxis.set_major_locator(MaxNLocator(5,integer=True));ax.yaxis.set_major_locator(MaxNLocator(5))
 ax.grid(axis='y',color='.89',lw=.55)

def point(ax,x,y,method,profile=1):
 color,marker,_=STYLE[method]
 fill='white' if profile==2 or method=='FP2020_down' else color
 # A larger open outline keeps both profiles visible when values coincide.
 return ax.plot(x,y,ls='none',marker=marker,color=color,mfc=fill,mec=color,
                ms=5 if profile==2 else 3.5,mew=.8,zorder=4 if profile==2 else 5)[0]

def curve(ax,x,y,method):
 color,_,ls=STYLE[method]
 return ax.plot(x,y,color=color,ls=ls,lw=1.15,zorder=2)[0]

def legend(fig,entries,ncol=3,y=.995):
 handles,labels=zip(*entries)
 fig.legend(handles,labels,handler_map={tuple:HandlerTuple(ndivide=1)},
            ncol=ncol,loc='upper center',bbox_to_anchor=(.5,y),frameon=False,
            columnspacing=1.5,handlelength=3.6)

def save(fig,name,scope):
 FIG.mkdir(exist_ok=True)
 assert all(a.get_xscale()=='linear' and a.get_yscale()=='linear' for a in fig.axes) if name in ('rayleigh_main','two_dimensional_main','rayleigh_block','binary_main') else True
 counts={'observation_points':0,'curve_points':0}
 for ax in fig.axes:
  for line in ax.lines:
   count=len(line.get_xdata())
   counts['observation_points' if line.get_linestyle()=='None' else 'curve_points']+=count
 receipt['figures'][name]={'interpretation':scope,**counts}
 if not os.environ.get('MANUSCRIPT_FIGURE_ONLY') or os.environ['MANUSCRIPT_FIGURE_ONLY']==name:
  fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
  fig.savefig(FIG/(name+'.png'),bbox_inches='tight')
 plt.close(fig)

PANELS=[('binary','A = 1','N'),('states','A = 4','N'),('gears','N = 2,049','A')]
fig,axs=plt.subplots(1,3,figsize=(7.2,2.9));fig.subplots_adjust(left=.085,right=.99,bottom=.19,top=.72,wspace=.35)
entries={}
for ax,(panel,title,key) in zip(axs,PANELS):
 rr=[r for r in ray if r['method']=='BLOCK' and panel in r['panels']]
 f=next(f for f in bs if f['panel']==panel and f['model']=='bound');co=f['coefficients']
 for profile in (1,):
  rs=sorted([r for r in rr if r['profile']==profile],key=lambda r:r[key])
  obs=ax.plot([r[key] for r in rs],[1000*r['index_loop_seconds'] for r in rs],ls='none',color='.15',marker='D',mfc='white' if profile==2 else '.15',ms=5 if profile==2 else 3.5,mew=.8,zorder=4 if profile==2 else 5)[0]
  entries[f'profile{profile}']=(obs,f'Observed, profile {profile}')
 xx=np.linspace(min(r[key] for r in rr),max(r[key] for r in rr),321)
 H=2048 if key=='A' else xx-1;A=xx if key=='A' else (1 if panel=='binary' else 4)
 fit=ax.plot(xx,1000*(co[0]+co[1]*H*A*A/1e6),color='.15',ls='-',lw=1.2)[0]
 entries['bound']=(fit,r'Fitted $HA^2$ count proxy')
 if panel=='gears':
  co=next(f for f in bs if f['panel']==panel and f['model']=='linear_and_quadratic')['coefficients']
  fit=ax.plot(xx,1000*(co[0]+co[1]*H*A/1e5+co[2]*H*A*A/1e6),color='.5',ls='--',lw=1.2)[0]
  entries['sensitivity']=(fit,r'Fitted $AH+HA^2$ sensitivity')
 ax.set_title(title);axes(ax,key);ax.set_ylim(bottom=0)
axs[0].set_ylabel('BLOCK index-loop time (ms)');legend(fig,[entries[k] for k in ('profile1','bound','sensitivity')],2)
save(fig,'rayleigh_block','Profile 1 markers and exact analytic count-proxy regression curves fitted to both profiles; no data-point joins.')

POL=[('WHITTLE_K','DAI-K','0','o','-'),('WHITTLE_P','DAI-P','.4','s','--'),
 ('LAGRANGIAN_K','LAG-K','.2','^','-.'),('LAGRANGIAN_NI','LAG-NI','.4','v',':'),
 ('LAGRANGIAN_FC','LAG-FC','0','D',(0,(4,1,1,1,1,1))),('MYOPIC_K','MYO-K','.2','X',(0,(6,2)))]
groups={r['group'] for r in primary}
fig,axs=plt.subplots(1,3,figsize=(7.2,3.1),sharey=True);fig.subplots_adjust(left=.09,right=.99,bottom=.20,top=.74,wspace=.22)
entries=[]
for col,(ax,alpha) in enumerate(zip(axs,('1/4','1/2','4/5'))):
 for p,label,c,m,ls in POL:
  rr=sorted([r for r in policy if r['group'] in groups and r['alpha']==alpha and r['policy']==p],key=lambda r:int(r['R']))
  line=ax.plot([int(r['R']) for r in rr],[float(r['cost_per_project_mean'])/float(r['bound_per_project_mean']) for r in rr],color=c,marker=m,ls=ls,mfc='white',mec=c,ms=4.5,lw=1.15)[0]
  if col==0:entries.append((line,label))
 ax.axhline(1,color='.65',ls='-',lw=.65,zorder=0);ax.set_xscale('log',base=4)
 ax.set_xticks([1,4,16,64],labels=['1','4','16','64']);ax.set_title(r'$\alpha='+alpha+'$');ax.set_xlabel('Timescale ratio R');ax.grid(axis='y',color='.9',lw=.5)
axs[0].set_ylabel('Mean cost / bound per project');legend(fig,entries)
save(fig,'all_policy_costs','Unchanged allocation means; actual plotted lines/markers form the legend. Connecting lines are descriptive, not regressions.')

fig,axs=plt.subplots(1,2,figsize=(7.2,3.2));fig.subplots_adjust(left=.09,right=.99,bottom=.19,top=.74,wspace=.25)
entries=[]
for col,(ax,R) in enumerate(zip(axs,(1,16))):
 for p,label,c,m,ls in POL:
  rr=sorted([r for r in scaling if r['R']==str(R) and r['policy']==p],key=lambda r:int(r['L']))
  error=ax.errorbar([int(r['L']) for r in rr],[float(r['mean']) for r in rr],yerr=[float(r['halfwidth']) for r in rr],color=c,marker=m,ls=ls,mfc='white',mec=c,ms=4.5,lw=1.15,capsize=2)
  if col==0:entries.append((error,label))
 ax.set(xscale='log',yscale='log',xlabel='Population L',title=f'R = {R}')
 ax.set_xticks([50,100,200,400,800,1600],labels=['50','100','200','400','800','1600']);ax.tick_params(axis='x',labelsize=7)
 ax.grid(axis='y',color='.9',lw=.5)
axs[0].set_ylabel('Mean residual per project');legend(fig,entries)
save(fig,'population_scaling','Unchanged means and confidence intervals; legend uses actual error-bar artists. Logarithmic axes retained for this nontiming diagnostic.')
