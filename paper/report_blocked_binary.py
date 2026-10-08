"""Reproduce reviewed binary figures, tables and diagnostics from frozen scalars.

No campaign is run and no raw timing is altered. Old binary summaries are
historical only. Run this before plot_manuscript_figures.py after legacy tools.
"""
from pathlib import Path
import collections
import hashlib
import importlib.util
import json
import statistics as stat
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'blocked_binary_evidence'
spec = importlib.util.spec_from_file_location('campaign_counts', EVIDENCE / 'campaign_count_models.py')
counts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(counts)
METHODS = ('FP_blocked_down', 'FP2020_up', 'FP2020_down', 'GGK_cubic', 'GGK_recompute')
LABELS = dict(zip(METHODS, ('Blocked FP down', 'Unblocked FP up', 'Unblocked FP down', 'GGK, no recomputation', 'GGK, default')))
SHORT = dict(zip(METHODS, ('Blocked FP', 'FP up', 'FP down', 'GGK no-rec.', 'GGK default')))
STYLES = {
    'FP_blocked_down': ('0', '^', '-', '0'),
    'FP2020_up': ('.48', 'o', ':', 'white'),
    'FP2020_down': ('.25', 's', '--', 'white'),
    'GGK_cubic': ('.42', 'D', '--', 'white'),
    'GGK_recompute': ('.12', 'v', '-.', '.12'),
}
rows = json.loads((EVIDENCE / 'binary_instances.json').read_text())
fits = json.loads((EVIDENCE / 'binary_count_models.json').read_text())['fits']

def chosen(method, phase):
    # Post-campaign descriptive choice: the extra term worsens blocked-loop
    # whole-size holdouts in all four ensemble/criterion strata.
    return 'leading_count' if (method, phase) == ('FP_blocked_down', 'index_loop') else 'leading_count_plus_quadratic'

def getfit(method, phase, family, criterion='average', model=None):
    return next(f for f in fits if (f['method'], f['phase'], f['family'], f['criterion'], f['model']) ==
                (method, phase, family, criterion, model or chosen(method, phase)))

def validate():
    raw = json.loads((EVIDENCE / 'measurements.json').read_text())
    assert len(rows) == 400 and len(raw) == 3200 and all(r['valid'] for r in raw)
    grouped = collections.defaultdict(list)
    for r in raw:
        grouped[r['case'], r['method'], r['phase']].append(r['seconds'])
        if r['method'].startswith('FP') and r['phase'] == 'end_to_end':
            assert r['order_matches_audited_path'] and r['compile_seconds'] == 0
    for r in rows:
        for phase in ('end_to_end', 'initialization', 'index_loop', 'output'):
            sample = grouped[r['case'], r['method'], phase]
            assert len(sample) == 2
            assert np.isclose(r[phase], stat.median(sample), rtol=1e-14, atol=0)
    bycase = {(r['case'], r['method']): r for r in rows}
    residuals = []
    for f in fits:
        rr = [bycase[c, f['method']] for c in f['cases']]
        actual = counts.fit_group(rr, f['method'], f['phase'], f['model'] == 'leading_count_plus_quadratic')
        for field in ('coefficients', 'predictions', 'weighted_R2', 'median_APE', 'maximum_APE', 'holdout_median_APE', 'holdout_maximum_APE'):
            assert np.allclose(actual[field], f[field], rtol=1e-12, atol=1e-12), (f['method'], field)
        for r, prediction in zip(rr, f['predictions']):
            residuals.append(dict(case=r['case'], N=r['N'], method=f['method'], phase=f['phase'],
                                  family=f['family'], criterion=f['criterion'], model=f['model'],
                                  observed=r[f['phase']], prediction=prediction,
                                  residual=r[f['phase']]-prediction,
                                  residual_pct=100*(r[f['phase']]-prediction)/r[f['phase']]))
    (ROOT / 'blocked_binary_residuals.json').write_text(json.dumps(residuals, indent=2)+'\n')
    primary = [r for r in raw if r['phase'] == 'end_to_end']
    summary = dict(verified_primary=len(primary), verified_phase_records=len(raw)-len(primary),
                   verified_fits=len(fits), verified_method_cases=len(rows),
                   maximum_FP_relative_index_error=max(r['maximum_relative_index_error'] for r in primary if r['method'].startswith('FP')),
                   selection='Leading model for blocked loop; count plus quadratic otherwise. Both families and all size holdouts retained. Post-campaign descriptive choice.',
                   source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in EVIDENCE.glob('*.json')},
                   timing_spread={phase: dict(median=stat.median(r[phase+'_range_pct'] for r in rows),
                                             maximum=max(r[phase+'_range_pct'] for r in rows)) for phase in ('end_to_end', 'index_loop')})
    (ROOT / 'blocked_binary_review.json').write_text(json.dumps(summary, indent=2)+'\n')

def plot_panels(methods, name, phases=('end_to_end', 'index_loop')):
    fig, axs = plt.subplots(len(phases), 2, figsize=(7.2, 5.25 if len(phases) == 2 else 3.0), squeeze=False)
    fig.subplots_adjust(left=.095, right=.99, bottom=.10 if len(phases)==2 else .17,
                        top=.875 if len(phases)==2 else .78, wspace=.26, hspace=.42)
    entries = {}
    for row, phase in enumerate(phases):
        for col, family in enumerate(('uniform', 'exponential')):
            ax = axs[row, col]
            for method in methods:
                rr = [r for r in rows if (r['method'],r['criterion'],r['family']) == (method,'average',family)]
                f = getfit(method,phase,family)
                color, marker, style, fill = STYLES[method]
                # Conditional display within each observed planned/rebuild
                # regime. Interpolate between compatible integer sizes only;
                # separate regimes never share a line. Omitting an occasional
                # incompatible integer is a display interpolation, not a claim
                # about the implementation's schedule at that omitted size.
                xx = np.arange(1000,8001)
                yy = np.array([f['coefficients'][0]+f['coefficients'][1]*counts.work(int(n),method,phase)/1e9 for n in xx])
                if len(f['coefficients']) == 3: yy += f['coefficients'][2]*xx.astype(float)**2/1e6
                if method == 'GGK_recompute':
                    regimes = {counts.ggk_count(r['N'],True)[1] for r in rr}
                    for regime in sorted(regimes):
                        present = np.array([counts.ggk_count(int(n),True)[1] == regime for n in xx])
                        line, = ax.plot(xx[present],yy[present],color=color,ls=style,lw=1.15,zorder=2)
                else:
                    line, = ax.plot(xx,yy,color=color,ls=style,lw=1.15,zorder=2)
                # The two unblocked orientations nearly coincide. A larger
                # open circle surrounds the smaller square without jittering
                # either observation or changing its coordinate.
                size = 5.3 if method == 'FP2020_up' else 3.6
                points, = ax.plot([r['N'] for r in rr],[r[phase] for r in rr],ls='none',marker=marker,
                                  color=color,mfc=fill,mec=color,mew=.8,ms=size,zorder=3)
                entries[method]=(line,points)
            ax.set_title(f'{family.capitalize()} ensemble')
            ax.set_xlabel('State count N')
            ax.set_ylim(bottom=0)
            ax.set_xticks([1000,2000,4000,6000,8000])
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.grid(axis='y',color='.89',lw=.55)
        axs[row,0].set_ylabel('End-to-end time (s)' if phase=='end_to_end' else 'Index-loop time (s)')
        # Identical limits make the two ensembles directly comparable.
        ymax=max(ax.get_ylim()[1] for ax in axs[row,:])
        for ax in axs[row,:]: ax.set_ylim(0,ymax)
    fig.legend([entries[m] for m in methods],[LABELS[m] for m in methods],
               handler_map={tuple:HandlerTuple(ndivide=1)},loc='upper center',bbox_to_anchor=(.5,.995),
               ncol=3,frameon=False,handlelength=3.5,columnspacing=1.4)
    dest=ROOT.parent/'figures'
    for suffix in ('pdf','png'): fig.savefig(dest/f'{name}.{suffix}',bbox_inches='tight')
    plt.close(fig)

def render():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,
                         'legend.fontsize':7.4,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':220,'pdf.fonttype':42,'ps.fonttype':42})
    principal=('FP_blocked_down','GGK_cubic','GGK_recompute')
    plot_panels(principal,'binary_main')
    plot_panels(METHODS[:3],'binary_blocking')
    plot_panels(principal,'binary_end_to_end',('end_to_end',))
    plot_panels(principal,'binary_index_loop',('index_loop',))

def tables():
    dest=ROOT.parent/'tables'
    for phase,name in [('end_to_end','binary_complete'),('index_loop','binary_loop_complete')]:
        title='end-to-end' if phase=='end_to_end' else 'index-loop'
        lines=[r'\begingroup\small\setlength{\tabcolsep}{3pt}',r'\begin{longtable}{rllrrrrrr}',
               rf'\caption{{Fresh random dense binary {title} times (seconds): medians across draws of each two-execution median. U/E: uniform/exponential; D/A: discounted/average. All five methods were rerun on the same inputs.}}\label{{tab:ec-{name.replace("_","-")}}}\\',
               r'\toprule $N$ & Ens. & Crit. & Draws & Blocked FP & FP up & FP down & GGK no-rec. & GGK default\\\midrule\endfirsthead',
               r'\multicolumn{9}{l}{\emph{Continued from previous page}}\\',
               r'\toprule $N$ & Ens. & Crit. & Draws & Blocked FP & FP up & FP down & GGK no-rec. & GGK default\\\midrule\endhead',
               r'\midrule\endfoot\bottomrule\endlastfoot']
        for n in (1000,2000,4000,6000,8000):
            for fam in ('uniform','exponential'):
                for crit in ('discounted','average'):
                    rr=[r for r in rows if (r['N'],r['family'],r['criterion'])==(n,fam,crit)]
                    times=[stat.median(r[phase] for r in rr if r['method']==m) for m in METHODS]
                    lines.append(f'{n} & {fam[0].upper()} & {crit[0].upper()} & {len(rr)//5} & '+' & '.join(f'{t:.3f}' for t in times)+r' \\')
        lines.append(r'\end{longtable}\endgroup')
        (dest/f'{name}.tex').write_text('\n'.join(lines)+'\n')
    def span(vals): return f'{min(vals):.2f}--{max(vals):.2f}'
    lines=[r'\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{4pt}',
           r'\caption{Average binary regression errors (percent), ranges across ensembles. Fit and holdout columns give median absolute percentage errors; the last column gives the maximum across all largest-size held-out observations. Leading and expanded models are both retained for every method and phase.}\label{tab:ec-binary-total-model}',
           r'\begin{tabular}{llrrrr}\toprule',
           r'Method & Phase & Leading & Expanded & Expanded & $N=8000$\\',
           r' & & holdout & fit & holdout & exp. holdout\\\midrule']
    for m in METHODS:
        for phase in ('end_to_end','index_loop'):
            a=[getfit(m,phase,f,model='leading_count') for f in ('uniform','exponential')]
            b=[getfit(m,phase,f,model='leading_count_plus_quadratic') for f in ('uniform','exponential')]
            largest=max(h['maximum_APE'] for f in b for h in f['holdouts'] if h['N']==8000)
            lines.append(SHORT[m]+(' & Total & ' if phase=='end_to_end' else ' & Loop & ')+span([f['holdout_median_APE'] for f in a])+' & '+span([f['median_APE'] for f in b])+' & '+span([f['holdout_median_APE'] for f in b])+f' & {largest:.2f}'+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\end{table}']
    (dest/'binary_total_model_diagnostics.tex').write_text('\n'.join(lines)+'\n')
    # Current criterion ratios are rebuilt by report_blocked_multigear.build.

if __name__=='__main__':
    validate();tables();render()
    print('Verified 3,200 scalar records and 80 fits; rebuilt fresh binary tables and figures.')
