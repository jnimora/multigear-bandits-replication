#!/usr/bin/env python3
"""Statistics fixed by the confirmation design (Python standard library only).

These functions analyze 32 independent replication-level observations. A time
period, arm, policy, overlapping window, or CRN cap partner is NOT a replication.
Intervals concern finite-window expectations under approximate normality of
replication means; they do not certify stationarity, tails, or asymptotic optimality.
"""
import math
import statistics

N = 32
# Student t, df=31. Generated with scipy.stats.t.ppf and checked independently.
T_POINTWISE = 2.039513446396408
T_PRIMARY_12 = 3.09350558266835
T_SECONDARY_48 = 3.6183900855997315


def _vector(values):
    x = [float(v) for v in values]
    if len(x) != N or any(not math.isfinite(v) for v in x):
        raise ValueError('Exactly 32 finite, independent replication-level values are required')
    return x


def _sample_cov(x, y):
    mx = statistics.fmean(x); my = statistics.fmean(y)
    return math.fsum((a-mx)*(b-my) for a,b in zip(x,y))/(len(x)-1)


def mean_interval(values, critical=T_POINTWISE):
    x=_vector(values); mean=statistics.fmean(x); sd=statistics.stdev(x)
    se=sd/math.sqrt(N); halfwidth=critical*se
    return dict(n=N,mean=mean,standard_error=se,critical=critical,
                lower=mean-halfwidth,upper=mean+halfwidth,halfwidth=halfwidth,
                zero_sample_variance=(sd==0),
                zero_sample_variance_is_not_a_rare_event_guarantee=True,
                interpretation='finite-window expectation; approximate normal replication means; no stationarity certificate')


def quadratic_nonpositive(A, B, C):
    """All real x with A*x*x+B*x+C <= 0; do not hide unbounded Fieller sets.

    Mathematical endpoints +/-infinity are represented as JSON-compatible strings.
    Near-zero nonzero coefficients are not silently set to zero.
    """
    if any(not math.isfinite(x) for x in (A,B,C)):
        raise ValueError('Nonfinite quadratic coefficient')
    if A==0:
        if B==0:
            return dict(type='all_real',intervals=[['-inf','+inf']]) if C<=0 else dict(type='empty',intervals=[])
        x=-C/B
        return dict(type='halfline',intervals=[['-inf',x] if B>0 else [x,'+inf']])
    # Scale first to reduce risk of overflow in the discriminant.
    scale=max(abs(A),abs(B),abs(C));A=A/scale;B=B/scale;C=C/scale
    disc=B*B-4*A*C
    if disc<0:
        return dict(type='all_real',intervals=[['-inf','+inf']]) if A<0 else dict(type='empty',intervals=[])
    if disc==0:
        return dict(type='all_real',intervals=[['-inf','+inf']]) if A<0 else dict(type='singleton',intervals=[[-B/(2*A),-B/(2*A)]])
    q=-0.5*(B+math.copysign(math.sqrt(disc),B))
    roots=sorted((q/A,C/q))
    if A>0:return dict(type='bounded',intervals=[roots])
    return dict(type='two_rays',intervals=[['-inf',roots[0]],[roots[1],'+inf']])


def paired_gain(myopic, whittle, critical=T_PRIMARY_12):
    """Gain=E[M-W]/E[M], not E[(M-W)/M]; paired Fieller set.

    Default critical value gives Bonferroni coverage for the 12 primary gain
    intervals, subject to the individual intervals' normal-mean approximation.
    Primary absolute-difference intervals below are POINTWISE, not a second
    jointly 95%-covered family with the ratio sets.
    """
    M=_vector(myopic);W=_vector(whittle)
    if any(x<0 for x in M+W):raise ValueError('AoII cost observations must be nonnegative')
    D=[m-w for m,w in zip(M,W)]
    dm=statistics.fmean(D);mm=statistics.fmean(M)
    vd=_sample_cov(D,D)/N;vm=_sample_cov(M,M)/N;cov=_sample_cov(D,M)/N
    k2=critical*critical
    A=mm*mm-k2*vm;B=-2*dm*mm+2*k2*cov;C=dm*dm-k2*vd
    region=quadratic_nonpositive(A,B,C)
    gain=dm/mm if mm!=0 else None
    bounded=region['type'] in ('bounded','singleton')
    halfwidth=(region['intervals'][0][1]-region['intervals'][0][0])/2 if bounded else None
    equivalence=bounded and -0.01<=region['intervals'][0][0] and region['intervals'][0][1]<=0.01
    material=bounded and region['intervals'][0][0]>=0.05
    return dict(n=N,mean_whittle=statistics.fmean(W),mean_myopic=mm,
                absolute_difference=mean_interval(D),gain_ratio_of_means=gain,
                gain_percent=None if gain is None else 100*gain,
                paired_fieller_set=region,critical=critical,fieller_coefficients=[A,B,C],
                gain_halfwidth=halfwidth,gain_halfwidth_target_met=bounded and halfwidth<=0.02,
                within_one_percentage_point_equivalence_margin=equivalence,
                lower_bound_at_least_five_percent_saving=material,
                outcome_is_not_a_computational_acceptance_criterion=True,
                finite_window_not_automatically_stationary=True)


def secondary_whittle_contrast(other,whittle):
    """Other minus Whittle, simultaneous within the 48 prespecified secondary tests."""
    x=_vector(other);y=_vector(whittle)
    return mean_interval([a-b for a,b in zip(x,y)],T_SECONDARY_48)


def residual_summary(residual, bound_per_project):
    x=_vector(residual);ell=float(bound_per_project)
    if ell<=0 or not math.isfinite(ell):raise ValueError('Positive finite bound required')
    if any(v<0 for v in x):raise ValueError('Unexpected negative certified residual; do not clip it')
    result=mean_interval(x)
    target=max(1e-4,1e-3*ell)
    result.update(halfwidth_target=target,halfwidth_target_met=result['halfwidth']<=target,
                  stationary_true_optimality_gap_measured=False,
                  rare_event_caution=result['zero_sample_variance'])
    return result


def paired_gain_change(myopic_reference, whittle_reference, myopic_alternative, whittle_alternative):
    """Alternative-minus-reference gain; prespecified pointwise delta-method check.

    Uses all within-replication covariance terms. This is not a primary test or
    a Fieller interval for a single ratio, and is labelled approximate.
    """
    mr=_vector(myopic_reference);wr=_vector(whittle_reference)
    ma=_vector(myopic_alternative);wa=_vector(whittle_alternative)
    means=list(map(statistics.fmean,(mr,wr,ma,wa)))
    MR,WR,MA,WA=means
    if MR<=0 or MA<=0:raise ValueError('Positive reference and alternative mean costs required')
    if any(v<0 for x in (mr,wr,ma,wa) for v in x):raise ValueError('Negative AoII cost')
    gradient=(-WR/(MR*MR),1/MR,WA/(MA*MA),-1/MA)
    influence=[sum(g*(v-mean) for g,v,mean in zip(gradient,values,means)) for values in zip(mr,wr,ma,wa)]
    se=statistics.stdev(influence)/math.sqrt(N)
    value=WR/MR-WA/MA;hw=T_POINTWISE*se
    return dict(n=N,gain_change=value,standard_error=se,lower=value-hw,upper=value+hw,
                within_one_percentage_point_margin=(value-hw>=-0.01 and value+hw<=0.01),
                method='paired multivariate delta method with full covariance; pointwise approximate',
                stationarity_or_uncapped_error_certified=False)
