"""
zne_fit.py

Zero-noise extrapolation fit: given observable values measured at several
qermit Folding.circuit noise-scaling factors (see qnexus_backend.
submit_zne_batch), fit a polynomial and extrapolate back to the
zero-noise (fold_factor=0) limit.

Done directly via numpy.polyfit rather than qermit's own
qermit.zero_noise_extrapolation.zne.Fit.* functions, since those take no
per-point weights and return a bare float with no uncertainty at all --
this project's existing shot-noise bootstrap standard errors
(shot_observables.bootstrap_observable_errors/bootstrap_mx_error) are used
here as inverse-variance fit weights, and numpy.polyfit's covariance
output gives a real, propagated standard error on the extrapolated
value, consistent with every other series this project plots with a
bootstrap error bar.

Uses cov='unscaled' (not the plain cov=True default): our weights are
real *absolute* standard errors (from bootstrap), not just relative
importance weights, so the covariance shouldn't be rescaled by the fit's
own residual variance the way cov=True does (which also requires more
data points than parameters -- deg+2 -- to have any residual degrees of
freedom to rescale by at all). 'unscaled' uses the weights directly as
inverse-variances, which is both the statistically correct choice here
and allows fitting with as few as deg+1 points (e.g. two fold factors
for a linear fit) -- verified directly against numpy.polyfit.
"""
import numpy as np


def zne_extrapolate(fold_factors, values, errors=None, deg=1):
    """Fit `values` vs. `fold_factors` to a degree-`deg` polynomial and
    return its value at fold_factor=0 (the zero-noise limit), plus that
    value's own propagated standard error.

    errors: per-point standard errors, e.g. from
    shot_observables.bootstrap_observable_errors/bootstrap_mx_error --
    used as inverse-variance weights (numpy.polyfit's `w=`) so noisier
    fold-scaled points count for less, with cov='unscaled' (see module
    docstring) so the returned error is a real propagated standard error
    even with the minimum deg+1 points. If None, or if any entry is
    None/0 (bootstrap error undefined/degenerate), an unweighted fit is
    used instead (cov=True, needs deg+2 points to have a residual-based
    error estimate at all).

    Returns (zero_noise_value, zero_noise_error).
    """
    x = np.asarray(fold_factors, dtype=float)
    y = np.asarray(values, dtype=float)

    w = None
    if errors is not None and all(e not in (None, 0) for e in errors):
        w = 1.0 / np.asarray(errors, dtype=float)

    min_points = deg + 1 if w is not None else deg + 2
    if len(fold_factors) < min_points:
        extra = "" if w is not None else " with a real (unweighted) covariance"
        raise ValueError(
            f"zne_extrapolate needs at least {min_points} fold factors to fit a "
            f"degree-{deg} polynomial{extra} -- got {len(fold_factors)}."
        )

    coeffs, cov = np.polyfit(x, y, deg, w=w, cov=('unscaled' if w is not None else True))
    # numpy.polyfit orders coefficients highest-degree first; the
    # fold_factor=0 value is the polynomial's constant term, i.e. the
    # LAST coefficient.
    return float(coeffs[-1]), float(np.sqrt(cov[-1, -1]))
