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
output (cov=True) gives a real, propagated standard error on the
extrapolated value, consistent with every other series this project
plots with a bootstrap error bar.
"""
import numpy as np


def zne_extrapolate(fold_factors, values, errors=None, deg=1):
    """Fit `values` vs. `fold_factors` to a degree-`deg` polynomial and
    return its value at fold_factor=0 (the zero-noise limit), plus that
    value's own propagated standard error.

    errors: per-point standard errors, e.g. from
    shot_observables.bootstrap_observable_errors/bootstrap_mx_error --
    used as inverse-variance weights (numpy.polyfit's `w=`) so noisier
    fold-scaled points count for less. If None, or if any entry is None/0
    (bootstrap error undefined/degenerate), an unweighted fit is used
    instead.

    Returns (zero_noise_value, zero_noise_error).
    """
    if len(fold_factors) < deg + 2:
        raise ValueError(
            f"zne_extrapolate needs at least deg+2 = {deg + 2} fold factors to fit a "
            f"degree-{deg} polynomial AND get a covariance-based error bar (a fit with "
            f"zero residual degrees of freedom has an undefined covariance) -- got "
            f"{len(fold_factors)}. Matches qermit's own Fit.polynomial constraint."
        )

    x = np.asarray(fold_factors, dtype=float)
    y = np.asarray(values, dtype=float)

    w = None
    if errors is not None and all(e not in (None, 0) for e in errors):
        w = 1.0 / np.asarray(errors, dtype=float)

    coeffs, cov = np.polyfit(x, y, deg, w=w, cov=True)
    # numpy.polyfit orders coefficients highest-degree first; the
    # fold_factor=0 value is the polynomial's constant term, i.e. the
    # LAST coefficient.
    return float(coeffs[-1]), float(np.sqrt(cov[-1, -1]))
