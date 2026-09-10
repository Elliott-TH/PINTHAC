"""
Annular fuel conduction: the Kirchhoff-transformed radial solve.

Moved verbatim from PinHT.py in Phase 1. Implements the scheme derived in
docs/reference/Annular_Heat_Transfer_Final.pdf section 1.1. Phase 4 adds the
iteration around it (Cramer's rule, warm start, convergence flag).
"""
import numpy as np
import scipy

from pinthac.backend import lib as compat


def Ann_Theta(kf_func, T_ref=300.0, T_max=3600.0, n=4000):
    """
    Builds Theta[kf](T) = integral_{T_ref}^{T} kf(T') dT', the Kirchhoff-
    transform conductivity integral used by Ann_HT to linearize the
    annular fuel conduction problem for a temperature-dependent kf(T)
    (e.g. Mat_Models.UO2.k_NFI). See
    SCA_PDFs/Annular_Heat_Transfer_Final.pdf sec 1.1: with
    Theta(T) = integral of kf, the nonlinear radial conduction equation
    becomes linear in Theta, exactly as it would be in T alone for a
    constant kf.

    Built once via cumulative trapezoidal integration over a fixed grid
    and returned as a fast interpolant, since the closure that calls
    this evaluates it every axial location on every training step.
    Offsetting from T_ref rather than 0 K only shifts Theta's (otherwise
    arbitrary) zero point -- Ann_HT only ever uses *differences* and
    sums of Theta, so any fixed reference is fine as long as it's used
    consistently for both surfaces, which sharing one interpolant here
    guarantees.

    kf_func: T [K] -> thermal conductivity [W/m-K], vectorized over T.
    Returns Theta(T), a scipy interp1d callable (extrapolates outside
    [T_ref, T_max], as SCA_Example.py's Property() lookup also does).
    """
    from scipy.interpolate import interp1d
    Tgrid = np.linspace(T_ref, T_max, n)
    kvals = kf_func(Tgrid)
    Theta_vals = np.concatenate(([0.0], np.cumsum(0.5*(kvals[1:] + kvals[:-1])*np.diff(Tgrid))))
    return interp1d(Tgrid, Theta_vals, kind='cubic', fill_value='extrapolate')


def Ann_HT(ri, ro, q3, Theta_i, Theta_o):
    """
    Closed-form solve for C1, C2 in the Kirchhoff-transformed annular
    fuel conduction problem (ri <= r <= ro, volumetric generation q3):

        Theta[kf](T(r)) = -q3/4*r**2 + C1*log(r) + C2

    Theta_i, Theta_o = Theta[kf] evaluated (via Ann_Theta) at the
    *already-known* inner/outer fuel surface temperatures -- forward
    evaluations only, never inverted (the scheme's whole point: no
    Newton solve needed to invert Theta back to a temperature). See
    SCA_PDFs/Annular_Heat_Transfer_Final.pdf sec 1.1, eq. 2 there
    (which already carries the correct -q3/4*r**2 sign, unlike the
    predecessor derivation PinHT previously matched).

    Note the surface temperatures feeding Theta_i, Theta_o are NOT
    interchangeable in how they're obtained from a guessed flux: the
    PDF's "Tfo(rj) = Tm,j + q''_j/htcj" is only correct at the outer
    surface. The inner coolant sits on the -r side of the fuel, so the
    physically-outward flux there is the *negative* of the q''(r)
    Fourier-convention function this module also uses (see Ann_qpp) --
    checked the same way as the predecessor derivation's sign slips: a
    fresh fixed-point iteration of the PDF's literal scheme (Section
    1.1/1.2) converges to a state with the inner wall *colder* than the
    inner coolant and a badly violated energy balance; using
    "Tfo(ri) = Ti - q''_i/htci" instead (a minus, mirroring the +r-vs-
    coolant-side geometry) converges to the energy-conserving solution.
    Callers (see Ann_SCA.closure) must apply that minus sign when they
    turn a flux/htc guess into the inner surface temperature that gets
    passed through Ann_Theta to make Theta_i.

    ri, ro (float): Fuel inner/outer radius [m]
    q3 (float or array): Volumetric heat generation rate [W/m^3]
    Theta_i, Theta_o (float or array): Theta[kf] at the inner/outer
        fuel surface temperature
    """
    lib = compat(ri, ro, q3, Theta_i, Theta_o)

    alpha_i = Theta_i + q3/4*ri**2
    alpha_o = Theta_o + q3/4*ro**2
    log_ri, log_ro = lib.log(ri), lib.log(ro)

    C1 = (alpha_i - alpha_o)/(log_ri - log_ro)
    C2 = (log_ri*alpha_o - log_ro*alpha_i)/(log_ri - log_ro)
    return C1, C2


def Ann_qpp(r, q3, C1):
    """Radial heat flux (+r direction) in the annular fuel region, given C1 from Ann_HT."""
    return q3/2*r - C1/r
