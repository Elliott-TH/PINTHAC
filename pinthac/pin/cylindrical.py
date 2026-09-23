"""Radial temperature profile of a solid cylindrical fuel pellet."""
from pinthac import backend


def Cyl_HT(r, q_vol, kint, C):
    """Radial temperature profile for a solid cylinder with uniform volumetric heat
    generation and constant thermal conductivity.

    Formulation:
        T(r) = C - q_vol*r^2/(4*kint)

        Sign convention confirmed against Annular_Heat_Transfer_Final.pdf
        Eq. (2), which carries the same -q'''*r^2/4 term for the general (annular) case
        this is the r_i=0 special case of -- see docs/OPEN_QUESTIONS.md Q17 item 4,
        where a different source document had the sign the other way and the PDF (and
        this code) were confirmed correct.

    Valid range:
        Not established -- see docs/OPEN_QUESTIONS.md (Q31). Standard cylindrical
        conduction; no fitted database to be out of range of.

    Uncertainty:
        Not established -- see docs/OPEN_QUESTIONS.md (Q31).

    Reference:
        Not established -- see docs/OPEN_QUESTIONS.md (Q31). Standard steady-state
        cylindrical (Fourier) conduction with uniform volumetric generation.

    Inputs (float, numpy array, or torch tensor; broadcastable against each other):
        r     : radial position, m
        q_vol : volumetric heat generation rate, W/m^3
        kint  : thermal conductivity, W/m-K
        C     : centerline (reference) temperature, K
    Returns:
        val : temperature at r, K, same type as r
    """
    val = C - q_vol * r**2 / (4*kint)
    return val


def Cyl_Theta(r, rfo, q3, Theta_fo):
    """Conductivity integral at radius r in a solid pellet, given its value at the surface.

    Formulation:
        The annular solution Theta = -q3/4*r^2 + C1*log(r) + C2 collapses for a solid
        pellet: log(r) diverges at the centreline, so C1 must be zero, and the single
        surface condition then fixes C2 outright. No 2x2 system, no iteration:

            C1 = 0
            C2 = Theta_fo + q3/4*rfo^2
            Theta(r) = Theta_fo + q3/4*(rfo^2 - r^2)

        At r = 0 this gives the centreline value Theta_fo + q3*rfo^2/4 directly, which
        is why peak fuel temperature in a solid pin costs one inversion rather than a
        solve.

    Valid range:
        0 <= r <= rfo. Uniform volumetric generation across the pellet.

    Uncertainty:
        Not applicable -- exact given Theta_fo. Whatever uncertainty exists is in the
        conductivity model behind Theta.

    Reference:
        PINTHA_Code_Summary.pdf section 4.3, and
        Annular_Heat_Transfer_Final.pdf Eq. (2) with ri -> 0.

    Inputs (float, numpy array, or torch tensor; broadcastable against each other):
        r        : radial position, m
        rfo      : pellet outer radius, m
        q3       : volumetric heat generation rate, W/m^3
        Theta_fo : conductivity integral at the pellet surface temperature, W/m
    Returns:
        Theta : conductivity integral at r, W/m, same type as the inputs
    """
    return Theta_fo + q3/4*(rfo**2 - r**2)


def Cyl_qpp(r, q3):
    """Radial heat flux at radius r in a solid pellet.

    Formulation:
        q''(r) = q3*r/2

        This is pin/annular.py::Ann_qpp with C1 = 0, which is what a solid pellet forces.
        It vanishes at the centreline, as symmetry requires.

    Valid range:
        0 <= r <= rfo. Uniform volumetric generation.

    Uncertainty:
        Not applicable -- an energy balance, not a correlation.

    Reference:
        PINTHA_Code_Summary.pdf section 4.3.

    Inputs (float, numpy array, or torch tensor; broadcastable against each other):
        r  : radial position, m
        q3 : volumetric heat generation rate, W/m^3
    Returns:
        qpp : heat flux in the +r direction, W/m^2, same type as r
    """
    return q3*r/2


def Cyl_qlin(r, q3):
    """Linear heat rate generated within radius r of a solid pellet.

    Formulation:
        q'(r) = 2*pi*r * q''(r) = pi*q3*r^2

        At r = rfo this returns the pin's whole linear heat rate, which is the identity
        worth testing against.

    Valid range:
        0 <= r <= rfo. Uniform volumetric generation.

    Uncertainty:
        Not applicable -- an energy balance, not a correlation.

    Reference:
        PINTHA_Code_Summary.pdf section 4.3.

    Inputs (float, numpy array, or torch tensor; broadcastable against each other):
        r  : radial position, m
        q3 : volumetric heat generation rate, W/m^3
    Returns:
        qlin : linear heat rate inside r, W/m, same type as r
    """
    import math
    xp = backend.lib(r, q3)
    pi = xp.pi if hasattr(xp, 'pi') else math.pi
    return pi*q3*r**2


def Cyl_T(r, rfo, q3, Theta_fo, Theta_func, k_func=None,
          T_lo=250.0, T_hi=4000.0, return_convergence=False):
    """Temperature at radius r in a solid pellet with temperature-dependent conductivity.

        Note the asymmetry with the annular case. There, the whole point of the scheme is
        that Theta is only ever evaluated forward. Here an inversion is unavoidable:
        the solution gives Theta at r, and a temperature is what is wanted. The saving
        grace is that it is one inversion at the end rather than one inside every
        iteration of a coupled solve.

    Formulation:
        Theta_target = Cyl_Theta(r, rfo, q3, Theta_fo)
        solve Theta_func(T) = Theta_target for T

        Theta is strictly increasing in T, because dTheta/dT = k(T) > 0 for any physical
        conductivity, so the root is unique and bisection cannot land on a wrong branch.
        That monotonicity is also why the plain bisect-then-Newton solver is enough here
        and the guarded bracketing of `torchsolve` is not needed -- unlike the
        supercritical wall-temperature solves, whose residual genuinely is non-monotone.

    Valid range:
        0 <= r <= rfo, and the true temperature inside [T_lo, T_hi]. Widen the bracket
        rather than assuming no solution exists.

    Uncertainty:
        Inherited from the conductivity model behind Theta_func.

    Reference:
        PINTHA_Code_Summary.pdf section 4.3.

    Inputs (float, numpy array, or torch tensor; broadcastable against each other):
        r          : radial position, m
        rfo        : pellet outer radius, m
        q3         : volumetric heat generation rate, W/m^3
        Theta_fo   : conductivity integral at the pellet surface temperature, W/m
        Theta_func : callable T [K] -> Theta [W/m]
        k_func     : optional callable T [K] -> k [W/m-K], the derivative of Theta_func
        T_lo, T_hi : bracket for the solve, K
        return_convergence : if True, also return whether the result sits inside the
                     bracket rather than pinned against an end of it

    Returns:
        T : temperature at r, K
        converged : only if return_convergence -- boolean, elementwise
    """
    import torch
    from pinthac.solvers import bisect_newton

    Theta_target = Cyl_Theta(r, rfo, q3, Theta_fo)
    if isinstance(Theta_target, torch.Tensor):
        target = Theta_target.to(dtype=torch.float64)
    else:
        target_array = backend.np.asarray(Theta_target, dtype=float)
        target = torch.as_tensor(target_array, dtype=torch.float64)

    def residual(T):
        return Theta_func(T) - target

    # Conductivity is the derivative of the conductivity integral.
    deriv = k_func
    lo = torch.full_like(target, T_lo)
    hi = torch.full_like(target, T_hi)
    T = bisect_newton(residual, lo, hi, deriv=deriv)

    if return_convergence:
        inside = (T > T_lo + 1.0) & (T < T_hi - 1.0)
        return T, inside
    return T
