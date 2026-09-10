"""
Annular fuel conduction: the Kirchhoff-transformed radial solve.

Moved verbatim from PinHT.py in Phase 1. Phase 2 brings it up to the docstring and
backend standard without changing any formula or sign convention. Implements the scheme
derived in docs/reference/Annular_Heat_Transfer_Final.pdf section 1.1. Phase 4 adds the
iteration around it (Cramer's rule, warm start, convergence flag).
"""
import numpy as np

from pinthac import backend


def Ann_Theta(kf_func, T_ref=300.0, T_max=3600.0, n=4000):
    """
    Build the Kirchhoff-transform conductivity integral Theta[kf](T).

    Why this model is here:
        Ann_HT needs Theta = integral of a temperature-dependent fuel conductivity kf(T)
        to linearize the otherwise-nonlinear annular fuel conduction equation -- exactly
        as it would already be linear in T alone for a constant kf. Built once via
        cumulative trapezoidal integration over a fixed grid and returned as a fast
        interpolant, since the closure that calls this (sca/annular.py::closure)
        evaluates it at every axial location on every training step.

    Formulation:
        Theta(T) = integral_{T_ref}^{T} kf(T') dT'

        Computed by cumulative trapezoidal integration of kf on a fixed grid of n
        points between T_ref and T_max, then wrapped in a cubic-spline interpolant.
        Offsetting from T_ref rather than 0 K only shifts Theta's (otherwise arbitrary)
        zero point -- Ann_HT only ever uses *differences* and sums of Theta, so any
        fixed reference is fine as long as it is used consistently for both surfaces,
        which sharing one interpolant here guarantees.

    Valid range:
        [T_ref, T_max] (defaults 300-3600 K); extrapolates via the cubic spline's own
        extrapolation outside that window (fill_value='extrapolate'), matching
        SCA_Example.py's Property() lookup convention. Not established beyond that --
        see docs/OPEN_QUESTIONS.md (Q31) for kf_func's own valid range, which this
        function inherits.

    Uncertainty:
        Not applicable to the integration itself (n=4000 trapezoidal points over a
        3300 K span resolves the integral far below the underlying kf model's own
        uncertainty); inherits whatever uncertainty kf_func itself carries.

    Reference:
        docs/reference/Annular_Heat_Transfer_Final.pdf section 1.1 for the
        Kirchhoff-transform derivation this integral feeds into (see Ann_HT).

    Backend contract note:
        Unlike every other function in this module, Ann_Theta does not itself accept or
        return a float/numpy array/torch tensor -- it returns a SciPy interp1d
        *callable*, built with plain NumPy internally regardless of what kf_func
        returns. That callable is not differentiable, and (per docs/AUDIT.md) silently
        strips the gradient from a torch input passed through it later rather than
        raising. This is a known architectural limitation carried over from PinHT.py,
        not something Phase 2's docstring/backend-contract/range/dead-code/test scope
        fixes -- replacing it with a differentiable table (e.g. backend.interp-based)
        is a bigger redesign than a physics-preserving cleanup, and this module's own
        original docstring already flags the follow-up ("Phase 4 adds the iteration
        around it") as later work.

    Inputs:
        kf_func : callable, temperature [K] (numpy array) -> thermal conductivity
                  [W/m-K] (numpy array), vectorized over T (e.g.
                  MatMod.UO2.k_NFI bound to fixed Bu/f_gad)
        T_ref   : integration lower bound / Theta's zero point, K (default 300.0)
        T_max   : integration upper bound, K (default 3600.0)
        n       : number of trapezoidal grid points (default 4000)
    Returns:
        Theta : scipy.interpolate.interp1d callable, K -> Theta [W/m] (numpy only, not
                differentiable -- see the backend contract note above)
    """
    from scipy.interpolate import interp1d
    Tgrid = np.linspace(T_ref, T_max, n)
    kvals = kf_func(Tgrid)
    Theta_vals = np.concatenate(([0.0], np.cumsum(0.5*(kvals[1:] + kvals[:-1])*np.diff(Tgrid))))
    return interp1d(Tgrid, Theta_vals, kind='cubic', fill_value='extrapolate')


def Ann_HT(ri, ro, q3, Theta_i, Theta_o):
    """
    Closed-form solve for the two integration constants in the Kirchhoff-transformed
    annular fuel conduction problem.

    Why this model is here:
        The linear (in Theta) two-point boundary value problem that results from
        substituting Theta[kf](T(r)) for T(r) in the annular conduction equation, given
        already-known fuel surface temperatures at both radii. Used by
        sca/annular.py::closure and ml/pinn.py every time the fuel surface flux split
        needs updating.

    Formulation:
        For ri <= r <= ro with uniform volumetric generation q3:
            Theta[kf](T(r)) = -q3/4*r^2 + C1*log(r) + C2

        alpha_i = Theta_i + q3/4*ri^2
        alpha_o = Theta_o + q3/4*ro^2
        C1 = (alpha_i - alpha_o) / (log(ri) - log(ro))
        C2 = (log(ri)*alpha_o - log(ro)*alpha_i) / (log(ri) - log(ro))

        Theta_i, Theta_o = Theta[kf] evaluated (via Ann_Theta) at the *already-known*
        inner/outer fuel surface temperatures -- forward evaluations only, never
        inverted (the scheme's whole point: no Newton solve is needed to invert Theta
        back to a temperature). The -q3/4*r^2 sign matches
        docs/reference/Annular_Heat_Transfer_Final.pdf section 1.1 Eq. (2), which
        already carries the correct sign, unlike an earlier PinHT derivation this one
        superseded.

        Sign-convention note the surface temperatures feeding Theta_i, Theta_o are NOT
        interchangeable in how they are obtained from a guessed flux: the reference
        PDF's "Tfo(rj) = Tm,j + q''_j/htcj" is only correct at the *outer* surface. The
        inner coolant sits on the -r side of the fuel, so the physically-outward flux
        there is the *negative* of the q''(r) Fourier-convention function this module
        also uses (see Ann_qpp) -- checked by two independent methods (an energy-balance
        invariant q_i + q_o = q'''*pi*(ro^2-ri^2), and both walls coming out hotter than
        their coolants) plus an independent sympy solve: a fixed-point iteration of the
        PDF's literal scheme converges to a state with the inner wall *colder* than the
        inner coolant and a badly violated energy balance, while
        "Tfo(ri) = Ti - q''_i/htci" (a minus, mirroring the +r-vs-coolant-side geometry)
        converges to the energy-conserving solution. Callers (see
        sca/annular.py::closure) must apply that minus sign when they turn a flux/htc
        guess into the inner surface temperature that gets passed through Ann_Theta to
        make Theta_i. This is a deliberate, documented departure from the reference
        PDF's literal Section 1.1/1.2 wording, approved in docs/DECISIONS.md (closing
        Q18): the signed +r Fourier convention with the minus at the inner surface, with
        the energy-balance invariant above as its own test (see tests/test_annular.py).

    Valid range:
        Not established -- see docs/OPEN_QUESTIONS.md (Q31). A closed-form algebraic
        solve, not a fitted correlation.

    Uncertainty:
        Not applicable -- exact given Theta_i, Theta_o (whatever uncertainty exists is
        in kf_func, inherited through Ann_Theta).

    Reference:
        docs/reference/Annular_Heat_Transfer_Final.pdf section 1.1, Eq. (2).

    Inputs (float, numpy array, or torch tensor; broadcastable against each other):
        ri, ro           : fuel inner/outer radius, m
        q3               : volumetric heat generation rate, W/m^3
        Theta_i, Theta_o : Theta[kf] at the inner/outer fuel surface temperature, W/m
                            (see the sign-convention note above for how these are
                            obtained from a guessed flux)
    Returns:
        (C1, C2) : integration constants, same type as ri
    """
    # ri/ro are the common case for a Python-float fixed pin geometry paired with
    # tensor Theta_i/Theta_o (an axial or batched solve) -- promoting them against
    # Theta_i is what lets xp.log(ri) below run under torch: torch.log rejects a bare
    # Python float outright, the same failure mode as pin/clad.py::T_ci's Rco/Rci (see
    # that docstring) and MatMod.UO2.k_NFI's Bu.
    ri = backend.promote(ri, Theta_i)
    ro = backend.promote(ro, Theta_i)
    xp = backend.lib(ri, ro, q3, Theta_i, Theta_o)

    alpha_i = Theta_i + q3/4*ri**2
    alpha_o = Theta_o + q3/4*ro**2
    log_ri, log_ro = xp.log(ri), xp.log(ro)

    C1 = (alpha_i - alpha_o)/(log_ri - log_ro)
    C2 = (log_ri*alpha_o - log_ro*alpha_i)/(log_ri - log_ro)
    return C1, C2


def Ann_qpp(r, q3, C1):
    """
    Radial heat flux in the annular fuel region, in the signed +r Fourier convention.

    Why this model is here:
        Converts Ann_HT's C1 into the actual heat flux at a given radius; per Ann_HT's
        sign-convention note, callers negate this at the inner surface to get the
        physically-outward flux into the inner coolant (see sca/annular.py::closure's
        `q_i = -Ann_qpp(ri, q3, C1)*Per_fuel_i`).

    Formulation:
        q''(r) = q3/2*r - C1/r

    Valid range:
        Not established -- see docs/OPEN_QUESTIONS.md (Q31). An algebraic consequence
        of Ann_HT's C1, not a fitted correlation.

    Uncertainty:
        Not applicable -- exact given C1.

    Reference:
        docs/reference/Annular_Heat_Transfer_Final.pdf section 1.1 (same derivation as
        Ann_HT).

    Inputs (float, numpy array, or torch tensor; broadcastable against each other):
        r  : radial position, m
        q3 : volumetric heat generation rate, W/m^3
        C1 : integration constant from Ann_HT
    Returns:
        val : radial heat flux in the +r direction, W/m^2, same type as r
    """
    return q3/2*r - C1/r


def Ann_flux_split(ri, ro, q_lin, Tm_i, Tm_o, htc_i, htc_o, Theta_func,
                   f_prev=None, n_iter=60, tol=1.0e-6, return_convergence=False):
    """
    Solve the annular fuel for how the generated heat splits between its two coolants.

    Why this model is here:
        Ann_HT solves the conduction problem given the two fuel surface temperatures, but
        those temperatures depend on the surface heat fluxes, which are what the
        conduction solve produces. That circle is the actual problem an annular pin
        poses, and this is the iteration that closes it. It is the scheme derived in
        docs/reference/Annular_Heat_Transfer_Final.pdf section 1.1 and drawn as its
        Figure 2.

    Formulation:
        Repeat until the surface fluxes stop moving:

          1. from the current linear-heat split, surface fluxes
                 q''_o = q'_o / (2*pi*ro)          q''_i = q'_i / (2*pi*ri)
          2. fuel surface temperatures, across the coolant-to-surface resistance
                 Tfo(ro) = Tm_o + q''_o/htc_o      Tfo(ri) = Tm_i + q''_i/htc_i
          3. the conductivity integral, evaluated FORWARD only
                 Theta_j = Theta[kf]( Tfo(r_j) )
          4. Cramer's rule on the resulting 2x2 linear system (this is Ann_HT)
                 C1 = (alpha_i - alpha_o) / (log(ri) - log(ro))
          5. updated fluxes from C1 alone -- C2 is not needed here
                 q''(r) = q'''/2*r - C1/r

        Never inverting Theta is the point of the whole scheme. An inverse would need a
        Newton solve per surface per iteration per axial node, and Theta is only
        available as a forward integral of kf.

        Sign convention: q''(r) is the signed +r Fourier flux, so it is negative at the
        inner surface where heat flows toward -r. The linear heat rates returned are
        magnitudes -- power per metre actually delivered into each coolant -- so
        q'_i = -2*pi*ri*q''(ri) and q'_o = +2*pi*ro*q''(ro). See Ann_HT's docstring for
        why the inner surface temperature uses the magnitude and not the signed flux.

        Energy balance is exact at every iterate, not just at convergence:

            q'_i + q'_o = 2*pi*ro*(q'''*ro/2 - C1/ro) - 2*pi*ri*(q'''*ri/2 - C1/ri)
                        = pi*q'''*(ro^2 - ri^2)

        the C1 terms cancelling identically. So this iteration never has to converge the
        total, only the split -- which is worth knowing when reading a convergence
        failure, because an unconverged result still conserves energy and is wrong only
        in how the heat is shared.

    The warm start:
        With f_prev given, the initial split reuses the heat ratio f = q'_i/q'_o from the
        previous axial node instead of halving:

            q'_o = q' / (1 + f_prev)        q'_i = q' * f_prev / (1 + f_prev)

        This is the option described in the reference PDF's section 1.2. It helps
        because the split varies slowly along a channel -- geometry is fixed and the two
        bulk temperatures drift gradually -- while an even split can be far off in a pin
        whose two channels carry very different mass flows. Starting from the previous
        node's answer typically leaves only a small correction, so the iteration count
        falls; and since the map is a contraction, a better start is strictly a cheaper
        one, never a different answer.

    Valid range:
        Requires htc_i, htc_o > 0 and ri < ro. No fitted range -- this is a solve, not a
        correlation.

    Uncertainty:
        Not applicable. The uncertainty in the result is whatever kf_func and the two
        heat transfer coefficients carry.

    Reference:
        docs/reference/Annular_Heat_Transfer_Final.pdf, section 1.1 and Figure 2.

    Inputs (float, numpy array, or torch tensor; broadcastable against each other):
        ri, ro     : fuel inner/outer radius, m
        q_lin      : total linear heat generation rate of the fuel annulus, W/m
        Tm_i, Tm_o : inner/outer bulk coolant temperature, K
        htc_i      : inner surface coolant-to-fuel heat transfer coefficient, W/m^2-K.
                     Combine convection, gap and clad as series resistances before
                     calling: 1/htc = 1/htc_conv + 1/htc_gap + 1/htc_clad.
        htc_o      : the same for the outer surface, W/m^2-K
        Theta_func : callable T [K] -> Theta[kf](T) [W/m], the conductivity integral.
                     Must be differentiable if the caller needs a gradient --
                     properties.matmod.UO2.Theta_Klimenko and Theta_NFI are; the
                     interpolant Ann_Theta builds is not.
        f_prev     : optional previous-node heat ratio q'_i/q'_o for the warm start
        n_iter     : fixed iteration count, applied to the whole batch
        tol        : convergence tolerance on the relative change in the split
        return_convergence : if True, also return the elementwise convergence flag

    Returns:
        q_i, q_o   : linear heat rate into the inner/outer coolant, W/m, both positive
                     in normal operation. q_i goes negative where the inner coolant is
                     hotter than the fuel inner surface and heats the fuel instead --
                     a real regime for an asymmetrically cooled pin, not a failure.
        Tfo_i, Tfo_o : fuel inner/outer surface temperature, K
        converged  : only if return_convergence -- boolean, elementwise
    """
    ri = backend.promote(ri, q_lin)
    ro = backend.promote(ro, q_lin)
    xp = backend.lib(ri, ro, q_lin, Tm_i, Tm_o, htc_i, htc_o)

    area = xp.pi * (ro**2 - ri**2)
    q3 = q_lin / area                       # volumetric generation rate, W/m^3

    # Step 0 of Figure 2: the initial guess, evenly split or warm started.
    if f_prev is None:
        q_i = 0.5 * q_lin
        q_o = 0.5 * q_lin
    else:
        q_o = q_lin / (1.0 + f_prev)
        q_i = q_lin * f_prev / (1.0 + f_prev)

    converged = backend.zeros_like(q_lin) > 1.0        # all False, right shape and kind
    scale = backend.maximum(abs(q_lin), 1.0)           # for a relative convergence test

    for _ in range(n_iter):
        # Step 1-2: surface fluxes, then surface temperatures across the coolant-side
        # resistance. Both use the magnitude of the flux leaving into that coolant.
        qpp_i = q_i / (2.0 * xp.pi * ri)
        qpp_o = q_o / (2.0 * xp.pi * ro)
        Tfo_i = Tm_i + qpp_i / htc_i
        Tfo_o = Tm_o + qpp_o / htc_o

        # Step 3: the conductivity integral, forward only.
        Theta_i = Theta_func(Tfo_i)
        Theta_o = Theta_func(Tfo_o)

        # Step 4: Cramer's rule for C1 and C2. Only C1 is needed to update the fluxes.
        C1, _C2 = Ann_HT(ri, ro, q3, Theta_i, Theta_o)

        # Step 5: new fluxes from C1, converted back to linear heat rates as magnitudes.
        q_i_new = -2.0 * xp.pi * ri * Ann_qpp(ri, q3, C1)
        q_o_new = 2.0 * xp.pi * ro * Ann_qpp(ro, q3, C1)

        converged = converged | ((abs(q_i_new - q_i) / scale < tol)
                                 & (abs(q_o_new - q_o) / scale < tol))
        q_i, q_o = q_i_new, q_o_new

    # Recompute the surface temperatures from the converged split so the returned
    # temperatures and fluxes describe the same state rather than being one step apart.
    Tfo_i = Tm_i + q_i / (2.0 * xp.pi * ri) / htc_i
    Tfo_o = Tm_o + q_o / (2.0 * xp.pi * ro) / htc_o

    if return_convergence:
        return q_i, q_o, Tfo_i, Tfo_o, converged
    return q_i, q_o, Tfo_i, Tfo_o
