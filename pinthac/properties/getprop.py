"""Thermophysical property lookup for water, sodium, lead, and LBE.

Water uses IAPWS-95 by default; formulation=97 selects IF97. Both return
the same property keys and units. _getprop97 also reports the IF97 region.
"""
import warnings

from pinthac import backend
from pinthac.ranges import RangeWarning
from pinthac.properties import iapws95 as iapws
from pinthac.properties import iapws97 as f97
from pinthac.properties import iapws_backend as w
from pinthac.properties import iapws_transport as tp
from pinthac.properties import liqprops as lm


def _getprop(substance, T, P, formulation=95):
    """Look up a substance's thermophysical properties at a given temperature and pressure.

    Formulation:
        "SCW" / "Water": water, from whichever of the two IAPWS formulations
            `formulation` selects.
            formulation=95 (the default) builds the IAPWS-95 Helmholtz state at
                (rho(T,P), T) and reads off rho, h, cp, mu, k.
            formulation=97 calls _getprop97(T, P) below, which picks the IF97 region from
                the state itself. Same keys, same units; see this module's docstring for
                which to reach for and iapws97.py for what differs.
            "Water" additionally adds the IAPWS R1-76 surface tension from
            properties/iapws_transport.py, the same way under either formulation since it
            belongs to neither ("SCW" operates above the critical point, where there is
            no interface and so no surface tension).
        "Lead" / "Pb": liqprops.Props(liqprops.Lead, T) (P is not used -- the liquid-metal
            correlations are pressure-independent, as documented in liqprops.py).
        "Sodium" / "Na": liqprops.Props(liqprops.Sodium, T) (P likewise unused).
        "LBE" / "PbBi": liqprops.Props(liqprops.LBE, T) (P likewise unused).

    Valid range:
        Whatever the underlying property library (iapws95.IAPWS95, liqprops.Sodium/Lead)
        validates -- this dispatcher performs no range check of its own.

    Uncertainty:
        Not applicable -- a dispatcher, not a correlation.

    Reference:
        Not applicable.

    Inputs:
        substance   : one of "SCW", "Water", "Lead", "Pb", "Sodium", "Na", "LBE",
                      "PbBi" (string)
        T           : temperature (float, numpy array, or torch tensor), K
        P           : pressure, MPa (float, numpy array, or torch tensor for
                      "SCW"/"Water"; unused, may be None, for the liquid metals)
        formulation : 95 or 97, which IAPWS formulation answers for water. Ignored for
                      the liquid metals, which have one correlation set each -- validated
                      anyway, so a typo is caught wherever it is written.
    Returns:
        props : dict of property name -> value (same type as T), keyed 'rho', 'h', 'cp',
                'mu', 'k' for every substance, plus 'sigma' for "Water" and every liquid
                metal. The keys do not depend on `formulation`.
    """
    if formulation not in (95, 97):
        raise ValueError(
            f"_getprop: unrecognized formulation {formulation!r} -- expected 95 "
            "(IAPWS-95, the default) or 97 (IAPWS-IF97)"
        )

    if substance in ("SCW", "Water"):
        if formulation == 95:
            rho = iapws.IAPWS95.rho_Tp(T, P)
            state = iapws.IAPWS95.helmholtz(rho, T)
            # Computed once and handed to lam(): the viscosity's critical enhancement
            # costs a second Helmholtz evaluation, and R15-11's conductivity divides by
            # that same mu.
            mu = iapws.IAPWS95.mu(state)

            props = {
                'rho': rho,
                'h': iapws.IAPWS95.h(state),
                'cp': iapws.IAPWS95.cp(state),
                'mu': mu,
                'k': iapws.IAPWS95.lam(state, mu=mu)
            }
        else:
            # Only the five shared keys, so that switching formulation cannot change the
            # shape of what a caller gets. _getprop97 also reports the region it used;
            # call it directly for that.
            props = {key: value for key, value in _getprop97(T, P).items()
                     if key in ('rho', 'h', 'cp', 'mu', 'k')}

        if substance == "Water":
            # Supercritical water has no interface, so only the subcritical branch gets a
            # surface tension. It comes from R1-76 either way -- it belongs to neither
            # equation of state, so `formulation` does not reach it.
            props['sigma'] = tp.SIGMA.sigma(T)

    elif substance in ("Lead", "Pb"):
        props = lm.Props(lm.Lead, T)

    elif substance in ("Sodium", "Na"):
        props = lm.Props(lm.Sodium, T)

    # liqprops implements LBE alongside Sodium and Lead, but this dispatcher used to
    # have no branch for it, so the one lead-bismuth model in the library was
    # unreachable through the interface every solver calls.
    elif substance in ("LBE", "PbBi"):
        props = lm.Props(lm.LBE, T)

    else:
        raise ValueError(
            f"_getprop: unrecognized substance {substance!r} -- expected one of "
            "'SCW', 'Water', 'Lead', 'Pb', 'Sodium', 'Na', 'LBE', 'PbBi'"
        )

    return props


# The (p, T) box each IF97 region equation may be evaluated over, as
# (p_lo, p_hi, T_lo, T_hi) in MPa and kelvin.
#
# These are NOT the region shapes -- the real boundaries are the saturation line and the
# B23 curve, and region() is what knows about those. _getprop97 evaluates a region on
# every point in the batch and selects afterwards, so a point that belongs to region 5
# still gets fed to region 1's equation, where (tau - 1.222) turns negative above 1134 K.
# Clamping into the box first keeps every unselected evaluation finite.
#
# The one rule these must obey is that each box CONTAINS its region, so the clamp is
# never active on a point that was actually selected. Getting that wrong is silent and
# expensive: region 3 reaches down to 16.5292 MPa at 623.15 K, and clamping its lower
# pressure bound to p_c = 22.064 instead -- as a first version of this table did -- moved
# the density at (623.3 K, 17 MPa) by 5.8 percent and at (640 K, 20 MPa) by a factor of
# three, because that state is on the vapor side of the dome and the clamp pushed it onto
# the liquid side.
#
# Only the temperature bounds are load-bearing for region 3; its equation is a Helmholtz
# energy and stays finite at any positive pressure, so its lower pressure bound is set as
# loosely as everyone else's.
_IF97_BOX = {
    1: (1.0e-8, 100.0, 273.15, f97.T_13),
    2: (1.0e-8, 100.0, 273.15, f97.T_25),
    3: (1.0e-8, 100.0, f97.T_13, f97.T_3max),
    5: (1.0e-8, f97.p_5max, f97.T_25, f97.T_max),
}


def _if97_state(code, p, T):
    """Evaluate one IF97 region at (p, T), clamped into that region's own evaluation box.

    Formulation:
        Not applicable -- dispatch, no physics of its own. See iapws97.R1/R2/R3/R5.

    Valid range:
        The clamp box in _IF97_BOX, which is wider than the region itself; see the
        comment there for why that is deliberate.

    Uncertainty:
        Not applicable.

    Reference:
        Not applicable.

    Inputs (1-D torch tensors, same shape):
        code : IF97 region number, 1, 2, 3 or 5
        p    : pressure, MPa
        T    : temperature, K

    Returns:
        cls : the region class (iapws97.R1, R2, R3 or R5)
        d   : that region's state dict, evaluated at the clamped (p, T)
    """
    p_lo, p_hi, T_lo, T_hi = _IF97_BOX[code]
    p_ = backend.clip(p, p_lo, p_hi)
    T_ = backend.clip(T, T_lo, T_hi)

    if code == 1:
        return f97.R1, f97.R1.gibbs(p_, T_)
    if code == 2:
        return f97.R2, f97.R2.gibbs(p_, T_)
    if code == 3:
        # The only region that costs an iteration: its basic equation takes density.
        return f97.R3, f97.R3.helmholtz(f97.R3.rho_pT(p_, T_), T_)
    return f97.R5, f97.R5.gibbs(p_, T_)


def _if97_props(cls, d):
    """Read the five 'Props' entries off one IF97 region state dict.

    Formulation:
        rho, h, cp come straight off the region equation. The two transport properties
        come from properties/iapws_transport.py, which needs (rho, T) plus (drho/dp)_T,
        cp and cv -- all of which the region equation also supplies.

        The viscosity is computed WITHOUT its critical enhancement. R12-08 Eq. (21) needs
        the isothermal compressibility at the state and again at T_R = 970.644 K at the
        same density, and there is no way to get the second one out of IF97: its
        equations are explicit in (p, T), so the pressure at (rho, 970.644 K) would
        itself need inverting. R12-08 Sec. 2.8 sanctions mu_2 = 1 for exactly this
        situation, and it costs nothing outside roughly 645.9 < T < 650.8 K.

        The thermal conductivity does keep its critical enhancement, because R15-11
        Eq. (25) supplies the reference compressibility as a fitted closed form rather
        than asking an equation of state for it. The caveat is that Eq. (18) divides by
        the viscosity, so where mu_2 would have mattered and was left out, lambda_2 comes
        out too large by the same proportion -- up to about 9 percent at rho_c. That
        band is the core of region 3, and iapws95.IAPWS95 is the right tool inside it.

    Valid range:
        Whatever the region class and the two transport releases validate.

    Uncertainty:
        That of the underlying releases, plus IF97's own deviation from IAPWS-95, plus
        the omitted mu_2 near the critical point as described above.

    Reference:
        IAPWS R7-97(2012); IAPWS R12-08; IAPWS R15-11.

    Inputs:
        cls : one of iapws97.R1, R2, R3, R5
        d   : that class's state dict

    Returns:
        props : dict keyed 'rho' [kg/m^3], 'h' [J/kg], 'cp' [J/kg-K], 'mu' [Pa-s],
                'k' [W/m-K]
    """
    # Every region reduces temperature by its own Tstar, so this recovers the same T the
    # state was built at -- including region 3, whose Tstar is the critical temperature.
    T = cls.Tstar / d['tau']

    rho = cls.rho(d)
    cp_kJ = cls.cp(d, units='kJ')
    cv_kJ = cls.cv(d, units='kJ')

    mu = tp.VISC.mu(rho, T)
    k = tp.COND.lam(rho, T, cls.drhodp(d), cp_kJ, cv_kJ, mu)

    return {'rho': rho, 'h': cls.h(d, units='J'), 'cp': cp_kJ * 1.0e3,
            'mu': mu, 'k': k}


def _getprop97(T, P, check_range=True):
    """Look up water properties from IAPWS-IF97, choosing the region from (T, P).

    Formulation:
        iapws97.region(P, T) labels each point 1, 2, 3, 5 (or 0, outside the
        formulation). Each region that the batch actually touches is then evaluated over
        the whole batch, on inputs clamped into that region's own evaluation box, and the
        answers are selected elementwise with where().

        Evaluating a region over the whole batch rather than over the points that belong
        to it keeps the array shapes fixed, which is what makes this work under torch on
        a GPU and keeps it differentiable: no boolean indexing, no data-dependent shapes,
        and every unselected branch finite because of the clamp. The cost is that a batch
        spanning two regions evaluates both equations everywhere. The one place that is
        expensive is region 3, whose density solve is sixty bisection steps, which is why
        a region is skipped entirely when no point in the batch is in it -- that test
        reads one boolean back to the host, and it changes how much work is done, never
        the answer.

        Properties come back in the same units and under the same keys as _getprop:
        'rho' kg/m^3, 'h' J/kg, 'cp' J/kg-K, 'mu' Pa-s, 'k' W/m-K, plus 'sigma' N/m and
        'region'. Points outside the formulation come back NaN rather than clamped to the
        nearest equation, with one warning per call.

    Valid range:
        The whole of IF97: 273.15 K to 1073.15 K at up to 100 MPa, and 1073.15 K to
        2273.15 K at up to 50 MPa. Anything else is region 0 and returns NaN.

        Not the whole of what the transport properties cover, and not the whole of what
        IAPWS-95 covers -- IF97 stops at 100 MPa where IAPWS-95 goes to 1000, and stops
        at 273.15 K where the saturation line goes down to the triple point.

    Uncertainty:
        See _if97_props. IF97 reproduces IAPWS-95 to the tolerances of R7-97 Sec. 12
        rather than exactly, and the region boundaries are visible as small
        discontinuities -- so this is a lookup, not something to differentiate through.

    Reference:
        IAPWS R7-97(2012), "Revised Release on the IAPWS Industrial Formulation 1997 for
        the Thermodynamic Properties of Water and Steam" (IAPWS_97.pdf).

    Inputs (float, numpy array, or torch tensor; broadcastable against each other):
        T           : temperature, K
        P           : pressure, MPa
        check_range : if True, warn when any point falls outside the formulation

    Returns:
        props : dict of property name -> value, each the same type and shape that T and P
                broadcast to, keyed 'rho', 'h', 'cp', 'mu', 'k', 'sigma' and 'region'
    """
    (T_, p_), state = w.prepare(T, P)
    region = f97.region(p_, T_)

    # NaN, not zero: a point outside the formulation has no answer, and a zero density
    # would propagate into a Reynolds number as a plausible-looking one.
    outside = backend.zeros_like(T_) + float('nan')
    props = {key: outside for key in ('rho', 'h', 'cp', 'mu', 'k')}

    for code in (1, 2, 3, 5):
        in_region = region == code
        # Skips work, never changes an answer -- see the Formulation note above.
        if not bool(in_region.any()):
            continue
        cls, d = _if97_state(code, p_, T_)
        got = _if97_props(cls, d)
        props = {key: backend.where(in_region, got[key], props[key]) for key in props}

    if check_range and bool((region == 0).any()):
        # One warning per call, never one per element -- the contract ranges.py sets.
        warnings.warn(
            "_getprop97: some states are outside IAPWS-IF97 entirely "
            "(273.15-1073.15 K below 100 MPa, 1073.15-2273.15 K below 50 MPa); "
            "those points are returned as NaN",
            RangeWarning, stacklevel=2)

    out = {key: w.restore(val, state) for key, val in props.items()}

    # Surface tension is a property of the saturation line, not of a region, so it is
    # added the same way _getprop's "Water" branch adds it. It is zero above T_c, where
    # there is no interface.
    out['sigma'] = tp.SIGMA.sigma(T)

    # Asked again on the caller's own inputs rather than converting the tensor labels
    # here: region() already knows how to hand an integer back in the right kind, and it
    # is two quadratics and a fourth root, so evaluating it twice costs nothing.
    out['region'] = f97.region(P, T)

    return out
