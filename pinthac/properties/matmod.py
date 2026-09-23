"""Fuel and cladding material property models: UO2, Zircaloy, HT-9 and (stub) D9 stainless."""
import math

import numpy as np

from pinthac import backend, ranges


def _erf(x):
    """Error function, dispatched by backend: torch has erf natively, numpy needs
    scipy.special.erf. Module-private -- only UO2.Theta_Klimenko needs it.

    Why this exists: backend.py deliberately stays small ("add helpers only where NumPy
    and Torch genuinely differ", CONTRIBUTING.md section 5) and erf is needed by exactly one
    function in the whole library, so the two-line dispatch lives here instead.

    Inputs:
        x : float, numpy array, or torch tensor
    Returns:
        erf(x), same type as x
    """
    xp = backend.lib(x)
    if xp is np:
        import scipy.special
        return scipy.special.erf(x)
    return xp.erf(x)


# Built from PNNL-35702's "Applicability and Uncertainty"
# subsections (one per model, cited by section number in the owning function's
# docstring) rather than from anything invented -- one plain dictionary per
# CONTRIBUTING.md section 7, keyed by the name each function's ranges.check() call below
# uses.
RANGES = {
    "uo2_k_nfi": {"T": (300.0, 2800.0), "Bu": (0.0, 90.0), "f_gad": (0.0, 0.10)},
    "uo2_eps": {"T": (300.0, 2500.0)},
    "uo2_thrm_expan": {"T": (300.0, None)},
    "uo2_swelling_solid": {"Bu": (0.0, 100.0)},
    "zircalloy_k": {"T": (285.0, 1770.0)},
    "zircalloy_cp": {"T": (285.0, 1300.0)},
    "zircalloy_thrm_expan_axial": {"T": (300.0, 1273.0)},
    "zircalloy_thrm_expan_diametral": {"T": (300.0, 1080.0)},
    "zircalloy_eps": {"T": (285.0, 1575.0), "t_ox": (0.0, 120.0E-6)},
    "zircalloy_E": {"T": (293.0, 1474.0), "phi": (0.0, 1.5E26)},
    "zircalloy_G": {"T": (293.0, 1474.0), "phi": (0.0, 1.5E26)},
    "zircalloy_meyer_hardness": {"T": (350.0, 875.0)},
    "zircalloy_axial_growth_high_fluence": {"phi": (0.0, 1.0E22)},
    "zircalloy_axial_growth_low_fluence": {"phi": (0.0, 8.5E21)},
    "zircalloy_creep": {"T": (570.0, 625.0), "sig": (40.0, 130.0)},
    "ht9_k": {"T": (293.0, 873.0)},
    "ht9_cp": {"T": (298.0, 873.0)},
    "ht9_thrm_expan": {"T": (298.15, 1073.15)},
    "ht9_E": {"T": (25.0, 600.0)},
    "ht9_G": {"T": (25.0, 600.0)},
    "ht9_yield_stress": {"T": (298.15, 873.15)},
    "ht9_creep": {"T": (298.15, 873.15)},
    "gas_k_He": {"T": (273.0, 2500.0)},
    "gas_k_Ar": {"T": (273.0, 2500.0)},
    "gas_k_N2": {"T": (273.0, 2500.0)},
    "gas_k_Kr": {"T": (273.0, 2300.0)},
    "gas_k_Xe": {"T": (273.0, 2200.0)},
    "gas_k_H2": {"T": (273.0, 2000.0)},
}


class Gas:
    def k(gas, T):
        """Fill-gas thermal conductivity, k = A * T^B per gas species.

        Formulation:
            k = A * T^B, with (A, B) tabulated per gas species below.

        Valid range:
            He, Ar, N2: 273 to 2500 K. Kr: 273 to 2300 K. Xe: 273 to 2200 K.
            H2: 273 to 2000 K. (PNNL-35702 section 4.1.3.) Air is in PNNL-35702's Table
            4-1 fitting-constant table (section 4.1.1) but has no range in section 4.1.3's
            applicability bullets -- not established for Air; see docs/OPEN_QUESTIONS.md
            (Q31).

        Uncertainty:
            Absolute standard error, PNNL-35702 section 4.1.3: He 8.99e-3, Ar 9.66e-4,
            Kr 8.86e-4, Xe 5.34e-4, H2 1.67e-2, N2 1.99e-3 W/m-K. Air: not established --
            see docs/OPEN_QUESTIONS.md (Q31), same gap as its valid range above.

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 4.1.1, Equation 4-1, Table 4-1.

        Inputs:
            gas : gas species symbol, string -- one of He, Ar, Kr, Xe, H2, N2, Air
            T   : temperature (float, numpy array, or torch tensor), K
        Returns:
            kval : gas thermal conductivity, W/m-K, same type as T
        """
        dic = {
                "He": [2.531E-3, 0.7146],
                "Ar": [4.092E-4, 0.6748],
                "Kr": [1.966E-4, 0.7006],
                "Xe": [9.825E-5, 0.7334],
                "H2": [1.349E-3, 0.8408],
                "N2": [2.984E-4, 0.7799],
                "Air": [1.945E-4, 0.8586]
            }
        A, B = dic[gas]
        # Air has no published range in MatLib_Info.pdf (see "Valid range" above), so
        # RANGES has no "gas_k_Air" entry and this lookup is simply a no-op for it --
        # ranges.check() only compares names present in its table.
        ranges.check("gas_k_" + gas, {"T": T}, RANGES.get("gas_k_" + gas, {}))
        kval = A * T**B
        return kval


class UO2:
    def k_NFI(T, Bu=0.0, f_gad=0.0):
        """Modified Frapcon-4 (NFI) model for UO2 thermal conductivity, accounting for
        burnup and gadolinia fraction.

        Formulation:
            f_Bu   = 0.00187 * Bu
            g_Bu   = 0.038 * Bu^0.28
            h_T    = 1 / (1 + 396*exp(-Q/T))
            K = 1 / (A + a*f_gad + B*T + f_Bu + (1 - 0.9*exp(-0.04*Bu))*g_Bu*h_T)
                + (E/T^2) * exp(-F/T)

        Valid range:
            Temperature 300 to 2800 K; gadolinia content 0 to 10 wt%; rod-average burnup
            0 to 90 GWd/MTU for UO2, 0 to 50 GWd/MTU for UO2-Gd2O3 (narrower burnup cap
            not separately enforced by RANGES, which uses the wider UO2 bound -- see
            PNNL-35702 section 2.1.1.3); as-fabricated density 90 to 98.6 %TD (this
            function returns k95 at the reference 95% TD and does not itself take a
            density argument, so the density bound is not checkable here).

        Uncertainty:
            Relative standard error, PNNL-35702 section 2.1.1.3: sigma = 8.3% for UO2,
            8.8% for UO2-Gd2O3 (f_gad > 0).

        Reference:
            Ohira, K., Itagaki, N. (1997), the Nuclear Fuel Industries (NFI) model,
            modified for burnup and gadolinia dependence by Lanning et al. (2005) -- both
            as cited in PNNL-35702 (Geelhood et al., 2024) section 2.1.1.1, Equations 2-1
            and 2-2.

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            T     : temperature, K
            Bu    : burnup, GWd/MTU (default 0.0, fresh fuel)
            f_gad : gadolinia weight fraction, dimensionless (default 0.0)
        Returns:
            K : thermal conductivity, W/m-K, same type as T
        """
        A = 0.0452
        B = 2.46E-4
        E = 3.5E9
        F = 16361
        a = 1.1599
        Q = 6380

        # Bu defaults to the plain Python float 0.0; torch.exp(-0.04*Bu) below then
        # raises TypeError on a tensor T unless Bu is promoted to match it first.
        T, Bu = backend.promote_all(T, Bu)
        xp = backend.lib(T, Bu)
        ranges.check("uo2_k_nfi", {"T": T, "Bu": Bu, "f_gad": f_gad}, RANGES["uo2_k_nfi"])
        f_Bu = 0.00187 * Bu
        g_Bu = 0.038 * Bu**(0.28)
        h_T = 1/(1+396*xp.exp(-Q/T))

        denom_1 = A + a*f_gad + B*T + f_Bu
        denom_2 = (1-0.9*xp.exp(-0.04*Bu))*g_Bu*h_T
        Term1 = 1/(denom_1+denom_2)
        Term2 = E/(T**2) * xp.exp(-F/T)
        K = Term1 + Term2
        return K

    def Theta_NFI(T, Bu=0.0, f_gad=0.0, T_ref=300.0, n=1000):
        """Fixed-step cumulative-trapezoid integral of UO2.k_NFI: Theta(T) = integral of
        k_NFI dT from T_ref to T, at fixed burnup and gadolinia content.

        Formulation:
            Theta(T) = integral_{T_ref}^{T} k_NFI(T', Bu, f_gad) dT', walked in n equal
            steps of the trapezoid rule (fixed iteration count, vectorized over the whole
            T/Bu/f_gad batch, in the same style as UO2.dens_B's bisection loop above):
                dT = (T - T_ref) / n
                Theta = sum over i=1..n of 0.5*(k_i-1 + k_i)*dT,  T_i = T_ref + i*dT
            Every step is ordinary tensor arithmetic through UO2.k_NFI itself -- no SciPy
            -- so the result carries a gradient with respect to T, Bu and f_gad under
            torch, unlike pin/annular.py::Ann_Theta's interp1d table. Because it calls
            k_NFI at every one of the n+1 nodes, k_NFI's own RANGES check (see its
            docstring) fires for any node outside 300-2800 K -- including, commonly, the
            T_ref=300 K starting node itself if T_ref is left at its default and the
            actual fuel state is colder, or the far end if T exceeds 2800 K.

        Valid range:
            Not established beyond what UO2.k_NFI's own docstring already states for the
            conductivity being integrated (T 300-2800 K, Bu 0-90 GWd/MTU, f_gad 0-0.10),
            which is enforced per-node through k_NFI's own ranges.check call rather than
            duplicated here.

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). tests/test_matmod.py
            checks this integral against scipy.integrate.quad of k_NFI itself instead,
            the same verification approach as Theta_Klimenko,
            item 3.

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31); same as k_NFI, which
            this integrates. The trapezoid scheme itself is a standard numerical method,
            not something drawn from a source that needs citing.

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            T     : upper integration limit (temperature), K
            Bu    : burnup, GWd/MTU (default 0.0, fresh fuel; held fixed over the
                    integration -- Theta is not an integral over burnup)
            f_gad : gadolinia weight fraction, dimensionless (default 0.0; likewise held
                    fixed)
            T_ref : lower integration limit / Theta's zero point, K (default 300.0)
            n     : number of fixed trapezoid steps, int (default 1000)
        Returns:
            Theta : integral of k_NFI from T_ref to T, W/m, same type as T
        """
        T, Bu, f_gad = backend.promote_all(T, Bu, f_gad)
        Theta = backend.zeros_like(T)
        T_prev = T_ref
        k_prev = UO2.k_NFI(T_prev, Bu, f_gad)
        for i in range(1, n+1):
            frac = i/n
            T_i = T_ref + frac*(T-T_ref)
            k_i = UO2.k_NFI(T_i, Bu, f_gad)
            Theta = Theta + 0.5*(k_prev+k_i)*(T_i-T_prev)
            T_prev, k_prev = T_i, k_i
        return Theta

    def k_Klimenko(T):
        """Klimenko-Zorin model for the thermal conductivity of 95%-dense UO2 fuel.

        Formulation:
            tau = T/1000
            k = 100 / (7.5408 + 17.692*tau + 3.6142*tau^2) + 6400*tau^(-5/2)*exp(-16.35/tau)

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Checked
            MatLib_Info.pdf (PNNL-35702) in full and
            Hughes_SCWR_1.pdf: neither names "Klimenko-Zorin" or contains
            this formula's coefficients (7.5408, 17.692, 3.6142, 6400, 16.35). Hughes
            (2014) Eq. (14) uses the same functional family without naming its source
            either (see the model references).

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            kval : thermal conductivity, W/m-K, same type as T
        """
        xp = backend.lib(T)
        tau = T/1000
        denom = 7.5408+17.692*tau+3.6142*tau**2
        exp_term = xp.exp(-16.35/tau)
        kval = 100/denom + 6400*exp_term * tau**(-5/2)
        return kval

    def Theta_Klimenko(T):
        """Analytic antiderivative of UO2.k_Klimenko: Theta(T) = integral of k dT.

        Formulation:
            An antiderivative, not a definite integral referenced to any particular T --
            only differences Theta(T2) - Theta(T1) are physical. The rational term
            integrates by partial fractions (the quadratic 7.5408+17.692*tau+3.6142*tau^2
            factors as 3.6142*(tau+0.471675)*(tau+4.42356)); the exponential term
            integrates by substituting s = 1/tau and integrating by parts once, which is
            where the erf term below comes from:
                tau = T/1000, a = 16.35
                Theta = 1000 * [ 7.00155*ln((tau+0.471675)/(tau+4.42356))
                                  + 6400*( exp(-a/tau)/(a*sqrt(tau))
                                           - sqrt(pi/a)/(2*a) * erf(sqrt(a/tau)) ) ]
            T is clamped to a 1 K floor first, matching Kint -- not folded, because
            folding a negative excursion onto its positive mirror would create a
            spurious extra root (Kint(-T) == Kint(T) either way; clamping keeps it
            monotonic instead).

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31); same caveat as
            k_Klimenko, which this integrates.

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). tests/test_matmod.py
            checks this integral against scipy.integrate.quad of k_Klimenko itself
            instead,
            verification, not a substitute for a published uncertainty.

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31); same as k_Klimenko. The
            antiderivative's coefficients were re-derived and numerically checked in
            commit 14172e4 (pinthac/sca/rod.py::Kint), not taken from a published source.

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : Theta(T), W/m, same type as T
        """
        T = backend.clip(T, lo=1.0)
        xp = backend.lib(T)
        tau = T/1000
        a = 16.35
        Term1 = 7.00155 * xp.log((tau+0.471675)/(tau+4.42356))
        const = math.sqrt(math.pi/a) / (2*a)
        Term2 = 6400 * (xp.exp(-a/tau)/(a*xp.sqrt(tau)) - const*_erf(xp.sqrt(a/tau)))
        val = 1000*(Term1+Term2)
        return val

    def eps(T):
        """Emissivity of UO2 fuel.

        Formulation:
            eps = 0.7856 + 1.5263e-5 * T

        Valid range:
            Gadolinia content 0 to 10 wt%; temperature 300 to 2500 K (PNNL-35702 section
            2.1.5.3). Rod-average burnup and as-fabricated density: no dependence
            observed.

        Uncertainty:
            sigma = 0.072, absolute standard error (PNNL-35702 section 2.1.5.3). This
            replaces the "+/- 6.8 percent" this docstring stated before development, whose
            source was never established (Q31) and which is not obviously the same
            number: eps itself only spans about 0.79 to 0.82 over 300-2500 K, so a 0.072
            absolute band and a 6.8 percent relative band are not equivalent here. Using
            PNNL-35702's figure since it is the one with a traceable source.

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 2.1.5.1, Equation 2-12. That
            equation's constant is 0.78557; this function uses 0.7856 (matches to the
            4th decimal, source of the small difference not established -- preserved as
            found per CONTRIBUTING.md, "change no physics").

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : emissivity, dimensionless, same type as T
        """
        ranges.check("uo2_eps", {"T": T}, RANGES["uo2_eps"])
        a = 0.7856
        b = 1.5263E-5
        val = a + b*T
        return val

    def thrm_expan(T):
        """Thermal expansion strain of solid UO2 fuel.

        Formulation:
            strain = K1*T - K2 + K3*exp(-Ed/(k*T))

        Valid range:
            Fuel types UO2, UO2-Gd2O3, MOX; gadolinia content 0 to 10 wt%; temperature
            300 K up to the fuel melting temperature (PNNL-35702 section 2.1.4.3 refers to
            its own melting-temperature model in section 2.1.3, which this module does not
            implement, so no numeric upper bound is enforced by RANGES here -- comparison
            data in section 2.1.4.2 covers room temperature to ~3000 K). Rod-average
            burnup and as-fabricated density: no dependence observed.

        Uncertainty:
            Relative standard error, PNNL-35702 section 2.1.4.3: sigma = 10.3% for UO2 and
            UO2-Gd2O3 (this module's fuel types).

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 2.1.4.1, Equation 2-9, Table 2-2
            (UO2/UO2-Gd2O3 column: K1, K2, K3, ED match exactly).

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            strain : linear thermal strain, dimensionless, same type as T
        """
        ranges.check("uo2_thrm_expan", {"T": T}, RANGES["uo2_thrm_expan"])
        K1 = 9.8E-6
        K2 = 2.61E-3
        K3 = 3.16E-1
        Ed = 1.32E-19
        k = 1.38E-23
        xp = backend.lib(T)
        strain = K1*T - K2 + K3*xp.exp(-Ed/(k*T))
        return strain

    def dens_max(T, rho_TD=95.0, T_sint=1873.15):
        """Maximum in-reactor pellet dimension change from densification.

        Formulation:
            Below 1000 K: -22.2*(100 - rho_TD)/(T_sint - 1453.15)
            At or above 1000 K: -66.6*(100 - rho_TD)/(T_sint - 1453.15)

        Valid range:
            Not established -- checked PNNL-35702 (Geelhood et al., 2024) section 2.1.7.3
            in full: it states the correlation is "applicable to the range of available
            data (i.e., fuels with pore size distributions similar to those included in
            the [Freshley et al., 1976] study)" without giving numeric bounds. See
            docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- PNNL-35702 section 2.1.7.3 states explicitly "Due to the
            scatter in the experimental data, it is difficult to establish a meaningful
            measure of uncertainty." See docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Rolstad et al. (1974), via PNNL-35702 (Geelhood et al., 2024) section 2.1.7.1,
            Equation 2-16 -- this is the "for rho_sint = 0 kg/m^3" (sintering-temperature-
            based) branch; this module does not take a re-sintering density change input,
            so the alternative "rho_sint > 0" branch in the same equation is not used.

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            T      : fuel temperature, K
            rho_TD : as-fabricated density, percent of theoretical density (default 95.0)
            T_sint : fabrication sintering temperature, K (default 1873.15)
        Returns:
            val : maximum dimension change, percent, same type as T
        """
        low = -22.2*(100-rho_TD)/(T_sint-1453.15)
        high = -66.6*(100-rho_TD)/(T_sint-1453.15)
        val = backend.where(T < 1000, low, high)
        return val

    def dens_B(dL_max, n_iter=100):
        """Offset constant B in the densification model, solved by bisection so that
        dL/L = 0 at zero burnup.

        Formulation:
            Root of exp(-3*B) + 2*exp(-35*B) - (-dL_max) = 0 in B, found by n_iter steps
            of bisection on [0, 50].

        Valid range:
            Not established -- see UO2.dens_max's docstring; same PNNL-35702 section
            2.1.7.3 checked, same "applicable to the range of available data" statement
            with no numeric bounds, since B is solved from dens_max's output.

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31); n_iter=100 bisection
            steps on [0, 50] resolve B to well below double precision's useful digits.

        Reference:
            Rolstad et al. (1974), via PNNL-35702 (Geelhood et al., 2024) section 2.1.7.1,
            Equation 2-15 -- B is the "constant determined by the code to fit the boundary
            condition; dL/L = 0 when Bu = 0" that equation calls for; PNNL-35702 does not
            give a closed form for it either, consistent with this module's bisection.

        Inputs:
            dL_max : maximum dimension change from UO2.dens_max, percent (float, numpy
                     array, or torch tensor)
            n_iter : number of bisection iterations, int (default 100)
        Returns:
            Bval : offset constant, same type as dL_max
        """
        xp = backend.lib(dL_max)
        target = -dL_max
        lo = backend.zeros_like(target)
        hi = backend.zeros_like(target) + 50.0
        for i in range(n_iter):
            mid = 0.5*(lo+hi)
            f = xp.exp(-3*mid) + 2*xp.exp(-35*mid) - target
            lo = backend.where(f > 0, mid, lo)
            hi = backend.where(f > 0, hi, mid)
        Bval = 0.5*(lo+hi)
        return Bval

    def densification(T, Bu, rho_TD=95.0, T_sint=1873.15):
        """Rolstad model for the densification of UO2, MOX and UO2-Gd2O3 fuel.

        Formulation:
            dL_max = UO2.dens_max(T, rho_TD, T_sint)
            B      = UO2.dens_B(dL_max)
            val    = dL_max + exp(-3*(Bu+B)) + 2*exp(-35*(Bu+B))

        Valid range:
            Not established -- see UO2.dens_max's docstring; same PNNL-35702 section
            2.1.7.3 checked, no numeric bounds given.

        Uncertainty:
            Not established -- see UO2.dens_max's docstring; PNNL-35702 section 2.1.7.3
            states uncertainty could not be meaningfully established for this model.

        Reference:
            Rolstad et al. (1974), via PNNL-35702 (Geelhood et al., 2024) section 2.1.7.1,
            Equation 2-15.

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            T      : fuel temperature, K
            Bu     : burnup, MWd/kgU
            rho_TD : as-fabricated density, percent of theoretical density (default 95.0)
            T_sint : fabrication sintering temperature, K (default 1873.15)
        Returns:
            val : dimension change, percent, same type as T
        """
        # Promote before any arithmetic touches these: each of them is routinely the
        # scalar while another is a whole batch, and a numpy intermediate formed from
        # the scalar cannot afterwards combine with the tensor.
        T, Bu = backend.promote_all(T, Bu)
        xp = backend.lib(T, Bu)
        dL_max = UO2.dens_max(T, rho_TD, T_sint)
        B = UO2.dens_B(dL_max)
        Term1 = xp.exp(-3*(Bu+B))
        Term2 = 2*xp.exp(-35*(Bu+B))
        val = dL_max + Term1 + Term2
        return val

    def swelling_solid(Bu, gad=False):
        """Solid fission-product swelling of UO2, MOX and UO2-Gd2O3 fuel.

        Formulation:
            Gadolinia-bearing fuel: val = 0.0005*Bu
            Otherwise, piecewise in Bu (GWd/MTU):
                Bu <= 6:            val = 0
                6 < Bu <= 80:       val = 0.00062*(Bu-6)
                Bu > 80:            val = 0.00062*(80-6) + 0.00086*(Bu-80)

        Valid range:
            Fuel types UO2, PuO2, MOX, UO2-Gd2O3; gadolinia content 0 to 10 wt%;
            rod-average burnup 0 to 100 GWd/MTU; as-fabricated density 90 to 98 %TD; no
            temperature dependence over the entire temperature range (PNNL-35702 section
            2.1.8.3).

        Uncertainty:
            Absolute standard error per 1 GWd/MTU, PNNL-35702 section 2.1.8.3 (given for
            Bu < 80 GWd/MTU): sigma = 0.00008 dV/V for UO2/MOX in one bullet and sigma =
            0.00016 dV/V for UO2/MOX in a second, separate bullet -- both stated for the
            same "Bu < 80 GWd/MTU" condition in the extracted text, so which applies where
            is not established from this document alone; UO2-Gd2O3: sigma = 0.00008 dV/V.
            See docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 2.1.8.1, Equations 2-17 (UO2/MOX)
            and 2-18 (UO2-Gd2O3, the gad=True branch here).

        Inputs:
            Bu  : pellet average burnup, GWd/MTU (float, numpy array, or torch tensor)
            gad : True for gadolinia-bearing fuel (bool, default False; a structural
                  choice, not a value to batch over -- see backend.is_torch's docstring)
        Returns:
            val : swelling, dimensionless volume fraction, same type as Bu
        """
        ranges.check("uo2_swelling_solid", {"Bu": Bu}, RANGES["uo2_swelling_solid"])
        if gad:
            val = 0.0005*Bu
            return val
        low = backend.zeros_like(Bu)
        mid = 0.00062*(Bu-6)
        high = 0.00062*(80-6) + 0.00086*(Bu-80)
        val = backend.where(Bu <= 6, low, backend.where(Bu <= 80, mid, high))
        return val

    def swelling_gas(T, Bu):
        """Gaseous fission-product swelling of UO2, UO2-Gd2O3 and MOX fuel.

        Formulation:
            Nonzero only for Bu >= 40 (ramped linearly in from 40 to 50 GWd/MTU) and
            1233 K <= T <= 2105 K:
                1233 <= T < 1643:  shape = -4.37e-2 + 4.55e-5*T
                1643 <= T <= 2105: shape =  7.40e-2 - 4.05e-5*T
                otherwise:         shape = 0
            val = ramp(Bu) * shape(T)

        Valid range:
            Active window is 1233 K <= T <= 2105 K and Bu >= 40 GWd/MTU (below 40, and
            outside the temperature window, the correlation returns zero by design
            rather than being invalid -- see the original docstring). No RANGES check is
            registered on T here for that reason: a warning on every normal (below-window)
            temperature would be a false positive, not a real range violation. PNNL-35702
            section 2.1.8.3 additionally states this swelling model (both the solid and
            gaseous terms together) is applicable over rod-average burnup 0 to 100
            GWd/MTU, gadolinia content 0 to 10 wt%, and as-fabricated density 90 to 98 %TD.

        Uncertainty:
            Not established for the gaseous term specifically -- PNNL-35702 section
            2.1.8.3 states an uncertainty only for the solid-swelling dV/V term (see
            UO2.swelling_solid's docstring); no analogous number is given for this
            gaseous dL/L term. See docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 2.1.8.1, Equations 2-19 through
            2-21.

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            T  : pellet ring temperature, K
            Bu : pellet average burnup, GWd/MTU
        Returns:
            val : swelling, dimensionless volume fraction, same type as T
        """
        T, Bu = backend.promote_all(T, Bu)
        low = -4.37E-2 + 4.55E-5*T
        high = 7.40E-2 - 4.05E-5*T
        zero = backend.zeros_like(T)
        shape = backend.where((T >= 1233) & (T < 1643), low,
                 backend.where((T >= 1643) & (T <= 2105), high, zero))
        ramp = backend.where(Bu < 40, 0.0, backend.where(Bu < 50, (Bu-40)/10, 1.0))
        val = ramp*shape
        return val


class Zircalloy:
    def k(T):
        """Thermal conductivity of Zircaloy-4, Zircaloy-2, M5, ZIRLO and Optimized ZIRLO.

        Formulation:
            Below 2098 K: k = 7.511 + 2.088e-2*T - 1.45e-5*T^2 + 7.668e-9*T^3
            At or above 2098 K: k = 36.0 (alpha-to-beta phase transition plateau)

        Valid range:
            285 to 1770 K (PNNL-35702 section 3.1.1.3). Rod-average burnup: no
            dependence observed.

        Uncertainty:
            sigma = 1.9 W/m-K, absolute (PNNL-35702 section 3.1.1.3, matches the value
            this docstring already stated).

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 3.1.1.1, Equations 3-1 and 3-2.

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : thermal conductivity, W/m-K, same type as T
        """
        ranges.check("zircalloy_k", {"T": T}, RANGES["zircalloy_k"])
        poly = 7.511 + 2.088E-2*T - 1.45E-5*T**2 + 7.668E-9*T**3
        val = backend.where(T >= 2098, 36.0, poly)
        return val

    def cp(T):
        """Specific heat capacity of Zircaloy-4, Zircaloy-2, M5, ZIRLO and Optimized ZIRLO.

        Formulation:
            Piecewise-linear interpolation of a 14-point T (290-1248 K) vs. Cp table,
            held flat outside the table (see pinthac.backend.interp).

        Valid range:
            290 K to 1248 K (the table's own extent, matching Table 3-2 exactly);
            extrapolated flat outside it. PNNL-35702 section 3.1.2.3's applicability
            bullet states a slightly wider 285 to 1300 K, which RANGES uses since it is
            the document's own stated validity range rather than merely the table's
            point extent -- the two numbers are both PNNL-35702's, for two different
            questions (where the table has data vs. where the model is considered valid).

        Uncertainty:
            Absolute standard error, PNNL-35702 section 3.1.2.3: sigma = 10 J/kg-K below
            1090 K, 25 J/kg-K between 1090 and 1248 K, 100 J/kg-K above 1248 K.

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 3.1.2.1, Table 3-2.

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : specific heat capacity, J/kg-K, same type as T
        """
        ranges.check("zircalloy_cp", {"T": T}, RANGES["zircalloy_cp"])
        T_tab = [290.0, 300.0, 400.0, 640.0, 1090.0, 1093.0, 1113.0, 1133.0,
                 1153.0, 1173.0, 1193.0, 1213.0, 1233.0, 1248.0]
        C_tab = [279.0, 281.0, 302.0, 331.0, 375.0, 502.0, 590.0, 615.0,
                 719.0, 816.0, 770.0, 619.0, 469.0, 356.0]
        val = backend.interp(T, T_tab, C_tab)
        return val

    def T_melt():
        """Melting temperature of Zircaloy-4, Zircaloy-2, M5, ZIRLO and Optimized ZIRLO.

        Formulation:
            Constant, 2123.15 K.

        Valid range:
            Not applicable -- a constant.

        Uncertainty:
            Not established -- PNNL-35702 (Geelhood et al., 2024) section 3.1.3.3 states
            explicitly "No uncertainty is given on the melting temperature." See
            docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 3.1.3.1, Equation 3-3.

        Inputs:
            None.
        Returns:
            val : melting temperature, K, Python float
        """
        val = 2123.15
        return val

    def rho():
        """Density of Zircaloy-4, Zircaloy-2, M5, ZIRLO and Optimized ZIRLO.

        Formulation:
            Constant, 6520.0 kg/m^3.

        Valid range:
            Not applicable -- a constant.

        Uncertainty:
            Not established -- PNNL-35702 (Geelhood et al., 2024) section 3.1.6.2 states
            "No uncertainty is given on the density." See docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 3.1.6.1, Equation 3-8.

        Inputs:
            None.
        Returns:
            val : density, kg/m^3, Python float
        """
        val = 6520.0
        return val

    def thrm_expan_axial(T):
        """Axial thermal expansion of Zircaloy-4, Zircaloy-2, M5, ZIRLO and Optimized ZIRLO.

        Formulation:
            Below 1073.15 K:  strain = -2.5060e-5 + 4.4410e-6*(T-273.15)
            Above 1273.15 K:  strain = -8.300e-3 + 9.70e-6*(T-273.15)
            In between: piecewise-linear interpolation of a 21-point table over the
            alpha-to-beta transition (see pinthac.backend.interp).

        Valid range:
            300 to 1273 K for axial expansion (PNNL-35702 section 3.1.4.3; the piecewise
            branches below cover 280-1073.15 K and >=1273.15 K, with the 1073.15-1273.15 K
            middle covered by the lookup table -- PNNL-35702 states 280 K as the low-end
            formula validity but 300 K as the applicability bullet's stated floor).
            Rod-average burnup: no dependence observed.

        Uncertainty:
            sigma = 4.8e-5 m/m, absolute (PNNL-35702 section 3.1.4.3, matches the value
            this docstring already stated).

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 3.1.4.1, Equation 3-4, Table 3-3.

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : axial linear strain, dimensionless, same type as T
        """
        ranges.check("zircalloy_thrm_expan_axial", {"T": T},
                      RANGES["zircalloy_thrm_expan_axial"])
        T_tab = [1073.15, 1083.15, 1093.15, 1103.15, 1113.15, 1123.15,
                 1133.15, 1143.15, 1153.15, 1163.15, 1173.15, 1183.15,
                 1193.15, 1203.15, 1213.15, 1223.15, 1233.15, 1243.15,
                 1253.15, 1263.15, 1273.15]
        e_tab = [0.00352774, 0.00353000, 0.00350000, 0.00346000, 0.00341000,
                 0.00333000, 0.00321000, 0.00307000, 0.00280000, 0.00250000,
                 0.00200000, 0.00150000, 0.00130000, 0.00116000, 0.00113000,
                 0.00110000, 0.00111000, 0.00113000, 0.00120000, 0.00130000,
                 0.00140000]
        low = -2.5060E-5 + 4.4410E-6*(T-273.15)
        high = -8.300E-3 + 9.70E-6*(T-273.15)
        table = backend.interp(T, T_tab, e_tab)
        val = backend.where(T <= 1073.15, low, backend.where(T >= 1273.15, high, table))
        return val

    def thrm_expan_diametral(T):
        """Circumferential (diametral) thermal expansion of Zircaloy-4, Zircaloy-2, M5,
        ZIRLO and Optimized ZIRLO.

        Formulation:
            Below 1073.15 K:  strain = -2.3730e-4 + 6.7210e-6*(T-273.15)
            Above 1273.15 K:  strain = -6.800e-3 + 9.70e-6*(T-273.15)
            In between: piecewise-linear interpolation of a 21-point table over the
            alpha-to-beta transition (see pinthac.backend.interp).

        Valid range:
            300 to 1080 K for circumferential expansion (PNNL-35702 section 3.1.4.3; see
            thrm_expan_axial's docstring for the same 280 K vs. 300 K floor note, which
            applies here identically). Rod-average burnup: no dependence observed.

        Uncertainty:
            sigma = 4.6e-4 m/m, absolute (PNNL-35702 section 3.1.4.3, matches the value
            this docstring already stated).

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 3.1.4.1, Equation 3-5, Table 3-3.

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : circumferential linear strain, dimensionless, same type as T
        """
        ranges.check("zircalloy_thrm_expan_diametral", {"T": T},
                      RANGES["zircalloy_thrm_expan_diametral"])
        T_tab = [1073.15, 1083.15, 1093.15, 1103.15, 1113.15, 1123.15,
                 1133.15, 1143.15, 1153.15, 1163.15, 1173.15, 1183.15,
                 1193.15, 1203.15, 1213.15, 1223.15, 1233.15, 1243.15,
                 1253.15, 1263.15, 1273.15]
        e_tab = [0.00513950, 0.00522000, 0.00525000, 0.00528000, 0.00528000,
                 0.00524000, 0.00522000, 0.00515000, 0.00508000, 0.00490000,
                 0.00470000, 0.00445000, 0.00410000, 0.00350000, 0.00313000,
                 0.00297000, 0.00292000, 0.00287000, 0.00286000, 0.00288000,
                 0.00290000]
        low = -2.3730E-4 + 6.7210E-6*(T-273.15)
        high = -6.800E-3 + 9.70E-6*(T-273.15)
        table = backend.interp(T, T_tab, e_tab)
        val = backend.where(T <= 1073.15, low, backend.where(T >= 1273.15, high, table))
        return val

    def eps(T, t_ox=0.0):
        """Emissivity of Zircaloy-4, Zircaloy-2, M5, ZIRLO and Optimized ZIRLO.

        Formulation:
            eps_1 = 0.325 + 1.246e5*t_ox            if t_ox < 3.88e-6
                  = 0.808642 - 50.0*t_ox             otherwise
            eps_2 = max(0.325, exp((1500-T)/300) * eps_1)
            val   = eps_2 if T > 1500 else eps_1

        Valid range:
            285 to 1575 K; oxide thickness 0 to 120 micrometers (PNNL-35702 section
            3.1.5.3). Rod-average burnup: no dependence observed.

        Uncertainty:
            sigma = 0.054, absolute (PNNL-35702 section 3.1.5.3, matches the value this
            docstring already stated).

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 3.1.5.1, Equations 3-6 and 3-7.

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            T    : temperature, K
            t_ox : inner surface oxide thickness, m (default 0.0)
        Returns:
            val : emissivity, dimensionless, same type as T
        """
        ranges.check("zircalloy_eps", {"T": T, "t_ox": t_ox}, RANGES["zircalloy_eps"])
        xp = backend.lib(T)
        T, t_ox = backend.promote_all(T, t_ox)
        eps_1 = backend.where(t_ox < 3.88E-6, 0.325+0.1246E6*t_ox,
                                       0.808642-50.0*t_ox)
        eps_2 = backend.maximum(0.325, xp.exp((1500-T)/300)*eps_1)
        val = backend.where(T > 1500, eps_2, eps_1)
        return val

    def E(T, CW=0.0, phi=0.0, d_ox=0.0012):
        """Young's modulus of Zircaloy-4, Zircaloy-2, M5, ZIRLO and Optimized ZIRLO.

        Formulation:
            c2 = 0.88 + 0.12*exp(-phi/1e25)
            Below 1090 K:
                E = (1.088e11 - 5.475e7*T + (6.61e11+5.912e8*T)*d_ox - 2.6e10*CW) / c2
            Above 1255 K:
                E = 9.21e10 - 4.05e7*T
            In between: linear interpolation between the two branches' values at
            1090 K and 1255 K.

        Valid range:
            293 to 1474 K; fast neutron flux up to 1.5e26 n/m^2 (PNNL-35702 section
            3.1.7.3 -- stated as a single value, "Fast neutron flux: 1.5e26 n/m2", read
            here as an upper bound on phi since the formula's phi dependence is
            monotonic in that direction).

        Uncertainty:
            sigma = 3.1e9 Pa, absolute (PNNL-35702 section 3.1.7.3).

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 3.1.7.1, Equation 3-9. d_ox's
            default (0.0012) matches the document's own statement that this is "hardwired
            to 0.0012 in MatLib".

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            T    : temperature, K
            CW   : effective cold work, ratio of areas (default 0.0)
            phi  : fast neutron (>1 MeV) fluence, n/m^2 (default 0.0)
            d_ox : average oxygen concentration, kg-O/kg-Zr (default 0.0012)
        Returns:
            val : Young's modulus, Pa, same type as T
        """
        ranges.check("zircalloy_E", {"T": T, "phi": phi}, RANGES["zircalloy_E"])
        xp = backend.lib(T)
        # phi defaults to the plain float 0.0; torch.exp rejects a bare float even when
        # T is a tensor, same failure mode as k_NFI's Bu (see that docstring's note).
        T, phi = backend.promote_all(T, phi)
        c2 = 0.88 + 0.12*xp.exp(-phi/1E25)
        c3 = -2.6E10
        low = (1.088E11 - 5.475E7*T + (6.61E11+5.912E8*T)*d_ox + c3*CW)/c2
        high = 9.21E10 - 4.05E7*T
        E_1090 = (1.088E11 - 5.475E7*1090.0
                  + (6.61E11+5.912E8*1090.0)*d_ox + c3*CW)/c2
        E_1255 = 9.21E10 - 4.05E7*1255.0
        mid = E_1090 + (E_1255-E_1090)*(T-1090.0)/(1255.0-1090.0)
        val = backend.where(T < 1090, low, backend.where(T > 1255, high, mid))
        return val

    def G(T, CW=0.0, phi=0.0, d_ox=0.0012):
        """Shear modulus of Zircaloy-4, Zircaloy-2, M5, ZIRLO and Optimized ZIRLO.

        Formulation:
            c2 = 0.88 + 0.12*exp(-phi/1e25)
            Below 1090 K:
                G = (4.04e10 - 2.168e7*T + (7.07e11-2.315e8*T)*d_ox + c3) / c2
            Above 1255 K:
                G = 3.49e10 - 1.66e7*T
            In between: linear interpolation between the two branches' values at
            1090 K and 1255 K.

        Valid range:
            293 to 1474 K; fast neutron flux up to 1.5e26 n/m^2 (PNNL-35702 section
            3.1.7.3, same figure and the same "single value read as an upper bound" note
            as Zircalloy.E -- MatLib gives one combined Applicability subsection for both
            Young's and shear modulus).

        Uncertainty:
            sigma = 6.2e9 Pa, absolute (PNNL-35702 section 3.1.7.3, "assumed to be twice
            that of the calculated Young's modulus").

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 3.1.7.1, Equation 3-10.

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            T    : temperature, K
            CW   : effective cold work, ratio of areas (default 0.0)
            phi  : fast neutron (>1 MeV) fluence, n/m^2 (default 0.0)
            d_ox : average oxygen concentration, kg-O/kg-Zr (default 0.0012)
        Returns:
            val : shear modulus, Pa, same type as T
        """
        ranges.check("zircalloy_G", {"T": T, "phi": phi}, RANGES["zircalloy_G"])
        xp = backend.lib(T)
        # phi defaults to the plain float 0.0; same fix as Zircalloy.E, above.
        T, phi = backend.promote_all(T, phi)
        c2 = 0.88 + 0.12*xp.exp(-phi/1E25)
        c3 = -0.867E10
        low = (4.04E10 - 2.168E7*T + (7.07E11-2.315E8*T)*d_ox + c3)/c2
        high = 3.49E10 - 1.66E7*T
        G_1090 = (4.04E10 - 2.168E7*1090.0
                  + (7.07E11-2.315E8*1090.0)*d_ox + c3)/c2
        G_1255 = 3.49E10 - 1.66E7*1255.0
        mid = G_1090 + (G_1255-G_1090)*(T-1090.0)/(1255.0-1090.0)
        val = backend.where(T < 1090, low, backend.where(T > 1255, high, mid))
        return val

    def meyer_hardness(T):
        """Meyer's hardness of Zircaloy-2, Zircaloy-4, M5, ZIRLO and Optimized ZIRLO.

        Formulation:
            Below or at 1235 K: val = exp(26.034 - 2.6394e-2*T + 4.3502e-5*T^2
                                             - 2.5621e-8*T^3)
            Above 1235 K: val = 1.0e5 (saturation plateau)

        Valid range:
            350 to 875 K (PNNL-35702 section 3.1.8.3). Rod-average burnup: no dependence
            observed.

        Uncertainty:
            Not established -- PNNL-35702 (Geelhood et al., 2024) section 3.1.8.3 states
            explicitly "An estimate of the uncertainty in this correlation has not been
            established due to the limited data." See docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 3.1.8.1, Equation 3-11.

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : Meyer's hardness (units as published), same type as T
        """
        ranges.check("zircalloy_meyer_hardness", {"T": T}, RANGES["zircalloy_meyer_hardness"])
        xp = backend.lib(T)
        arg = (26.034 - 2.6394E-2*T + 4.3502E-5*T**2 - 2.5621E-8*T**3)
        val = backend.where(T <= 1235, xp.exp(arg), 1.0E5)
        return val

    def axial_growth(alloy, phi):
        """Axial irradiation growth of zirconium-based fuel-rod cladding.

        Formulation:
            val = A * phi^n, with (A, n) tabulated per alloy below.

        Valid range:
            Temperature 530 to 620 K (not an input to this function, so not enforceable
            here -- stated for completeness); fast neutron fluence 0 to 1e22 n/cm^2 for
            Zircaloy-2 and M5, 0 to 8.5e21 n/cm^2 for Zircaloy-4, ZIRLO and Optimized
            ZIRLO (PNNL-35702 section 3.1.9.3). Confirmed applicable to fuel rod cladding
            only, matching what this docstring already said: "these correlations are only
            valid for fuel rod cladding axial irradiation growth and may not represent
            guide tube growth" (PNNL-35702 section 3.1.9, introductory paragraph).

        Uncertainty:
            Relative standard error unless noted, PNNL-35702 section 3.1.9.3: Zircaloy-2
            sigma = 20.9%, Zircaloy-4 sigma = 22.3%, M5 sigma = 18.6%; ZIRLO and Optimized
            ZIRLO sigma = 0.0005 m/m absolute (stated as constant with fluence for these
            two, unlike the other three).

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 3.1.9.1, Equations 3-12 through
            3-15. Section 3.1.9.2 also notes Optimized ZIRLO reuses the ZIRLO correlation
            directly (proprietary data showed similar or slightly lower growth), which is
            why this table gives them the same (A, n) pair.

        Inputs:
            alloy : cladding alloy, string -- one of Zircaloy-2, Zircaloy-4, ZIRLO,
                    Optimized ZIRLO, M5
            phi   : fast neutron (>1 MeV) fluence, n/cm^2 (float, numpy array, or torch
                    tensor)
        Returns:
            val : axial growth strain, dimensionless, same type as phi
        """
        dic = {
                "Zircaloy-2": [1.09E-21, 0.845],
                "Zircaloy-4": [2.18E-21, 0.845],
                "ZIRLO": [9.7893E-25, 0.98239],
                "Optimized ZIRLO": [9.7893E-25, 0.98239],
                "M5": [7.013E-21, 0.81787]
            }
        A, n = dic[alloy]
        # Two fluence caps apply (PNNL-35702 section 3.1.9.3): the wider one for
        # Zircaloy-2/M5, the narrower one for Zircaloy-4/ZIRLO/Optimized ZIRLO.
        high_fluence_alloys = ("Zircaloy-2", "M5")
        range_name = ("zircalloy_axial_growth_high_fluence" if alloy in high_fluence_alloys
                       else "zircalloy_axial_growth_low_fluence")
        ranges.check(range_name, {"phi": phi}, RANGES[range_name])
        val = A*phi**n
        return val

    def sigma_eff(Pi, Po, ri, ro, r=None):
        """Effective cladding stress from thick-wall (Lame) principal stresses.

        Formulation:
            sig_r = (Pi*ri^2 - Po*ro^2 + ri^2*ro^2*(Po-Pi)/r^2) / (ro^2 - ri^2)
            sig_t = (Pi*ri^2 - Po*ro^2 - ri^2*ro^2*(Po-Pi)/r^2) / (ro^2 - ri^2)
            sig_l = (Pi*ri^2 - Po*ro^2) / (ro^2 - ri^2)
            val   = sqrt(0.5*((sig_l-sig_t)^2 + (sig_t-sig_r)^2 + (sig_r-sig_l)^2))

        Valid range:
            Not applicable -- exact thick-wall stress formulas, not a fitted correlation.

        Uncertainty:
            Not applicable -- exact given Pi, Po, ri, ro (whatever measurement or model
            uncertainty exists is in those inputs, not in this formula).

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 3.1.10.1, Equations 3-23 through
            3-26 -- standard thick-wall (Lame) cylinder stress formulas, given there as
            part of the zirconium-alloy creep-rate model's effective-stress input.

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            Pi : inner pressure, MPa
            Po : outer pressure, MPa
            ri : inner radius, cm
            ro : outer radius, cm
            r  : radius within the tube, cm (default: mid-wall, 0.5*(ri+ro))
        Returns:
            val : effective (von Mises) stress, MPa, same type as Pi
        """
        xp = backend.lib(Pi, Po, ri, ro)
        if r is None:
            r = 0.5*(ri+ro)
        denom = ro**2 - ri**2
        common = Pi*ri**2 - Po*ro**2
        extra = ri**2*ro**2*(Po-Pi)/r**2
        sig_r = (common+extra)/denom
        sig_t = (common-extra)/denom
        sig_l = common/denom
        val = xp.sqrt(0.5*((sig_l-sig_t)**2+(sig_t-sig_r)**2+(sig_r-sig_l)**2))
        return val

    def cw_type(alloy):
        """Cold-work class (RXA/SRA) used by the strain-rate correlations below.

        Formulation:
            Table lookup, no equation.

        Valid range:
            Not applicable -- a lookup table.

        Uncertainty:
            Not applicable.

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 3.1.10.1. That section states
            "The RXA correlation is used for Zircaloy-2 and M5. The SRA correlation is
            used for Zircaloy-4. An adjustment to the SRA correlation is used for ZIRLO."
            -- matching this table for those four -- but for Optimized ZIRLO that same
            introductory sentence says "An adjustment to the RXA correlation is used",
            while the model description a page later says "The Zircaloy SRA model is used
            for ZIRLO and Optimized ZIRLO with a reduction factor of 0.8". PNNL-35702
            contradicts itself on this one alloy; this table follows the second, more
            specific statement (the one next to the equations and the 0.8 factor that
            Zircalloy.creep_strain/creep_rate also implement), not the first. Logged as a
            documentation quirk in docs/OPEN_QUESTIONS.md rather than a code defect, since
            the code is self-consistent with the equation-level statement.

        Inputs:
            alloy : cladding alloy, string -- one of Zircaloy-2, Zircaloy-4, ZIRLO,
                    Optimized ZIRLO, M5
        Returns:
            val : cold-work class, string, "RXA" or "SRA"
        """
        dic = {
                "Zircaloy-2": "RXA",
                "Zircaloy-4": "SRA",
                "M5": "RXA",
                "ZIRLO": "SRA",
                "Optimized ZIRLO": "SRA"
            }
        val = dic[alloy]
        return val

    def strain_rate_thermal(T, sig, Phi, cw="SRA"):
        """Thermal creep strain rate of zirconium-based cladding.

        Formulation:
            E   = 1.148e5 - 59.9*T
            a_i = 650*(1 - 0.56*(1 - exp(-1.4e-27*Phi^1.3)))
            val = A*(E/T)*sinh(a_i*sig/E)^n * exp(-Q/(R*T)), with (A, n) set by cw
                  (Q = 201000, R = 8.314)

        Valid range:
            Temperature 570 to 625 K; effective stress 40 to 130 MPa (PNNL-35702 section
            3.1.10.3, "applicable to the range of available data" for the strain
            correlations generally). Phi (fluence, feeding the a_i term of Equation 3-17b)
            has no stated unit or range anywhere in PNNL-35702's section 3.1.10 -- its
            "Where," clause for Equation 3-17b lists ai as a "Fluence term" without giving
            Phi's unit, unlike every other symbol in that clause. Not established; RANGES
            checks T and sig here, not Phi. See docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Relative standard error, PNNL-35702 section 3.1.10.3: sigma = 21.6% for
            Zircaloy-2 and M5 (RXA), 14.5% for Zircaloy-4, ZIRLO and Optimized ZIRLO
            (SRA) -- this is Zircalloy.creep_rate's uncertainty, since strain_rate_thermal
            alone is not separately validated in PNNL-35702.

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 3.1.10.1, Equations 3-16 and
            3-17a/b, Table 3-4. The fluence term a_i is noted there as adapted from
            Limback and Andersson (1996), "parameters changed from original Limback
            equation".

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            T   : temperature, K
            sig : effective stress, MPa
            Phi : fast neutron (>1 MeV) fluence, n/cm^2
            cw  : cold-work class, string, "SRA" or "RXA" (default "SRA")
        Returns:
            val : thermal creep strain rate, 1/hr, same type as T
        """
        dic = {
                "SRA": [1.08E9, 2.0],
                "RXA": [5.47E8, 3.5]
            }
        # Promote before any arithmetic touches these: each of them is routinely the
        # scalar while another is a whole batch, and a numpy intermediate formed from
        # the scalar cannot afterwards combine with the tensor.
        T, sig, Phi = backend.promote_all(T, sig, Phi)
        xp = backend.lib(T, sig, Phi)
        ranges.check("zircalloy_creep", {"T": T, "sig": sig}, RANGES["zircalloy_creep"])
        A, n = dic[cw]
        Q = 201000.0
        R = 8.314
        E = 1.148E5 - 59.9*T
        a_i = 650*(1 - 0.56*(1 - xp.exp(-1.4E-27*Phi**1.3)))
        val = A*(E/T)*xp.sinh(a_i*sig/E)**n * xp.exp(-Q/(R*T))
        return val

    def strain_rate_irrad(T, sig, flux, cw="SRA"):
        """Irradiation creep strain rate of zirconium-based cladding.

        Formulation:
            f_T = f_lo                  if T <= 570
                = f_hi                  if T >= 625
                = f_a + f_b*T           otherwise
            val = c0 * flux^c1 * sig^c2 * f_T, with (c0, f_lo, f_a, f_b, f_hi) set by cw
                  (c1 = 0.85, c2 = 1.0)

        Valid range:
            Temperature 570 to 625 K; effective stress 40 to 130 MPa (PNNL-35702 section
            3.1.10.3, checked via the "zircalloy_creep" range table, shared with
            strain_rate_thermal). Flux: PNNL-35702's own model-description "Where," clause
            for Equation 3-18 states phi's unit as n/m^2-s (matching this function's
            docstring below), but section 3.1.10.3's applicability bullet gives the range
            "1e17 to 2e18 n/cm2-s" -- a factor-of-1e4 unit mismatch between PNNL-35702's
            own formula and its own stated applicability range. Not resolved by guessing
            which is right; RANGES therefore does not check flux here. See
            docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Relative standard error, PNNL-35702 section 3.1.10.3: sigma = 21.6% for
            Zircaloy-2 and M5 (RXA), 14.5% for Zircaloy-4, ZIRLO and Optimized ZIRLO
            (SRA) -- same figure as strain_rate_thermal, both feed the combined
            Zircalloy.creep_rate that PNNL-35702 actually validates.

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 3.1.10.1, Equation 3-18, Table
            3-4.

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            T    : temperature, K
            sig  : effective stress, MPa
            flux : fast neutron (>1 MeV) flux, n/m^2-s
            cw   : cold-work class, string, "SRA" or "RXA" (default "SRA")
        Returns:
            val : irradiation creep strain rate, 1/hr, same type as T
        """
        dic = {
                "SRA": [4.0985E-24, 0.7283, -7.0237, 0.0136, 1.4763],
                "RXA": [1.87473E-24, 0.7994, -3.18562, 0.006699132, 1.1840]
            }
        c0, f_lo, f_a, f_b, f_hi = dic[cw]
        c1 = 0.85
        c2 = 1.0
        # f_T comes out of `where` over T alone, so it is a numpy value whenever T is the
        # scalar and sig or flux is the array. Promote before combining them.
        T, sig, flux = backend.promote_all(T, sig, flux)
        ranges.check("zircalloy_creep", {"T": T, "sig": sig}, RANGES["zircalloy_creep"])
        f_T = backend.where(T <= 570, f_lo, backend.where(T >= 625, f_hi, f_a+f_b*T))
        val = c0*flux**c1 * sig**c2 * f_T
        return val

    def strain_sat_primary(eps_dot):
        """Saturated primary hoop creep strain of zirconium-based cladding.

        Formulation:
            val = 0.0216*eps_dot^0.109 * (2 - tanh(3.55e4*eps_dot))^(-2.05)

        Valid range:
            Not separately stated -- PNNL-35702 section 3.1.10.3 gives an applicability
            range for "the strain correlations" as a whole (T 570-625 K, sig 40-130 MPa),
            not for this intermediate quantity by itself; eps_dot is derived from
            strain_rate_thermal/strain_rate_irrad, whose own T and sig ranges already
            enforce that. See docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31); PNNL-35702 gives an
            uncertainty for the combined creep strain (Zircalloy.creep_strain's docstring)
            but not for this intermediate saturated-primary-strain term alone.

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 3.1.10.1, Equation 3-19.

        Inputs:
            eps_dot : combined thermal and irradiation strain rate, 1/hr (float, numpy
                      array, or torch tensor)
        Returns:
            val : saturated primary hoop strain, dimensionless, same type as eps_dot
        """
        xp = backend.lib(eps_dot)
        val = 0.0216*eps_dot**0.109 * (2-xp.tanh(3.55E4*eps_dot))**(-2.05)
        return val

    def creep_strain(alloy, T, sig, flux, Phi, t):
        """Total hoop creep strain of zirconium-based cladding.

        Formulation:
            eps_dot = strain_rate_thermal(T,sig,Phi,cw) + strain_rate_irrad(T,sig,flux,cw)
            eps_sp  = strain_sat_primary(eps_dot)
            val = eps_sp*(1 - exp(-52*sqrt(t*eps_dot))) + eps_dot*t
            ZIRLO / Optimized ZIRLO: val *= 0.8

        Valid range:
            Temperature 570 to 625 K; effective stress 40 to 130 MPa (PNNL-35702 section
            3.1.10.3), enforced through strain_rate_thermal/strain_rate_irrad's own
            ranges.check calls. See those two functions' docstrings for the flux/fluence
            unit caveats.

        Uncertainty:
            Relative standard error, PNNL-35702 section 3.1.10.3: sigma = 21.6% for
            Zircaloy-2 and M5, 14.5% for Zircaloy-4, ZIRLO and Optimized ZIRLO.

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 3.1.10.1, Equations 3-19 through
            3-21. Section 3.1.10.1 also states the ZIRLO/Optimized ZIRLO 0.8 reduction
            factor and its own source: "ZIRLO exhibits about 80% of SRA Zircaloy-4
            creepdown [Sabol et al., 1994]" -- matching this function's `if alloy in
            ("ZIRLO", "Optimized ZIRLO"): val = 0.8*val` exactly.

        Inputs (float, numpy array, or torch tensor for T, sig, flux, Phi, t;
                broadcastable against each other):
            alloy : cladding alloy, string -- one of Zircaloy-2, Zircaloy-4, M5, ZIRLO,
                    Optimized ZIRLO
            T     : temperature, K
            sig   : effective stress, MPa
            flux  : fast neutron (>1 MeV) flux, n/m^2-s
            Phi   : fast neutron (>1 MeV) fluence, n/cm^2
            t     : time, hr
        Returns:
            val : total hoop creep strain, dimensionless, same type as T
        """
        xp = backend.lib(T, sig)
        cw = Zircalloy.cw_type(alloy)
        eps_th = Zircalloy.strain_rate_thermal(T, sig, Phi, cw)
        eps_irr = Zircalloy.strain_rate_irrad(T, sig, flux, cw)
        eps_dot = eps_th + eps_irr
        eps_sp = Zircalloy.strain_sat_primary(eps_dot)
        val = eps_sp*(1-xp.exp(-52*xp.sqrt(t*eps_dot))) + eps_dot*t
        if alloy in ("ZIRLO", "Optimized ZIRLO"):
            val = 0.8*val
        return val

    def creep_rate(alloy, T, sig, flux, Phi, t):
        """Total hoop creep rate of zirconium-based cladding.

        Formulation:
            eps_dot = strain_rate_thermal(T,sig,Phi,cw) + strain_rate_irrad(T,sig,flux,cw)
            eps_sp  = strain_sat_primary(eps_dot)
            val = 26*eps_sp*sqrt(eps_dot)/sqrt(t) * exp(-52*sqrt(t*eps_dot)) + eps_dot
            ZIRLO / Optimized ZIRLO: val *= 0.8

        Valid range:
            Temperature 570 to 625 K; effective stress 40 to 130 MPa (PNNL-35702 section
            3.1.10.3), enforced through strain_rate_thermal/strain_rate_irrad's own
            ranges.check calls -- same as Zircalloy.creep_strain.

        Uncertainty:
            sigma = 21.6 percent for Zircaloy-2 and M5, 14.5 percent for Zircaloy-4,
            ZIRLO and Optimized ZIRLO (PNNL-35702 section 3.1.10.3, matches the value this
            docstring already stated).

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 3.1.10.1, Equation 3-22 (the time
            derivative of Equation 3-21, which is what this function implements per its
            own comment below).

        Inputs (float, numpy array, or torch tensor for T, sig, flux, Phi, t;
                broadcastable against each other):
            alloy : cladding alloy, string -- one of Zircaloy-2, Zircaloy-4, M5, ZIRLO,
                    Optimized ZIRLO
            T     : temperature, K
            sig   : effective stress, MPa
            flux  : fast neutron (>1 MeV) flux, n/m^2-s
            Phi   : fast neutron (>1 MeV) fluence, n/cm^2
            t     : time, hr
        Returns:
            val : total hoop creep rate, 1/hr, same type as T
        """
        xp = backend.lib(T, sig)
        cw = Zircalloy.cw_type(alloy)
        eps_th = Zircalloy.strain_rate_thermal(T, sig, Phi, cw)
        eps_irr = Zircalloy.strain_rate_irrad(T, sig, flux, cw)
        eps_dot = eps_th + eps_irr
        eps_sp = Zircalloy.strain_sat_primary(eps_dot)
        Term1 = 26*eps_sp*xp.sqrt(eps_dot)/xp.sqrt(t)
        Term2 = xp.exp(-52*xp.sqrt(t*eps_dot))
        val = Term1*Term2 + eps_dot
        if alloy in ("ZIRLO", "Optimized ZIRLO"):
            val = 0.8*val
        return val


class HT9:
    def k(T):
        """Thermal conductivity of HT-9 cladding.

        Formulation:
            k = A0 + A1*T

        Valid range:
            293 to 873 K (PNNL-35702 section 3.3.1.3).

        Uncertainty:
            Not established -- PNNL-35702 (Geelhood et al., 2024) section 3.3.1.3 states
            "No uncertainty is given." See docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Akiyama, M. (1991), via PNNL-35702 (Geelhood et al., 2024) section 3.3.1.1,
            Equation 3-39.

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : thermal conductivity, W/m-K, same type as T
        """
        ranges.check("ht9_k", {"T": T}, RANGES["ht9_k"])
        A0 = 22.47
        A1 = 4.397E-3
        val = A0 + A1*T
        return val

    def cp(T):
        """Specific heat capacity of HT-9 cladding.

        Formulation:
            Below 800.15 K: cp = 416.642 + 0.167*T
            At or above 800.15 K: cp = 69.910 + 0.600*T

        Valid range:
            298 to 873 K, unirradiated (PNNL-35702 section 3.3.2.3).

        Uncertainty:
            Not established -- PNNL-35702 (Geelhood et al., 2024) section 3.3.2.3 states
            "No uncertainty for the specific heat capacity is reported." See
            docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Yamanouchi et al. (1992), via PNNL-35702 (Geelhood et al., 2024) section
            3.3.2.1, Equation 3-40, Table 3-11.

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : specific heat capacity, J/kg-K, same type as T
        """
        ranges.check("ht9_cp", {"T": T}, RANGES["ht9_cp"])
        low = 416.642 + 0.167*T
        high = 69.910 + 0.600*T
        val = backend.where(T < 800.15, low, high)
        return val

    def T_melt():
        """Melting temperature of HT-9 cladding, taken as the HT-9/metallic-fuel eutectic.

        Formulation:
            Constant, 973.0 K.

        Valid range:
            Not applicable -- a constant.

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31); PNNL-35702 gives no
            uncertainty on this constant.

        Reference:
            Baker, R.B., Wilson, C.N. (1992), via PNNL-35702 (Geelhood et al., 2024)
            section 3.3.3, Equation 3-41.

        Inputs:
            None.
        Returns:
            val : eutectic temperature, K, Python float
        """
        val = 973.0
        return val

    def rho():
        """Density of HT-9 cladding.

        Formulation:
            Constant, 7750.0 kg/m^3.

        Valid range:
            Not applicable -- a constant.

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31); PNNL-35702 gives no
            uncertainty on this constant.

        Reference:
            Akiyama, M. (1991), via PNNL-35702 (Geelhood et al., 2024) section 3.3.6.1,
            Equation 3-44.

        Inputs:
            None.
        Returns:
            val : density, kg/m^3, Python float
        """
        val = 7750.0
        return val

    def eps():
        """Emissivity of HT-9 cladding, no temperature or burnup dependence.

        Formulation:
            Constant, 0.9.

        Valid range:
            Not applicable -- a constant. PNNL-35702 section 3.3.5.1 notes no temperature
            or burnup dependence observed.

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31); PNNL-35702 gives no
            uncertainty on this constant.

        Reference:
            Dutt, D.S., Baker, R.B. (1974), via PNNL-35702 (Geelhood et al., 2024)
            section 3.3.5.1, Equation 3-43.

        Inputs:
            None.
        Returns:
            val : emissivity, dimensionless, Python float
        """
        val = 0.9
        return val

    def thrm_expan(T):
        """Thermal expansion of HT-9 cladding, assumed isotropic.

        Formulation:
            strain = A1 + A2*T + A3*T^2

        Valid range:
            298.15 to 1073.15 K (PNNL-35702 section 3.3.4.3).

        Uncertainty:
            Not established -- PNNL-35702 (Geelhood et al., 2024) section 3.3.4.3 states
            "No uncertainty is given." See docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Yamanouchi et al. (1992), via PNNL-35702 (Geelhood et al., 2024) section
            3.3.4.1, Equation 3-42, Table 3-12. PNNL-35702 labels this equation's output
            "alpha = Thermal expansion coefficient, K^-1" (units 1/K), not a dimensionless
            strain; this function's docstring (both before and after development) calls its
            return value a dimensionless "strain" instead, matching how Zircalloy.thrm_expan_*
            and UO2.thrm_expan are used elsewhere. Whether the coefficients were fit to a
            true CTE or to a strain under the "alpha" name is not resolved here -- flagged
            in docs/OPEN_QUESTIONS.md rather than reinterpreted, since either reading is
            physically plausible from the formula alone and this is a units/labeling
            question, not something to guess at.

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            strain : linear thermal strain, dimensionless, same type as T
        """
        ranges.check("ht9_thrm_expan", {"T": T}, RANGES["ht9_thrm_expan"])
        A1 = -2.882E-3
        A2 = 9.226E-6
        A3 = 1.842E-9
        strain = A1 + A2*T + A3*T**2
        return strain

    def E(T):
        """Young's modulus of HT-9 cladding.

        Formulation:
            E = A0 + A1*T

        Valid range:
            298.15 to 873.15 K per PNNL-35702 section 3.3.7.3 -- converted to this
            function's Celsius T argument (see the note below): 25.0 to 600.0 C.

        Uncertainty:
            Not established -- PNNL-35702 (Geelhood et al., 2024) section 3.3.7.3 states
            "No uncertainty is given." See docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Akiyama, M. (1991), via PNNL-35702 (Geelhood et al., 2024) section 3.3.7.1,
            Equation 3-45. PNNL-35702's own "Where," clause states T's unit for this
            equation as degrees C, confirming the note below is not an artifact of this
            repository's transcription.

        Inputs:
            T : temperature, C (note: unlike every other T argument in this module, the
                original source documents this one in Celsius, not kelvin -- preserved
                as found) (float, numpy array, or torch tensor)
        Returns:
            val : Young's modulus, Pa, same type as T
        """
        ranges.check("ht9_E", {"T": T}, RANGES["ht9_E"])
        A0 = 2.137E11
        A1 = -1.0274E8
        val = A0 + A1*T
        return val

    def G(T):
        """Shear modulus of HT-9 cladding.

        Formulation:
            G = A0 + A1*T

        Valid range:
            298.15 to 873.15 K per PNNL-35702 section 3.3.8.3 -- converted to this
            function's Celsius T argument, same as HT9.E's note above: 25.0 to 600.0 C.

        Uncertainty:
            Not established -- PNNL-35702 (Geelhood et al., 2024) section 3.3.8.3 states
            "No uncertainty is given." See docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 3.3.8.1, Equation 3-46 (T given
            in degrees C there too, per its own "Where," clause).

        Inputs:
            T : temperature, C (note: unlike every other T argument in this module, the
                original source documents this one in Celsius, not kelvin -- preserved
                as found) (float, numpy array, or torch tensor)
        Returns:
            val : shear modulus, Pa, same type as T
        """
        ranges.check("ht9_G", {"T": T}, RANGES["ht9_G"])
        A0 = 8.964E10
        A1 = -5.378E7
        val = A0 + A1*T
        return val

    def meyer_hardness(T):
        """Meyer's hardness of HT-9 cladding, taken as the zirconium-based cladding model.

        Formulation:
            val = Zircalloy.meyer_hardness(T)

        Valid range:
            350 to 875 K, same as Zircalloy.meyer_hardness (PNNL-35702 section 3.1.8.3
            explicitly lists "HT9 (see Section 3.3.9)" among the alloys this same range
            and correlation applies to). Enforced by Zircalloy.meyer_hardness's own
            ranges.check call, since this function delegates to it directly.

        Uncertainty:
            Not established -- see Zircalloy.meyer_hardness's docstring; PNNL-35702
            section 3.1.8.3 states this could not be established due to limited data.

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 3.3.9: "The Meyer's hardness
            model for HT-9 cladding utilizes the same Meyer's hardness model for
            zirconium-based cladding (see Section 3.1.8)."

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : Meyer's hardness (units as published), same type as T
        """
        val = Zircalloy.meyer_hardness(T)
        return val

    def strain_rate_primary(T, sig, t):
        """Primary thermal creep strain rate of HT-9 cladding.

        Formulation:
            val = (C1*sig*exp(-Q1/(R*T)) + C2*sig^4*exp(-Q2/(R*T))
                   + C3*sqrt(sig)*exp(-Q3/(R*T))) * C4*exp(-C4*t)

        Valid range:
            298.15 to 873.15 K (PNNL-35702 section 3.3.10.2, "the strain rate model" as a
            whole; no separate range given for effective stress or time).

        Uncertainty:
            Not established -- PNNL-35702 (Geelhood et al., 2024) section 3.3.10.2 states
            "No uncertainty is given." See docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Akiyama, M. (1991), via PNNL-35702 (Geelhood et al., 2024) section 3.3.10.1,
            Equation 3-48a, Table 3-13.

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            T   : temperature, K
            sig : effective stress, MPa
            t   : time, s
        Returns:
            val : primary thermal creep strain rate, 1/s, same type as T
        """
        # Promote before any arithmetic touches these: each of them is routinely the
        # scalar while another is a whole batch, and a numpy intermediate formed from
        # the scalar cannot afterwards combine with the tensor.
        T, sig, t = backend.promote_all(T, sig, t)
        xp = backend.lib(T, sig)
        ranges.check("ht9_creep", {"T": T}, RANGES["ht9_creep"])
        C1 = 13.4
        C2 = 8.43E-3
        C3 = 4.08E18
        C4 = 1.6E-6
        Q1 = 15027.0
        Q2 = 26451.0
        Q3 = 89167.0
        R = 1.987
        Term1 = C1*sig*xp.exp(-Q1/(R*T))
        Term2 = C2*sig**4*xp.exp(-Q2/(R*T))
        Term3 = C3*xp.sqrt(sig)*xp.exp(-Q3/(R*T))
        val = (Term1+Term2+Term3)*C4*xp.exp(-C4*t)
        return val

    def strain_rate_secondary(T, sig):
        """Secondary (steady-state) thermal creep strain rate of HT-9 cladding.

        Formulation:
            val = C5*sig^2*exp(-Q4/(R*T)) + C6*sig^5*exp(-Q5/(R*T))

        Valid range:
            298.15 to 873.15 K (PNNL-35702 section 3.3.10.2, same "strain rate model"
            range as strain_rate_primary).

        Uncertainty:
            Not established -- see strain_rate_primary's docstring; same PNNL-35702
            section 3.3.10.2 statement that no uncertainty is given.

        Reference:
            Akiyama, M. (1991), via PNNL-35702 (Geelhood et al., 2024) section 3.3.10.1,
            Equation 3-48b, Table 3-13.

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            T   : temperature, K
            sig : effective stress, MPa
        Returns:
            val : secondary thermal creep strain rate, 1/s, same type as T
        """
        xp = backend.lib(T, sig)
        ranges.check("ht9_creep", {"T": T}, RANGES["ht9_creep"])
        C5 = 1.17E9
        C6 = 8.33E9
        Q4 = 83142.0
        Q5 = 108276.0
        R = 1.987
        Term1 = C5*sig**2*xp.exp(-Q4/(R*T))
        Term2 = C6*sig**5*xp.exp(-Q5/(R*T))
        val = Term1 + Term2
        return val

    def strain_rate_tertiary(T, sig, t):
        """Tertiary thermal creep strain rate of HT-9 cladding.

        Formulation:
            val = 4*sig^10 * (C7*exp(-Q6/(R*T))*t)^3

        Valid range:
            298.15 to 873.15 K (PNNL-35702 section 3.3.10.2, same "strain rate model"
            range as strain_rate_primary).

        Uncertainty:
            Not established -- see strain_rate_primary's docstring; same PNNL-35702
            section 3.3.10.2 statement that no uncertainty is given.

        Reference:
            Akiyama, M. (1991), via PNNL-35702 (Geelhood et al., 2024) section 3.3.10.1,
            Equation 3-48c, Table 3-13.

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            T   : temperature, K
            sig : effective stress, MPa
            t   : time, s
        Returns:
            val : tertiary thermal creep strain rate, 1/s, same type as T
        """
        xp = backend.lib(T, sig)
        ranges.check("ht9_creep", {"T": T}, RANGES["ht9_creep"])
        C7 = 2.12E7
        Q6 = 94233.3
        R = 1.987
        val = 4*sig**10 * (C7*xp.exp(-Q6/(R*T))*t)**3
        return val

    def strain_rate_thermal(T, sig, t):
        """Total thermal creep strain rate of HT-9 cladding.

        Formulation:
            val = strain_rate_primary(T,sig,t) + strain_rate_secondary(T,sig)
                  + strain_rate_tertiary(T,sig,t)

        Valid range:
            298.15 to 873.15 K (PNNL-35702 section 3.3.10.2), enforced through each of
            the three summed terms' own ranges.check calls.

        Uncertainty:
            Not established -- see strain_rate_primary's docstring; same PNNL-35702
            section 3.3.10.2 statement that no uncertainty is given for the strain rate
            model as a whole.

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 3.3.10.1, Equation 3-47.

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            T   : temperature, K
            sig : effective stress, MPa
            t   : time, s
        Returns:
            val : total thermal creep strain rate, 1/s, same type as T
        """
        eps_1 = HT9.strain_rate_primary(T, sig, t)
        eps_2 = HT9.strain_rate_secondary(T, sig)
        eps_3 = HT9.strain_rate_tertiary(T, sig, t)
        val = eps_1 + eps_2 + eps_3
        return val

    def strain_rate_irrad(T, sig, flux):
        """Irradiation creep strain rate of HT-9 cladding.

        Formulation:
            val = (B0 + A1*exp(-Q_irr/(R*T))) * flux * sig^1.3 * 1e-22

        Valid range:
            298.15 to 873.15 K (PNNL-35702 section 3.3.10.2, "the strain rate model" as a
            whole); no numeric range given for effective stress or flux.

        Uncertainty:
            Not established -- see strain_rate_primary's docstring; same PNNL-35702
            section 3.3.10.2 statement that no uncertainty is given.

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 3.3.10.1, Equation 3-49.

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            T    : temperature, K
            sig  : effective stress, MPa
            flux : fast neutron (>1 MeV) flux, n/cm^2-s
        Returns:
            val : irradiation creep strain rate, 1/s, same type as T
        """
        xp = backend.lib(T, sig)
        ranges.check("ht9_creep", {"T": T}, RANGES["ht9_creep"])
        B0 = 1.83E-4
        A1 = 2.59E14
        Q_irr = 73000.0
        R = 1.987
        val = (B0 + A1*xp.exp(-Q_irr/(R*T)))*flux*sig**1.3 * 1E-22
        return val

    def strain_rate(T, sig, flux, t):
        """Total creep strain rate of HT-9 cladding.

        Formulation:
            val = strain_rate_thermal(T,sig,t) + strain_rate_irrad(T,sig,flux)

        Valid range:
            298.15 to 873.15 K (PNNL-35702 section 3.3.10.2), enforced through
            strain_rate_thermal and strain_rate_irrad's own ranges.check calls.

        Uncertainty:
            Not established -- see strain_rate_primary's docstring; same PNNL-35702
            section 3.3.10.2 statement that no uncertainty is given.

        Reference:
            PNNL-35702 (Geelhood et al., 2024), section 3.3.10.1, Equation 3-50.

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            T    : temperature, K
            sig  : effective stress, MPa
            flux : fast neutron (>1 MeV) flux, n/cm^2-s
            t    : time, s
        Returns:
            val : total creep strain rate, 1/s, same type as T
        """
        eps_th = HT9.strain_rate_thermal(T, sig, t)
        eps_irr = HT9.strain_rate_irrad(T, sig, flux)
        val = eps_th + eps_irr
        return val

    def yield_stress(T):
        """Yield stress of HT-9 cladding; the ultimate tensile stress is assumed equal to
        the yield stress.

        Formulation:
            val = A1 + A2*T + A3*T^2 + A4*T^3

        Valid range:
            298.15 to 873.15 K (PNNL-35702 section 3.3.11.2).

        Uncertainty:
            Not established -- PNNL-35702 (Geelhood et al., 2024) section 3.3.11.2 states
            "No uncertainty is given." See docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Akiyama, M. (1991), via PNNL-35702 (Geelhood et al., 2024) section 3.3.11.1,
            Equation 3-51, Table 3-14.

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : yield stress, Pa, same type as T
        """
        ranges.check("ht9_yield_stress", {"T": T}, RANGES["ht9_yield_stress"])
        A1 = 1.290E9
        A2 = -3.561E6
        A3 = 6.371E3
        A4 = -3.959
        val = A1 + A2*T + A3*T**2 + A4*T**3
        return val


class D9_SS:
    """D9 austenitic stainless steel, the cladding of the SCWR lattice in Hughes et al."""

    @staticmethod
    def k(T=None):
        """Thermal conductivity of D9 stainless steel cladding, constant.

        Formulation:
            k = 18.9 W/m-K, independent of temperature.

            Hughes et al. evaluate it once at 650 K, their average clad temperature, and
            hold it constant through their single-channel analysis. This reproduces that
            choice rather than extrapolating it into a temperature dependence the source
            does not provide.

        Valid range:
            Evaluated at 650 K. Reasonable across the clad temperatures of a
            supercritical-water channel, which is the range Hughes uses it over; there is
            no basis in the source for a wider claim. `T` is accepted and ignored, so
            this stays interchangeable with the temperature-dependent conductivities of
            the other cladding materials in this module.

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q16). The source quotes a
            single value with no band.

        Reference:
            Leibowitz, L. and Blomquist, R.A. (1988), as cited by Hughes, Pelaez,
            Schubring & Jordan, Nucl. Eng. Des. 270 (2014) 412-420, section 4.

        Inputs:
            T : temperature, K. Accepted for interface consistency and not used.
        Returns:
            k : thermal conductivity, 18.9 W/m-K
        """
        return 18.9

    @staticmethod
    def rho():
        """Density of D9 stainless steel cladding.

        Formulation:
            rho = 8100 kg/m^3, independent of temperature.

        Valid range:
            As-modelled value for the SCWR lattice of Hughes et al. Table 1.

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q16).

        Reference:
            Hughes et al., Nucl. Eng. Des. 270 (2014) 412-420, Table 1
            ("Clad/water box density").

        Returns:
            rho : density, 8100 kg/m^3
        """
        return 8100.0
