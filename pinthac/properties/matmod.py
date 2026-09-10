"""
Fuel and cladding material property models: UO2, Zircaloy, HT-9 and (stub) D9 stainless.

Moved from MatMod.py in Phase 1, the best-styled file in the pre-cleanup repository.
Phase 2 brings it up to the docstring and backend-dispatch standard without touching any
formula, constant or exponent. Every function here is a flat namespace member --
`UO2.k_Klimenko(1200.0)`, no `self`, no instantiation -- per CLAUDE.md section 2.

Most of these correlations carry only a model family name in the original source
(Frapcon-4, MATPRO, Rolstad, Akiyama, Yamanouchi, Klimenko-Zorin) and no bibliographic
reference, valid range, or uncertainty beyond what a handful of docstrings already state
in passing (an emissivity error band, a few thermal-expansion sigmas). Per CLAUDE.md
section 6, nothing is invented to fill the gaps: fields with no source in the code or in
docs/reference/ are marked "Not established -- see docs/OPEN_QUESTIONS.md", and the
missing citations are logged there as a single consolidated entry (Q31) rather than one
question per function.
"""
from pinthac import backend


RANGES = {}
# No correlation in this module has a published validated input range in the source
# code, docs/reference/, or docs/PHYSICS_REVIEW.md -- only model *names* survive from
# the original import, not the fitted database each was regressed against. RANGES stays
# empty rather than inventing bounds (see docs/OPEN_QUESTIONS.md Q31); no function below
# calls ranges.check() as a result. A later phase that recovers the source papers should
# populate this table and wire the calls back in.


class Gas:
    def k(gas, T):
        """
        Fill-gas thermal conductivity, k = A * T^B per gas species.

        Why this model is here:
            Feeds the gas-gap conductance (pin/gap.py::htc_gap) with the conductivity of
            whatever gas fills the fuel-clad gap -- helium in a fresh rod, increasingly a
            fission-gas mixture as burnup accumulates.

        Formulation:
            k = A * T^B, with (A, B) tabulated per gas species below.

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as "Matlib".

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
        kval = A * T**B
        return kval


class UO2:
    def k_NFI(T, Bu=0.0, f_gad=0.0):
        """
        Modified Frapcon-4 (NFI) model for UO2 thermal conductivity, accounting for
        burnup and gadolinia fraction.

        Why this model is here:
            The burnup- and gadolinia-aware UO2 conductivity model used where fuel
            history matters; MatMod.UO2.k_Klimenko is the simpler fresh-fuel alternative
            used elsewhere in this repository (e.g. pin/annular.py's Theta integral).

        Formulation:
            f_Bu   = 0.00187 * Bu
            g_Bu   = 0.038 * Bu^0.28
            h_T    = 1 / (1 + 396*exp(-Q/T))
            K = 1 / (A + a*f_gad + B*T + f_Bu + (1 - 0.9*exp(-0.04*Bu))*g_Bu*h_T)
                + (E/T^2) * exp(-F/T)

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as "Modified Frapcon-4".

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
        f_Bu = 0.00187 * Bu
        g_Bu = 0.038 * Bu**(0.28)
        h_T = 1/(1+396*xp.exp(-Q/T))

        denom_1 = A + a*f_gad + B*T + f_Bu
        denom_2 = (1-0.9*xp.exp(-0.04*Bu))*g_Bu*h_T
        Term1 = 1/(denom_1+denom_2)
        Term2 = E/(T**2) * xp.exp(-F/T)
        K = Term1 + Term2
        return K

    def k_Klimenko(T):
        """
        Klimenko-Zorin model for the thermal conductivity of 95%-dense UO2 fuel.

        Why this model is here:
            The fresh-fuel UO2 conductivity model used through the annular and rod SCA
            paths (pin/annular.py's Theta integral is built from this via k_NFI's sibling
            call site in sca/annular.py); Hughes (2014) Eq. (14) uses the same functional
            family for the conductivity integral (see docs/PHYSICS_REVIEW.md).

        Formulation:
            tau = T/1000
            k = 100 / (7.5408 + 17.692*tau + 3.6142*tau^2) + 6400*tau^(-5/2)*exp(-16.35/tau)

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as "Klimenko-Zorin"; Hughes (2014) uses the same form
            without naming the exponential term's source either.

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

    def eps(T):
        """
        Emissivity of UO2 fuel.

        Why this model is here:
            The fuel-surface emissivity used in the gap radiation term
            (pin/gap.py::htc_gap's eps_f argument).

        Formulation:
            eps = 0.7856 + 1.5263e-5 * T

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            +/- 6.8 percent, as stated in the original source.

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as "MATPRO".

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : emissivity, dimensionless, same type as T
        """
        a = 0.7856
        b = 1.5263E-5
        val = a + b*T
        return val

    def thrm_expan(T):
        """
        Thermal expansion strain of solid UO2 fuel.

        Why this model is here:
            Feeds pellet dimensional-change calculations (gap closure, densification)
            elsewhere in the fuel-performance chain.

        Formulation:
            strain = K1*T - K2 + K3*exp(-Ed/(k*T))

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as "Frapcon-4".

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            strain : linear thermal strain, dimensionless, same type as T
        """
        K1 = 9.8E-6
        K2 = 2.61E-3
        K3 = 3.16E-1
        Ed = 1.32E-19
        k = 1.38E-23
        xp = backend.lib(T)
        strain = K1*T - K2 + K3*xp.exp(-Ed/(k*T))
        return strain

    def dens_max(T, rho_TD=95.0, T_sint=1873.15):
        """
        Maximum in-reactor pellet dimension change from densification.

        Why this model is here:
            Sets the asymptote UO2.densification relaxes towards, from the fabrication
            sintering temperature and as-fabricated density alone.

        Formulation:
            Below 1000 K: -22.2*(100 - rho_TD)/(T_sint - 1453.15)
            At or above 1000 K: -66.6*(100 - rho_TD)/(T_sint - 1453.15)

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as the "Rolstad model".

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
        """
        Offset constant B in the densification model, solved by bisection so that
        dL/L = 0 at zero burnup.

        Why this model is here:
            UO2.densification needs B as a free offset that anchors its exponential decay
            to start from zero dimension change; there is no closed form for it, so it is
            bisected once per call from dL_max.

        Formulation:
            Root of exp(-3*B) + 2*exp(-35*B) - (-dL_max) = 0 in B, found by n_iter steps
            of bisection on [0, 50].

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31); n_iter=100 bisection
            steps on [0, 50] resolve B to well below double precision's useful digits.

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as the "Rolstad model".

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
        """
        Rolstad model for the densification of UO2, MOX and UO2-Gd2O3 fuel.

        Why this model is here:
            The pellet dimension-change history used by gap-closure and fuel-stack
            length calculations, combining UO2.dens_max's asymptote with a two-term
            exponential burnup decay anchored by UO2.dens_B.

        Formulation:
            dL_max = UO2.dens_max(T, rho_TD, T_sint)
            B      = UO2.dens_B(dL_max)
            val    = dL_max + exp(-3*(Bu+B)) + 2*exp(-35*(Bu+B))

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as the "Rolstad model".

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
        """
        Solid fission-product swelling of UO2, MOX and UO2-Gd2O3 fuel.

        Why this model is here:
            The volumetric-swelling contribution used alongside UO2.swelling_gas for
            pellet dimension change under irradiation.

        Formulation:
            Gadolinia-bearing fuel: val = 0.0005*Bu
            Otherwise, piecewise in Bu (GWd/MTU):
                Bu <= 6:            val = 0
                6 < Bu <= 80:       val = 0.00062*(Bu-6)
                Bu > 80:            val = 0.00062*(80-6) + 0.00086*(Bu-80)

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Inputs:
            Bu  : pellet average burnup, GWd/MTU (float, numpy array, or torch tensor)
            gad : True for gadolinia-bearing fuel (bool, default False; a structural
                  choice, not a value to batch over -- see backend.is_torch's docstring)
        Returns:
            val : swelling, dimensionless volume fraction, same type as Bu
        """
        if gad:
            val = 0.0005*Bu
            return val
        low = backend.zeros_like(Bu)
        mid = 0.00062*(Bu-6)
        high = 0.00062*(80-6) + 0.00086*(Bu-80)
        val = backend.where(Bu <= 6, low, backend.where(Bu <= 80, mid, high))
        return val

    def swelling_gas(T, Bu):
        """
        Gaseous fission-product swelling of UO2, UO2-Gd2O3 and MOX fuel.

        Why this model is here:
            The second swelling contribution, active only above 40 GWd/MTU and in a
            fixed temperature window where fission gas release accelerates.

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
            rather than being invalid -- see the original docstring).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

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
        """
        Thermal conductivity of Zircaloy-4, Zircaloy-2, M5, ZIRLO and Optimized ZIRLO.

        Why this model is here:
            The clad conductivity model used through the gap and cladding conduction
            chain (pin/clad.py::T_ci, sca/annular.py's cladding_gap_step).

        Formulation:
            Below 2098 K: k = 7.511 + 2.088e-2*T - 1.45e-5*T^2 + 7.668e-9*T^3
            At or above 2098 K: k = 36.0 (alpha-to-beta phase transition plateau)

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            sigma = 1.9 W/m-K, as stated in the original source.

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as "Matlib".

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : thermal conductivity, W/m-K, same type as T
        """
        poly = 7.511 + 2.088E-2*T - 1.45E-5*T**2 + 7.668E-9*T**3
        val = backend.where(T >= 2098, 36.0, poly)
        return val

    def cp(T):
        """
        Specific heat capacity of Zircaloy-4, Zircaloy-2, M5, ZIRLO and Optimized ZIRLO.

        Why this model is here:
            Published as a lookup table rather than a formula (the alpha-to-beta
            transition around 1093-1250 K is not smooth), so this is a piecewise-linear
            interpolation rather than a closed-form correlation.

        Formulation:
            Piecewise-linear interpolation of a 14-point T (290-1248 K) vs. Cp table,
            held flat outside the table (see pinthac.backend.interp).

        Valid range:
            290 K to 1248 K (the table's own extent); extrapolated flat outside it.

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as "Matlib".

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : specific heat capacity, J/kg-K, same type as T
        """
        T_tab = [290.0, 300.0, 400.0, 640.0, 1090.0, 1093.0, 1113.0, 1133.0,
                 1153.0, 1173.0, 1193.0, 1213.0, 1233.0, 1248.0]
        C_tab = [279.0, 281.0, 302.0, 331.0, 375.0, 502.0, 590.0, 615.0,
                 719.0, 816.0, 770.0, 619.0, 469.0, 356.0]
        val = backend.interp(T, T_tab, C_tab)
        return val

    def T_melt():
        """
        Melting temperature of Zircaloy-4, Zircaloy-2, M5, ZIRLO and Optimized ZIRLO.

        Why this model is here:
            A fixed reference constant used by cladding-failure checks elsewhere.

        Formulation:
            Constant, 2123.15 K.

        Valid range:
            Not applicable -- a constant.

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as "Matlib".

        Inputs:
            None.
        Returns:
            val : melting temperature, K, Python float
        """
        val = 2123.15
        return val

    def rho():
        """
        Density of Zircaloy-4, Zircaloy-2, M5, ZIRLO and Optimized ZIRLO.

        Why this model is here:
            A fixed reference constant used by mass and thermal-inertia calculations
            elsewhere.

        Formulation:
            Constant, 6520.0 kg/m^3.

        Valid range:
            Not applicable -- a constant.

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as "Matlib".

        Inputs:
            None.
        Returns:
            val : density, kg/m^3, Python float
        """
        val = 6520.0
        return val

    def thrm_expan_axial(T):
        """
        Axial thermal expansion of Zircaloy-4, Zircaloy-2, M5, ZIRLO and Optimized ZIRLO.

        Why this model is here:
            The axial dimension-change model used alongside thrm_expan_diametral for
            clad dimensional-change calculations.

        Formulation:
            Below 1073.15 K:  strain = -2.5060e-5 + 4.4410e-6*(T-273.15)
            Above 1273.15 K:  strain = -8.300e-3 + 9.70e-6*(T-273.15)
            In between: piecewise-linear interpolation of a 21-point table over the
            alpha-to-beta transition (see pinthac.backend.interp).

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            sigma = 4.8e-5 m/m, as stated in the original source.

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as "Matlib".

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : axial linear strain, dimensionless, same type as T
        """
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
        """
        Circumferential (diametral) thermal expansion of Zircaloy-4, Zircaloy-2, M5,
        ZIRLO and Optimized ZIRLO.

        Why this model is here:
            The circumferential dimension-change model used alongside
            thrm_expan_axial for gap-closure and clad dimensional-change calculations.

        Formulation:
            Below 1073.15 K:  strain = -2.3730e-4 + 6.7210e-6*(T-273.15)
            Above 1273.15 K:  strain = -6.800e-3 + 9.70e-6*(T-273.15)
            In between: piecewise-linear interpolation of a 21-point table over the
            alpha-to-beta transition (see pinthac.backend.interp).

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            sigma = 4.6e-4 m/m, as stated in the original source.

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as "Matlib".

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : circumferential linear strain, dimensionless, same type as T
        """
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
        """
        Emissivity of Zircaloy-4, Zircaloy-2, M5, ZIRLO and Optimized ZIRLO.

        Why this model is here:
            The clad-side emissivity used in the gap radiation term
            (pin/gap.py::htc_gap's eps_c argument), accounting for oxide-layer buildup
            and the sharp rise above 1500 K.

        Formulation:
            eps_1 = 0.325 + 1.246e5*t_ox            if t_ox < 3.88e-6
                  = 0.808642 - 50.0*t_ox             otherwise
            eps_2 = max(0.325, exp((1500-T)/300) * eps_1)
            val   = eps_2 if T > 1500 else eps_1

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            sigma = 0.054, as stated in the original source.

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as "Matlib".

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            T    : temperature, K
            t_ox : inner surface oxide thickness, m (default 0.0)
        Returns:
            val : emissivity, dimensionless, same type as T
        """
        xp = backend.lib(T)
        # t_ox defaults to the plain float 0.0 and does not otherwise depend on T; left
        # unpromoted, a torch T would make this where() dispatch to numpy instead (none
        # of its three arguments are tensors), producing a bare numpy value that later
        # silently strips the gradient when multiplied against a torch quantity below.
        T, t_ox = backend.promote_all(T, t_ox)
        eps_1 = backend.where(t_ox < 3.88E-6, 0.325+0.1246E6*t_ox,
                                       0.808642-50.0*t_ox)
        eps_2 = backend.maximum(0.325, xp.exp((1500-T)/300)*eps_1)
        val = backend.where(T > 1500, eps_2, eps_1)
        return val

    def E(T, CW=0.0, phi=0.0, d_ox=0.0012):
        """
        Young's modulus of Zircaloy-4, Zircaloy-2, M5, ZIRLO and Optimized ZIRLO.

        Why this model is here:
            The elastic modulus feeding Zircalloy.sigma_eff-based stress calculations
            elsewhere, temperature-, cold-work-, fluence- and oxide-dependent, linearly
            interpolated across the alpha-to-beta transition.

        Formulation:
            c2 = 0.88 + 0.12*exp(-phi/1e25)
            Below 1090 K:
                E = (1.088e11 - 5.475e7*T + (6.61e11+5.912e8*T)*d_ox - 2.6e10*CW) / c2
            Above 1255 K:
                E = 9.21e10 - 4.05e7*T
            In between: linear interpolation between the two branches' values at
            1090 K and 1255 K.

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as "PNNL-35702".

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            T    : temperature, K
            CW   : effective cold work, ratio of areas (default 0.0)
            phi  : fast neutron (>1 MeV) fluence, n/m^2 (default 0.0)
            d_ox : average oxygen concentration, kg-O/kg-Zr (default 0.0012)
        Returns:
            val : Young's modulus, Pa, same type as T
        """
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
        """
        Shear modulus of Zircaloy-4, Zircaloy-2, M5, ZIRLO and Optimized ZIRLO.

        Why this model is here:
            The shear modulus companion to Zircalloy.E, same temperature/cold-
            work/fluence/oxide dependence and interpolation scheme. PNNL-35702 adds the
            bare constant c3 here rather than c3*CW, unlike E -- preserved as found.

        Formulation:
            c2 = 0.88 + 0.12*exp(-phi/1e25)
            Below 1090 K:
                G = (4.04e10 - 2.168e7*T + (7.07e11-2.315e8*T)*d_ox + c3) / c2
            Above 1255 K:
                G = 3.49e10 - 1.66e7*T
            In between: linear interpolation between the two branches' values at
            1090 K and 1255 K.

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as "PNNL-35702".

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            T    : temperature, K
            CW   : effective cold work, ratio of areas (default 0.0)
            phi  : fast neutron (>1 MeV) fluence, n/m^2 (default 0.0)
            d_ox : average oxygen concentration, kg-O/kg-Zr (default 0.0012)
        Returns:
            val : shear modulus, Pa, same type as T
        """
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
        """
        Meyer's hardness of Zircaloy-2, Zircaloy-4, M5, ZIRLO and Optimized ZIRLO.

        Why this model is here:
            Used by HT9.meyer_hardness as a fallback (HT-9 is modeled with the
            zirconium-based correlation below 1235 K where no HT-9-specific data exists).

        Formulation:
            Below or at 1235 K: val = exp(26.034 - 2.6394e-2*T + 4.3502e-5*T^2
                                             - 2.5621e-8*T^3)
            Above 1235 K: val = 1.0e5 (saturation plateau)

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : Meyer's hardness (units as published), same type as T
        """
        xp = backend.lib(T)
        arg = (26.034 - 2.6394E-2*T + 4.3502E-5*T**2 - 2.5621E-8*T**3)
        val = backend.where(T <= 1235, xp.exp(arg), 1.0E5)
        return val

    def axial_growth(alloy, phi):
        """
        Axial irradiation growth of zirconium-based fuel-rod cladding.

        Why this model is here:
            Dimensional-change contribution from fast-neutron irradiation, separate from
            the thermal-expansion models above; alloy-specific coefficients.

        Formulation:
            val = A * phi^n, with (A, n) tabulated per alloy below.

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Stated in the original
            source to apply to fuel rod cladding only (not other zirconium components).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

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
        val = A*phi**n
        return val

    def sigma_eff(Pi, Po, ri, ro, r=None):
        """
        Effective cladding stress from thick-wall (Lame) principal stresses.

        Why this model is here:
            The von Mises effective stress used by the creep-strain-rate correlations
            below, evaluated at the mid-wall radius by default.

        Formulation:
            sig_r = (Pi*ri^2 - Po*ro^2 + ri^2*ro^2*(Po-Pi)/r^2) / (ro^2 - ri^2)
            sig_t = (Pi*ri^2 - Po*ro^2 - ri^2*ro^2*(Po-Pi)/r^2) / (ro^2 - ri^2)
            sig_l = (Pi*ri^2 - Po*ro^2) / (ro^2 - ri^2)
            val   = sqrt(0.5*((sig_l-sig_t)^2 + (sig_t-sig_r)^2 + (sig_r-sig_l)^2))

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Standard thick-wall
            (Lame) cylinder stress formulas.

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
        """
        Cold-work class (RXA/SRA) used by the strain-rate correlations below.

        Why this model is here:
            The thermal- and irradiation-creep correlations are fitted separately for
            recrystallization-annealed (RXA) and stress-relief-annealed (SRA) material;
            this maps each alloy to its class.

        Formulation:
            Table lookup, no equation.

        Valid range:
            Not applicable -- a lookup table.

        Uncertainty:
            Not applicable.

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

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
        """
        Thermal creep strain rate of zirconium-based cladding.

        Why this model is here:
            One of the two contributions (with strain_rate_irrad) summed by
            Zircalloy.creep_strain / creep_rate into the total hoop creep rate.

        Formulation:
            E   = 1.148e5 - 59.9*T
            a_i = 650*(1 - 0.56*(1 - exp(-1.4e-27*Phi^1.3)))
            val = A*(E/T)*sinh(a_i*sig/E)^n * exp(-Q/(R*T)), with (A, n) set by cw
                  (Q = 201000, R = 8.314)

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

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
        A, n = dic[cw]
        Q = 201000.0
        R = 8.314
        E = 1.148E5 - 59.9*T
        a_i = 650*(1 - 0.56*(1 - xp.exp(-1.4E-27*Phi**1.3)))
        val = A*(E/T)*xp.sinh(a_i*sig/E)**n * xp.exp(-Q/(R*T))
        return val

    def strain_rate_irrad(T, sig, flux, cw="SRA"):
        """
        Irradiation creep strain rate of zirconium-based cladding.

        Why this model is here:
            The second contribution (with strain_rate_thermal) summed by
            Zircalloy.creep_strain / creep_rate into the total hoop creep rate.

        Formulation:
            f_T = f_lo                  if T <= 570
                = f_hi                  if T >= 625
                = f_a + f_b*T           otherwise
            val = c0 * flux^c1 * sig^c2 * f_T, with (c0, f_lo, f_a, f_b, f_hi) set by cw
                  (c1 = 0.85, c2 = 1.0)

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

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
        f_T = backend.where(T <= 570, f_lo, backend.where(T >= 625, f_hi, f_a+f_b*T))
        val = c0*flux**c1 * sig**c2 * f_T
        return val

    def strain_sat_primary(eps_dot):
        """
        Saturated primary hoop creep strain of zirconium-based cladding.

        Why this model is here:
            The asymptotic primary-creep strain used by Zircalloy.creep_strain /
            creep_rate to scale their primary-creep transient term.

        Formulation:
            val = 0.0216*eps_dot^0.109 * (2 - tanh(3.55e4*eps_dot))^(-2.05)

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

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
        """
        Total hoop creep strain of zirconium-based cladding.

        Why this model is here:
            Combines the primary-creep transient (strain_sat_primary) with the steady
            thermal-plus-irradiation strain rate into a single accumulated strain.

        Formulation:
            eps_dot = strain_rate_thermal(T,sig,Phi,cw) + strain_rate_irrad(T,sig,flux,cw)
            eps_sp  = strain_sat_primary(eps_dot)
            val = eps_sp*(1 - exp(-52*sqrt(t*eps_dot))) + eps_dot*t
            ZIRLO / Optimized ZIRLO: val *= 0.8

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

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
        """
        Total hoop creep rate of zirconium-based cladding.

        Why this model is here:
            The time-derivative companion to Zircalloy.creep_strain, giving the
            instantaneous rather than accumulated hoop creep.

        Formulation:
            eps_dot = strain_rate_thermal(T,sig,Phi,cw) + strain_rate_irrad(T,sig,flux,cw)
            eps_sp  = strain_sat_primary(eps_dot)
            val = 26*eps_sp*sqrt(eps_dot)/sqrt(t) * exp(-52*sqrt(t*eps_dot)) + eps_dot
            ZIRLO / Optimized ZIRLO: val *= 0.8

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            sigma = 21.6 percent for Zircaloy-2 and M5, 14.5 percent for Zircaloy-4,
            ZIRLO and Optimized ZIRLO, as stated in the original source.

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

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
        """
        Thermal conductivity of HT-9 cladding.

        Why this model is here:
            The clad conductivity model for HT-9-clad (fast-reactor) pins, analogous to
            Zircalloy.k for LWR cladding.

        Formulation:
            k = A0 + A1*T

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as "Akiyama".

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : thermal conductivity, W/m-K, same type as T
        """
        A0 = 22.47
        A1 = 4.397E-3
        val = A0 + A1*T
        return val

    def cp(T):
        """
        Specific heat capacity of HT-9 cladding.

        Why this model is here:
            The clad heat-capacity model for HT-9-clad pins.

        Formulation:
            Below 800.15 K: cp = 416.642 + 0.167*T
            At or above 800.15 K: cp = 69.910 + 0.600*T

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as "Yamanouchi".

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : specific heat capacity, J/kg-K, same type as T
        """
        low = 416.642 + 0.167*T
        high = 69.910 + 0.600*T
        val = backend.where(T < 800.15, low, high)
        return val

    def T_melt():
        """
        Melting temperature of HT-9 cladding, taken as the HT-9/metallic-fuel eutectic.

        Why this model is here:
            A fixed reference constant used by cladding-failure checks elsewhere.

        Formulation:
            Constant, 973.0 K.

        Valid range:
            Not applicable -- a constant.

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Inputs:
            None.
        Returns:
            val : eutectic temperature, K, Python float
        """
        val = 973.0
        return val

    def rho():
        """
        Density of HT-9 cladding.

        Why this model is here:
            A fixed reference constant used by mass and thermal-inertia calculations
            elsewhere.

        Formulation:
            Constant, 7750.0 kg/m^3.

        Valid range:
            Not applicable -- a constant.

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Inputs:
            None.
        Returns:
            val : density, kg/m^3, Python float
        """
        val = 7750.0
        return val

    def eps():
        """
        Emissivity of HT-9 cladding, no temperature or burnup dependence.

        Why this model is here:
            The clad-side emissivity for HT-9-clad pins' gas-gap radiation term.

        Formulation:
            Constant, 0.9.

        Valid range:
            Not applicable -- a constant.

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Inputs:
            None.
        Returns:
            val : emissivity, dimensionless, Python float
        """
        val = 0.9
        return val

    def thrm_expan(T):
        """
        Thermal expansion of HT-9 cladding, assumed isotropic.

        Why this model is here:
            The dimensional-change model for HT-9-clad pins, analogous to Zircalloy's
            axial/diametral pair but with no reported anisotropy.

        Formulation:
            strain = A1 + A2*T + A3*T^2

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as "Yamanouchi".

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            strain : linear thermal strain, dimensionless, same type as T
        """
        A1 = -2.882E-3
        A2 = 9.226E-6
        A3 = 1.842E-9
        strain = A1 + A2*T + A3*T**2
        return strain

    def E(T):
        """
        Young's modulus of HT-9 cladding.

        Why this model is here:
            The elastic modulus for HT-9-clad pins, analogous to Zircalloy.E.

        Formulation:
            E = A0 + A1*T

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as "Akiyama".

        Inputs:
            T : temperature, C (note: unlike every other T argument in this module, the
                original source documents this one in Celsius, not kelvin -- preserved
                as found) (float, numpy array, or torch tensor)
        Returns:
            val : Young's modulus, Pa, same type as T
        """
        A0 = 2.137E11
        A1 = -1.0274E8
        val = A0 + A1*T
        return val

    def G(T):
        """
        Shear modulus of HT-9 cladding.

        Why this model is here:
            The shear modulus for HT-9-clad pins, analogous to Zircalloy.G.

        Formulation:
            G = A0 + A1*T

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Inputs:
            T : temperature, C (note: unlike every other T argument in this module, the
                original source documents this one in Celsius, not kelvin -- preserved
                as found) (float, numpy array, or torch tensor)
        Returns:
            val : shear modulus, Pa, same type as T
        """
        A0 = 8.964E10
        A1 = -5.378E7
        val = A0 + A1*T
        return val

    def meyer_hardness(T):
        """
        Meyer's hardness of HT-9 cladding, taken as the zirconium-based cladding model.

        Why this model is here:
            No HT-9-specific hardness correlation exists in the source; the zirconium
            model is reused as the best available substitute.

        Formulation:
            val = Zircalloy.meyer_hardness(T)

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : Meyer's hardness (units as published), same type as T
        """
        val = Zircalloy.meyer_hardness(T)
        return val

    def strain_rate_primary(T, sig, t):
        """
        Primary thermal creep strain rate of HT-9 cladding.

        Why this model is here:
            One of three contributions (with strain_rate_secondary and
            strain_rate_tertiary) summed by HT9.strain_rate_thermal.

        Formulation:
            val = (C1*sig*exp(-Q1/(R*T)) + C2*sig^4*exp(-Q2/(R*T))
                   + C3*sqrt(sig)*exp(-Q3/(R*T))) * C4*exp(-C4*t)

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as "Akiyama".

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
        """
        Secondary (steady-state) thermal creep strain rate of HT-9 cladding.

        Why this model is here:
            One of three contributions (with strain_rate_primary and
            strain_rate_tertiary) summed by HT9.strain_rate_thermal.

        Formulation:
            val = C5*sig^2*exp(-Q4/(R*T)) + C6*sig^5*exp(-Q5/(R*T))

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as "Akiyama".

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            T   : temperature, K
            sig : effective stress, MPa
        Returns:
            val : secondary thermal creep strain rate, 1/s, same type as T
        """
        xp = backend.lib(T, sig)
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
        """
        Tertiary thermal creep strain rate of HT-9 cladding.

        Why this model is here:
            One of three contributions (with strain_rate_primary and
            strain_rate_secondary) summed by HT9.strain_rate_thermal.

        Formulation:
            val = 4*sig^10 * (C7*exp(-Q6/(R*T))*t)^3

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as "Akiyama".

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            T   : temperature, K
            sig : effective stress, MPa
            t   : time, s
        Returns:
            val : tertiary thermal creep strain rate, 1/s, same type as T
        """
        xp = backend.lib(T, sig)
        C7 = 2.12E7
        Q6 = 94233.3
        R = 1.987
        val = 4*sig**10 * (C7*xp.exp(-Q6/(R*T))*t)**3
        return val

    def strain_rate_thermal(T, sig, t):
        """
        Total thermal creep strain rate of HT-9 cladding.

        Why this model is here:
            Sums the primary, secondary and tertiary creep-rate contributions into the
            thermal (non-irradiation) creep rate used by HT9.strain_rate.

        Formulation:
            val = strain_rate_primary(T,sig,t) + strain_rate_secondary(T,sig)
                  + strain_rate_tertiary(T,sig,t)

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

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
        """
        Irradiation creep strain rate of HT-9 cladding.

        Why this model is here:
            The irradiation contribution summed with strain_rate_thermal by
            HT9.strain_rate into the total creep rate.

        Formulation:
            val = (B0 + A1*exp(-Q_irr/(R*T))) * flux * sig^1.3 * 1e-22

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Inputs (float, numpy array, or torch tensor; broadcastable against each other):
            T    : temperature, K
            sig  : effective stress, MPa
            flux : fast neutron (>1 MeV) flux, n/cm^2-s
        Returns:
            val : irradiation creep strain rate, 1/s, same type as T
        """
        xp = backend.lib(T, sig)
        B0 = 1.83E-4
        A1 = 2.59E14
        Q_irr = 73000.0
        R = 1.987
        val = (B0 + A1*xp.exp(-Q_irr/(R*T)))*flux*sig**1.3 * 1E-22
        return val

    def strain_rate(T, sig, flux, t):
        """
        Total creep strain rate of HT-9 cladding.

        Why this model is here:
            Sums the thermal (HT9.strain_rate_thermal) and irradiation
            (HT9.strain_rate_irrad) creep rates into a single total.

        Formulation:
            val = strain_rate_thermal(T,sig,t) + strain_rate_irrad(T,sig,flux)

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

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
        """
        Yield stress of HT-9 cladding; the ultimate tensile stress is assumed equal to
        the yield stress.

        Why this model is here:
            The strength limit used by cladding stress-margin checks elsewhere.

        Formulation:
            val = A1 + A2*T + A3*T^2 + A4*T^3

        Valid range:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Uncertainty:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31).

        Reference:
            Not established -- see docs/OPEN_QUESTIONS.md (Q31). Referred to in the
            original source only as "Akiyama".

        Inputs:
            T : temperature (float, numpy array, or torch tensor), K
        Returns:
            val : yield stress, Pa, same type as T
        """
        A1 = 1.290E9
        A2 = -3.561E6
        A3 = 6.371E3
        A4 = -3.959
        val = A1 + A2*T + A3*T**2 + A4*T**3
        return val


class D9_SS:
    def k(T):
        """
        D9 stainless steel cladding thermal conductivity -- not implemented.

        Why this model is here:
            A placeholder only. Per docs/DECISIONS.md, no temperature-dependent D9
            conductivity model was available at Phase 0; a constant value (k = 18.9 W/m-K
            at 650 K, Leibowitz & Blomquist 1988, via Hughes et al. 2014 Table 1 -- see
            docs/PHYSICS_REVIEW.md) is documented as available for a later phase to add,
            but adding it is a physics addition outside Phase 2's docstring/backend/
            dead-code scope (Phase 2 changes no physics) and is deferred rather than
            done here -- see the Phase 2 report.

        Formulation:
            Not implemented.

        Valid range:
            Not applicable.

        Uncertainty:
            Not applicable.

        Reference:
            Leibowitz & Blomquist (1988), via Hughes et al. (2014) Table 1 -- for the
            constant value noted above, not yet wired in.

        Inputs:
            T : temperature, K
        Returns:
            Does not return; raises NotImplementedError.
        """
        raise NotImplementedError(
            "D9 stainless steel has no temperature-dependent conductivity model in this "
            "repository -- see docs/DECISIONS.md ('D9 cladding') and docs/PHYSICS_REVIEW.md."
        )
