"""IAPWS-IF97: the Industrial Formulation 1997 for the Thermodynamic Properties of Water
and Steam.

Why this module is here:
    IF97 is the fast, non-iterative counterpart to the IAPWS-95 scientific formulation in
    iapws95.py. IAPWS-95 is a single Helmholtz surface in (rho, T) that has to be
    inverted whenever the state is given as (p, T) or (p, h), which is almost always;
    IF97 instead divides the (p, T) plane into five regions, each with its own explicit
    equation in the variables a plant engineer actually has, plus backward equations that
    give T(p,h) and T(p,s) with no iteration at all. That is why it is the formulation
    every system code uses for steady-state property lookups, and why it is worth having
    beside IAPWS-95 rather than instead of it.

    Accuracy is the trade. IF97 reproduces IAPWS-95 to within the tolerances of Table 23
    and Table 28 rather than exactly, and the region boundaries are visible as small
    discontinuities. Where the answer has to be thermodynamically consistent -- inside a
    solver, or anywhere a derivative is taken -- use iapws95.py. Where a fast, repeatable
    (p, T) or (p, h) lookup is wanted, use this.

The five regions, in the same order as the release:

    Region 1 : compressed liquid, 273.15 K <= T <= 623.15 K, p_sat(T) <= p <= 100 MPa
    Region 2 : steam, 273.15 K <= T <= 1073.15 K, p up to the B23 line / 100 MPa
    Region 3 : the region around the critical point, bounded below by 623.15 K and above
               by the B23 line. Its basic equation is a Helmholtz energy in (rho, T),
               not a Gibbs energy in (p, T), so it is the one region that has to be
               inverted for density.
    Region 4 : the saturation line itself -- one equation, p_sat(T), and its inverse.
    Region 5 : high-temperature steam, 1073.15 K <= T <= 2273.15 K, p <= 50 MPa.

Shape of the interface, which is the same as iapws95.py's:
    Each region class has one function that evaluates the dimensionless potential and
    every derivative of it that any property needs, and returns them in a dict -- gibbs()
    for regions 1, 2 and 5, helmholtz() for region 3. Every property is then a cheap,
    closed-form read off that dict:

        d = R1.gibbs(3.0, 300.0)      # p [MPa], T [K]
        R1.h(d)                        # J/kg
        R1.rho(d)                      # kg/m^3

    Doing it this way means a state is evaluated once no matter how many properties are
    wanted from it, and it keeps the published derivative tables (Tables 4, 13, 14, 32,
    40 and 41) in one place each, where they can be checked against the release.

    Transport properties are deliberately NOT here. Viscosity, thermal conductivity and
    surface tension are separate IAPWS releases on a different variable basis, and they
    live in properties/iapws_transport.py.

Reference:
    IAPWS R7-97(2012), "Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam" (IAPWS_97.pdf). Equation,
    table and section numbers in the docstrings below all refer to that document.
"""
import os

import numpy as np
import torch

from pinthac.properties import iapws_backend as w

# The published coefficient tables are data, not code, so they sit in their own directory
# beside this module rather than alongside the source files.
pth = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iapws_data')

device = w.device
accelerator = w.accelerator

# ====================
# Reference Constants
# ====================
# IF97 Eq. (1) fixes its own specific gas constant, and it is NOT the IAPWS-95 value
# (0.46151805 kJ/kg/K). The difference is 8e-6 relative, far below the formulation's own
# tolerances, but each release must be evaluated with the constant it was fitted with.

R = 0.461526         # Specific gas constant [kJ kg^-1 K^-1], Eq. (1)
Tc = 647.096         # Critical temperature [K], Eq. (2)
pc = 22.064          # Critical pressure [MPa], Eq. (3)
rhoc = 322.0         # Critical density [kg m^-3], Eq. (4)
Tt = 273.16          # Triple-point temperature [K]
pt = 611.657e-6      # Triple-point pressure [MPa]

# ====================
# Region boundaries
# ====================

T_min = 273.15       # Lower temperature limit of regions 1 and 2 [K]
T_13 = 623.15        # Region 1 / region 3 boundary temperature [K]
T_25 = 1073.15       # Region 2 / region 5 boundary temperature [K]
# The two ends of the B23 line, Sec. 4: it runs from 623.15 K at 16.5292 MPa to 863.15 K
# at 100 MPa. Those are therefore also the lowest pressure and the highest temperature
# anywhere in region 3, which is what makes them worth naming.
p_13 = 16.5291643    # Lowest pressure in region 3 [MPa]
T_3max = 863.15      # Highest temperature in region 3 [K]
T_max = 2273.15      # Upper temperature limit of region 5 [K]
p_max = 100.0        # Upper pressure limit of regions 1, 2 and 3 [MPa]
p_5max = 50.0        # Upper pressure limit of region 5 [MPa]


class B23:
    # =========================================
    # Table 1: B23-equation coefficients
    # =========================================
    n = torch.tensor([
        0.34805185628969e3, -0.11671859879975e1, 0.10192970039326e-2,
        0.57254459862746e3, 0.13918839778870e2
    ], dtype=w.dtype)

    @classmethod
    def p(cls, T):
        """Pressure on the boundary between regions 2 and 3, IF97 Eq. (5).

        Formulation:
            pi = n1 + n2*theta + n3*theta^2,   theta = T/1 K,  pi = p/1 MPa

        Valid range:
            623.15 K <= T <= 863.15 K, which maps to 16.5292 MPa <= p <= 100 MPa.

        Uncertainty:
            Not applicable -- a defined boundary, not a fitted property.

        Reference:
            IAPWS-IF97 Eq. (5) and Table 1.

        Inputs:
            T : temperature, K (float, numpy array, or torch tensor)

        Returns:
            p : boundary pressure, MPa, same type as the input
        """
        (T_,), state = w.prepare(T)
        n = w.on(cls.n, T_)

        theta = T_ / 1.0
        pi = n[0] + n[1] * theta + n[2] * theta**2

        return w.restore(pi * 1.0, state)

    @classmethod
    def T(cls, p):
        """Temperature on the boundary between regions 2 and 3, IF97 Eq. (6).

        Formulation:
            theta = n4 + sqrt((pi - n5)/n3),   pi = p/1 MPa,  theta = T/1 K

        Valid range:
            16.5292 MPa <= p <= 100 MPa.

        Uncertainty:
            Not applicable -- a defined boundary, not a fitted property.

        Reference:
            IAPWS-IF97 Eq. (6) and Table 1.

        Inputs:
            p : pressure, MPa (float, numpy array, or torch tensor)

        Returns:
            T : boundary temperature, K, same type as the input
        """
        (p_,), state = w.prepare(p)
        n = w.on(cls.n, p_)

        pi = p_ / 1.0
        # Below 16.5292 MPa the square root has no real value because the boundary does
        # not exist there. Clamping rather than letting it go NaN keeps a batch that
        # straddles the lower limit usable, and keeps the backward pass finite.
        root = torch.clamp((pi - n[4]) / n[2], min=0.0)
        theta = n[3] + torch.sqrt(root)

        return w.restore(theta * 1.0, state)


class R1:
    # =========================================
    # Table 2: Region 1 Gibbs coefficients
    # =========================================
    pstar = 16.53        # Reducing pressure [MPa]
    Tstar = 1386.0       # Reducing temperature [K]

    Coeff = torch.from_numpy(np.loadtxt(f'{pth}/IAPWS_97_Region1.txt')).to(w.dtype)
    I, J, n = Coeff[:, 0:1], Coeff[:, 1:2], Coeff[:, 2:3]

    # Table 6: backward equation T(p,h), Eq. (11)
    Cph = torch.from_numpy(np.loadtxt(f'{pth}/Region1_ph.txt')).to(w.dtype)
    # Table 8: backward equation T(p,s), Eq. (13)
    Cps = torch.from_numpy(np.loadtxt(f'{pth}/Region1_ps.txt')).to(w.dtype)

    @classmethod
    def gibbs(cls, p, T):
        """Dimensionless Gibbs free energy of region 1 and all of its derivatives, Eq. (7).

        Formulation:
            gamma(pi, tau) = sum_i n_i * (7.1 - pi)^I_i * (tau - 1.222)^J_i
            with pi = p/16.53 MPa and tau = 1386 K/T. The derivatives are Table 4,
            taken with respect to pi and tau, and the subscripts _p and _t below mean
            exactly that -- not derivatives with respect to p and T.

        Valid range:
            273.15 K <= T <= 623.15 K with p_sat(T) <= p <= 100 MPa, and the metastable
            extension a little beyond the saturation line. Nothing is range-checked here;
            use region() to decide which equation applies.

        Uncertainty:
            Reproduces IAPWS-95 to within the tolerances of IF97 Sec. 12.

        Reference:
            IAPWS-IF97 Eq. (7), Table 2 (coefficients) and Table 4 (derivatives).

        Inputs (float, numpy array, or torch tensor; broadcastable):
            p : pressure, MPa
            T : temperature, K

        Returns:
            d : dict of pi, tau, g, g_p, g_pp, g_t, g_tt, g_pt -- each the same type and
                shape the inputs broadcast to. Pass it to the accessors below.
        """
        (p_, T_), state = w.prepare(p, T)
        I, J, n = (w.on(t, p_) for t in (cls.I, cls.J, cls.n))

        pi = p_ / cls.pstar
        tau = cls.Tstar / T_

        # Both bases stay strictly positive over region 1 and its metastable surroundings:
        # A = 7.1 - pi is positive for p < 117 MPa, and B = tau - 1.222 exceeds 1 for
        # T < 623.15 K. The I = 0 and J = 0 terms below raise them to negative powers,
        # which is harmless only because of that.
        A = 7.1 - pi
        B = tau - 1.222

        g = (n * A**I * B**J).sum(dim=0)
        g_p = -(n * I * A**(I - 1) * B**J).sum(dim=0)
        g_pp = (n * I * (I - 1) * A**(I - 2) * B**J).sum(dim=0)
        g_t = (n * J * A**I * B**(J - 1)).sum(dim=0)
        g_tt = (n * J * (J - 1) * A**I * B**(J - 2)).sum(dim=0)
        g_pt = -(n * I * J * A**(I - 1) * B**(J - 1)).sum(dim=0)

        vals = {'pi': pi, 'tau': tau, 'g': g, 'g_p': g_p, 'g_pp': g_pp,
                'g_t': g_t, 'g_tt': g_tt, 'g_pt': g_pt}
        return {key: w.restore(val, state) for key, val in vals.items()}

    @classmethod
    def v(cls, d):
        """Specific volume [m^3/kg] from a gibbs() state. v*p/(RT) = pi*gamma_pi."""
        T = cls.Tstar / d['tau']
        p = cls.pstar * d['pi']
        # R [kJ/kg/K] * T [K] / p [MPa] is kJ/kg/MPa, which is 1e-3 m^3/kg.
        return R * T / p * d['pi'] * d['g_p'] * 1.0e-3

    @classmethod
    def rho(cls, d):
        """Density [kg/m^3] from a gibbs() state."""
        return 1.0 / cls.v(d)

    @classmethod
    def drhodp(cls, d):
        """Isothermal compressibility (drho/dp)_T [kg/m^3/MPa] from a gibbs() state.

        This is the derivative properties/iapws_transport.py needs for both critical
        enhancements, which is why it is a named accessor rather than left to the caller.
        dv/dp = R*T*gamma_pipi / pstar^2 * 1e-3, and drho/dp = -rho^2 * dv/dp.
        """
        T = cls.Tstar / d['tau']
        dvdp = R * T * d['g_pp'] / cls.pstar**2 * 1.0e-3   # m^3/kg/MPa, < 0 for a liquid
        return -cls.rho(d)**2 * dvdp

    @classmethod
    def u(cls, d, units='J'):
        """Specific internal energy [J/kg by default]. u/(RT) = tau*gamma_tau - pi*gamma_pi."""
        scale = w.energy_units[units]
        T = cls.Tstar / d['tau']
        return scale * R * T * (d['tau'] * d['g_t'] - d['pi'] * d['g_p'])

    @classmethod
    def s(cls, d, units='J'):
        """Specific entropy [J/kg-K by default]. s/R = tau*gamma_tau - gamma."""
        scale = w.energy_units[units]
        return scale * R * (d['tau'] * d['g_t'] - d['g'])

    @classmethod
    def h(cls, d, units='J'):
        """Specific enthalpy [J/kg by default]. h/(RT) = tau*gamma_tau."""
        scale = w.energy_units[units]
        T = cls.Tstar / d['tau']
        return scale * R * T * d['tau'] * d['g_t']

    @classmethod
    def cp(cls, d, units='J'):
        """Isobaric heat capacity [J/kg-K by default]. cp/R = -tau^2*gamma_tautau."""
        scale = w.energy_units[units]
        return scale * (-R * d['tau']**2 * d['g_tt'])

    @classmethod
    def cv(cls, d, units='J'):
        """Isochoric heat capacity [J/kg-K by default].

        cv/R = -tau^2*gamma_tautau + (gamma_pi - tau*gamma_pitau)^2 / gamma_pipi

        The second term is positive here because gamma_pipi is negative for a liquid,
        which makes cv smaller than cp as it must be.
        """
        scale = w.energy_units[units]
        num = (d['g_p'] - d['tau'] * d['g_pt'])**2
        return scale * R * (-d['tau']**2 * d['g_tt'] + num / d['g_pp'])

    @classmethod
    def c(cls, d):
        """Speed of sound [m/s] from a gibbs() state.

        w^2/(RT) = pi^2*gamma_pi^2
                   / [ (pi*gamma_pi - tau*pi*gamma_pitau)^2/(tau^2*gamma_tautau)
                       - pi^2*gamma_pipi ]

        R is in kJ/kg/K, so the factor of 1000 turns R*T into m^2/s^2.
        """
        T = cls.Tstar / d['tau']
        num = d['pi']**2 * d['g_p']**2
        den = ((d['pi'] * d['g_p'] - d['tau'] * d['pi'] * d['g_pt'])**2
               / (d['tau']**2 * d['g_tt']) - d['pi']**2 * d['g_pp'])
        val = R * 1000.0 * T * num / den
        return w.sqrt(val)

    @classmethod
    def Prop(cls, p, T, prop):
        """Legacy one-shot accessor: build the region 1 state and read one property off it.

        Kept because this was the module's original interface. New code should call
        gibbs() once and then the named accessors, which is both faster when several
        properties are wanted and explicit about units.

        Inputs:
            p    : pressure, MPa
            T    : temperature, K
            prop : one of 'vol', 'rho', 'u', 's', 'h', 'cp', 'cv', 'w'

        Returns:
            the requested property in this module's default units -- m^3/kg, kg/m^3,
            J/kg, J/kg-K, J/kg, J/kg-K, J/kg-K, m/s respectively. Note that these are
            SI base units; the pre-cleanup version of this function returned the
            kilojoule forms.
        """
        d = cls.gibbs(p, T)
        if prop == 'vol':
            return cls.v(d)
        if prop == 'rho':
            return cls.rho(d)
        if prop == 'u':
            return cls.u(d)
        if prop == 's':
            return cls.s(d)
        if prop == 'h':
            return cls.h(d)
        if prop == 'cp':
            return cls.cp(d)
        if prop == 'cv':
            return cls.cv(d)
        if prop == 'w':
            return cls.c(d)
        raise ValueError(f"R1.Prop: unrecognized property {prop!r} -- expected one of "
                         "'vol', 'rho', 'u', 's', 'h', 'cp', 'cv', 'w'")

    @classmethod
    def Tph(cls, p, h):
        """Backward equation T(p,h) for region 1, IF97 Eq. (11).

        Formulation:
            T/1 K = sum_i n_i * (p/1 MPa)^I_i * (h/2500 kJ/kg + 1)^J_i

        Valid range:
            The whole of region 1: 273.15 K <= T <= 623.15 K, p_sat(T) <= p <= 100 MPa.

        Uncertainty:
            Maximum deviation from the basic equation 0.023 K, RMS 0.0038 K (Sec. 5.2.1).

        Reference:
            IAPWS-IF97 Eq. (11) and Table 6.

        Inputs (float, numpy array, or torch tensor; broadcastable):
            p : pressure, MPa
            h : specific enthalpy, kJ/kg

        Returns:
            T : temperature, K, same type as the inputs
        """
        (p_, h_), state = w.prepare(p, h)
        C = w.on(cls.Cph, p_)
        I, J, n = C[:, 0:1], C[:, 1:2], C[:, 2:3]

        pi = p_ / 1.0
        eta = h_ / 2500.0
        T = (n * pi**I * (eta + 1.0)**J).sum(dim=0)

        return w.restore(T, state)

    @classmethod
    def Tps(cls, p, s):
        """Backward equation T(p,s) for region 1, IF97 Eq. (13).

        Formulation:
            T/1 K = sum_i n_i * (p/1 MPa)^I_i * (s/1 kJ/kg-K + 2)^J_i

        Valid range:
            The whole of region 1.

        Uncertainty:
            Maximum deviation from the basic equation 0.039 K, RMS 0.0084 K (Sec. 5.2.2).

        Reference:
            IAPWS-IF97 Eq. (13) and Table 8.

        Inputs (float, numpy array, or torch tensor; broadcastable):
            p : pressure, MPa
            s : specific entropy, kJ/kg-K

        Returns:
            T : temperature, K, same type as the inputs
        """
        (p_, s_), state = w.prepare(p, s)
        C = w.on(cls.Cps, p_)
        I, J, n = C[:, 0:1], C[:, 1:2], C[:, 2:3]

        pi = p_ / 1.0
        sigma = s_ / 1.0
        T = (n * pi**I * (sigma + 2.0)**J).sum(dim=0)

        return w.restore(T, state)


class R2:
    # =========================================
    # Table 10: ideal-gas part coefficients
    # Table 11: residual part coefficients
    # =========================================
    pstar = 1.0          # Reducing pressure [MPa]
    Tstar = 540.0        # Reducing temperature [K]

    Coeff_I = torch.from_numpy(np.loadtxt(f'{pth}/Region2_Ideal.txt')).to(w.dtype)
    Coeff_R = torch.from_numpy(np.loadtxt(f'{pth}/Region2_Res.txt')).to(w.dtype)

    J0, n0 = Coeff_I[:, 0:1], Coeff_I[:, 1:2]
    Ir, Jr, nr = Coeff_R[:, 0:1], Coeff_R[:, 1:2], Coeff_R[:, 2:3]

    # Tables 20-22: backward equations T(p,h) for subregions 2a, 2b, 2c, Eqs. (22)-(24)
    Cph_a = torch.from_numpy(np.loadtxt(f'{pth}/Region2_ph_2a.txt')).to(w.dtype)
    Cph_b = torch.from_numpy(np.loadtxt(f'{pth}/Region2_ph_2b.txt')).to(w.dtype)
    Cph_c = torch.from_numpy(np.loadtxt(f'{pth}/Region2_ph_2c.txt')).to(w.dtype)

    # Tables 25-27: backward equations T(p,s) for subregions 2a, 2b, 2c, Eqs. (25)-(27)
    Cps_a = torch.from_numpy(np.loadtxt(f'{pth}/Region2_ps_2a.txt')).to(w.dtype)
    Cps_b = torch.from_numpy(np.loadtxt(f'{pth}/Region2_ps_2b.txt')).to(w.dtype)
    Cps_c = torch.from_numpy(np.loadtxt(f'{pth}/Region2_ps_2c.txt')).to(w.dtype)

    n_B2bc = torch.tensor([
        0.90584278514723e3, -0.67955786399241, 0.12809002730136e-3,
        0.26526571908428e4, 0.45257578905948e1
    ], dtype=w.dtype)

    # The 2b/2c boundary only exists above this pressure -- it starts at the point
    # (6.54670 MPa, 2500 kJ/kg) where it meets the h = 2500 kJ/kg line that separates 2a
    # from the rest. Below it, everything at p > 4 MPa is subregion 2b. Eq. (21) takes the
    # square root of (pi - 4.5258), so evaluating it below this pressure is not merely
    # unnecessary but undefined, and a NaN compared against with >= silently answers
    # "false" and sends the point to 2c.
    p_2bc_min = 6.54670

    @classmethod
    def gibbs(cls, p, T):
        """Dimensionless Gibbs free energy of region 2 and all of its derivatives, Eq. (15).

        Formulation:
            gamma = gamma_o + gamma_r
            gamma_o = ln(pi) + sum_i n_i^o * tau^(J_i^o)                       Eq. (16)
            gamma_r = sum_i n_i * pi^I_i * (tau - 0.5)^J_i                     Eq. (17)
            with pi = p/1 MPa and tau = 540 K/T. The derivatives are Tables 13 and 14,
            taken with respect to pi and tau.

        Valid range:
            273.15 K <= T <= 1073.15 K, from p -> 0 up to p_sat(T) below 623.15 K and up
            to the B23 line above it, capped at 100 MPa. Nothing is range-checked here;
            use region() to decide which equation applies.

        Uncertainty:
            Reproduces IAPWS-95 to within the tolerances of IF97 Sec. 12.

        Reference:
            IAPWS-IF97 Eqs. (15)-(17), Tables 10-11 (coefficients), 13-14 (derivatives).

        Inputs (float, numpy array, or torch tensor; broadcastable):
            p : pressure, MPa, strictly greater than zero
            T : temperature, K

        Returns:
            d : dict of pi, tau, the ideal-gas derivatives g0, g0_p, g0_pp, g0_t, g0_tt
                and the residual derivatives gr, gr_p, gr_pp, gr_t, gr_tt, gr_pt -- each
                the same type and shape the inputs broadcast to. gamma_o has no mixed
                derivative: it is a sum of a function of pi and a function of tau, so
                gamma_o_pitau is identically zero and is not carried.
        """
        (p_, T_), state = w.prepare(p, T)
        J0, n0 = (w.on(t, p_) for t in (cls.J0, cls.n0))
        Ir, Jr, nr = (w.on(t, p_) for t in (cls.Ir, cls.Jr, cls.nr))

        pi = p_ / cls.pstar
        tau = cls.Tstar / T_

        # ---- Ideal-gas part, Table 13 ----
        g0 = torch.log(pi) + (n0 * tau**J0).sum(dim=0)
        g0_p = 1.0 / pi
        g0_pp = -1.0 / pi**2
        g0_t = (n0 * J0 * tau**(J0 - 1)).sum(dim=0)
        g0_tt = (n0 * J0 * (J0 - 1) * tau**(J0 - 2)).sum(dim=0)

        # ---- Residual part, Table 14 ----
        # tau - 0.5 stays positive throughout region 2: tau >= 540/1073.15 = 0.503.
        B = tau - 0.5
        gr = (nr * pi**Ir * B**Jr).sum(dim=0)
        gr_p = (nr * Ir * pi**(Ir - 1) * B**Jr).sum(dim=0)
        gr_pp = (nr * Ir * (Ir - 1) * pi**(Ir - 2) * B**Jr).sum(dim=0)
        gr_t = (nr * Jr * pi**Ir * B**(Jr - 1)).sum(dim=0)
        gr_tt = (nr * Jr * (Jr - 1) * pi**Ir * B**(Jr - 2)).sum(dim=0)
        gr_pt = (nr * Ir * Jr * pi**(Ir - 1) * B**(Jr - 1)).sum(dim=0)

        vals = {'pi': pi, 'tau': tau,
                'g0': g0, 'g0_p': g0_p, 'g0_pp': g0_pp, 'g0_t': g0_t, 'g0_tt': g0_tt,
                'gr': gr, 'gr_p': gr_p, 'gr_pp': gr_pp, 'gr_t': gr_t, 'gr_tt': gr_tt,
                'gr_pt': gr_pt}
        return {key: w.restore(val, state) for key, val in vals.items()}

    @classmethod
    def v(cls, d):
        """Specific volume [m^3/kg]. v*p/(RT) = pi*(gamma_o_pi + gamma_r_pi)."""
        T = cls.Tstar / d['tau']
        p = cls.pstar * d['pi']
        return R * T / p * d['pi'] * (d['g0_p'] + d['gr_p']) * 1.0e-3

    @classmethod
    def rho(cls, d):
        """Density [kg/m^3]."""
        return 1.0 / cls.v(d)

    @classmethod
    def drhodp(cls, d):
        """Isothermal compressibility (drho/dp)_T [kg/m^3/MPa].

        dv/dp = R*T*(gamma_o_pipi + gamma_r_pipi)/pstar^2 * 1e-3, and drho/dp is
        -rho^2 dv/dp. Supplied for properties/iapws_transport.py, which needs it for both
        critical enhancements.
        """
        T = cls.Tstar / d['tau']
        dvdp = R * T * (d['g0_pp'] + d['gr_pp']) / cls.pstar**2 * 1.0e-3
        return -cls.rho(d)**2 * dvdp

    @classmethod
    def u(cls, d, units='J'):
        """Specific internal energy [J/kg by default]."""
        scale = w.energy_units[units]
        T = cls.Tstar / d['tau']
        return scale * R * T * (d['tau'] * (d['g0_t'] + d['gr_t'])
                                - d['pi'] * (d['g0_p'] + d['gr_p']))

    @classmethod
    def s(cls, d, units='J'):
        """Specific entropy [J/kg-K by default]."""
        scale = w.energy_units[units]
        return scale * R * (d['tau'] * (d['g0_t'] + d['gr_t']) - (d['g0'] + d['gr']))

    @classmethod
    def h(cls, d, units='J'):
        """Specific enthalpy [J/kg by default]. h/(RT) = tau*(gamma_o_tau + gamma_r_tau)."""
        scale = w.energy_units[units]
        T = cls.Tstar / d['tau']
        return scale * R * T * d['tau'] * (d['g0_t'] + d['gr_t'])

    @classmethod
    def cp(cls, d, units='J'):
        """Isobaric heat capacity [J/kg-K by default].

        cp/R = -tau^2 * (gamma_o_tautau + gamma_r_tautau).
        """
        scale = w.energy_units[units]
        return scale * (-R * d['tau']**2 * (d['g0_tt'] + d['gr_tt']))

    @classmethod
    def cv(cls, d, units='J'):
        """Isochoric heat capacity [J/kg-K by default].

        cv/R = -tau^2*(gamma_o_tautau + gamma_r_tautau)
               - (1 + pi*gamma_r_pi - tau*pi*gamma_r_pitau)^2 / (1 - pi^2*gamma_r_pipi)
        """
        scale = w.energy_units[units]
        num = (1.0 + d['pi'] * d['gr_p'] - d['tau'] * d['pi'] * d['gr_pt'])**2
        den = 1.0 - d['pi']**2 * d['gr_pp']
        return scale * R * (-d['tau']**2 * (d['g0_tt'] + d['gr_tt']) - num / den)

    @classmethod
    def c(cls, d):
        """Speed of sound [m/s].

        w^2/(RT) = [1 + 2*pi*gamma_r_pi + pi^2*gamma_r_pi^2]
                   / [ (1 - pi^2*gamma_r_pipi)
                       + (1 + pi*gamma_r_pi - tau*pi*gamma_r_pitau)^2
                         / (tau^2*(gamma_o_tautau + gamma_r_tautau)) ]
        """
        T = cls.Tstar / d['tau']
        num = 1.0 + 2.0 * d['pi'] * d['gr_p'] + d['pi']**2 * d['gr_p']**2
        inner = (1.0 + d['pi'] * d['gr_p'] - d['tau'] * d['pi'] * d['gr_pt'])**2
        den = ((1.0 - d['pi']**2 * d['gr_pp'])
               + inner / (d['tau']**2 * (d['g0_tt'] + d['gr_tt'])))
        val = R * 1000.0 * T * num / den
        return w.sqrt(val)

    @classmethod
    def Prop(cls, p, T, prop):
        """Legacy one-shot accessor for region 2. See R1.Prop for the units and the reason
        this is kept; new code should call gibbs() once and use the named accessors.
        """
        d = cls.gibbs(p, T)
        if prop == 'vol':
            return cls.v(d)
        if prop == 'rho':
            return cls.rho(d)
        if prop == 'u':
            return cls.u(d)
        if prop == 's':
            return cls.s(d)
        if prop == 'h':
            return cls.h(d)
        if prop == 'cp':
            return cls.cp(d)
        if prop == 'cv':
            return cls.cv(d)
        if prop == 'w':
            return cls.c(d)
        raise ValueError(f"R2.Prop: unrecognized property {prop!r} -- expected one of "
                         "'vol', 'rho', 'u', 's', 'h', 'cp', 'cv', 'w'")

    @classmethod
    def h_B2bc(cls, p):
        """Enthalpy on the boundary between subregions 2b and 2c, IF97 Eq. (21).

        Formulation:
            eta = n4 + sqrt((pi - n5)/n3),   pi = p/1 MPa,  eta = h/1 kJ/kg

        Valid range:
            p >= 6.54670 MPa, where the boundary begins. Below that the square root has
            no real value and the caller should use subregion 2b outright; see
            p_2bc_min. The argument is clamped at zero here so that a batch spanning
            that pressure stays finite in both the forward and the backward pass.

        Uncertainty:
            Not applicable -- a defined subregion boundary, not a fitted property.

        Reference:
            IAPWS-IF97 Eq. (21) and Table 19.

        Inputs:
            p : pressure, MPa (float, numpy array, or torch tensor)

        Returns:
            h : boundary enthalpy, kJ/kg, same type as the input
        """
        (p_,), state = w.prepare(p)
        n = w.on(cls.n_B2bc, p_)

        pi = p_ / 1.0
        root = torch.clamp((pi - n[4]) / n[2], min=0.0)
        eta = n[3] + torch.sqrt(root)

        return w.restore(eta * 1.0, state)

    @classmethod
    def _backward(cls, C, pbase, xbase):
        """Evaluate one of the six region 2 backward polynomials.

        All six have the shape sum_i n_i * pbase^I_i * xbase^J_i and differ only in the
        coefficient table and in how the two bases are formed from p and h or s -- and
        those differ in more than a scale factor, which is why the caller forms them and
        passes them in rather than handing over a shift. Eq. (23) raises (pi - 2) and
        Eq. (24) raises (pi + 25), not pi itself, and Eqs. (26) and (27) raise
        (10 - sigma) and (2 - sigma), not sigma plus something.

        Kept private because the useful entry points are Tph and Tps, which also have to
        choose the subregion.

        Inputs (1-D torch tensors, same shape):
            C     : coefficient table, columns I, J, n
            pbase : the pressure base this equation raises to I_i, dimensionless
            xbase : the enthalpy or entropy base this equation raises to J_i

        Returns:
            T : temperature, K
        """
        C = w.on(C, pbase)
        I, J, n = C[:, 0:1], C[:, 1:2], C[:, 2:3]
        return (n * pbase**I * xbase**J).sum(dim=0)

    @classmethod
    def Tph(cls, p, h):
        """Backward equations T(p,h) for region 2, IF97 Eqs. (22)-(24).

        Formulation:
            eta = h/2000 kJ/kg and pi = p/1 MPa throughout, and then
              2a, p <= 4 MPa                     : T = sum n_i * pi^I_i      * (eta-2.1)^J_i
              2b, p > 4 MPa, h >= h_B2bc(p)      : T = sum n_i * (pi-2)^I_i  * (eta-2.6)^J_i
              2c, p > 4 MPa, h <  h_B2bc(p)      : T = sum n_i * (pi+25)^I_i * (eta-1.8)^J_i
            Note that 2b and 2c shift the pressure base as well as the enthalpy base;
            they are not the same polynomial with different tables. Because the 2b/2c
            boundary only starts at 6.54670 MPa, everything between 4 and 6.54670 MPa is
            2b regardless of enthalpy.

            All three polynomials are evaluated at every point and the answer selected
            with where(), rather than each being evaluated only on the points that need
            it. That keeps the batch a fixed shape with no device synchronization, at the
            cost of two extra polynomial evaluations. The pressure handed to each branch
            is clamped into that branch's own range first: subregion 2c carries pi to the
            power -7, so feeding it a low pressure it will never be selected for would
            overflow to infinity and poison the gradient of the branch that was selected.

        Valid range:
            The whole of region 2.

        Uncertainty:
            Maximum deviation from the basic equation 0.6 K in 2a, 0.01 K in 2b and
            0.02 K in 2c; RMS 0.13, 0.0021 and 0.0079 K (Table 23).

        Reference:
            IAPWS-IF97 Eqs. (20)-(24), Tables 19-22.

        Inputs (float, numpy array, or torch tensor; broadcastable):
            p : pressure, MPa
            h : specific enthalpy, kJ/kg

        Returns:
            T : temperature, K, same type as the inputs
        """
        (p_, h_), state = w.prepare(p, h)

        hstar = 2000.0
        eta = h_ / hstar

        # Each branch sees only pressures it is defined for.
        pi_a = torch.clamp(p_, max=4.0)
        pi_b = torch.clamp(p_, min=4.0)
        pi_c = torch.clamp(p_, min=cls.p_2bc_min)

        T_a = cls._backward(cls.Cph_a, pi_a, eta - 2.1)
        T_b = cls._backward(cls.Cph_b, pi_b - 2.0, eta - 2.6)
        T_c = cls._backward(cls.Cph_c, pi_c + 25.0, eta - 1.8)

        n = w.on(cls.n_B2bc, p_)
        h_bnd = n[3] + torch.sqrt(torch.clamp((p_ - n[4]) / n[2], min=0.0))
        is_2c = (p_ > cls.p_2bc_min) & (h_ < h_bnd)

        T = torch.where(p_ <= 4.0, T_a, torch.where(is_2c, T_c, T_b))
        return w.restore(T, state)

    @classmethod
    def Tps(cls, p, s):
        """Backward equations T(p,s) for region 2, IF97 Eqs. (25)-(27).

        Formulation:
            pi = p/1 MPa throughout, sigma is s over that subregion's own s*, and
              2a, p <= 4 MPa, s* = 2      : T = sum n_i * pi^I_i * (sigma-2)^J_i
              2b, p > 4, s >= 5.85, s* = 0.7853 : T = sum n_i * pi^I_i * (10-sigma)^J_i
              2c, p > 4, s <  5.85, s* = 2.9251 : T = sum n_i * pi^I_i * (2-sigma)^J_i
            Note that 2b and 2c subtract sigma from a constant rather than adding to it.
            The 2b/2c split is on entropy alone, so unlike Tph it needs no auxiliary
            boundary equation and holds at every pressure above 4 MPa. As in Tph, all
            three are evaluated and selected with where(), on pressures clamped into each
            branch's own range.

        Valid range:
            The whole of region 2.

        Uncertainty:
            Maximum deviation from the basic equation 0.5 K in 2a, 0.01 K in 2b and
            0.02 K in 2c; RMS 0.12, 0.0022 and 0.0072 K (Table 28).

        Reference:
            IAPWS-IF97 Eqs. (25)-(27), Tables 25-27.

        Inputs (float, numpy array, or torch tensor; broadcastable):
            p : pressure, MPa
            s : specific entropy, kJ/kg-K

        Returns:
            T : temperature, K, same type as the inputs
        """
        (p_, s_), state = w.prepare(p, s)

        pi_a = torch.clamp(p_, max=4.0)
        pi_b = torch.clamp(p_, min=4.0)

        T_a = cls._backward(cls.Cps_a, pi_a, s_ / 2.0 - 2.0)
        T_b = cls._backward(cls.Cps_b, pi_b, 10.0 - s_ / 0.7853)
        T_c = cls._backward(cls.Cps_c, pi_b, 2.0 - s_ / 2.9251)

        T = torch.where(p_ <= 4.0, T_a, torch.where(s_ >= 5.85, T_b, T_c))
        return w.restore(T, state)


class R3:
    # =========================================
    # Table 30: Region 3 Helmholtz coefficients
    # =========================================
    rhostar = rhoc       # Reducing density [kg m^-3]
    Tstar = Tc           # Reducing temperature [K]

    # The i = 1 term of Eq. (28) is n1*ln(delta) rather than a power, so it is carried
    # here and the data file holds terms 2 to 40 only.
    n1 = 0.10658070028513e1

    Coeff = torch.from_numpy(np.loadtxt(f'{pth}/Region3.txt')).to(w.dtype)
    I, J, n = Coeff[:, 0:1], Coeff[:, 1:2], Coeff[:, 2:3]

    @classmethod
    def helmholtz(cls, rho, T):
        """Dimensionless Helmholtz free energy of region 3 and its derivatives, Eq. (28).

        Formulation:
            phi(delta, tau) = n1*ln(delta) + sum_{i=2..40} n_i * delta^I_i * tau^J_i
            with delta = rho/322 kg/m^3 and tau = 647.096 K/T. The derivatives are
            Table 32, taken with respect to delta and tau; the subscripts _d and _t below
            mean exactly that.

        Valid range:
            623.15 K <= T <= T_B23(p) with p_B23(T) <= p <= 100 MPa, which in density
            terms is roughly 113 to 765 kg/m^3. The equation also gives reasonable values
            in the metastable regions just outside the saturation line. Nothing is
            range-checked here; use region() to decide which equation applies.

        Uncertainty:
            Reproduces IAPWS-95 to within the tolerances of IF97 Sec. 12, and reproduces
            the critical parameters of Eqs. (2)-(4) exactly.

        Reference:
            IAPWS-IF97 Eq. (28), Table 30 (coefficients), Table 32 (derivatives).

        Inputs (float, numpy array, or torch tensor; broadcastable):
            rho : density, kg/m^3, strictly greater than zero
            T   : temperature, K

        Returns:
            d : dict of delta, tau, phi, phi_d, phi_dd, phi_t, phi_tt, phi_dt -- each the
                same type and shape the inputs broadcast to.
        """
        (rho_, T_), state = w.prepare(rho, T)
        I, J, n = (w.on(t, rho_) for t in (cls.I, cls.J, cls.n))

        delta = rho_ / cls.rhostar
        tau = cls.Tstar / T_

        phi = cls.n1 * torch.log(delta) + (n * delta**I * tau**J).sum(dim=0)
        phi_d = cls.n1 / delta + (n * I * delta**(I - 1) * tau**J).sum(dim=0)
        phi_dd = -cls.n1 / delta**2 + (n * I * (I - 1) * delta**(I - 2) * tau**J).sum(dim=0)
        phi_t = (n * J * delta**I * tau**(J - 1)).sum(dim=0)
        phi_tt = (n * J * (J - 1) * delta**I * tau**(J - 2)).sum(dim=0)
        phi_dt = (n * I * J * delta**(I - 1) * tau**(J - 1)).sum(dim=0)

        vals = {'delta': delta, 'tau': tau, 'phi': phi, 'phi_d': phi_d, 'phi_dd': phi_dd,
                'phi_t': phi_t, 'phi_tt': phi_tt, 'phi_dt': phi_dt}
        return {key: w.restore(val, state) for key, val in vals.items()}

    @classmethod
    def p(cls, d, units='MPa'):
        """Pressure [MPa by default]. p/(rho*R*T) = delta*phi_delta."""
        scale = w.pressure_units[units]
        rho = cls.rhostar * d['delta']
        T = cls.Tstar / d['tau']
        return scale * rho * R * T * d['delta'] * d['phi_d']

    @classmethod
    def p_rho(cls, d, units='MPa'):
        """(dp/drho)_T [MPa/(kg/m^3) by default] from a helmholtz() state.

        dp/drho = R*T*(2*delta*phi_delta + delta^2*phi_deltadelta). This is what the
        density solve in rho_pT() uses for its Newton step and what iapws_transport.py needs
        inverted as (drho/dp)_T.
        """
        scale = w.pressure_units[units]
        T = cls.Tstar / d['tau']
        return scale * R * T * (2.0 * d['delta'] * d['phi_d']
                                + d['delta']**2 * d['phi_dd'])

    @classmethod
    def drhodp(cls, d):
        """Isothermal compressibility (drho/dp)_T [kg/m^3/MPa], for iapws_transport.py."""
        return 1.0 / cls.p_rho(d, units='MPa')

    @classmethod
    def v(cls, d):
        """Specific volume [m^3/kg]."""
        return 1.0 / (cls.rhostar * d['delta'])

    @classmethod
    def rho(cls, d):
        """Density [kg/m^3] -- what was fed in, returned for symmetry with R1 and R2."""
        return cls.rhostar * d['delta']

    @classmethod
    def u(cls, d, units='J'):
        """Specific internal energy [J/kg by default]. u/(RT) = tau*phi_tau."""
        scale = w.energy_units[units]
        T = cls.Tstar / d['tau']
        return scale * R * T * d['tau'] * d['phi_t']

    @classmethod
    def s(cls, d, units='J'):
        """Specific entropy [J/kg-K by default]. s/R = tau*phi_tau - phi."""
        scale = w.energy_units[units]
        return scale * R * (d['tau'] * d['phi_t'] - d['phi'])

    @classmethod
    def h(cls, d, units='J'):
        """Specific enthalpy [J/kg by default]. h/(RT) = tau*phi_tau + delta*phi_delta."""
        scale = w.energy_units[units]
        T = cls.Tstar / d['tau']
        return scale * R * T * (d['tau'] * d['phi_t'] + d['delta'] * d['phi_d'])

    @classmethod
    def cv(cls, d, units='J'):
        """Isochoric heat capacity [J/kg-K by default]. cv/R = -tau^2*phi_tautau."""
        scale = w.energy_units[units]
        return scale * (-R * d['tau']**2 * d['phi_tt'])

    @classmethod
    def cp(cls, d, units='J'):
        """Isobaric heat capacity [J/kg-K by default].

        cp/R = -tau^2*phi_tautau
               + (delta*phi_delta - delta*tau*phi_deltatau)^2
                 / (2*delta*phi_delta + delta^2*phi_deltadelta)

        The denominator is dp/drho in reduced form, so cp diverges where the isotherm
        goes flat -- which at the critical point is exactly what it should do.
        """
        scale = w.energy_units[units]
        num = (d['delta'] * d['phi_d'] - d['delta'] * d['tau'] * d['phi_dt'])**2
        den = 2.0 * d['delta'] * d['phi_d'] + d['delta']**2 * d['phi_dd']
        return scale * R * (-d['tau']**2 * d['phi_tt'] + num / den)

    @classmethod
    def c(cls, d):
        """Speed of sound [m/s].

        w^2/(RT) = 2*delta*phi_delta + delta^2*phi_deltadelta
                   - (delta*phi_delta - delta*tau*phi_deltatau)^2 / (tau^2*phi_tautau)
        """
        T = cls.Tstar / d['tau']
        num = (d['delta'] * d['phi_d'] - d['delta'] * d['tau'] * d['phi_dt'])**2
        val = R * 1000.0 * T * (2.0 * d['delta'] * d['phi_d'] + d['delta']**2 * d['phi_dd']
                                - num / (d['tau']**2 * d['phi_tt']))
        return w.sqrt(val)

    @classmethod
    def rho_pT(cls, p, T, iters=60):
        """Density [kg/m^3] at a given (p, T) in region 3, by solving Eq. (28) for density.

        Formulation:
            Bisection on p(rho, T) - p over a bracket chosen by which side of the
            saturation line the state is on, followed by one Newton correction that
            carries the autograd graph.

            Bisection rather than a bare Newton because p(rho, T) is very flat in rho
            near the critical point -- that flatness is what region 3 exists to describe
            -- so a Newton step there can be enormous and leave the region entirely.
            Bisection cannot diverge, and the bracket is chosen per point:

                T >= Tc                   : one branch, 1 to 900 kg/m^3
                T <  Tc and p >= p_sat(T) : liquid branch, rho_c to 900 kg/m^3
                T <  Tc and p <  p_sat(T) : vapor branch,  1 to rho_c

            The final Newton step is what makes this differentiable. The bisection loop
            is not: it is a chain of comparisons, and its result is detached. Adding one
            step of rho <- rho - (p(rho,T) - p)/(dp/drho)_T on top leaves the value alone
            (the residual is already at machine precision) but gives torch the exact
            implicit derivatives of the converged root,
                (drho/dp)_T = 1/(dp/drho)_T  and  (drho/dT)_p = -(dp/dT)_rho/(dp/drho)_T,
            which is what a surrogate trained through this needs.

        Valid range:
            Region 3 as defined by region(): 623.15 K <= T <= T_B23(p), p up to 100 MPa.
            Outside it the bracket will not contain the root and the answer is an
            endpoint, not a solution. Nothing is range-checked here.

        Uncertainty:
            Exact to the basic equation to within the bisection tolerance, which after
            60 halvings of a 900 kg/m^3 bracket is far below floating-point resolution.

        Reference:
            IAPWS-IF97 Eq. (28); the bracketing strategy is this library's, not the
            release's.

        Inputs (float, numpy array, or torch tensor; broadcastable):
            p     : pressure, MPa
            T     : temperature, K
            iters : bisection halvings

        Returns:
            rho : density, kg/m^3, same type as the inputs
        """
        (p_, T_), state = w.prepare(p, T)

        # The saturation line only exists below Tc; above it there is one branch.
        p_sat = R4._p(torch.clamp(T_, max=Tc - 1.0e-6))
        liquid_like = (T_ >= Tc) | (p_ >= p_sat)
        vapor_like = (T_ < Tc) & (p_ < p_sat)

        lo = torch.where(liquid_like & (T_ < Tc),
                         torch.full_like(T_, rhoc), torch.full_like(T_, 1.0))
        hi = torch.where(vapor_like, torch.full_like(T_, rhoc), torch.full_like(T_, 900.0))

        # Detached throughout: a comparison carries no useful derivative, and the final
        # Newton step below restores the one that matters.
        with torch.no_grad():
            T_fix = T_.detach()
            p_fix = p_.detach()
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                p_mid = cls._p(mid, T_fix)
                too_low = p_mid < p_fix
                lo = torch.where(too_low, mid, lo)
                hi = torch.where(too_low, hi, mid)
            rho_star = 0.5 * (lo + hi)

        # One differentiable Newton step. The denominator is detached so that the
        # derivative it produces is the implicit one and nothing else; the numerator
        # carries the graph and is numerically zero, so the value does not move.
        res = cls._p(rho_star, T_) - p_
        with torch.no_grad():
            dp_drho = cls._p_rho(rho_star, T_.detach())
        rho = rho_star - res / dp_drho

        return w.restore(rho, state)

    @classmethod
    def _p(cls, rho, T):
        """Pressure [MPa] straight from (rho, T), staying in torch tensors.

        The public path is helmholtz() then p(), but that round-trips the whole state
        dict through restore() on every call, which is wasted work inside a solver loop
        that only wants the pressure. This is the same equation with the type handling
        left out, so it is private and takes 1-D float64 tensors only.
        """
        I, J, n = (w.on(t, rho) for t in (cls.I, cls.J, cls.n))
        delta = rho / cls.rhostar
        tau = cls.Tstar / T
        phi_d = cls.n1 / delta + (n * I * delta**(I - 1) * tau**J).sum(dim=0)
        return rho * R * T * delta * phi_d * 1.0e-3     # kPa -> MPa

    @classmethod
    def _p_rho(cls, rho, T):
        """(dp/drho)_T [MPa/(kg/m^3)] straight from (rho, T). Private, see _p."""
        I, J, n = (w.on(t, rho) for t in (cls.I, cls.J, cls.n))
        delta = rho / cls.rhostar
        tau = cls.Tstar / T
        phi_d = cls.n1 / delta + (n * I * delta**(I - 1) * tau**J).sum(dim=0)
        phi_dd = -cls.n1 / delta**2 + (n * I * (I - 1) * delta**(I - 2) * tau**J).sum(dim=0)
        return R * T * (2.0 * delta * phi_d + delta**2 * phi_dd) * 1.0e-3


class R4:
    # =========================================
    # Table 34: Region 4 saturation coefficients
    # =========================================
    Coeff = torch.from_numpy(np.loadtxt(f'{pth}/Region4.txt')).to(w.dtype)

    T_max = T_13         # Above this, the saturation line runs through region 3 [K]

    @classmethod
    def _p(cls, T):
        """Saturation pressure [MPa] straight from T, staying in torch tensors.

        Private counterpart of p(), used inside R3.rho_pT() where the type round-trip
        would be wasted work. Takes a 1-D float64 tensor and returns one.
        """
        n = w.on(cls.Coeff, T)

        theta = T / 1.0 + n[8] / (T / 1.0 - n[9])            # Eq. (29b)
        A = theta**2 + n[0] * theta + n[1]
        B = n[2] * theta**2 + n[3] * theta + n[4]
        C = n[5] * theta**2 + n[6] * theta + n[7]

        # The discriminant is positive over the whole saturation line; clamping only
        # guards a caller who reaches outside it, and keeps the backward pass finite.
        disc = torch.clamp(B**2 - 4.0 * A * C, min=0.0)
        return (2.0 * C / (-B + torch.sqrt(disc)))**4        # Eq. (30)

    @classmethod
    def p(cls, T):
        """Saturation pressure, IF97 Eq. (30) -- the region 4 basic equation.

        Formulation:
            theta   = T/1 K + n9/(T/1 K - n10)
            A, B, C = quadratics in theta with the Table 34 coefficients
            p_sat   = [2C / (-B + sqrt(B^2 - 4AC))]^4   MPa

        Valid range:
            273.15 K <= T <= 647.096 K, from the triple point to the critical point.

        Uncertainty:
            Agrees with the IAPWS-95 saturation line to better than 0.02 percent in
            pressure over the whole range.

        Reference:
            IAPWS-IF97 Eqs. (29)-(30) and Table 34.

        Inputs:
            T : temperature, K (float, numpy array, or torch tensor)

        Returns:
            p_sat : saturation pressure, MPa, same type as the input
        """
        (T_,), state = w.prepare(T)
        return w.restore(cls._p(T_), state)

    @classmethod
    def T(cls, p):
        """Saturation temperature, IF97 Eq. (31) -- the explicit inverse of Eq. (30).

        Formulation:
            beta    = (p/1 MPa)^(1/4)
            E, F, G = quadratics in beta with the Table 34 coefficients
            D       = 2G / (-F - sqrt(F^2 - 4EG))
            T_sat   = [n10 + D - sqrt((n10 + D)^2 - 4(n9 + n10*D))] / 2

        Valid range:
            611.213 Pa <= p <= 22.064 MPa, from the triple point to the critical point.

        Uncertainty:
            The exact inverse of Eq. (30) to within rounding.

        Reference:
            IAPWS-IF97 Eq. (31) and Table 34.

        Inputs:
            p : pressure, MPa (float, numpy array, or torch tensor)

        Returns:
            T_sat : saturation temperature, K, same type as the input
        """
        (p_,), state = w.prepare(p)
        n = w.on(cls.Coeff, p_)

        beta = (p_ / 1.0)**0.25
        E = beta**2 + n[2] * beta + n[5]
        F = n[0] * beta**2 + n[3] * beta + n[6]
        G = n[1] * beta**2 + n[4] * beta + n[7]

        disc_D = torch.clamp(F**2 - 4.0 * E * G, min=0.0)
        D = 2.0 * G / (-F - torch.sqrt(disc_D))

        disc_T = torch.clamp((n[9] + D)**2 - 4.0 * (n[8] + n[9] * D), min=0.0)
        T = (n[9] + D - torch.sqrt(disc_T)) / 2.0

        return w.restore(T, state)

    @classmethod
    def h_f(cls, T, units='J'):
        """Saturated-liquid enthalpy [J/kg by default] at temperature T.

        Formulation:
            h_f(T) = R1.h(R1.gibbs(p_sat(T), T))

        Valid range:
            273.15 K <= T <= 623.15 K. Above 623.15 K the saturation line leaves region 1
            and runs through region 3, where the saturated densities themselves have to be
            solved for; that case is not covered here. Use iapws95.IAPWS95.saturation(),
            which solves the phase-equilibrium conditions directly and is valid all the
            way to the critical point.

        Uncertainty:
            That of region 1 and of Eq. (30) combined; see IF97 Sec. 12.

        Reference:
            IAPWS-IF97 Sec. 8 and Sec. 5.

        Inputs:
            T     : temperature, K (float, numpy array, or torch tensor)
            units : 'J', 'kJ' or 'MJ' per kg

        Returns:
            h_f : saturated-liquid enthalpy, same type as the input
        """
        return R1.h(R1.gibbs(cls.p(T), T), units=units)

    @classmethod
    def h_g(cls, T, units='J'):
        """Saturated-vapor enthalpy [J/kg by default] at temperature T.

        The region 2 counterpart of h_f: the saturated vapor at T is region 2 evaluated on
        the saturation line. Same 623.15 K ceiling and the same reason for it.

        Inputs:
            T     : temperature, K (float, numpy array, or torch tensor)
            units : 'J', 'kJ' or 'MJ' per kg

        Returns:
            h_g : saturated-vapor enthalpy, same type as the input
        """
        return R2.h(R2.gibbs(cls.p(T), T), units=units)


# Backwards-compatible spelling: region 4 used to be called RSAT in this module.
RSAT = R4


class R5:
    # =========================================
    # Table 37: ideal-gas part coefficients
    # Table 38: residual part coefficients
    # =========================================
    pstar = 1.0          # Reducing pressure [MPa]
    Tstar = 1000.0       # Reducing temperature [K]

    Coeff_I = torch.from_numpy(np.loadtxt(f'{pth}/Region5_Ideal.txt')).to(w.dtype)
    Coeff_R = torch.from_numpy(np.loadtxt(f'{pth}/Region5_Res.txt')).to(w.dtype)

    J0, n0 = Coeff_I[:, 0:1], Coeff_I[:, 1:2]
    Ir, Jr, nr = Coeff_R[:, 0:1], Coeff_R[:, 1:2], Coeff_R[:, 2:3]

    @classmethod
    def gibbs(cls, p, T):
        """Dimensionless Gibbs free energy of region 5 and its derivatives, Eq. (32).

        Formulation:
            gamma = gamma_o + gamma_r
            gamma_o = ln(pi) + sum_{i=1..6} n_i^o * tau^(J_i^o)                Eq. (33)
            gamma_r = sum_{i=1..6} n_i * pi^I_i * tau^J_i                      Eq. (34)
            with pi = p/1 MPa and tau = 1000 K/T. Note that region 5's residual part
            raises tau itself, not (tau - 0.5) as region 2 does. The derivatives are
            Tables 40 and 41.

        Valid range:
            1073.15 K <= T <= 2273.15 K, 0 < p <= 50 MPa. The equation is for pure
            undissociated water; at these temperatures real steam dissociates, and that
            has to be accounted for separately.

        Uncertainty:
            See IF97 Sec. 12; the underlying data at these temperatures are sparse.

        Reference:
            IAPWS-IF97 Eqs. (32)-(34), Tables 37-38 (coefficients), 40-41 (derivatives).

        Inputs (float, numpy array, or torch tensor; broadcastable):
            p : pressure, MPa, strictly greater than zero
            T : temperature, K

        Returns:
            d : dict of pi, tau, g0, g0_p, g0_pp, g0_t, g0_tt, gr, gr_p, gr_pp, gr_t,
                gr_tt, gr_pt -- the same keys region 2 uses, so the accessors read the
                same way.
        """
        (p_, T_), state = w.prepare(p, T)
        J0, n0 = (w.on(t, p_) for t in (cls.J0, cls.n0))
        Ir, Jr, nr = (w.on(t, p_) for t in (cls.Ir, cls.Jr, cls.nr))

        pi = p_ / cls.pstar
        tau = cls.Tstar / T_

        # ---- Ideal-gas part, Table 40 ----
        g0 = torch.log(pi) + (n0 * tau**J0).sum(dim=0)
        g0_p = 1.0 / pi
        g0_pp = -1.0 / pi**2
        g0_t = (n0 * J0 * tau**(J0 - 1)).sum(dim=0)
        g0_tt = (n0 * J0 * (J0 - 1) * tau**(J0 - 2)).sum(dim=0)

        # ---- Residual part, Table 41 ----
        gr = (nr * pi**Ir * tau**Jr).sum(dim=0)
        gr_p = (nr * Ir * pi**(Ir - 1) * tau**Jr).sum(dim=0)
        gr_pp = (nr * Ir * (Ir - 1) * pi**(Ir - 2) * tau**Jr).sum(dim=0)
        gr_t = (nr * Jr * pi**Ir * tau**(Jr - 1)).sum(dim=0)
        gr_tt = (nr * Jr * (Jr - 1) * pi**Ir * tau**(Jr - 2)).sum(dim=0)
        gr_pt = (nr * Ir * Jr * pi**(Ir - 1) * tau**(Jr - 1)).sum(dim=0)

        vals = {'pi': pi, 'tau': tau,
                'g0': g0, 'g0_p': g0_p, 'g0_pp': g0_pp, 'g0_t': g0_t, 'g0_tt': g0_tt,
                'gr': gr, 'gr_p': gr_p, 'gr_pp': gr_pp, 'gr_t': gr_t, 'gr_tt': gr_tt,
                'gr_pt': gr_pt}
        return {key: w.restore(val, state) for key, val in vals.items()}


    @classmethod
    def v(cls, d):
        """Specific volume [m^3/kg]."""
        T = cls.Tstar / d['tau']
        p = cls.pstar * d['pi']
        return R * T / p * d['pi'] * (d['g0_p'] + d['gr_p']) * 1.0e-3

    @classmethod
    def rho(cls, d):
        """Density [kg/m^3]."""
        return 1.0 / cls.v(d)

    @classmethod
    def drhodp(cls, d):
        """Isothermal compressibility (drho/dp)_T [kg/m^3/MPa], for iapws_transport.py."""
        T = cls.Tstar / d['tau']
        dvdp = R * T * (d['g0_pp'] + d['gr_pp']) / cls.pstar**2 * 1.0e-3
        return -cls.rho(d)**2 * dvdp

    @classmethod
    def u(cls, d, units='J'):
        """Specific internal energy [J/kg by default]."""
        scale = w.energy_units[units]
        T = cls.Tstar / d['tau']
        return scale * R * T * (d['tau'] * (d['g0_t'] + d['gr_t'])
                                - d['pi'] * (d['g0_p'] + d['gr_p']))

    @classmethod
    def s(cls, d, units='J'):
        """Specific entropy [J/kg-K by default]."""
        scale = w.energy_units[units]
        return scale * R * (d['tau'] * (d['g0_t'] + d['gr_t']) - (d['g0'] + d['gr']))

    @classmethod
    def h(cls, d, units='J'):
        """Specific enthalpy [J/kg by default]."""
        scale = w.energy_units[units]
        T = cls.Tstar / d['tau']
        return scale * R * T * d['tau'] * (d['g0_t'] + d['gr_t'])

    @classmethod
    def cp(cls, d, units='J'):
        """Isobaric heat capacity [J/kg-K by default]. cp/R = -tau^2*(g0_tt + gr_tt)."""
        scale = w.energy_units[units]
        return scale * (-R * d['tau']**2 * (d['g0_tt'] + d['gr_tt']))

    @classmethod
    def cv(cls, d, units='J'):
        """Isochoric heat capacity [J/kg-K by default]. Same relation as region 2."""
        scale = w.energy_units[units]
        num = (1.0 + d['pi'] * d['gr_p'] - d['tau'] * d['pi'] * d['gr_pt'])**2
        den = 1.0 - d['pi']**2 * d['gr_pp']
        return scale * R * (-d['tau']**2 * (d['g0_tt'] + d['gr_tt']) - num / den)

    @classmethod
    def c(cls, d):
        """Speed of sound [m/s]. Same relation as region 2."""
        T = cls.Tstar / d['tau']
        num = 1.0 + 2.0 * d['pi'] * d['gr_p'] + d['pi']**2 * d['gr_p']**2
        inner = (1.0 + d['pi'] * d['gr_p'] - d['tau'] * d['pi'] * d['gr_pt'])**2
        den = ((1.0 - d['pi']**2 * d['gr_pp'])
               + inner / (d['tau']**2 * (d['g0_tt'] + d['gr_tt'])))
        val = R * 1000.0 * T * num / den
        return w.sqrt(val)


def region(p, T):
    """Which IF97 region a state belongs to.

    Formulation:
        273.15 K <= T <= 623.15 K  : region 1 if p >= p_sat(T), else region 2
        623.15 K <  T <= 863.15 K  : region 3 if p >  p_B23(T), else region 2
        863.15 K <  T <= 1073.15 K : region 2
        1073.15 K < T <= 2273.15 K : region 5
        Region 4 -- the saturation line itself -- is never returned. It is a line of zero
        area in the (p, T) plane, so no floating-point state lands exactly on it; call
        R4.p or R4.T directly when the saturation line is what is wanted.

    Valid range:
        273.15 K <= T <= 2273.15 K, 0 < p <= 100 MPa (50 MPa in region 5). A state
        outside the formulation entirely is reported as region 0 rather than silently
        assigned to the nearest equation.

    Uncertainty:
        Not applicable -- a case split, not a correlation.

    Reference:
        IAPWS-IF97 Fig. 1, Eq. (5) and Sec. 8.

    Inputs (float, numpy array, or torch tensor; broadcastable):
        p : pressure, MPa
        T : temperature, K

    Returns:
        region : 1, 2, 3 or 5, or 0 for a state outside the formulation. Integer-valued,
                 and of the same kind as the inputs -- a torch long tensor for torch
                 input, a NumPy integer array for NumPy or list input, a Python int for a
                 scalar.
    """
    (p_, T_), state = w.prepare(p, T)

    p_sat = R4._p(torch.clamp(T_, min=T_min, max=Tc - 1.0e-6))
    p_b23 = B23.p(T_)

    cold = T_ <= T_13
    warm = (T_ > T_13) & (T_ <= T_25)
    hot = (T_ > T_25) & (T_ <= T_max)

    reg = torch.zeros_like(T_)
    reg = torch.where(cold & (p_ >= p_sat), torch.full_like(T_, 1.0), reg)
    reg = torch.where(cold & (p_ < p_sat), torch.full_like(T_, 2.0), reg)
    reg = torch.where(warm & (p_ > p_b23), torch.full_like(T_, 3.0), reg)
    reg = torch.where(warm & (p_ <= p_b23), torch.full_like(T_, 2.0), reg)
    reg = torch.where(hot, torch.full_like(T_, 5.0), reg)

    # Outside the formulation's own limits, in either variable, nothing applies.
    inside = ((T_ >= T_min) & (T_ <= T_max) & (p_ > 0.0)
              & torch.where(hot, p_ <= p_5max, p_ <= p_max))
    reg = torch.where(inside, reg, torch.zeros_like(reg))

    # restore() would hand back floats; a region label is an integer, so the cast is done
    # here rather than pretending 3.0 and 3 are the same thing.
    reg = reg.reshape(state['shape'])
    if state['kind'] == 'torch':
        return reg.to(torch.long)
    if state['kind'] == 'numpy':
        return reg.detach().cpu().numpy().astype(np.int64)
    return int(reg.item())
