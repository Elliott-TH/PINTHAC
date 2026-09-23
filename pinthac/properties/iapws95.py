"""IAPWS-95: the scientific formulation for the thermodynamic properties of ordinary water,
as a single differentiable Helmholtz free-energy surface that runs batched on the GPU.

Why this module is here:
    IAPWS-95 is the reference equation of state for water -- one Helmholtz energy
    phi(delta, tau) covering liquid, vapor and the whole supercritical region, from which
    every thermodynamic property follows by differentiation. That makes it the right
    thing to put underneath a solver: the properties are thermodynamically consistent
    with each other by construction, and because every property is an analytic derivative
    of one function, the whole surface is differentiable with respect to its inputs. IF97
    (properties/iapws97.py) is faster for a plain lookup, but it is a piecewise fit with
    visible seams at the region boundaries, and a solver that takes derivatives will feel
    them.

    The reason it is written in torch rather than NumPy is throughput. A supercritical
    channel solve evaluates properties at every axial node at every iteration, and a
    surrogate trained on this evaluates them a million points at a time. The whole
    formulation here is array arithmetic against stacked coefficient tables, so a batch
    of a million states costs one pass.

How it is used:
    Every property is read off one state evaluation, so the expensive part happens once:

        d = IAPWS95.helmholtz(rho, T)     # rho [kg/m^3], T [K]
        IAPWS95.p(d)                       # MPa
        IAPWS95.h(d)                       # J/kg
        IAPWS95.cp(d)                      # J/kg-K

    When the state is given as (T, p) rather than (rho, T), invert first with rho_Tp();
    when it is given as (h, p), with T_hp(). Both are differentiable in every argument --
    see rho_Tp for how, and why it matters.

    Transport properties are separate IAPWS releases and live in properties/iapws_transport.py;
    mu() and lam() below are thin wrappers that hand that module the thermodynamic
    derivatives it needs and that only IAPWS-95 can supply.

Reference:
    IAPWS R6-95(2018), "Revised Release on the IAPWS Formulation 1995 for the
    Thermodynamic Properties of Ordinary Water Substance for General and Scientific Use"
    (IAPWS95-2018.pdf). Wagner, W. and Pruss, A., J. Phys. Chem. Ref. Data
    31, 387 (2002) for the ancillary equations. Equation and table numbers in the
    docstrings below refer to the release.
"""
import torch

from pinthac.properties import iapws_backend as w
from pinthac.properties import iapws_transport as transport

device = w.device
accelerator = w.accelerator

# ====================
# Critical Properties
# ====================

Tc = 647.096         # Critical temperature [K], Eq. (1)
rhoc = 322.0         # Critical density [kg m^-3], Eq. (2)
R = 0.46151805       # Specific gas constant [kJ kg^-1 K^-1], Eq. (3)
pc = 22.064          # Critical pressure [MPa]
Tt = 273.16          # Triple-point temperature [K]


class IAPWS95:
    # =========================================
    # Table 1: ideal-gas part coefficients
    # (n_i^o, gamma_i^o), Eq. (5)
    # =========================================
    C0 = w.stack(
        [-8.32044648374970, 6.68321052759320, 3.00632,
         0.012436, 0.97315, 1.27950, 0.96956, 0.24873],
        [0.0, 0.0, 0.0,
         1.28728967, 3.53734222, 7.74073708, 9.24437796, 27.5075105])

    # =========================================
    # Table 2: residual part coefficients,
    # Eq. (6), verified against R6-95(2018).
    # The four groups are stacked one table
    # each so that a formula takes one trip to
    # the caller's device, and so that the
    # columns of a group cannot drift apart.
    # =========================================

    # ---- Group 1, terms 1-7: n, d, t ----
    C1 = w.stack(
        [0.012533547935523, 7.8957634722828, -8.7803203303561, 0.31802509345418,
         -0.26145533859358, -0.0078199751687981, 0.0088089493102134],
        [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0],
        [-0.5, 0.875, 1.0, 0.5, 0.75, 0.375, 1.0])

    # ---- Group 2, terms 8-51: n, d, t, c ----
    C2 = w.stack(
        [-0.66856572307965, 0.20433810950965, -6.6212605039687e-05, -0.19232721156002,
         -0.25709043003438, 0.16074868486251, -0.040092828925807, 3.9343422603254e-07,
         -7.5941377088144e-06, 0.00056250979351888, -1.5608652257135e-05, 1.1537996422951e-09,
         3.6582165144204e-07, -1.3251180074668e-12, -6.2639586912454e-10, -0.10793600908932,
         0.017611491008752, 0.22132295167546, -0.40247669763528, 0.58083399985759,
         0.0049969146990806, -0.031358700712549, -0.74315929710341, 0.4780732991548,
         0.020527940895948, -0.13636435110343, 0.014180634400617, 0.0083326504880713,
         -0.029052336009585, 0.038615085574206, -0.020393486513704, -0.0016554050063734,
         0.0019955571979541, 0.00015870308324157, -1.638856834253e-05, 0.043613615723811,
         0.034994005463765, -0.076788197844621, 0.022446277332006, -6.2689710414685e-05,
         -5.5711118565645e-10, -0.19905718354408, 0.31777497330738, -0.11841182425981],
        [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 7.0, 9.0, 10.0, 11.0, 13.0, 15.0,
         1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 9.0, 9.0, 9.0,
         9.0, 9.0, 10.0, 10.0, 12.0, 3.0, 4.0, 4.0, 5.0, 14.0, 3.0, 6.0, 6.0, 6.0],
        [4.0, 6.0, 12.0, 1.0, 5.0, 4.0, 2.0, 13.0, 9.0, 3.0, 4.0, 11.0, 4.0, 13.0, 1.0,
         7.0, 1.0, 9.0, 10.0, 10.0, 3.0, 7.0, 10.0, 10.0, 6.0, 10.0, 10.0, 1.0, 2.0, 3.0,
         4.0, 8.0, 6.0, 9.0, 8.0, 16.0, 22.0, 23.0, 23.0, 10.0, 50.0, 44.0, 46.0, 50.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
         2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 6.0, 6.0, 6.0, 6.0])

    # ---- Group 3, terms 52-54: n, d, t, alpha, beta, gamma, epsilon ----
    C3 = w.stack(
        [-31.306260323435, 31.546140237781, -2521.3154341695],
        [3.0, 3.0, 3.0],
        [0.0, 1.0, 4.0],
        [20.0, 20.0, 20.0],
        [150.0, 150.0, 250.0],
        [1.21, 1.21, 1.25],
        [1.0, 1.0, 1.0])

    # ---- Group 4, terms 55-56: n, a, b, B, C, D, A, beta ----
    C4 = w.stack(
        [-0.14874640856724, 0.31806110878444],
        [3.5, 3.5],
        [0.85, 0.95],
        [0.2, 0.2],
        [28.0, 32.0],
        [700.0, 800.0],
        [0.32, 0.32],
        [0.3, 0.3])

    @classmethod
    def Phi0(cls, delta, tau):
        """Ideal-gas part of the dimensionless Helmholtz energy and its derivatives, Eq. (5).

        Formulation:
            phi^o = ln(delta) + n1 + n2*tau + n3*ln(tau)
                    + sum_{i=4..8} n_i * ln(1 - exp(-gamma_i*tau))
            with the derivatives of Table 4. phi^o has no mixed derivative: it is a
            function of delta plus a function of tau, so phi^o_deltatau is identically
            zero and is returned as such.

        Valid range:
            All delta > 0 and tau > 0.

        Uncertainty:
            Not applicable on its own; see helmholtz().

        Reference:
            IAPWS R6-95(2018) Eq. (5), Table 1 (coefficients), Table 4 (derivatives).

        Inputs (1-D torch tensors, same shape):
            delta : reduced density rho/rho_c, dimensionless
            tau   : inverse reduced temperature T_c/T, dimensionless

        Returns:
            phi, phi_d, phi_dd, phi_t, phi_tt, phi_dt -- each a 1-D tensor
        """
        n, gam = w.on(cls.C0, delta)

        n1, n2, n3 = n[0], n[1], n[2]          # n1^o, n2^o, n3^o
        ng, gamg = n[3:], gam[3:]              # the five Einstein terms, i = 4..8

        phi = (torch.log(delta) + n1 + n2 * tau + n3 * torch.log(tau)
               + (ng * torch.log(1.0 - torch.exp(-gamg * tau))).sum(dim=0))

        phi_d = 1.0 / delta
        phi_dd = -1.0 / delta**2

        phi_t = (n2 + n3 / tau
                 + (ng * gamg * ((1.0 - torch.exp(-gamg * tau))**-1 - 1.0)).sum(dim=0))
        phi_tt = (-n3 / tau**2
                  - (ng * gamg**2 * torch.exp(-gamg * tau)
                     * (1.0 - torch.exp(-gamg * tau))**-2).sum(dim=0))

        phi_dt = torch.zeros_like(phi)

        return phi, phi_d, phi_dd, phi_t, phi_tt, phi_dt

    @classmethod
    def Phir(cls, delta, tau):
        """Residual part of the dimensionless Helmholtz energy and its derivatives, Eq. (6).

        Formulation:
            phi^r = sum_1 n*delta^d*tau^t                                   (group 1)
                  + sum_2 n*delta^d*tau^t*exp(-delta^c)                     (group 2)
                  + sum_3 n*delta^d*tau^t*exp(-alpha(delta-eps)^2 - beta(tau-gamma)^2)
                  + sum_4 n*Delta^b*delta*psi                               (group 4)
            with the derivatives of Table 5.

        Valid range:
            All delta > 0 and tau > 0; see helmholtz() for the range over which the
            release itself is valid.

        Uncertainty:
            Not applicable on its own; see helmholtz().

        Reference:
            IAPWS R6-95(2018) Eq. (6), Table 2 (coefficients), Table 5 (derivatives).

        Inputs (1-D torch tensors, same shape):
            delta : reduced density rho/rho_c, dimensionless
            tau   : inverse reduced temperature T_c/T, dimensionless

        Returns:
            phi, phi_d, phi_dd, phi_t, phi_tt, phi_dt -- each a 1-D tensor
        """
        n1, d1, t1 = w.on(cls.C1, delta)
        n2, d2, t2, c2 = w.on(cls.C2, delta)
        n3, d3, t3, alpha3, beta3, gamma3, epsilon3 = w.on(cls.C3, delta)
        n4, a4, b4, B4, C4, D4, A4, beta4 = w.on(cls.C4, delta)

        # ---- Group 1: simple polynomial terms ----
        phi_1 = (n1 * delta**d1 * tau**t1).sum(dim=0)
        phi_1_d = (n1 * d1 * delta**(d1 - 1) * tau**t1).sum(dim=0)
        phi_1_dd = (n1 * d1 * (d1 - 1) * delta**(d1 - 2) * tau**t1).sum(dim=0)
        phi_1_t = (n1 * t1 * delta**d1 * tau**(t1 - 1)).sum(dim=0)
        phi_1_tt = (n1 * t1 * (t1 - 1) * delta**d1 * tau**(t1 - 2)).sum(dim=0)
        phi_1_dt = (n1 * d1 * t1 * delta**(d1 - 1) * tau**(t1 - 1)).sum(dim=0)

        # ---- Group 2: exponential terms, exp(-delta^c) ----
        E2 = torch.exp(-delta**c2)
        phi_2 = (n2 * delta**d2 * tau**t2 * E2).sum(dim=0)
        phi_2_d = (n2 * E2 * delta**(d2 - 1) * tau**t2
                   * (d2 - c2 * delta**c2)).sum(dim=0)
        phi_2_dd = (n2 * E2 * delta**(d2 - 2) * tau**t2
                    * ((d2 - c2 * delta**c2) * (d2 - 1 - c2 * delta**c2)
                       - c2**2 * delta**c2)).sum(dim=0)
        phi_2_t = (n2 * t2 * delta**d2 * tau**(t2 - 1) * E2).sum(dim=0)
        phi_2_tt = (n2 * t2 * (t2 - 1) * delta**d2 * tau**(t2 - 2) * E2).sum(dim=0)
        phi_2_dt = (n2 * t2 * tau**(t2 - 1) * E2 * delta**(d2 - 1)
                    * (d2 - c2 * delta**c2)).sum(dim=0)

        # ---- Group 3: Gaussian bell terms ----
        E3 = torch.exp(-alpha3 * (delta - epsilon3)**2 - beta3 * (tau - gamma3)**2)
        phi_3 = (n3 * delta**d3 * tau**t3 * E3).sum(dim=0)

        dEd = -2 * alpha3 * (delta - epsilon3) * E3
        d2Ed = (2 * alpha3 * (delta - epsilon3))**2 * E3 - 2 * alpha3 * E3
        dEt = -2 * beta3 * (tau - gamma3) * E3
        d2Et = (2 * beta3 * (tau - gamma3))**2 * E3 - 2 * beta3 * E3
        dEdt = 4 * alpha3 * beta3 * (delta - epsilon3) * (tau - gamma3) * E3

        phi_3_d = (n3 * delta**d3 * tau**t3
                   * (d3 / delta * E3 + dEd)).sum(dim=0)
        phi_3_dd = (n3 * tau**t3 * delta**(d3 - 2)
                    * (d3 * (d3 - 1) * E3
                       + 2 * d3 * delta * dEd
                       + delta**2 * d2Ed)).sum(dim=0)
        phi_3_t = (n3 * delta**d3 * tau**(t3 - 1)
                   * (t3 * E3 + tau * dEt)).sum(dim=0)
        phi_3_tt = (n3 * delta**d3 * tau**(t3 - 2)
                    * (t3 * (t3 - 1) * E3
                       + 2 * t3 * tau * dEt
                       + tau**2 * d2Et)).sum(dim=0)
        phi_3_dt = (n3 * delta**(d3 - 1) * tau**(t3 - 1)
                    * (d3 * t3 * E3
                       + d3 * tau * dEt
                       + t3 * delta * dEd
                       + delta * tau * dEdt)).sum(dim=0)

        # ---- Group 4: non-analytic (critical-region) terms ----
        # These are singular at delta = 1 exactly, and not removably so in floating point:
        # d2Delta_dd2 below divides by (delta-1), and carries ((delta-1)^2) raised to
        # 1/(2*beta) - 2 = -1/3. Both blow up at the critical density, and the resulting
        # NaN propagates into phi_dd and so into pressure derivatives, cp and the speed of
        # sound -- IAPWS95.p_rho and cp both returned NaN at exactly rho = 322.0 kg/m^3.
        #
        # Nudging delta off exactly 1 by 1e-11 removes it. That is 3.2e-9 kg/m^3 in
        # density, which is four orders of magnitude inside the region both R12-08 and
        # R15-11 already flag as unreliable ("approximately within 0.01 kg/m^3 of rho_c on
        # the critical isotherm"), so it cannot move any physically meaningful result --
        # it only replaces a NaN with the value from immediately beside it. The shadowed
        # name applies to this group alone; groups 1 to 3 are analytic at delta = 1 and
        # keep the exact value.
        delta = torch.where(torch.abs(delta - 1.0) < 1.0e-11, delta + 1.0e-11, delta)

        theta = (1 - tau) + A4 * ((delta - 1)**2) ** (1 / (2 * beta4))
        Delta = theta**2 + B4 * ((delta - 1)**2) ** a4
        psi = torch.exp(-C4 * (delta - 1)**2 - D4 * (tau - 1)**2)

        X = ((delta - 1)**2) ** (1 / (2 * beta4) - 1)

        dtheta_dd = A4 / beta4 * X * (delta - 1)
        dDelta_dd = (delta - 1) * (A4 * theta * (2 / beta4) * X
                                   + 2 * B4 * a4 * ((delta - 1)**2)**(a4 - 1))
        d2Delta_dd2 = ((1 / (delta - 1)) * dDelta_dd
                       + (delta - 1)**2 * (4 * B4 * a4 * (a4 - 1) * ((delta - 1)**2)**(a4 - 2)
                                           + 2 * (A4 / beta4)**2 * X**2
                                           + A4 * theta * (4 / beta4) * (1 / (2 * beta4) - 1)
                                           * ((delta - 1)**2)**(1 / (2 * beta4) - 2)))

        dDeltab_dd = b4 * Delta**(b4 - 1) * dDelta_dd
        d2Deltab_dd2 = (b4 * (b4 - 1) * Delta**(b4 - 2) * dDelta_dd**2
                        + b4 * Delta**(b4 - 1) * d2Delta_dd2)
        dDeltab_dt = -2 * theta * b4 * Delta**(b4 - 1)
        d2Deltab_dt2 = 2 * b4 * Delta**(b4 - 1) + 4 * theta**2 * b4 * (b4 - 1) * Delta**(b4 - 2)
        d2Deltab_ddt = -2 * b4 * Delta**(b4 - 1) * dtheta_dd \
                       - 2 * theta * b4 * (b4 - 1) * Delta**(b4 - 2) * dDelta_dd

        dpsi_dd = -2 * C4 * (delta - 1) * psi
        d2psi_dd2 = (2 * C4 * (delta - 1)**2 - 1) * 2 * C4 * psi
        dpsi_dt = -2 * D4 * (tau - 1) * psi
        d2psi_dt2 = (2 * D4 * (tau - 1)**2 - 1) * 2 * D4 * psi
        d2psi_ddt = 4 * C4 * D4 * (delta - 1) * (tau - 1) * psi

        phi_4 = (n4 * Delta**b4 * delta * psi).sum(dim=0)

        phi_4_d = (n4 * (Delta**b4 * (psi + delta * dpsi_dd)
                         + dDeltab_dd * delta * psi)).sum(dim=0)

        phi_4_dd = (n4 * (Delta**b4 * (2 * dpsi_dd + delta * d2psi_dd2)
                          + 2 * dDeltab_dd * (psi + delta * dpsi_dd)
                          + d2Deltab_dd2 * delta * psi)).sum(dim=0)

        phi_4_t = (n4 * delta * (dDeltab_dt * psi + Delta**b4 * dpsi_dt)).sum(dim=0)

        phi_4_tt = (n4 * delta * (d2Deltab_dt2 * psi
                                  + 2 * dDeltab_dt * dpsi_dt
                                  + Delta**b4 * d2psi_dt2)).sum(dim=0)

        phi_4_dt = (n4 * (Delta**b4 * (dpsi_dt + delta * d2psi_ddt)
                          + delta * dDeltab_dd * dpsi_dt
                          + dDeltab_dt * (psi + delta * dpsi_dd)
                          + delta * d2Deltab_ddt * psi)).sum(dim=0)

        phi = phi_1 + phi_2 + phi_3 + phi_4
        phi_d = phi_1_d + phi_2_d + phi_3_d + phi_4_d
        phi_dd = phi_1_dd + phi_2_dd + phi_3_dd + phi_4_dd
        phi_t = phi_1_t + phi_2_t + phi_3_t + phi_4_t
        phi_tt = phi_1_tt + phi_2_tt + phi_3_tt + phi_4_tt
        phi_dt = phi_1_dt + phi_2_dt + phi_3_dt + phi_4_dt

        return phi, phi_d, phi_dd, phi_t, phi_tt, phi_dt

    @classmethod
    def helmholtz(cls, rho, T):
        """Evaluate the IAPWS-95 Helmholtz surface at (rho, T), once.

        Formulation:
            f(rho,T)/(R*T) = phi^o(delta, tau) + phi^r(delta, tau),
            delta = rho/322 kg/m^3, tau = 647.096 K/T, Eq. (4).

        Valid range:
            From the melting line to 1273 K at pressures to 1000 MPa, and the equation
            also behaves sensibly in the metastable regions bordering the saturation
            line. Nothing is range-checked here.

        Uncertainty:
            See R6-95(2018) Sec. 7 -- density to about 0.0001 percent in the liquid at
            ambient conditions, degrading near the critical point and in the
            high-temperature gas.

        Reference:
            IAPWS R6-95(2018) Eqs. (4)-(6).

        Inputs (float, numpy array, or torch tensor; broadcastable):
            rho : density, kg/m^3
            T   : temperature, K

        Returns:
            d : dict of delta, tau, and the six derivatives each of phi^o and phi^r,
                keyed phi0, phi0_d, phi0_dd, phi0_t, phi0_tt, phi0_dt and phir, phir_d,
                phir_dd, phir_t, phir_tt, phir_dt. Each entry is the same type and shape
                that rho and T broadcast to, and a torch entry carries the autograd graph
                back to rho and T. Pass the dict to the property accessors below.
        """
        (rho_, T_), state = w.prepare(rho, T)
        vals = cls._state(rho_, T_)
        return {key: w.restore(val, state) for key, val in vals.items()}

    @classmethod
    def _state(cls, rho, T):
        """The same state dict helmholtz() returns, but in raw 1-D torch tensors.

        The solvers below -- saturation(), rho_Tp(), T_hp() -- evaluate this hundreds of
        times inside their iteration loops, where the type round-trip helmholtz() does on
        every call would be wasted work and, for a NumPy caller, would break the loop's
        arithmetic. Private, and takes float64 tensors only.
        """
        delta = rho / rhoc
        tau = Tc / T

        phi0, phi0_d, phi0_dd, phi0_t, phi0_tt, phi0_dt = cls.Phi0(delta, tau)
        phir, phir_d, phir_dd, phir_t, phir_tt, phir_dt = cls.Phir(delta, tau)

        return {
            'delta': delta, 'tau': tau,
            'phi0': phi0, 'phi0_d': phi0_d, 'phi0_dd': phi0_dd,
            'phi0_t': phi0_t, 'phi0_tt': phi0_tt, 'phi0_dt': phi0_dt,
            'phir': phir, 'phir_d': phir_d, 'phir_dd': phir_dd,
            'phir_t': phir_t, 'phir_tt': phir_tt, 'phir_dt': phir_dt,
        }

    # =========================================
    # Properties, Table 3 of R6-95(2018).
    # Each is closed-form arithmetic on the dict
    # helmholtz() returned, so none of them
    # re-evaluates the 56-term sum.
    # =========================================

    @classmethod
    def p(cls, d, units='MPa'):
        """Pressure [MPa by default] from a helmholtz() state.

        p/(rho*R*T) = 1 + delta*phi^r_delta.

        Inputs:
            d     : state dict from helmholtz()
            units : 'Pa', 'kPa' or 'MPa'
        Returns:
            p : pressure, same type as the state
        """
        scale = w.pressure_units[units]
        T = Tc / d['tau']
        return scale * rhoc * d['delta'] * R * T * (1.0 + d['delta'] * d['phir_d'])

    @classmethod
    def p_rho(cls, d, units='MPa'):
        """(dp/drho)_T [MPa/(kg/m^3) by default] from a helmholtz() state.

        dp/drho = R*T*(1 + 2*delta*phi^r_delta + delta^2*phi^r_deltadelta).

        This is the derivative the density solve uses for its Newton step and the one
        properties/iapws_transport.py needs inverted as the isothermal compressibility, which
        is why it is a named accessor and not left to the caller to differentiate.
        """
        scale = w.pressure_units[units]
        T = Tc / d['tau']
        return scale * R * T * (1.0 + 2.0 * d['delta'] * d['phir_d']
                                + d['delta']**2 * d['phir_dd'])

    @classmethod
    def drhodp(cls, d):
        """Isothermal compressibility (drho/dp)_T [kg/m^3/MPa], for iapws_transport.py."""
        return 1.0 / cls.p_rho(d, units='MPa')

    @classmethod
    def s(cls, d, units='J'):
        """Specific entropy [J/kg-K by default].

        s/R = tau*(phi^o_tau + phi^r_tau) - phi^o - phi^r
        """
        scale = w.energy_units[units]
        return scale * R * (d['tau'] * (d['phi0_t'] + d['phir_t'])
                            - d['phi0'] - d['phir'])

    @classmethod
    def h(cls, d, units='J'):
        """Specific enthalpy [J/kg by default].

        h/(R*T) = 1 + tau*(phi^o_tau + phi^r_tau) + delta*phi^r_delta
        """
        scale = w.energy_units[units]
        T = Tc / d['tau']
        return scale * R * T * (1.0 + d['tau'] * (d['phi0_t'] + d['phir_t'])
                                + d['delta'] * d['phir_d'])

    @classmethod
    def u(cls, d, units='J'):
        """Specific internal energy [J/kg by default].

        u/(R*T) = tau*(phi^o_tau + phi^r_tau)
        """
        scale = w.energy_units[units]
        T = Tc / d['tau']
        return scale * R * T * d['tau'] * (d['phi0_t'] + d['phir_t'])

    @classmethod
    def cv(cls, d, units='J'):
        """Isochoric heat capacity [J/kg-K by default].

        cv/R = -tau^2 * (phi^o_tautau + phi^r_tautau)
        """
        scale = w.energy_units[units]
        return scale * (-R * d['tau']**2 * (d['phi0_tt'] + d['phir_tt']))

    @classmethod
    def cp(cls, d, units='J'):
        """Isobaric heat capacity [J/kg-K by default].

        cp/R = -tau^2*(phi^o_tautau + phi^r_tautau)
               + (1 + delta*phi^r_delta - delta*tau*phi^r_deltatau)^2
                 / (1 + 2*delta*phi^r_delta + delta^2*phi^r_deltadelta)

        The denominator is (dp/drho)_T in reduced form, so cp diverges where the isotherm
        goes flat. That is not a defect: it is the reason the pseudo-critical peak in cp
        exists, and it is what supercritical heat transfer is built around.
        """
        scale = w.energy_units[units]
        num = (1.0 + d['delta'] * d['phir_d'] - d['delta'] * d['tau'] * d['phir_dt'])**2
        den = 1.0 + 2.0 * d['delta'] * d['phir_d'] + d['delta']**2 * d['phir_dd']
        return scale * (-R * d['tau']**2 * (d['phi0_tt'] + d['phir_tt']) + R * num / den)

    @classmethod
    def c(cls, d):
        """Speed of sound [m/s] from a helmholtz() state.

        w^2/(R*T) = 1 + 2*delta*phi^r_delta + delta^2*phi^r_deltadelta
                    + (1 + delta*phi^r_delta - delta*tau*phi^r_deltatau)^2 / (cv/R)

        R is in kJ/kg/K, so the factor of 1000 turns R*T into m^2/s^2.
        """
        cv_R = -d['tau']**2 * (d['phi0_tt'] + d['phir_tt'])
        num = (1.0 + d['delta'] * d['phir_d'] - d['delta'] * d['tau'] * d['phir_dt'])**2
        den = 1.0 + 2.0 * d['delta'] * d['phir_d'] + d['delta']**2 * d['phir_dd']
        val = R * 1000.0 * (Tc / d['tau']) * (den + num / cv_R)
        return w.sqrt(val)

    # =========================================
    # Transport properties.
    # The formulations live in iapws_transport.py --
    # they are separate IAPWS releases, not part
    # of R6-95. What IAPWS-95 supplies is the
    # thermodynamic input they need: (drho/dp)_T,
    # cp and cv, over the whole surface rather
    # than one IF97 region.
    # =========================================

    @classmethod
    def mu(cls, d, enhancement=True):
        """Dynamic viscosity [Pa-s] at a helmholtz() state, IAPWS R12-08.

        Formulation:
            See transport.VISC.mu. This wrapper computes (drho/dp)_T at T and at T_R and
            passes both down.

            The second compressibility costs an extra Helmholtz evaluation at (rho, T_R),
            so mu is about twice the price with the enhancement on -- worth knowing
            before benchmarking property throughput. Pass enhancement=False for the
            industrial simplification mu2 = 1, which R12-08 Sec. 2.8 and Sec. 3 sanction
            outside the near-critical region, where it agrees with the full form to
            better than the correlation's own uncertainty.

            Leaving it on by default is deliberate: mu feeds the thermal-conductivity
            critical enhancement (R15-11 Eq. 18 divides by it), so a mu missing its own
            enhancement silently inflates lambda near the critical point.

        Valid range:
            That of R12-08; see transport.VISC.mu.

        Uncertainty:
            That of R12-08; see transport.VISC.mu.

        Reference:
            IAPWS R12-08 (IAWPS_Viscosity.pdf).

        Inputs:
            d           : state dict from helmholtz()
            enhancement : include the critical enhancement mu2

        Returns:
            mu : dynamic viscosity, Pa-s, same type as the state
        """
        rho = rhoc * d['delta']
        T = Tc / d['tau']
        if not enhancement:
            return transport.VISC.mu(rho, T)

        drhodp_T = cls.drhodp(d)
        T_R = 1.5 * Tc
        # rho arrives as whatever the caller passed in -- float, numpy or torch -- so
        # build the matching constant by arithmetic rather than with a library-specific
        # full_like, which would pin this to one backend.
        d_R = cls.helmholtz(rho, rho * 0.0 + T_R)
        drhodp_TR = cls.drhodp(d_R)
        return transport.VISC.mu(rho, T, drhodp_T=drhodp_T, drhodp_TR=drhodp_TR)

    @classmethod
    def lam(cls, d, mu=None):
        """Thermal conductivity [W/m-K] at a helmholtz() state, IAPWS R15-11.

        Formulation:
            See transport.COND.lam.

            The viscosity is the expensive input: with its critical enhancement on, it
            costs a second Helmholtz evaluation at the reference temperature T_R. A
            caller who already has mu at this state -- getprop._getprop does, since it
            returns both -- passes it in rather than paying for it twice.

        Valid range:
            That of R15-11; see transport.COND.lam.

        Uncertainty:
            That of R15-11; see transport.COND.lam.

        Reference:
            IAPWS R15-11 (ThCond.pdf).

        Inputs:
            d  : state dict from helmholtz()
            mu : optional, dynamic viscosity [Pa-s] already computed at this state. It
                 must carry its own critical enhancement -- R15-11 Eq. (18) divides by
                 it, so a mu without one inflates lambda near the critical point.

        Returns:
            lam : thermal conductivity, W/m-K, same type as the state
        """
        rho = rhoc * d['delta']
        T = Tc / d['tau']
        if mu is None:
            mu = cls.mu(d)
        return transport.COND.lam(rho, T, cls.drhodp(d),
                                  cls.cp(d, units='kJ'), cls.cv(d, units='kJ'), mu)

    # =========================================
    # Saturation line.
    # IAPWS-95 has no explicit saturation
    # equation: the two-phase boundary is where
    # the Maxwell criterion holds, and it has to
    # be solved for. The ancillary equations of
    # Wagner and Pruss (2002) Eqs. (2.5)-(2.7)
    # seed that solve; they are not the answer.
    # =========================================

    A_SAT = w.stack(
        [-7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502],
        [1.0, 1.5, 3.0, 3.5, 4.0, 7.5])

    B_SAT = w.stack(
        [1.99274064, 1.09965342, -0.510839303, -1.75493479, -45.5170352, -6.74694450e5],
        [1 / 3, 2 / 3, 5 / 3, 16 / 3, 43 / 3, 110 / 3])

    C_SAT = w.stack(
        [-2.03150240, -2.68302940, -5.38626492, -17.2991605, -44.7586581, -63.9201063],
        [2 / 6, 4 / 6, 8 / 6, 18 / 6, 37 / 6, 71 / 6])

    @classmethod
    def _sat_ancillary(cls, T):
        """Approximate p_sat, rho_f and rho_g from the IAPWS-95 ancillary equations.

        Formulation:
            ln(p_sat/p_c)   = (T_c/T) * sum a_i * theta^(t_i)
            rho_f/rho_c     = 1 + sum b_i * theta^(u_i)
            ln(rho_g/rho_c) = sum c_i * theta^(v_i),      theta = 1 - T/T_c

        Valid range:
            273.16 K <= T <= T_c. theta must be non-negative -- a non-integer power of a
            negative number is NaN -- which is why saturation() clamps T below T_c before
            calling this.

        Uncertainty:
            A few hundredths of a percent, and it does not propagate: these values are
            only a seed.

        Reference:
            Wagner, W. and Pruss, A., J. Phys. Chem. Ref. Data 31, 387 (2002),
            Eqs. (2.5), (2.6) and (2.7).

        Inputs:
            T : temperature, K, 1-D torch tensor

        Returns:
            p0, rhof0, rhog0 : saturation pressure [MPa] and the two saturated densities
                               [kg/m^3], each a 1-D tensor
        """
        a, ta = w.on(cls.A_SAT, T)
        b, tb = w.on(cls.B_SAT, T)
        c, tcv = w.on(cls.C_SAT, T)

        theta = 1.0 - T / Tc

        ln_p_pc = (Tc / T) * (a * theta**ta).sum(dim=0)
        rho_f_rat = 1.0 + (b * theta**tb).sum(dim=0)
        ln_rho_g = (c * theta**tcv).sum(dim=0)

        return pc * torch.exp(ln_p_pc), rhoc * rho_f_rat, rhoc * torch.exp(ln_rho_g)

    @classmethod
    def saturation(cls, T, iters=20):
        """The coexisting liquid and vapor state at temperature T.

        Formulation:
            Solve, for rho_f and rho_g at fixed T,
                F1 = p(rho_f,T) - p(rho_g,T)                    = 0
                F2 = g(rho_f,T) - g(rho_g,T)                    = 0,  g = h - T*s
            by Newton's method on the 2x2 system, seeded from _sat_ancillary().

            Newton is quadratic from this seed, and twenty iterations reaches the
            equation of state's own precision across the whole line -- the cold end,
            where the vapor density is five orders of magnitude below the liquid one,
            is the slow case and takes about twice as many steps as the middle. The loop
            runs a fixed count with no convergence test, because testing one would mean
            reading a tensor back to the host every iteration and stalling the GPU. The correction step afterwards is what carries the derivative with
            respect to T -- see rho_Tp() for why that construction gives the exact
            implicit derivative.

        Valid range:
            273.16 K <= T <= T_c = 647.096 K. T is clamped into that interval, since
            there is no saturation state above the critical point to return.

        Uncertainty:
            The two saturated densities come out exact to the equation of state, to about
            1e-10 relative against the R6-95 Table 8 check values. The saturation pressure
            is not that good at the cold end and cannot be: below about 300 K the liquid
            pressure is a difference of terms nine digits larger than itself, so float64
            leaves roughly 1e-8 relative noise in it whatever the solver does. R6-95's own
            Table 7 carries a footnote saying exactly this. Over the whole line p_sat
            agrees with the reference implementation to better than 3e-8 relative, and to
            1e-10 above 350 K.

        Reference:
            IAPWS R6-95(2018) Sec. 6.2 (phase-equilibrium condition, Table 3).

        Inputs:
            T     : temperature, K (float, numpy array, or torch tensor)
            iters : Newton iterations

        Returns:
            d : dict with keys 'T' [K], 'p' [MPa], 'rho_f' and 'rho_g' [kg/m^3], each the
                same type and shape as the input
        """
        (T_,), state = w.prepare(T)
        T_ = torch.clamp(T_, min=Tt, max=Tc - 1.0e-6)

        _, rhof0, rhog0 = cls._sat_ancillary(T_)

        with torch.no_grad():
            T_fix = T_.detach()
            rho_f, rho_g = rhof0.detach().clone(), rhog0.detach().clone()
            for _ in range(iters):
                rho_f, rho_g = cls._sat_step(rho_f, rho_g, T_fix)

        # One more step, this time carrying the graph, so that d(rho_f)/dT and
        # d(rho_g)/dT come out of the implicit function theorem instead of being lost.
        rho_f, rho_g = cls._sat_step(rho_f, rho_g, T_)
        psat = cls.p(cls._state(rho_f, T_), units='MPa')

        vals = {'T': T_, 'p': psat, 'rho_f': rho_f, 'rho_g': rho_g}
        return {key: w.restore(val, state) for key, val in vals.items()}

    @classmethod
    def _sat_step(cls, rho_f, rho_g, T):
        """One Newton step of the saturation solve. Private; see saturation() for the system.

        The step is taken with a detached Jacobian and a residual that keeps whatever
        graph T carries, so calling this once more after convergence leaves the value
        alone and installs the correct derivative with respect to T.
        """
        d_f = cls._state(rho_f, T)
        d_g = cls._state(rho_g, T)

        F1 = cls.p(d_f, units='MPa') - cls.p(d_g, units='MPa')
        g_f = cls.h(d_f, units='kJ') - T * cls.s(d_f, units='kJ')
        g_g = cls.h(d_g, units='kJ') - T * cls.s(d_g, units='kJ')
        F2 = g_f - g_g

        with torch.no_grad():
            pr_f = cls.p_rho(d_f, units='MPa')
            pr_g = cls.p_rho(d_g, units='MPa')
            # dg/drho = v * dp/drho at constant T, and p_rho is in MPa so the 1e3 puts
            # the Gibbs residual in the kJ/kg that F2 is measured in.
            J11, J12 = pr_f, -pr_g
            J21, J22 = 1.0e3 * pr_f / rho_f, -1.0e3 * pr_g / rho_g
            det = J11 * J22 - J12 * J21

        d_rho_f = -(F1 * J22 - F2 * J12) / det
        d_rho_g = -(F2 * J11 - F1 * J21) / det

        rho_f = torch.clamp(rho_f + d_rho_f, rhoc * (1.0 + 1.0e-6), rhoc * 3.5)
        rho_g = torch.clamp(rho_g + d_rho_g, rhoc * 1.0e-8, rhoc * (1.0 - 1.0e-6))
        return rho_f, rho_g

    @classmethod
    def p_sat(cls, T):
        """Saturation pressure [MPa] at temperature T [K]. See saturation()."""
        return cls.saturation(T)['p']

    @classmethod
    def T_sat(cls, p, iters=60):
        """Saturation temperature [K] at pressure p [MPa].

        Formulation:
            Bisection on the ancillary saturation pressure over [T_t, T_c], then three
            Newton steps on the exact p_sat, then one correction step

                T <- T - (p_sat(T) - p) / (dp_sat/dT)

            with dp_sat/dT taken from the Clausius-Clapeyron relation,
                dp_sat/dT = (s_g - s_f) / (v_g - v_f),
            evaluated on the converged saturation state. Using Clapeyron rather than
            differentiating the saturation solve is what makes T_sat differentiable in p
            at all: bisection is a chain of comparisons and carries no gradient, but the
            correction step installs dT/dp = 1/(dp_sat/dT) exactly.

        Valid range:
            611.657 Pa <= p <= 22.064 MPa. Outside it the bracket does not contain a root
            and the answer is an endpoint.

        Uncertainty:
            Exact to the equation of state.

        Reference:
            IAPWS R6-95(2018); the Clapeyron correction is this library's construction.

        Inputs:
            p     : pressure, MPa (float, numpy array, or torch tensor)
            iters : bisection halvings

        Returns:
            T_sat : saturation temperature, K, same type as the input
        """
        (p_,), state = w.prepare(p)

        with torch.no_grad():
            p_fix = p_.detach()

            lo = torch.full_like(p_fix, Tt)
            hi = torch.full_like(p_fix, Tc - 1.0e-6)
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                p_anc, _, _ = cls._sat_ancillary(mid)
                too_hot = p_anc > p_fix
                hi = torch.where(too_hot, mid, hi)
                lo = torch.where(too_hot, lo, mid)
            T_star = 0.5 * (lo + hi)

            # Stage two: Newton on the exact p_sat, with dp_sat/dT from Clapeyron. From
            # a seed this good three steps is already past the equation of state's own
            # precision.
            for _ in range(3):
                T_star = T_star - (cls._p_sat(T_star) - p_fix) / cls._dpsat_dT(T_star)
                T_star = torch.clamp(T_star, min=Tt, max=Tc - 1.0e-6)

            dpdT = cls._dpsat_dT(T_star)

        T = T_star - (cls._p_sat(T_star) - p_) / dpdT
        return w.restore(T, state)

    @classmethod
    def _dpsat_dT(cls, T):
        """Slope of the saturation line [MPa/K] from the Clausius-Clapeyron relation.

        dp_sat/dT = (s_g - s_f) / (v_g - v_f), evaluated on the converged saturation
        state. Private; it is what makes T_sat's Newton step and its derivative in p
        exact without differentiating through the saturation solve.
        """
        sat = cls._saturation(T)
        v_f, v_g = 1.0 / sat['rho_f'], 1.0 / sat['rho_g']
        s_f = cls.s(cls._state(sat['rho_f'], T), units='kJ')
        s_g = cls.s(cls._state(sat['rho_g'], T), units='kJ')
        return 1.0e-3 * (s_g - s_f) / (v_g - v_f)          # kPa/K -> MPa/K

    @classmethod
    def _saturation(cls, T, iters=20):
        """Tensor-level saturation solve. Private; see saturation()."""
        _, rhof0, rhog0 = cls._sat_ancillary(T)
        rho_f, rho_g = rhof0, rhog0
        for _ in range(iters):
            rho_f, rho_g = cls._sat_step(rho_f, rho_g, T)
        return {'rho_f': rho_f, 'rho_g': rho_g}

    @classmethod
    def _p_sat(cls, T):
        """Tensor-level saturation pressure [MPa]. Private; see p_sat()."""
        sat = cls._saturation(torch.clamp(T, min=Tt, max=Tc - 1.0e-6))
        return cls.p(cls._state(sat['rho_f'], T), units='MPa')

    # =========================================
    # Inversions.
    # helmholtz() takes (rho, T), but a solver
    # knows (T, p) or (h, p). These two invert
    # the surface, and both stay differentiable
    # in every argument -- see rho_Tp for how.
    # =========================================

    @classmethod
    def rho_Tp(cls, T, p, newton_iters=60):
        """Density [kg/m^3] at a given temperature and pressure.

        Formulation:
            Seed, then damped Newton on p(rho, T) - p = 0 with the analytic (dp/drho)_T,
            then one differentiable correction step.

            The seed is physically motivated: the saturated-liquid ancillary density for a
            compressed liquid, the ideal-gas density otherwise. A global bisection over
            the full [1e-3, 1300] kg/m^3 range -- which an earlier version of this
            function used -- is NOT safe. Away from the true (T, rho) branch, at
            subcritical temperatures, the IAPWS-95 residual terms (several carry tau
            exponents up to 50) stop cancelling cleanly in floating point and the computed
            pressure swings by ten or more orders of magnitude between adjacent densities.
            A bracket search reading that noise as a monotonic signal can lock onto a
            spurious root near the critical density instead of the physical one -- it
            silently returned rho = 322 kg/m^3 for T = 300 K, p = 1 MPa instead of the
            correct 997. Seeding close to the true branch and damping the Newton step
            keeps every evaluation on the well-behaved side of that region.

            The Newton loop itself runs under no_grad and uses the analytic derivative, so
            it costs one state evaluation per iteration and no graph. What makes the
            result differentiable is the single correction step afterwards:

                rho = rho* - (p(rho*, T) - p) / (dp/drho)_T

            At convergence the numerator is zero to machine precision, so the value does
            not move; but the numerator carries T and p, and the denominator is detached,
            so torch reads off exactly the implicit derivatives of the converged root,
                (drho/dp)_T = 1 / (dp/drho)_T
                (drho/dT)_p = -(dp/dT)_rho / (dp/drho)_T
            This matters because the DeepONet power surrogate is trained through this
            function: without the correction step, rho would be a constant as far as
            autograd is concerned, and every gradient that reaches a property would be
            wrong rather than merely inaccurate.

        Valid range:
            Assumes a single real root in rho at the given (T, p) -- that is, a
            supercritical pressure or a state clearly off the two-phase dome. This is not
            a phase-aware solver; inside the dome use saturation().

        Uncertainty:
            Exact to the equation of state, to Newton's convergence.

        Reference:
            IAPWS R6-95(2018); the seeding and damping strategy is this library's.

        Inputs (float, numpy array, or torch tensor; broadcastable):
            T            : temperature, K
            p            : pressure, MPa
            newton_iters : damped Newton iterations

        Returns:
            rho : density, kg/m^3, same type as the inputs
        """
        (T_, p_), state = w.prepare(T, p)

        with torch.no_grad():
            rho_star = cls._rho_Tp(T_.detach(), p_.detach(), newton_iters)
            dp_drho = cls.p_rho(cls._state(rho_star, T_.detach()), units='MPa')

        res = cls.p(cls._state(rho_star, T_), units='MPa') - p_
        return w.restore(rho_star - res / dp_drho, state)

    @classmethod
    def _rho_Tp(cls, T, p, newton_iters=60):
        """Tensor-level damped Newton solve for density. Private; see rho_Tp().

        The step is clamped to half the current density in either direction. Undamped,
        a Newton step taken where the isotherm is nearly flat -- which is most of the
        near-critical region -- can be several hundred kg/m^3 and land at a negative
        density, from which there is no way back.
        """
        psat0, rhof0, _ = cls._sat_ancillary(torch.clamp(T, min=Tt, max=Tc - 1.0e-6))
        rho_ig = torch.clamp(p * 1000.0 / (R * T), min=1.0e-3)
        liquid_like = (T < Tc) & (p >= psat0)
        rho = torch.clamp(torch.where(liquid_like, rhof0, rho_ig), 1.0e-3, 1300.0)

        for _ in range(newton_iters):
            d = cls._state(rho, T)
            F = cls.p(d, units='MPa') - p
            dF = cls.p_rho(d, units='MPa')
            step = torch.clamp(F / dF, -0.5 * rho, 0.5 * rho)
            rho = torch.clamp(rho - step, 1.0e-4, 1300.0)
        return rho

    @classmethod
    def T_hp(cls, h, p, iters=60, T_lo=273.16, T_hi=1300.0):
        """Temperature [K] at a given specific enthalpy and pressure.

        Formulation:
            Bisection on h(T)|_p over [T_lo, T_hi], where each evaluation is a nested
            rho_Tp() solve, followed by one differentiable correction step

                T <- T - (h(T*, p) - h) / cp

            since (dh/dT)_p is exactly cp. As in rho_Tp(), the bisection is detached and
            the correction installs the derivatives: dT/dh = 1/cp at fixed p, and
            dT/dp = -(dh/dp)_T / cp at fixed h, the latter arriving through the
            differentiable rho_Tp() inside the residual.

            Bisection rather than Newton because h(T)|_p has a very steep, very narrow
            rise through the pseudo-critical line -- at 25 MPa, cp peaks above 100
            kJ/kg-K -- and a Newton step taken just off that peak overshoots by hundreds
            of kelvin. Bisection does not care how steep the function is, only that it is
            monotonic.

        Valid range:
            Assumes h is monotonically increasing in T at fixed p, which holds away from
            the two-phase dome -- in particular on the supercritical isobars the
            single-channel analysis runs on. Inside the dome h is flat in T and this
            returns the bracket's midpoint, not a meaningful temperature.

        Uncertainty:
            Exact to the equation of state, to the bisection tolerance: after 60 halvings
            of a 1027 K bracket, far below floating-point resolution.

        Reference:
            IAPWS R6-95(2018); the inversion strategy is this library's.

        Inputs (float, numpy array, or torch tensor; broadcastable):
            h            : specific enthalpy, kJ/kg
            p            : pressure, MPa
            iters        : bisection halvings
            T_lo, T_hi   : bracket, K

        Returns:
            T : temperature, K, same type as the inputs
        """
        (h_, p_), state = w.prepare(h, p)

        with torch.no_grad():
            h_fix, p_fix = h_.detach(), p_.detach()
            lo = torch.full_like(h_fix, T_lo)
            hi = torch.full_like(h_fix, T_hi)
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                h_mid = cls.h(cls._state(cls._rho_Tp(mid, p_fix), mid), units='kJ')
                too_cold = h_mid < h_fix
                lo = torch.where(too_cold, mid, lo)
                hi = torch.where(too_cold, hi, mid)
            T_star = 0.5 * (lo + hi)
            cp_star = cls.cp(cls._state(cls._rho_Tp(T_star, p_fix), T_star), units='kJ')

        # The residual is rebuilt with the differentiable rho_Tp so that the pressure
        # dependence of h at fixed T reaches the correction step.
        rho_ = cls.rho_Tp(T_star, p_)
        res = cls.h(cls._state(rho_, T_star), units='kJ') - h_
        return w.restore(T_star - res / cp_star, state)
