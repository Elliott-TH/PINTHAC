"""Transport and interfacial properties of ordinary water: viscosity, thermal conductivity
and surface tension.

Why this module is here:
    These three models used to live at the bottom of iapws97.py, which was wrong on both
    counts. None of them is part of IAPWS-IF97 -- they are three separate IAPWS releases
    (R12-08, R15-11 and R1-76), published years apart, on a different independent-variable
    basis. IF97 is a set of equations in (p, T); viscosity and thermal conductivity are
    equations in (rho, T), and they are meant to be evaluated with densities and
    derivatives from IAPWS-95, not from IF97. Keeping them inside the IF97 module implied
    a validity range they do not have, and made the IAPWS-95 module import IF97 purely to
    reach past it to a formulation that was never IF97's.

    Both formulations need thermodynamic input that only an equation of state can
    provide -- the isothermal compressibility (drho/dp)_T, and for conductivity also cp
    and cv. This module does not compute those. The caller passes them in, which is what
    lets the same code serve an IAPWS-95 state, an IF97 Region 1 state, or a lookup
    table, and is why this module imports an equation of state only at the bottom, for
    the four (p, T) convenience wrappers.

Contents:
    VISC  : dynamic viscosity, IAPWS R12-08
    COND  : thermal conductivity, IAPWS R15-11
    SIGMA : surface tension along the saturation line, IAPWS R1-76
    mu_R1, mu_R2, lam_R1, lam_R2 : the same two transport properties at (p, T), with the
            density and derivatives taken from IF97 Region 1 or Region 2

Reference:
    IAPWS R12-08, "Release on the IAPWS Formulation 2008 for the Viscosity of Ordinary
        Water Substance" (IAWPS_Viscosity.pdf).
    IAPWS R15-11, "Release on the IAPWS Formulation 2011 for the Thermal Conductivity of
        Ordinary Water Substance" (ThCond.pdf).
    IAPWS R1-76 (2014), "Revised Release on Surface Tension of Ordinary Water Substance".
"""
import os

import numpy as np
import torch

from pinthac.properties import iapws_backend as w

# The published coefficient tables are data, not code, so they sit in their own directory
# beside this module rather than alongside the source files.
pth = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iapws_data')

# ====================
# Reference Constants
# ====================
# R12-08 Eqs. (1)-(3) and R15-11 Sec. 2.2 use the same reducing parameters.

Tstar = 647.096        # Reducing temperature [K]
rhostar = 322.0        # Reducing density [kg m^-3]
pstar = 22.064         # Reducing pressure [MPa]
mustar = 1.0e-6        # Reducing viscosity [Pa s]
lamstar = 1.0e-3       # Reducing thermal conductivity [W m^-1 K^-1]
Rg = 0.46151805        # Specific gas constant [kJ kg^-1 K^-1], R15-11 Eq. (6)

# Below this the correlation length contributes nothing measurable to either critical
# enhancement, and the enhancement is switched off. It is a positive floor rather than
# zero on purpose: dchi**(nu/gamma) has an infinite slope at dchi = 0, and torch would
# multiply that infinity by the zero gradient of the clamp that produced the zero, giving
# NaN in the backward pass of a state that is nowhere near the critical point. Clamping
# to a small positive number instead keeps the derivative finite and the value identical.
DCHI_FLOOR = 1.0e-30


class VISC:
    # =========================================
    # Table 1: dilute-gas coefficients H_i
    # (R12-08 Eq. 11)
    # =========================================
    H = torch.tensor([1.67752, 2.20462, 0.6366564, -0.241605],
                     dtype=w.dtype).unsqueeze(1)

    # =========================================
    # Table 2: finite-density coefficients H_ij
    # (R12-08 Eq. 12). Row i indexes the
    # temperature term, column j the density
    # term; the table is mostly zeros, so it is
    # written out by position rather than as a
    # 6x7 block of mostly 0.0.
    # =========================================
    Hij = torch.zeros((6, 7), dtype=w.dtype)
    Hij[0, 0] = 5.20094e-1;  Hij[1, 0] = 8.50895e-2
    Hij[2, 0] = -1.08374;    Hij[3, 0] = -2.89555e-1
    Hij[0, 1] = 2.22531e-1;  Hij[1, 1] = 9.99115e-1
    Hij[2, 1] = 1.88797;     Hij[3, 1] = 1.26613
    Hij[5, 1] = 1.20573e-1;  Hij[0, 2] = -2.81378e-1
    Hij[1, 2] = -9.06851e-1; Hij[2, 2] = -7.72479e-1
    Hij[3, 2] = -4.89837e-1; Hij[4, 2] = -2.57040e-1
    Hij[0, 3] = 1.61913e-1;  Hij[1, 3] = 2.57399e-1
    Hij[0, 4] = -3.25372e-2; Hij[3, 4] = 6.98452e-2
    Hij[4, 5] = 8.72102e-3;  Hij[3, 6] = -4.35673e-3
    Hij[5, 6] = -5.93264e-4

    # =========================================
    # Table 3: critical-region constants
    # =========================================
    x_mu = 0.068          # Critical exponent for viscosity [-]
    qC_inv = 1.9e-9       # Inverse of the first cutoff wave number [m]
    qD_inv = 1.1e-9       # Inverse of the second cutoff wave number [m]
    nu = 0.630            # Critical exponent [-]
    gamma_c = 1.239       # Critical exponent [-]
    xi0 = 0.13e-9         # Amplitude of the correlation length [m]
    Gamma0 = 0.06         # Amplitude of the dimensionless compressibility [-]
    TR_bar = 1.5          # Reduced reference temperature, T_R/T* [-]
    xi_split = 0.3817016416e-9   # Correlation length where Eq. (15) hands over to (16) [m]

    @classmethod
    def mu0(cls, T_bar):
        """Dilute-gas limit of the viscosity, R12-08 Eq. (11).

        Formulation:
            mu0_bar = 100 * sqrt(T_bar) / sum_{i=0..3} H_i / T_bar^i

        Valid range:
            The whole range of the release; it is a limit, not a region.

        Uncertainty:
            Part of Eq. (10); see mu() for the uncertainty of the product.

        Reference:
            IAPWS R12-08 Eq. (11) and Table 1.

        Inputs:
            T_bar : reduced temperature T/T*, dimensionless, 1-D torch tensor

        Returns:
            mu0_bar : dimensionless dilute-gas viscosity
        """
        H = w.on(cls.H, T_bar)                          # shape (4, 1)
        i = torch.arange(4, dtype=T_bar.dtype, device=T_bar.device).unsqueeze(1)
        denom = (H * T_bar**(-i)).sum(dim=0)
        return 100.0 * torch.sqrt(T_bar) / denom

    @classmethod
    def mu1(cls, rho_bar, T_bar):
        """Finite-density contribution to the viscosity, R12-08 Eq. (12).

        Formulation:
            mu1_bar = exp( rho_bar * sum_{i=0..5} (1/T_bar - 1)^i
                                   * sum_{j=0..6} H_ij * (rho_bar - 1)^j )

        Valid range:
            The whole range of the release.

        Uncertainty:
            Part of Eq. (10); see mu().

        Reference:
            IAPWS R12-08 Eq. (12) and Table 2.

        Inputs:
            rho_bar : reduced density rho/rho*, dimensionless, 1-D torch tensor
            T_bar   : reduced temperature T/T*, dimensionless, same shape

        Returns:
            mu1_bar : dimensionless finite-density factor
        """
        Hij = w.on(cls.Hij, rho_bar)                    # shape (6, 7)

        t_term = 1.0 / T_bar - 1.0
        rho_term = rho_bar - 1.0

        i = torch.arange(6, dtype=T_bar.dtype, device=T_bar.device).unsqueeze(1)
        j = torch.arange(7, dtype=T_bar.dtype, device=T_bar.device).unsqueeze(1)
        t_pows = t_term.unsqueeze(0)**i                 # (6, N)
        rho_pows = rho_term.unsqueeze(0)**j             # (7, N)

        # inner[i] is the density polynomial belonging to temperature term i.
        inner = (Hij.unsqueeze(-1) * rho_pows.unsqueeze(0)).sum(dim=1)   # (6, N)
        outer = (t_pows * inner).sum(dim=0)
        return torch.exp(rho_bar * outer)

    @classmethod
    def mu2(cls, rho_bar, T_bar, drhodp_T, drhodp_TR):
        """Critical enhancement of the viscosity, R12-08 Eqs. (14)-(21).

            It also matters well beyond viscosity itself: the thermal-conductivity
            critical enhancement, R15-11 Eq. (18), divides by this viscosity, so a mu
            missing its enhancement makes lambda_2 too large by the same proportion.

        Formulation:
            dchi = rho_bar * (zeta_T - zeta_TR * T_R_bar / T_bar),   clamped at >= 0
            xi   = xi0 * (dchi / Gamma0)^(nu/gamma)
            Y    = Eq. (15) for xi <= 0.3817016416 nm, Eq. (16) above it
            mu2  = exp(x_mu * Y)

        Valid range:
            645.91 K < T < 650.77 K and 245.8 < rho < 405.3 kg/m^3 is where it matters.
            Everywhere else it evaluates to essentially 1, so it is safe to leave on.

        Uncertainty:
            Part of Eq. (10); see mu().

        Reference:
            IAPWS R12-08 Eqs. (14)-(21) and Table 3.

        Inputs (1-D torch tensors, all the same shape):
            rho_bar   : reduced density rho/rho*, dimensionless
            T_bar     : reduced temperature T/T*, dimensionless
            drhodp_T  : (drho/dp)_T at T, kg/m^3/MPa
            drhodp_TR : (drho/dp)_T at T_R = 1.5*T* = 970.644 K, kg/m^3/MPa

        Returns:
            mu2 : dimensionless enhancement factor, >= 1
        """
        # Eq. (21): dchi from the two compressibilities, made dimensionless by p*/rho*.
        zeta_T = drhodp_T * pstar / rhostar
        zeta_TR = drhodp_TR * pstar / rhostar
        dchi = rho_bar * (zeta_T - zeta_TR * cls.TR_bar / T_bar)
        # Eq. (21) says a negative dchi must be set to zero. See DCHI_FLOOR for why the
        # floor is a small positive number rather than the zero the release writes.
        dchi = torch.clamp(dchi, min=DCHI_FLOOR)

        # Eq. (20): correlation length.
        xi = cls.xi0 * (dchi / cls.Gamma0) ** (cls.nu / cls.gamma_c)

        qC_xi = xi / cls.qC_inv
        qD_xi = xi / cls.qD_inv

        # Eq. (15), the small-xi branch: a series that stays well conditioned as xi -> 0,
        # where the Eq. (16) form would divide by (qC_xi)^3.
        Y_small = (qC_xi * qD_xi**5 / 5.0
                   * (1.0 - qC_xi + qC_xi**2 - (765.0 / 504.0) * qD_xi**2))

        # Eq. (16)-(19), the large-xi branch.
        qC_safe = torch.clamp(qC_xi, min=1.0e-12)

        # Eq. (17) is psi_D = arccos[(1 + qD_xi^2)^(-1/2)], and that is arctan(qD_xi):
        # a right triangle with opposite qD_xi and adjacent 1 has exactly that cosine.
        # The arctan form is used because the published form is unusable in a backward
        # pass. Away from the critical point qD_xi underflows to zero, (1 + 0)^(-1/2) is
        # exactly 1, and the slope of arccos at 1 is infinite -- so every state in the
        # dilute limit came back with a NaN gradient for mu even though its value was
        # right. arctan has slope 1 there, and is more accurate for small qD_xi besides.
        psi_D = torch.arctan(qD_xi)                                                # Eq. (17)

        # Eq. (19). The sign of (qC_xi - 1) selects which form of L(w) applies, so take
        # the magnitude here and branch on qC_xi below.
        w_ = torch.sqrt(torch.abs((qC_safe - 1.0) / (qC_safe + 1.0))) * torch.tan(psi_D / 2.0)
        w_abs = torch.abs(w_)
        L_gt = torch.log((1.0 + w_abs) / torch.clamp(1.0 - w_abs, min=1.0e-15))
        L_le = 2.0 * torch.arctan(w_abs)
        L_w = torch.where(qC_safe > 1.0, L_gt, L_le)                               # Eq. (18)

        Y_large = (torch.sin(3.0 * psi_D) / 12.0
                   - torch.sin(2.0 * psi_D) / (4.0 * qC_safe)
                   + (1.0 - 1.25 * qC_safe**2) * torch.sin(psi_D) / qC_safe**2
                   - ((1.0 - 1.5 * qC_safe**2) * psi_D
                      - torch.abs(qC_safe**2 - 1.0)**1.5 * L_w) / qC_safe**3)

        Y = torch.where(xi <= cls.xi_split, Y_small, Y_large)
        return torch.exp(cls.x_mu * Y)                                             # Eq. (14)

    @classmethod
    def mu(cls, rho, T, drhodp_T=None, drhodp_TR=None):
        """Dynamic viscosity of ordinary water, IAPWS R12-08 Eq. (10).

        Formulation:
            mu = mu* * mu0_bar(T_bar) * mu1_bar(rho_bar, T_bar) * mu2_bar
            with mu* = 1 uPa s, T_bar = T/647.096 K, rho_bar = rho/322 kg/m^3.

            The critical enhancement mu2 is included only when the caller supplies the
            isothermal compressibility at two temperatures -- the state's own T and the
            fixed reference T_R = 1.5*T* = 970.644 K -- because R12-08 says both must
            come from IAPWS-95, and this module deliberately does not depend on one.
            Omitting them gives the industrial simplification mu2 = 1, which R12-08
            Sec. 2.8 and Sec. 3 sanction outside the near-critical region and which
            Table 4's own check values are quoted for.

        Valid range:
            273.16 K <= T <= 1173.15 K over the pressure range of IAPWS-95, along the
            melting curve up to 300 MPa at the cold end. R12-08 Sec. 2.9.

        Uncertainty:
            About 0.2 percent for liquid water near ambient, rising to roughly 1 to 2
            percent in the dense supercritical region and 3 percent near the critical
            point; see R12-08 Fig. 1 for the full map.

        Reference:
            IAPWS R12-08, Eq. (10), Tables 1-3 (IAWPS_Viscosity.pdf).

        Inputs (float, numpy array, or torch tensor; broadcastable):
            rho       : density, kg/m^3
            T         : temperature, K
            drhodp_T  : optional, (drho/dp)_T at T, kg/m^3/MPa
            drhodp_TR : optional, (drho/dp)_T at T_R = 970.644 K, kg/m^3/MPa

        Returns:
            mu : dynamic viscosity, Pa-s, same type as the inputs
        """
        if drhodp_T is None or drhodp_TR is None:
            (rho_, T_), state = w.prepare(rho, T)
            enhance = None
        else:
            (rho_, T_, dT_, dTR_), state = w.prepare(rho, T, drhodp_T, drhodp_TR)
            enhance = (dT_, dTR_)

        T_bar = T_ / Tstar
        rho_bar = rho_ / rhostar

        mu0_bar = cls.mu0(T_bar)
        mu1_bar = cls.mu1(rho_bar, T_bar)
        if enhance is None:
            mu2_bar = torch.ones_like(mu0_bar)
        else:
            mu2_bar = cls.mu2(rho_bar, T_bar, enhance[0], enhance[1])

        return w.restore(mustar * mu0_bar * mu1_bar * mu2_bar, state)


class COND:
    # =========================================
    # Table 1: dilute-gas coefficients L_k
    # Table 2: finite-density coefficients L_ij
    # Table 6: reference-compressibility A_ij
    # (loaded from iapws_data/, see pth above)
    # =========================================
    Lk = torch.from_numpy(np.loadtxt(f'{pth}/ThCond_Lk.txt')).to(w.dtype).unsqueeze(1)
    Lij = torch.from_numpy(np.loadtxt(f'{pth}/ThCond_Lij.txt')).to(w.dtype)
    Aij = torch.from_numpy(np.loadtxt(f'{pth}/ThCond_Aij.txt')).to(w.dtype)

    # =========================================
    # Table 3: critical-enhancement constants
    # =========================================
    Lambda = 177.8514      # Numerical prefactor of Eq. (18) [-]
    qD_inv = 0.40e-9       # Inverse cutoff wave number [m] (R12-08 uses 1.1 nm for mu)
    xi0 = 0.13e-9          # Amplitude of the correlation length [m]
    Gamma0 = 0.06          # Amplitude of the dimensionless compressibility [-]
    nu = 0.630             # Critical exponent [-]
    gamma_c = 1.239        # Critical exponent [-]
    TR_bar = 1.5           # Reduced reference temperature, T_R/T* [-]
    y_floor = 1.2e-7       # Below this, Eq. (19) is truncation error only, so Z = 0 [-]

    # Density boundaries that select the A_ij column, Eq. (26).
    rho_bounds = torch.tensor([0.310559006, 0.776397516, 1.242236025, 1.863354037],
                              dtype=w.dtype)

    @classmethod
    def lam0(cls, T_bar):
        """Dilute-gas limit of the thermal conductivity, R15-11 Eq. (16).

        Formulation:
            lam0_bar = sqrt(T_bar) / sum_{k=0..4} L_k / T_bar^k

        Valid range:
            The whole range of the release; it is a limit, not a region.

        Uncertainty:
            Part of Eq. (15); see lam().

        Reference:
            IAPWS R15-11 Eq. (16) and Table 1 (ThCond.pdf).

        Inputs:
            T_bar : reduced temperature T/T*, dimensionless, 1-D torch tensor

        Returns:
            lam0_bar : dimensionless dilute-gas thermal conductivity
        """
        Lk = w.on(cls.Lk, T_bar)                        # shape (5, 1)
        k = torch.arange(5, dtype=T_bar.dtype, device=T_bar.device).unsqueeze(1)
        denom = (Lk * T_bar**(-k)).sum(dim=0)
        return torch.sqrt(T_bar) / denom

    @classmethod
    def lam1(cls, rho_bar, T_bar):
        """Finite-density contribution to the thermal conductivity, R15-11 Eq. (17).

        Formulation:
            lam1_bar = exp( rho_bar * sum_{i=0..4} (1/T_bar - 1)^i
                                    * sum_{j=0..5} L_ij * (rho_bar - 1)^j )

        Valid range:
            The whole range of the release.

        Uncertainty:
            Part of Eq. (15); see lam().

        Reference:
            IAPWS R15-11 Eq. (17) and Table 2.

        Inputs (1-D torch tensors, same shape):
            rho_bar : reduced density rho/rho*, dimensionless
            T_bar   : reduced temperature T/T*, dimensionless

        Returns:
            lam1_bar : dimensionless finite-density factor
        """
        Lij = w.on(cls.Lij, rho_bar)                    # shape (5, 6)

        i = torch.arange(5, dtype=T_bar.dtype, device=T_bar.device).unsqueeze(1)
        j = torch.arange(6, dtype=T_bar.dtype, device=T_bar.device).unsqueeze(1)
        t_pows = (1.0 / T_bar - 1.0).unsqueeze(0)**i    # (5, N)
        rho_pows = (rho_bar - 1.0).unsqueeze(0)**j      # (6, N)

        inner = (Lij.unsqueeze(-1) * rho_pows.unsqueeze(0)).sum(dim=1)   # (5, N)
        outer = (t_pows * inner).sum(dim=0)
        return torch.exp(rho_bar * outer)

    @classmethod
    def zeta_TR(cls, rho_bar):
        """Reference dimensionless isothermal compressibility at T_R, R15-11 Eq. (25).

        Formulation:
            zeta(T_R, rho_bar) = 1 / sum_{i=0..5} A_ij * rho_bar^i,
            where the column j is chosen by which of five density bands rho_bar falls
            in, Eq. (26).

        Valid range:
            The density range of the release; the five bands tile it with no gap.

        Uncertainty:
            Not applicable -- an auxiliary fit internal to Eq. (18).

        Reference:
            IAPWS R15-11 Eqs. (25)-(26) and Table 6.

        Inputs:
            rho_bar : reduced density rho/rho*, dimensionless, 1-D torch tensor

        Returns:
            zeta_TR : dimensionless isothermal compressibility on the T_R isotherm
        """
        Aij = w.on(cls.Aij, rho_bar)                            # shape (6, 5)
        bounds = w.on(cls.rho_bounds, rho_bar)

        i = torch.arange(6, dtype=rho_bar.dtype, device=rho_bar.device).unsqueeze(1)
        rho_pows = rho_bar.unsqueeze(0)**i                      # (6, N)

        # The band index is a count of how many boundaries rho_bar has passed. Written as
        # a sum rather than a chain of ifs so it stays elementwise over a batch.
        j = (rho_bar.unsqueeze(0) > bounds.unsqueeze(1)).sum(dim=0)
        A_col = Aij[:, j]                                       # (6, N)
        return 1.0 / (A_col * rho_pows).sum(dim=0)

    @classmethod
    def lam2(cls, rho_bar, T_bar, drhodp_T, cp, cv, mu):
        """Critical enhancement of the thermal conductivity, R15-11 Eqs. (18)-(25).

        Formulation:
            dchi = rho_bar * (zeta_T - zeta(T_R,rho_bar) * T_R_bar / T_bar), clamped >= 0
            xi   = xi0 * (dchi / Gamma0)^(nu/gamma)                          Eq. (22)
            y    = xi / qD_inv                                               Eq. (20)
            Z(y) = Eq. (19), set to zero for y < 1.2e-7
            lam2_bar = Lambda * rho_bar * T_bar * cp_bar / mu_bar * Z(y)     Eq. (18)

        Valid range:
            Everywhere in the release; away from the critical region it evaluates to
            essentially zero rather than being switched off.

        Uncertainty:
            Part of Eq. (15); see lam().

        Reference:
            IAPWS R15-11 Eqs. (18)-(25) and Table 3.

        Inputs (1-D torch tensors, all the same shape):
            rho_bar  : reduced density rho/rho*, dimensionless
            T_bar    : reduced temperature T/T*, dimensionless
            drhodp_T : (drho/dp)_T, kg/m^3/MPa
            cp       : isobaric heat capacity, kJ/kg-K
            cv       : isochoric heat capacity, kJ/kg-K
            mu       : dynamic viscosity, Pa-s

        Returns:
            lam2_bar : dimensionless critical enhancement
        """
        zeta_T = drhodp_T * pstar / rhostar                # at temperature T, Eq. (24)
        zeta_R = cls.zeta_TR(rho_bar)                      # at T_R,           Eq. (25)

        dchi = rho_bar * (zeta_T - cls.TR_bar / T_bar * zeta_R)
        # Eq. (23) says a negative dchi must be set to zero; see DCHI_FLOOR for why the
        # floor is positive. At the Table 4 liquid check points dchi really is negative,
        # so this branch is exercised by the published verification and not just by
        # pathological input.
        dchi = torch.clamp(dchi, min=DCHI_FLOOR)

        xi = cls.xi0 * (dchi / cls.Gamma0) ** (cls.nu / cls.gamma_c)   # Eq. (22)
        y = xi / cls.qD_inv                                            # Eq. (20)

        # Eq. (19) divides by y and by rho_bar, both of which reach zero in the dilute
        # limit the release's own check values include. Evaluate the expression on
        # floored copies and select afterwards, so neither the value nor the gradient
        # ever sees the division.
        y_safe = torch.clamp(y, min=cls.y_floor)
        rho_safe = torch.clamp(rho_bar, min=1.0e-12)

        kappa = cp / cv
        arg = 1.0 / (1.0 / y_safe + y_safe**2 / (3.0 * rho_safe**2))
        Zy = (2.0 / (torch.pi * y_safe)) * ((1.0 - 1.0 / kappa) * torch.arctan(y_safe)
                                            + y_safe / kappa
                                            - (1.0 - torch.exp(-arg)))
        Zy = torch.where(y < cls.y_floor, torch.zeros_like(Zy), Zy)

        mu_bar = mu / mustar
        cp_bar = cp / Rg                                   # dimensionless, R15-11 Eq. (6)
        return cls.Lambda * rho_bar * T_bar * cp_bar / mu_bar * Zy     # Eq. (18)

    @classmethod
    def lam(cls, rho, T, drhodp_T, cp, cv, mu):
        """Thermal conductivity of ordinary water, IAPWS R15-11 Eq. (15).

        Formulation:
            lambda = lam* * (lam0_bar * lam1_bar + lam2_bar),  lam* = 1 mW/m/K

        Valid range:
            From the melting curve to 1173.15 K, at pressures to 100 MPa (and to 1000 MPa
            below 500 K). R15-11 Sec. 2.8.

        Uncertainty:
            Roughly 1 percent for liquid water and 1.5 percent for steam away from the
            critical region, degrading to about 5 percent near the critical point;
            see R15-11 Fig. 1.

        Reference:
            IAPWS R15-11, Eq. (15), Tables 1-3 and 6 (ThCond.pdf).

        Inputs (float, numpy array, or torch tensor; broadcastable):
            rho      : density, kg/m^3
            T        : temperature, K
            drhodp_T : isothermal compressibility (drho/dp)_T, kg/m^3/MPa
            cp       : isobaric heat capacity, kJ/kg-K
            cv       : isochoric heat capacity, kJ/kg-K
            mu       : dynamic viscosity, Pa-s

        Returns:
            lam : thermal conductivity, W/m-K, same type as the inputs
        """
        (rho_, T_, drhodp_, cp_, cv_, mu_), state = w.prepare(rho, T, drhodp_T, cp, cv, mu)

        T_bar = T_ / Tstar
        rho_bar = rho_ / rhostar

        lam0_bar = cls.lam0(T_bar)
        lam1_bar = cls.lam1(rho_bar, T_bar)
        lam2_bar = cls.lam2(rho_bar, T_bar, drhodp_, cp_, cv_, mu_)

        return w.restore(lamstar * (lam0_bar * lam1_bar + lam2_bar), state)


class SIGMA:
    # =========================================
    # IAPWS R1-76 (2014), Eq. (1)
    # =========================================
    Tc = 647.096       # Critical temperature [K]
    B = 235.8e-3       # Amplitude [N m^-1]
    b = -0.625         # Linear correction coefficient [-]
    mu_exp = 1.256     # Critical exponent [-]

    @classmethod
    def sigma(cls, T):
        """Surface tension of ordinary water against its own vapor, IAPWS R1-76.

        Formulation:
            sigma = B * tau^mu * (1 + b * tau),   tau = 1 - T/Tc

        Valid range:
            273.15 K (and down to 248.15 K for supercooled water) to Tc = 647.096 K.
            Above Tc there is no interface; this returns zero there rather than the NaN
            that tau^1.256 of a negative tau would otherwise produce.

        Uncertainty:
            About 0.5 percent over most of the range, rising near the critical point.

        Reference:
            IAPWS R1-76 (2014), "Revised Release on Surface Tension of Ordinary Water
            Substance", Eq. (1).

        Inputs:
            T : temperature, K (float, numpy array, or torch tensor)

        Returns:
            sigma : surface tension, N/m, same type as the input
        """
        (T_,), state = w.prepare(T)

        tau = 1.0 - T_ / cls.Tc
        # tau^1.256 of a negative tau is NaN, and torch carries that NaN through the
        # backward pass of a where() even when the forward pass discarded it -- so the
        # supercritical branch is clamped rather than selected after the fact.
        tau_safe = torch.clamp(tau, min=0.0)
        sig = cls.B * tau_safe**cls.mu_exp * (1.0 + cls.b * tau_safe)

        return w.restore(sig, state)


# =========================================
# Convenience wrappers on IF97.
# R12-08 and R15-11 both take (rho, T) plus
# thermodynamic derivatives; these four fetch
# them from an IF97 region so a caller with
# (p, T) does not have to assemble the state
# by hand. The import sits here, at the bottom,
# rather than at the top of the module: the
# formulations above deliberately depend on no
# equation of state at all, and only these
# wrappers do.
# =========================================
from pinthac.properties import iapws97 as f97      # noqa: E402


def mu_R1(p, T):
    """Dynamic viscosity [Pa-s] of compressed liquid water at (p, T), via IF97 Region 1.

        The critical enhancement is left off. It needs the isothermal compressibility at
        T_R = 970.644 K as well as at T, and 970 K is far outside Region 1 -- there is no
        honest way to get it from this equation. That costs nothing here: Region 1 stops
        at 623.15 K, nowhere near the region where the enhancement is measurable.

    Formulation:
        rho from IF97 Region 1, then VISC.mu(rho, T) with mu2 = 1.

    Valid range:
        IF97 Region 1: 273.15 K <= T <= 623.15 K, p_sat(T) <= p <= 100 MPa.

    Uncertainty:
        That of R12-08, plus IF97's own density deviation from IAPWS-95.

    Reference:
        IAPWS R12-08 Eq. (10); IAPWS-IF97 Eq. (7).

    Inputs (float, numpy array, or torch tensor; broadcastable):
        p : pressure, MPa
        T : temperature, K

    Returns:
        mu : dynamic viscosity, Pa-s, same type as the inputs
    """
    d = f97.R1.gibbs(p, T)
    return VISC.mu(f97.R1.rho(d), T)


def mu_R2(p, T):
    """Dynamic viscosity [Pa-s] of steam at (p, T), via IF97 Region 2.

    The Region 2 counterpart of mu_R1, with the same reasoning and the same omission of
    the critical enhancement.

    Valid range:
        IF97 Region 2: up to 1073.15 K, bounded below by the saturation line and the B23
        line. Use region() to check that (p, T) is actually in Region 2.

    Inputs (float, numpy array, or torch tensor; broadcastable):
        p : pressure, MPa
        T : temperature, K

    Returns:
        mu : dynamic viscosity, Pa-s, same type as the inputs
    """
    d = f97.R2.gibbs(p, T)
    return VISC.mu(f97.R2.rho(d), T)


def lam_R1(p, T):
    """Thermal conductivity [W/m-K] of compressed liquid water at (p, T), via IF97 Region 1.

    Formulation:
        rho, (drho/dp)_T, cp and cv from IF97 Region 1, mu from mu_R1, then COND.lam.

    Valid range:
        IF97 Region 1: 273.15 K <= T <= 623.15 K, p_sat(T) <= p <= 100 MPa.

    Uncertainty:
        That of R15-11, plus IF97's own deviation from IAPWS-95 in the four inputs.

    Reference:
        IAPWS R15-11 Eq. (15); IAPWS-IF97 Eq. (7).

    Inputs (float, numpy array, or torch tensor; broadcastable):
        p : pressure, MPa
        T : temperature, K

    Returns:
        lam : thermal conductivity, W/m-K, same type as the inputs
    """
    d = f97.R1.gibbs(p, T)
    rho = f97.R1.rho(d)
    return COND.lam(rho, T, f97.R1.drhodp(d),
                    f97.R1.cp(d, units='kJ'), f97.R1.cv(d, units='kJ'),
                    VISC.mu(rho, T))


def lam_R2(p, T):
    """Thermal conductivity [W/m-K] of steam at (p, T), via IF97 Region 2.

    The Region 2 counterpart of lam_R1, with the same reasoning.

    Valid range:
        IF97 Region 2: up to 1073.15 K, bounded below by the saturation line and the B23
        line. Use region() to check that (p, T) is actually in Region 2.

    Inputs (float, numpy array, or torch tensor; broadcastable):
        p : pressure, MPa
        T : temperature, K

    Returns:
        lam : thermal conductivity, W/m-K, same type as the inputs
    """
    d = f97.R2.gibbs(p, T)
    rho = f97.R2.rho(d)
    return COND.lam(rho, T, f97.R2.drhodp(d),
                    f97.R2.cp(d, units='kJ'), f97.R2.cv(d, units='kJ'),
                    VISC.mu(rho, T))


# Backwards-compatible spelling: this class used to be `Sigma` inside iapws97.py, and
# getprop.py reaches it through the 'sigma' key of the Water branch.
Sigma = SIGMA
