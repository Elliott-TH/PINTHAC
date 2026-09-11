import functools
import numpy as np
import torch
from pinthac.properties import iapws97 as w97
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# No print here. CLAUDE.md section 5 rule 6 forbids side effects at import time, and a
# library that announces itself on import is one whose output a caller cannot control:
# it lands in the middle of a user's own stdout, in every subprocess, and in the middle
# of every example's pasted output. `device` is a module attribute -- anything that wants
# to know which one was chosen can read it.

# ====================
# Critical Properties
# ====================

Tc = 647.096       # Critical temperature [K]
rhoc = 322.0        # Critical density [kg m^-3]
R = 0.46151805       # Specific gas constant [kJ kg^-1 K^-1]
pc = 22.064          # Critical pressure [MPa]
Tt = 273.16          # Triple-point temperature [K]



class IAPWS95:
    # =========================================
    # Table 1: Ideal-Gas Part Coefficients
    # =========================================
    n0 = torch.tensor([
        -8.32044648374970, 6.68321052759320, 3.00632,
        0.012436, 0.97315, 1.27950, 0.96956, 0.24873
    ], dtype=torch.float64, device=device).unsqueeze(1)

    gamma0 = torch.tensor([
        0.0, 0.0, 0.0,
        1.28728967, 3.53734222, 7.74073708, 9.24437796, 27.5075105
    ], dtype=torch.float64, device=device).unsqueeze(1)

    # =========================================
    # Table 2: Residual Part Coefficients
    # (verified against IAPWS R6-95(2018) Table 2)
    # =========================================
    n1 = torch.tensor([
        0.012533547935523, 7.8957634722828, -8.7803203303561, 0.31802509345418, -0.26145533859358, -0.0078199751687981, 0.0088089493102134
    ], dtype=torch.float64, device=device).unsqueeze(1)

    d1 = torch.tensor([
        1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0
    ], dtype=torch.float64, device=device).unsqueeze(1)

    t1 = torch.tensor([
        -0.5, 0.875, 1.0, 0.5, 0.75, 0.375, 1.0
    ], dtype=torch.float64, device=device).unsqueeze(1)



    n2 = torch.tensor([
        -0.66856572307965, 0.20433810950965, -6.6212605039687e-05, -0.19232721156002,
        -0.25709043003438, 0.16074868486251, -0.040092828925807, 3.9343422603254e-07,
        -7.5941377088144e-06, 0.00056250979351888, -1.5608652257135e-05, 1.1537996422951e-09,
        3.6582165144204e-07, -1.3251180074668e-12, -6.2639586912454e-10, -0.10793600908932,
        0.017611491008752, 0.22132295167546, -0.40247669763528, 0.58083399985759,
        0.0049969146990806, -0.031358700712549, -0.74315929710341, 0.4780732991548,
        0.020527940895948, -0.13636435110343, 0.014180634400617, 0.0083326504880713,
        -0.029052336009585, 0.038615085574206, -0.020393486513704, -0.0016554050063734,
        0.0019955571979541, 0.00015870308324157, -1.638856834253e-05, 0.043613615723811,
        0.034994005463765, -0.076788197844621, 0.022446277332006, -6.2689710414685e-05,
        -5.5711118565645e-10, -0.19905718354408, 0.31777497330738, -0.11841182425981
    ], dtype=torch.float64, device=device).unsqueeze(1)

    d2 = torch.tensor([
        1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 7.0, 9.0, 10.0, 11.0, 13.0, 15.0,
        1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 9.0, 9.0, 9.0,
        9.0, 9.0, 10.0, 10.0, 12.0, 3.0, 4.0, 4.0, 5.0, 14.0, 3.0, 6.0, 6.0, 6.0
    ], dtype=torch.float64, device=device).unsqueeze(1)

    t2 = torch.tensor([
        4.0, 6.0, 12.0, 1.0, 5.0, 4.0, 2.0, 13.0, 9.0, 3.0, 4.0, 11.0, 4.0, 13.0, 1.0,
        7.0, 1.0, 9.0, 10.0, 10.0, 3.0, 7.0, 10.0, 10.0, 6.0, 10.0, 10.0, 1.0, 2.0, 3.0,
        4.0, 8.0, 6.0, 9.0, 8.0, 16.0, 22.0, 23.0, 23.0, 10.0, 50.0, 44.0, 46.0, 50.0
    ], dtype=torch.float64, device=device).unsqueeze(1)

    c2 = torch.tensor([
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 6.0, 6.0, 6.0, 6.0
    ], dtype=torch.float64, device=device).unsqueeze(1)



    n3 = torch.tensor([
        -31.306260323435, 31.546140237781, -2521.3154341695
    ], dtype=torch.float64, device=device).unsqueeze(1)

    d3 = torch.tensor([
        3.0, 3.0, 3.0
    ], dtype=torch.float64, device=device).unsqueeze(1)

    t3 = torch.tensor([
        0.0, 1.0, 4.0
    ], dtype=torch.float64, device=device).unsqueeze(1)

    alpha3 = torch.tensor([
        20.0, 20.0, 20.0
    ], dtype=torch.float64, device=device).unsqueeze(1)

    beta3 = torch.tensor([
        150.0, 150.0, 250.0
    ], dtype=torch.float64, device=device).unsqueeze(1)

    gamma3 = torch.tensor([
        1.21, 1.21, 1.25
    ], dtype=torch.float64, device=device).unsqueeze(1)

    epsilon3 = torch.tensor([
        1.0, 1.0, 1.0
    ], dtype=torch.float64, device=device).unsqueeze(1)



    n4 = torch.tensor([
        -0.14874640856724, 0.31806110878444
    ], dtype=torch.float64, device=device).unsqueeze(1)

    a4 = torch.tensor([
        3.5, 3.5
    ], dtype=torch.float64, device=device).unsqueeze(1)

    b4 = torch.tensor([
        0.85, 0.95
    ], dtype=torch.float64, device=device).unsqueeze(1)

    B4 = torch.tensor([
        0.2, 0.2
    ], dtype=torch.float64, device=device).unsqueeze(1)

    C4 = torch.tensor([
        28.0, 32.0
    ], dtype=torch.float64, device=device).unsqueeze(1)

    D4 = torch.tensor([
        700.0, 800.0
    ], dtype=torch.float64, device=device).unsqueeze(1)

    A4 = torch.tensor([
        0.32, 0.32
    ], dtype=torch.float64, device=device).unsqueeze(1)

    beta4 = torch.tensor([
        0.3, 0.3
    ], dtype=torch.float64, device=device).unsqueeze(1)
    # =========================================
    # Ideal-gas part: Eq. (5), Table 4
    # =========================================
    @classmethod
    def Phi0(cls, rho, T):
        """phi^o(delta, tau) and its derivatives."""
        #rho = rho.unsqueeze(0)
        #T = T.unsqueeze(0)
        delta = rho / rhoc
        tau = Tc / T

        n, gam = cls.n0, cls.gamma0
        n1_, n2_, n3_ = n[0], n[1], n[2]      # n1^o, n2^o, n3^o
        ng, gamg = n[3:], gam[3:]             # terms 4-8

        phi = (torch.log(delta) + n1_ + n2_*tau + n3_*torch.log(tau)
               + (ng * torch.log(1 - torch.exp(-gamg*tau))).sum(dim=0))

        phi_d = 1/delta
        phi_dd = -1/delta**2

        phi_t = (n2_ + n3_/tau
                 + (ng * gamg * ((1 - torch.exp(-gamg*tau))**-1 - 1)).sum(dim=0))
        phi_tt = (-n3_/tau**2
                  - (ng * gamg**2 * torch.exp(-gamg*tau)
                     * (1 - torch.exp(-gamg*tau))**-2).sum(dim=0))

        phi_dt = torch.zeros_like(phi)

        return phi, phi_d, phi_dd, phi_t, phi_tt, phi_dt

    # =========================================
    # Residual part: Eq. (6), Table 5
    # =========================================
    @classmethod
    def Phir(cls, rho, T):
        """phi^r(delta, tau) and its derivatives."""
        #rho = rho.unsqueeze(0)
        #T = T.unsqueeze(0)
        delta = rho / rhoc
        tau = Tc / T

        n1_, d1_, t1_ = cls.n1, cls.d1, cls.t1
        n2_, d2_, t2_, c2_ = cls.n2, cls.d2, cls.t2, cls.c2
        n3_, d3_, t3_, alpha3_, beta3_, gamma3_, epsilon3_ = (
            cls.n3, cls.d3, cls.t3, cls.alpha3, cls.beta3, cls.gamma3, cls.epsilon3)
        n4_, a4_, b4_, B4_, C4_, D4_, A4_, beta4_ = (
            cls.n4, cls.a4, cls.b4, cls.B4, cls.C4, cls.D4, cls.A4, cls.beta4)

        # ---- Group 1: simple polynomial terms ----
        phi_1 = (n1_ * delta**d1_ * tau**t1_).sum(dim=0)
        phi_1_d = (n1_ * d1_ * delta**(d1_-1) * tau**t1_).sum(dim=0)
        phi_1_dd = (n1_ * d1_ * (d1_-1) * delta**(d1_-2) * tau**t1_).sum(dim=0)
        phi_1_t = (n1_ * t1_ * delta**d1_ * tau**(t1_-1)).sum(dim=0)
        phi_1_tt = (n1_ * t1_ * (t1_-1) * delta**d1_ * tau**(t1_-2)).sum(dim=0)
        phi_1_dt = (n1_ * d1_ * t1_ * delta**(d1_-1) * tau**(t1_-1)).sum(dim=0)

        # ---- Group 2: exponential terms, exp(-delta^c) ----
        E2 = torch.exp(-delta**c2_)
        phi_2 = (n2_ * delta**d2_ * tau**t2_ * E2).sum(dim=0)
        phi_2_d = (n2_ * E2 * delta**(d2_-1) * tau**t2_
                   * (d2_ - c2_*delta**c2_)).sum(dim=0)
        phi_2_dd = (n2_ * E2 * delta**(d2_-2) * tau**t2_
                    * ((d2_ - c2_*delta**c2_) * (d2_ - 1 - c2_*delta**c2_)
                       - c2_**2 * delta**c2_)).sum(dim=0)
        phi_2_t = (n2_ * t2_ * delta**d2_ * tau**(t2_-1) * E2).sum(dim=0)
        phi_2_tt = (n2_ * t2_ * (t2_-1) * delta**d2_ * tau**(t2_-2) * E2).sum(dim=0)
        phi_2_dt = (n2_ * t2_ * tau**(t2_-1) * E2 * delta**(d2_-1)
                    * (d2_ - c2_*delta**c2_)).sum(dim=0)

        # ---- Group 3: Gaussian bell terms ----
        E3 = torch.exp(-alpha3_*(delta-epsilon3_)**2 - beta3_*(tau-gamma3_)**2)
        phi_3 = (n3_ * delta**d3_ * tau**t3_ * E3).sum(dim=0)

        dEd = -2*alpha3_*(delta-epsilon3_) * E3
        d2Ed = (2*alpha3_*(delta-epsilon3_))**2 * E3 - 2*alpha3_*E3
        dEt = -2*beta3_*(tau-gamma3_) * E3
        d2Et = (2*beta3_*(tau-gamma3_))**2 * E3 - 2*beta3_*E3
        dEdt = 4*alpha3_*beta3_*(delta-epsilon3_)*(tau-gamma3_) * E3

        phi_3_d = (n3_ * delta**d3_ * tau**t3_
                   * (d3_/delta * E3 + dEd)).sum(dim=0)
        phi_3_dd = (n3_ * tau**t3_ * delta**(d3_-2)
                    * (d3_*(d3_-1)*E3
                       + 2*d3_*delta*dEd
                       + delta**2*d2Ed)).sum(dim=0)
        phi_3_t = (n3_ * delta**d3_ * tau**(t3_-1)
                   * (t3_*E3 + tau*dEt)).sum(dim=0)
        phi_3_tt = (n3_ * delta**d3_ * tau**(t3_-2)
                    * (t3_*(t3_-1)*E3
                       + 2*t3_*tau*dEt
                       + tau**2*d2Et)).sum(dim=0)
        phi_3_dt = (n3_ * delta**(d3_-1) * tau**(t3_-1)
                    * (d3_*t3_*E3
                       + d3_*tau*dEt
                       + t3_*delta*dEd
                       + delta*tau*dEdt)).sum(dim=0)

        # ---- Group 4: nonanalytic (critical-region) terms ----
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

        theta = (1 - tau) + A4_ * ((delta-1)**2) ** (1/(2*beta4_))
        Delta = theta**2 + B4_ * ((delta-1)**2) ** a4_
        psi = torch.exp(-C4_*(delta-1)**2 - D4_*(tau-1)**2)

        dtheta_dd = A4_/beta4_ * ((delta-1)**2) ** (1/(2*beta4_) - 1) * (delta-1)
        dDelta_dd = (delta-1) * (B4_*a4_*((delta-1)**2)**(a4_-1)
                                  + 2*theta*A4_/beta4_*((delta-1)**2)**(1/(2*beta4_)-1))
        dDelta_dt = -2*theta

        dDeltab_dd = b4_ * Delta**(b4_-1) * dDelta_dd
        d2Delta_dd2 = (1/(delta-1)) * dDelta_dd \
                      + (delta-1)**2 * (B4_*a4_*(a4_-1)*4*((delta-1)**2)**(a4_-2)
                                        + 2*A4_/beta4_*((delta-1)**2)**(1/(2*beta4_)-1)
                                          * (A4_/beta4_*((delta-1)**2)**(1/(2*beta4_)-1)
                                             + theta*(1/beta4_-1)*2*((delta-1)**2)**(1/(2*beta4_)-2)*(delta-1)**0))
        d2Deltab_dd2 = (b4_*(b4_-1)*Delta**(b4_-2)*dDelta_dd**2
                        + b4_*Delta**(b4_-1)*d2Delta_dd2)
        dDeltab_dt = -2*theta*b4_*Delta**(b4_-1)
        d2Deltab_dt2 = 2*b4_*Delta**(b4_-1) + 4*theta**2*b4_*(b4_-1)*Delta**(b4_-2)
        d2Deltab_ddt = -2*b4_*Delta**(b4_-1)*dtheta_dd \
                       - 2*theta*b4_*(b4_-1)*Delta**(b4_-2)*dDelta_dd

        dpsi_dd = -2*C4_*(delta-1)*psi
        d2psi_dd2 = (2*C4_*(delta-1)**2 - 1) * 2*C4_*psi
        dpsi_dt = -2*D4_*(tau-1)*psi
        d2psi_dt2 = (2*D4_*(tau-1)**2 - 1) * 2*D4_*psi
        d2psi_ddt = 4*C4_*D4_*(delta-1)*(tau-1)*psi

        phi_4 = (n4_ * Delta**b4_ * delta * psi).sum(dim=0)

        phi_4_d = (n4_ * (Delta**b4_ * (psi + delta*dpsi_dd)
                           + dDeltab_dd * delta * psi)).sum(dim=0)

        phi_4_dd = (n4_ * (Delta**b4_ * (2*dpsi_dd + delta*d2psi_dd2)
                            + 2*dDeltab_dd * (psi + delta*dpsi_dd)
                            + d2Deltab_dd2 * delta * psi)).sum(dim=0)

        phi_4_t = (n4_ * delta * (dDeltab_dt*psi + Delta**b4_*dpsi_dt)).sum(dim=0)

        phi_4_tt = (n4_ * delta * (d2Deltab_dt2*psi
                                    + 2*dDeltab_dt*dpsi_dt
                                    + Delta**b4_*d2psi_dt2)).sum(dim=0)

        phi_4_dt = (n4_ * (Delta**b4_ * (dpsi_dt + delta*d2psi_ddt)
                            + delta*dDeltab_dd*dpsi_dt
                            + dDeltab_dt * (psi + delta*dpsi_dd)
                            + delta*d2Deltab_ddt*psi)).sum(dim=0)

        phi = phi_1 + phi_2 + phi_3 + phi_4
        phi_d = phi_1_d + phi_2_d + phi_3_d + phi_4_d
        phi_dd = phi_1_dd + phi_2_dd + phi_3_dd + phi_4_dd
        phi_t = phi_1_t + phi_2_t + phi_3_t + phi_4_t
        phi_tt = phi_1_tt + phi_2_tt + phi_3_tt + phi_4_tt
        phi_dt = phi_1_dt + phi_2_dt + phi_3_dt + phi_4_dt

        return phi, phi_d, phi_dd, phi_t, phi_tt, phi_dt
    
    @classmethod
    def helmholtz(cls, rho, T):
        """Compute all phi^o and phi^r derivatives once.
        Returns a dict — arrays if rho/T are arrays."""
        if isinstance(rho,torch.Tensor):
            intype = 'torch'
            size = rho.shape
            orig_device = rho.device
            rho = rho.reshape(-1)
            T = T.reshape(-1)
        elif isinstance(rho,np.ndarray):
            rho = torch.from_numpy(rho)
            T = torch.from_numpy(T)
            intype = 'np'
            size = rho.shape
            rho = rho.reshape(-1)
            T = T.reshape(-1)
        elif isinstance(rho or T,float) or isinstance(rho or T,int):
            rho = torch.tensor([rho])
            T = torch.tensor([T])
            intype='single'
        elif isinstance(rho,list):
            rho = torch.tensor(rho)
            T = torch.tensor(T)
            intype='list'

        rho = rho.unsqueeze(0).to(device)
        T   = T.unsqueeze(0).to(device)
        delta = rho / rhoc
        tau   = Tc / T

        phi0,  phi0_d,  phi0_dd,  phi0_t,  phi0_tt,  phi0_dt  = cls.Phi0(rho, T)
        phir,  phir_d,  phir_dd,  phir_t,  phir_tt,  phir_dt  = cls.Phir(rho, T)

        vals = {
            'delta': delta.squeeze(0), 'tau': tau.squeeze(0),
            'phi0': phi0, 'phi0_d': phi0_d, 'phi0_dd': phi0_dd,
            'phi0_t': phi0_t, 'phi0_tt': phi0_tt, 'phi0_dt': phi0_dt,
            'phir': phir, 'phir_d': phir_d, 'phir_dd': phir_dd,
            'phir_t': phir_t, 'phir_tt': phir_tt, 'phir_dt': phir_dt,
        }

        if intype == 'torch':
            vals = {key: val.reshape(size).to(orig_device) for key,val in vals.items()}
        elif intype == 'np':
            vals = {key: val.detach().cpu().numpy() for key, val in vals.items()}
            vals = {key: val.reshape(size) for key,val in vals.items()}
        elif intype == 'single':
            vals = {key: val.item() for key, val in vals.items()}
        elif intype == 'list':
            vals = {key: val.tolist() for key, val in vals.items()}
        
        return vals

    @classmethod
    def p(cls, d,units='MPa'):
        unit = {
            'Pa':1000,
            'kPa':1,
            'MPa':1/1000
        }
        scale = unit[units]
        """p = rho * R * T * (1 + delta * phi^r_delta)"""
        return rhoc * d['delta'] * R * (Tc / d['tau']) * (1 + d['delta'] * d['phir_d']) * scale
    
    @classmethod
    def p_rho(cls, d, units='MPa'):
        unit = {
            'Pa':1000,
            'kPa':1,
            'MPa':1/1000
        }
        scale = unit[units]
        """p = rho * R * T * (1 + delta * phi^r_delta)"""
        T=(Tc / d['tau'])
        return (R*T + 2*R*T*d['delta'] * d['phir_d'] + R*T*d['delta']**2*d['phir_dd']) * scale

    @classmethod
    def s(cls, d,units='J'):
        unit = {
            'J':1000,
            'kJ':1,
            'MJ':1/1000
        }
        scale = unit[units]
        """s = R * (tau*(phi^o_tau + phi^r_tau) - phi^o - phi^r)"""
        return scale * R * (d['tau']*(d['phi0_t'] + d['phir_t']) - d['phi0'] - d['phir'])

    @classmethod
    def h(cls, d,units='J'):
        unit = {
            'J':1000,
            'kJ':1,
            'MJ':1/1000
        }
        scale = unit[units]
        """h = R*T * (1 + tau*(phi^o_tau + phi^r_tau) + delta*phi^r_delta)"""
        return scale * R * (Tc/d['tau']) * (1 + d['tau']*(d['phi0_t'] + d['phir_t'])
                                    + d['delta']*d['phir_d'])

    @classmethod
    def cv(cls, d,units='J'):
        unit = {
            'J':1000,
            'kJ':1,
            'MJ':1/1000
        }
        scale = unit[units]
        """cv = -R * tau^2 * (phi^o_tautau + phi^r_tautau)"""
        return (-R * d['tau']**2 * (d['phi0_tt'] + d['phir_tt']) ) * scale

    @classmethod
    def cp(cls, d,units='J'):
        unit = {
            'J':1000,
            'kJ':1,
            'MJ':1/1000
        }
        scale = unit[units]
        num = (1 + d['delta']*d['phir_d'] - d['delta']*d['tau']*d['phir_dt'])**2
        den = 1 + 2*d['delta']*d['phir_d'] + d['delta']**2*d['phir_dd'] \
            - d['tau']**2*(d['phi0_tt'] + d['phir_tt'])
        # Note: den includes -tau^2*(phi0_tt+phir_tt) which is cv/R
        denom = 1 + 2*d['delta']*d['phir_d'] + d['delta']**2*d['phir_dd']
        return (-R * d['tau']**2 * (d['phi0_tt'] + d['phir_tt']) + R * num/denom)*scale

    @classmethod
    def c(cls, d):
        cv_R = -d['tau']**2 * (d['phi0_tt'] + d['phir_tt'])
        num  = (1 + d['delta']*d['phir_d'] - d['delta']*d['tau']*d['phir_dt'])**2
        den  = 1 + 2*d['delta']*d['phir_d'] + d['delta']**2*d['phir_dd']
        val = R*1000 * (Tc/d['tau']) * (den + num/cv_R)  # m/s^2 (R in kJ -> *1000)
        # d['tau'] carries whichever type helmholtz() returned (torch/np/float/list)
        return torch.sqrt(val) if isinstance(val, torch.Tensor) else np.sqrt(val)

    @staticmethod
    def _match_type(val, like):
        """Cast a torch result back to whatever container type `like`
        came in as (mirrors the type handling helmholtz() already does),
        so mu()/lam()/saturation() behave like every other property
        accessor on this class regardless of scalar/array/tensor input."""
        if isinstance(like, torch.Tensor):
            return val.reshape(like.shape).to(like.device)
        if isinstance(like, np.ndarray):
            return val.detach().cpu().numpy().reshape(like.shape)
        if isinstance(like, list):
            return val.tolist()
        return val.item()

    # =========================================
    # Transport properties (IAPWS_97.VISC/COND)
    # Both formulations only need (rho, T) plus,
    # for thermal conductivity, (drho/dp)_T, cp,
    # cv, mu -- all of which the IAPWS-95 EOS
    # supplies directly, so these are valid over
    # the whole surface, not just Region 1/2.
    # =========================================
    @classmethod
    def mu(cls, d, enhancement=True):
        """Dynamic viscosity [Pa.s], IAPWS 2008 formulation (R12-08), Eq. (10).

        The critical enhancement mu2 needs the isothermal compressibility at two
        temperatures -- the state's own T, and the fixed reference T_R = 1.5*Tc =
        970.644 K -- and R12-08 says both must come from IAPWS-95. VISC lives in the
        IAPWS-97 module, which this module imports, so it cannot reach back here for
        them; they are computed on this side and passed down.

        The second one costs an extra Helmholtz evaluation at (rho, T_R), so mu is
        about twice the price with the enhancement on. That is worth knowing before
        benchmarking property throughput. Pass enhancement=False for the industrial
        simplification mu2 = 1, which R12-08 Sec. 2.8 and Sec. 3 sanction outside the
        near-critical region -- there it agrees with the full form to better than the
        correlation's own uncertainty.

        Leaving it on by default is deliberate: mu feeds the thermal-conductivity
        critical enhancement (R15-11 Eq. 18 divides by it), so a mu missing its own
        enhancement silently inflates lambda near the critical point.
        """
        rho = rhoc * d['delta']
        T = Tc / d['tau']
        if not enhancement:
            val = w97.VISC.mu(rho, T)
            return cls._match_type(val, d['delta'])

        drhodp_T = 1.0 / cls.p_rho(d, units='MPa')            # (drho/dp)_T at T
        T_R = 1.5 * Tc
        # rho arrives as whatever the caller passed in -- numpy or torch -- so build the
        # matching constant array by arithmetic rather than with a library-specific
        # full_like, which would pin this to one backend.
        d_R = cls.helmholtz(rho, rho * 0.0 + T_R)
        drhodp_TR = 1.0 / cls.p_rho(d_R, units='MPa')          # (drho/dp)_T at T_R
        val = w97.VISC.mu(rho, T, drhodp_T=drhodp_T, drhodp_TR=drhodp_TR)
        return cls._match_type(val, d['delta'])

    @classmethod
    def lam(cls, d):
        """Thermal conductivity [W/m/K], IAPWS 2011 formulation (R15-11)."""
        rho = rhoc * d['delta']
        T = Tc / d['tau']
        drhodp = 1.0 / cls.p_rho(d, units='MPa')   # (drho/dp)_T [kg/m^3/MPa]
        cp = cls.cp(d, units='kJ')
        cv = cls.cv(d, units='kJ')
        mu = cls.mu(d)
        val = w97.COND.lam(rho, T, drhodp, cp, cv, mu)
        return cls._match_type(val, d['delta'])

    # =========================================
    # Saturation region (Region 4): the phase-
    # coexistence conditions p'=p'', g'=g'' are
    # solved directly against the Helmholtz
    # energy (Newton, autograd Jacobian), seeded
    # by the IAPWS-95 ancillary equations Eq.
    # (2.5a-c) of Wagner & Pruss (2002).
    # =========================================
    @classmethod
    def _sat_ancillary(cls, T):
        """Initial guess for (p_sat, rho_f, rho_g) from the IAPWS-95
        ancillary equations. Good to a few hundredths of a percent on
        their own -- used here only to seed the exact Newton solve
        below, so that accuracy doesn't matter beyond landing in the
        basin of convergence."""
        theta = 1.0 - T / Tc

        a  = torch.tensor([-7.85951783, 1.84408259, -11.7866497,
                            22.6807411, -15.9618719, 1.80122502],
                           dtype=torch.float64, device=T.device)
        ta = torch.tensor([1.0, 1.5, 3.0, 3.5, 4.0, 7.5],
                           dtype=torch.float64, device=T.device)
        b  = torch.tensor([1.99274064, 1.09965342, -0.510839303,
                            -1.75493479, -45.5170352, -6.74694450e5],
                           dtype=torch.float64, device=T.device)
        tb = torch.tensor([1/3, 2/3, 5/3, 16/3, 43/3, 110/3],
                           dtype=torch.float64, device=T.device)
        c  = torch.tensor([-2.03150240, -2.68302940, -5.38626492,
                            -17.2991605, -44.7586581, -63.9201063],
                           dtype=torch.float64, device=T.device)
        tcv = torch.tensor([2/6, 4/6, 8/6, 18/6, 37/6, 71/6],
                            dtype=torch.float64, device=T.device)

        theta_a = theta.unsqueeze(0) ** ta.unsqueeze(1)   # (6, N)
        theta_b = theta.unsqueeze(0) ** tb.unsqueeze(1)
        theta_c = theta.unsqueeze(0) ** tcv.unsqueeze(1)

        ln_p_pc   = (Tc / T) * (a.unsqueeze(1) * theta_a).sum(dim=0)
        rho_f_rat = 1.0 + (b.unsqueeze(1) * theta_b).sum(dim=0)
        ln_rho_g  = (c.unsqueeze(1) * theta_c).sum(dim=0)

        p0    = pc * torch.exp(ln_p_pc)
        rhof0 = rhoc * rho_f_rat
        rhog0 = rhoc * torch.exp(ln_rho_g)
        return p0, rhof0, rhog0

    @classmethod
    def saturation(cls, T, iters=40, tol=1e-11):
        """Solve for the coexisting liquid/vapor state at temperature T
        by driving p(rho_f,T)-p(rho_g,T) and g(rho_f,T)-g(rho_g,T) to
        zero with Newton's method (2x2 system, Jacobian via autograd).
        Valid for Tt <= T <= Tc. Returns a dict {'T','p','rho_f','rho_g'}
        in the same array convention as helmholtz()/p()/etc (K, MPa,
        kg/m^3)."""
        if isinstance(T, torch.Tensor):
            intype, size = 'torch', T.shape
            orig_device = T.device
            Tt_ = T.reshape(-1).to(torch.float64).to(device)
        elif isinstance(T, np.ndarray):
            intype, size = 'np', T.shape
            Tt_ = torch.from_numpy(T).reshape(-1).to(torch.float64).to(device)
        elif isinstance(T, list):
            Tt_ = torch.tensor(T, dtype=torch.float64, device=device)
            intype, size = 'list', Tt_.shape
        else:
            Tt_ = torch.tensor([float(T)], dtype=torch.float64, device=device)
            intype, size = 'single', Tt_.shape

        # theta**(non-integer) of a negative number is NaN, so keep T
        # strictly inside (Tt, Tc) -- there is no saturation state above
        # the critical point anyway.
        Tt_ = torch.clamp(Tt_, min=Tt, max=Tc - 1e-6)

        p0, rhof0, rhog0 = cls._sat_ancillary(Tt_)
        rho_f = rhof0.clone().requires_grad_(True)
        rho_g = rhog0.clone().requires_grad_(True)

        for it in range(iters):
            d_f = cls.helmholtz(rho_f, Tt_)
            d_g = cls.helmholtz(rho_g, Tt_)
            F1 = cls.p(d_f) - cls.p(d_g)
            F2 = (cls.h(d_f) - Tt_*cls.s(d_f)) - (cls.h(d_g) - Tt_*cls.s(d_g))

            dF1_drf = torch.autograd.grad(F1, rho_f, grad_outputs=torch.ones_like(F1), retain_graph=True)[0]
            dF1_drg = torch.autograd.grad(F1, rho_g, grad_outputs=torch.ones_like(F1), retain_graph=True)[0]
            dF2_drf = torch.autograd.grad(F2, rho_f, grad_outputs=torch.ones_like(F2), retain_graph=True)[0]
            dF2_drg = torch.autograd.grad(F2, rho_g, grad_outputs=torch.ones_like(F2), retain_graph=False)[0]

            det = dF1_drf*dF2_drg - dF1_drg*dF2_drf
            d_rho_f = -(F1*dF2_drg - F2*dF1_drg) / det
            d_rho_g = -(F2*dF1_drf - F1*dF2_drf) / det

            with torch.no_grad():
                rho_f += d_rho_f
                rho_g += d_rho_g
                # Stay on the correct branch: liquid above rhoc, vapor below.
                rho_f.clamp_(rhoc*(1.0 + 1e-6), rhoc*3.5)
                rho_g.clamp_(rhoc*1e-8, rhoc*(1.0 - 1e-6))

            if torch.max(torch.abs(d_rho_f)).item() < tol and torch.max(torch.abs(d_rho_g)).item() < tol:
                break

        with torch.no_grad():
            psat = cls.p(cls.helmholtz(rho_f, Tt_))

        vals = {'T': Tt_.detach(), 'p': psat.detach(),
                'rho_f': rho_f.detach(), 'rho_g': rho_g.detach()}

        if intype == 'torch':
            vals = {key: val.reshape(size).to(orig_device) for key, val in vals.items()}
        elif intype == 'np':
            vals = {key: val.cpu().numpy().reshape(size) for key, val in vals.items()}
        elif intype == 'single':
            vals = {key: val.item() for key, val in vals.items()}
        elif intype == 'list':
            vals = {key: val.tolist() for key, val in vals.items()}

        return vals

    @classmethod
    def p_sat(cls, T):
        """Saturation pressure [MPa] at temperature T [K]."""
        return cls.saturation(T)['p']

    @classmethod
    def T_sat(cls, p, iters=50):
        """Saturation temperature [K] at pressure p [MPa], via bisection
        on p_sat(T), which is monotonic over [Tt, Tc]."""
        if isinstance(p, torch.Tensor):
            intype, size = 'torch', p.shape
            orig_device = p.device
            p_ = p.reshape(-1).to(torch.float64).to(device)
        elif isinstance(p, np.ndarray):
            intype, size = 'np', p.shape
            p_ = torch.from_numpy(p).reshape(-1).to(torch.float64).to(device)
        elif isinstance(p, list):
            p_ = torch.tensor(p, dtype=torch.float64, device=device)
            intype, size = 'list', p_.shape
        else:
            p_ = torch.tensor([float(p)], dtype=torch.float64, device=device)
            intype, size = 'single', p_.shape

        lo = torch.full_like(p_, Tt)
        hi = torch.full_like(p_, Tc - 1e-6)
        for _ in range(iters):
            mid = 0.5*(lo + hi)
            p_mid = cls.saturation(mid)['p']
            hi = torch.where(p_mid > p_, mid, hi)
            lo = torch.where(p_mid > p_, lo, mid)
        Tsat = 0.5*(lo + hi)

        if intype == 'torch':
            return Tsat.reshape(size).to(orig_device)
        elif intype == 'np':
            return Tsat.cpu().numpy().reshape(size)
        elif intype == 'single':
            return Tsat.item()
        elif intype == 'list':
            return Tsat.tolist()

    # =========================================
    # (T,p) -> rho inversion, so IAPWS95 can be
    # driven the same way as iapws.IAPWS95(T=,P=)
    # -- needed since helmholtz() itself only
    # takes (rho, T). Bisection (bracket, always
    # converges) then a Newton polish (autograd)
    # for full precision.
    # NOTE: assumes a single real root in rho at
    # the given (T,p), i.e. p is supercritical
    # (p > pc) or otherwise clearly off the
    # two-phase dome -- this is not a phase-aware
    # solver like saturation() above.
    # =========================================
    @classmethod
    def rho_Tp(cls, T, p, newton_iters=60):
        """Density [kg/m^3] at given T [K], p [MPa].

        Seeds a physically-motivated initial guess -- the saturated-liquid
        ancillary density for compressed liquid, ideal-gas density
        otherwise -- then refines with damped Newton iteration.

        A global bisection over the full [1e-3, 1300] kg/m^3 range (the
        previous approach) is NOT safe: away from the true (T,rho) branch,
        at subcritical temperatures the IAPWS-95 residual terms (several
        carry tau exponents up to 50) stop cancelling cleanly in floating
        point and the computed pressure swings by 10+ orders of magnitude
        between adjacent densities. A bracket search reading that noise as
        a monotonic signal can lock onto a spurious root near the critical
        density instead of the physical one -- e.g. it silently returned
        rho=322 kg/m^3 (the critical density) for T=300 K, p=1 MPa instead
        of the correct ~997 kg/m^3. Seeding close to the true branch and
        damping the Newton step keeps every evaluation on the well-behaved
        side of that unstable region.
        """
        if isinstance(T, torch.Tensor):
            intype, size = 'torch', T.shape
            orig_device = T.device
            T_ = T.reshape(-1).to(torch.float64).to(device)
        elif isinstance(T, np.ndarray):
            intype, size = 'np', T.shape
            T_ = torch.from_numpy(T).reshape(-1).to(torch.float64).to(device)
        elif isinstance(T, list):
            T_ = torch.tensor(T, dtype=torch.float64, device=device)
            intype, size = 'list', T_.shape
        else:
            T_ = torch.tensor([float(T)], dtype=torch.float64, device=device)
            intype, size = 'single', T_.shape
        p_ = torch.as_tensor(p, dtype=torch.float64, device=device).reshape(-1).expand_as(T_)

        psat0, rhof0, _ = cls._sat_ancillary(torch.clamp(T_, min=Tt, max=Tc - 1e-6))
        rho_ig = torch.clamp(p_ * 1000.0 / (R * T_), min=1e-3)
        liquid_like = (T_ < Tc) & (p_ >= psat0)
        rho0 = torch.clamp(torch.where(liquid_like, rhof0, rho_ig), 1e-3, 1300.0)

        rho = rho0.clone().requires_grad_(True)
        for _ in range(newton_iters):
            F = cls.p(cls.helmholtz(rho, T_), units='MPa') - p_
            dF = torch.autograd.grad(F, rho, grad_outputs=torch.ones_like(F))[0]
            with torch.no_grad():
                step = torch.clamp(F / dF, -0.5 * rho, 0.5 * rho)
                rho -= step
                rho.clamp_(1e-4, 1300.0)

        rho = rho.detach()
        if intype == 'torch':
            return rho.reshape(size).to(orig_device)
        elif intype == 'np':
            return rho.cpu().numpy().reshape(size)
        elif intype == 'single':
            return rho.item()
        elif intype == 'list':
            return rho.tolist()

    @classmethod
    def T_hp(cls, h, p, iters=60, T_lo=273.16, T_hi=1300.0):
        """Temperature [K] at given specific enthalpy h [kJ/kg] and
        pressure p [MPa], via bisection on h(T)|_p (built from
        rho_Tp() + helmholtz()/h() above). Assumes h is monotonically
        increasing in T at fixed p, true away from the two-phase dome
        -- e.g. for the supercritical isobars used throughout SCA."""
        if isinstance(h, torch.Tensor):
            intype, size = 'torch', h.shape
            orig_device = h.device
            h_ = h.reshape(-1).to(torch.float64).to(device)
        elif isinstance(h, np.ndarray):
            intype, size = 'np', h.shape
            h_ = torch.from_numpy(h).reshape(-1).to(torch.float64).to(device)
        elif isinstance(h, list):
            h_ = torch.tensor(h, dtype=torch.float64, device=device)
            intype, size = 'list', h_.shape
        else:
            h_ = torch.tensor([float(h)], dtype=torch.float64, device=device)
            intype, size = 'single', h_.shape
        p_ = torch.as_tensor(p, dtype=torch.float64, device=device).reshape(-1).expand_as(h_)

        T_lo_ = torch.full_like(h_, T_lo)
        T_hi_ = torch.full_like(h_, T_hi)
        for _ in range(iters):
            mid = 0.5*(T_lo_ + T_hi_)
            rho_mid = cls.rho_Tp(mid, p_)
            h_mid = cls.h(cls.helmholtz(rho_mid, mid), units='kJ')
            T_lo_ = torch.where(h_mid < h_, mid, T_lo_)
            T_hi_ = torch.where(h_mid < h_, T_hi_, mid)
        T = 0.5*(T_lo_ + T_hi_)

        if intype == 'torch':
            return T.reshape(size).to(orig_device)
        elif intype == 'np':
            return T.cpu().numpy().reshape(size)
        elif intype == 'single':
            return T.item()
        elif intype == 'list':
            return T.tolist()


'''
# scalar
d = IAPWS95.helmholtz(1000, 300)
print(IAPWS95.p(d,'MPa'))  
print(IAPWS95.s(d,'J'))     # 10.0003858 MPa  ✓

# batched — every property call is free arithmetic, no recomputation
rhos = np.linspace(100, 1000, 500)
Ts   = np.full(500, 500.0)
d    = IAPWS95.helmholtz(rhos, Ts)
p    = IAPWS95.p(d)        # shape [500,]
s    = IAPWS95.s(d)         # shape [500,]
h    = IAPWS95.h(d)        # shape [500,]
print(h)

rhos = torch.linspace(100, 1000, 500)
Ts   = 500*torch.ones_like(rhos)
d    = IAPWS95.helmholtz(rhos, Ts)
p    = IAPWS95.p(d)        # shape [500,]
s    = IAPWS95.s(d)         # shape [500,]
h    = IAPWS95.h(d)        # shape [500,]
print(h)

print(len(rhos))
X = torch.cartesian_prod(rhos,Ts)
Rho, T = X[:,0:1], X[:,1:2]
print(len(Rho))
#Rho = Rho.reshape(250000)
#T = T.reshape(250000)
d    = IAPWS95.helmholtz(Rho, T)
p    = IAPWS95.p(d)        # shape [500,]
s    = IAPWS95.s(d)         # shape [500,]
h    = IAPWS95.h(d)        # shape [500,]
print(h)
'''