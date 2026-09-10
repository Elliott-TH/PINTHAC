import numpy as np
import torch
import functools
import os
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The published coefficient tables are data, not code, so they sit in their own
# directory beside this module rather than alongside the source files.
pth = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iapws_data')

#====================
#Critical Properties
#====================

Tc = 647.096 #Critical temperature [K]
rhoc = 322  #Critical density	[kg m⁻³]
#R = 0.46151805 #Specific gas constant [kJ kg⁻¹ K⁻¹]
R = 0.461526
Tt = 273.16 #Triple-point temperature [K]
pt = 0.000611654771 #Triple-point pressure [MPa]
pc = 22.064 #MPa

def iapws_input(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_args = []
        for a in args:
            if isinstance(a, torch.Tensor):
                new_args.append(a.reshape(-1).to(dtype=torch.float64, device=device))
            elif isinstance(a, str) or isinstance(a, type):  # skip str and cls
                new_args.append(a)
            else:
                new_args.append(torch.atleast_1d(torch.tensor(a, dtype=torch.float64, device=device)))
        result = func(*new_args, **kwargs)
        if isinstance(result, torch.Tensor):
            return result.squeeze()
        return result
    return wrapper

class R1():
    #============
    # Region One
    #============

    #Range
    #273.15 K < T < 623.15 K
    #p_s(T) < p < 100 MPa

    pstar = 16.53
    Tstar = 1386

    Coeff = np.loadtxt(f'{pth}/IAPWS_97_Region1.txt')
    Coeff = torch.from_numpy(Coeff).to(device)
    I, J, n = Coeff[:,0:1], Coeff[:,1:2], Coeff[:,2:3]


    Tph_coeff = np.loadtxt(f'{pth}/Region1_ph.txt')
    Tph_coeff = torch.from_numpy(Tph_coeff).to(device)
    Iph, Jph, nph = Tph_coeff[:,0:1], Tph_coeff[:,1:2], Tph_coeff[:,2:3]

    Tps_coeff = np.loadtxt(f'{pth}/Region1_ps.txt')
    Tps_coeff = torch.from_numpy(Tps_coeff).to(device)
    Ips, Jps, nps = Tps_coeff[:,0:1], Tps_coeff[:,1:2], Tps_coeff[:,2:3]


    @classmethod
    @iapws_input
    def Gamma(cls,p,T):
        T = T.unsqueeze(0)
        p = p.unsqueeze(0)
        pi = p/cls.pstar
        tau = cls.Tstar/T
        n, I, J = cls.n, cls.I,cls.J
        A = 7.1 - pi
        B = tau - 1.222
        g = (n * (A)**I * (B)**J).sum(dim=0)
        g_p = (-n * I * (A)**(I-1) * (B)**J).sum(dim=0)
        g_pp = (-n * I *(I-1)* (A)**(I-2) * (B)**J).sum(dim=0)
        g_t = (n * J * (A)**I * (B)**(J-1)).sum(dim=0)
        g_tt = (n * J * (J-1) * (A)**I * (B)**(J-2)).sum(dim=0)
        g_pt = (-n*J*I*A**(I-1)*B**(J-1)).sum(dim=0)
        return g, g_p, g_pp, g_t, g_tt, g_pt

    @classmethod
    @iapws_input
    def Prop(cls,p,T,prop):
        pi = p/cls.pstar
        tau = cls.Tstar/T
        g, g_p, g_pp, g_t, g_tt, g_pt = cls.Gamma(p,T)

        def vol():
            val = pi * g_p
            out = R*T/p * val * 1e-3
            return out
        def u():
            val = tau*g_t - pi*g_p
            out = R*T*val
            return out

        def h():
            val = tau*g_t
            out = R*T*val
            return out

        def s():
            val = R*(tau*g_t - g)
            return val

        def cp():
            val = -R*tau**2 * g_tt
            return val

        def cv():
            numr = (g_p - tau*g_pt)**2
            val = R*(numr/g_pp-tau**2*g_tt)
            return val

        props = {
            "vol": vol,
            "u": u,
            "s": s,
            "h": h,
            "cp":cp,
            "cv":cv
        }
        return props[prop]()

    @classmethod
    @iapws_input
    def Tph(cls,p,h):
        Tstar = 1 #K
        pstar = 1
        p = p.unsqueeze(0)#p[None,:]
        h = h.unsqueeze(0)#h[None,:]
        pi = p/pstar
        hstar = 2500
        eta = h/hstar
        val = cls.nph * pi**cls.Iph * (eta+1)**cls.Jph
        val_sum = val.sum(dim=0)
        out = Tstar * val_sum
        return out

    @classmethod
    @iapws_input
    def Tps(cls,p,s):
        Tstar = 1 #K
        s_star = 1
        pstar = 1
        p = p.unsqueeze(0)#p[None,:]
        s = s.unsqueeze(0)#s[None,:]
        pi = p/pstar
        sigma = s/s_star
        val = cls.nps * pi**cls.Ips * (sigma+2)**cls.Jps
        val_sum = val.sum(dim=0)
        out = Tstar * val_sum
        return out
'''
pval = torch.tensor([6.89])
Tval = torch.tensor([265+273.15])
hval = torch.tensor([1159.3766])
print(R1.Prop(pval,Tval,'h'))
print(R1.Tph(pval,hval)-273.15)
'''
class R2:
    Coeff_Ideal = np.loadtxt(f'{pth}/Region2_Ideal.txt')
    Coeff_Res = np.loadtxt(f'{pth}/Region2_Res.txt')
    Coeff_I = torch.from_numpy(Coeff_Ideal).to(device)
    Coeff_R = torch.from_numpy(Coeff_Res).to(device)

    #Ideal model Coeffs
    J_id, n_id = Coeff_I[:,0:1], Coeff_I[:,1:2]

    #Res Coeffs
    I_r, J_r, n_r = Coeff_R[:,0:1], Coeff_R[:,1:2], Coeff_R[:,2:3]
    @classmethod
    @iapws_input
    def Ideal(cls,p,T):
        p = p.unsqueeze(0)
        T = T.unsqueeze(0)
        pstar = 1
        Tstar = 540
        tau = Tstar/T
        pi = p/pstar
        n, J = cls.n_id, cls.J_id
        seq = n* tau**(J)

        g = torch.log(pi) + seq.sum(dim=0) #gamma
        g_p = 1/pi
        g_pp = -1/pi**2
        g_t = (n*J*tau**(J-1)).sum(dim=0)
        g_tt = (n*J*(J-1)*tau**(J-2)).sum(dim=0)

        return g, g_p, g_pp, g_t, g_tt

    @classmethod
    @iapws_input
    def Residual(cls,p,T):
        T = T.unsqueeze(0)
        p = p.unsqueeze(0)
        pstar = 1
        Tstar = 540
        tau = Tstar/T
        pi = p/pstar
        n, J, I = cls.n_r, cls.J_r, cls.I_r

        t_1 = tau-0.5
        g = (n*pi**I*t_1**J).sum(dim=0)
        g_p = (n*I*pi**(I-1)*t_1**J).sum(dim=0)
        g_pp = (n*I*(I-1)*pi**(I-2)*t_1**J).sum(dim=0)
        g_t = (n*J*pi**I*t_1**(J-1)).sum(dim=0)
        g_tt = (n*J*(J-1)*pi**(I)*t_1**(J-2)).sum(dim=0)
        g_pt = (n*I*J*pi**(I-1)*t_1**(J-1)).sum(dim=0)
        return g, g_p, g_pp, g_t, g_tt, g_pt

    @classmethod
    @iapws_input
    def Prop(cls, p,T,prop):
        g0, g0_p, g0_pp, g0_t, g0_tt = cls.Ideal(p,T)
        gr, gr_p, gr_pp, gr_t, gr_tt, gr_pt = cls.Residual(p,T)
        pstar = 1
        Tstar = 540
        pi = p/pstar
        tau = Tstar/T
        def vol():
            val = R*T/p * pi * (g0_p + gr_p) * 1e-3
            return val
        def u():
            val = R*T*(tau*(g0_t+gr_t)-pi*(g0_p + gr_p))
            return val
        def s():
            val = R*(tau*(g0_t+gr_t)-(g0+gr))
            return val
        def h():
            val = R*T*(tau*(g0_t+gr_t))
            return val

        def cp():
            val = -R*(g0_tt+gr_tt)
            return val
        def cv():
            numr = (1+pi*gr_p-tau*pi*gr_pt)**2
            denom = 1-pi**2 * gr_pp
            val = -R*((g0_tt+gr_tt)-numr/denom)
            return val
        props = {
            "vol": vol,
            "u": u,
            "s": s,
            "h": h,
            "cp":cp,
            "cv":cv
        }
        return props[prop]()

    # Load backward coefficients
    Tph_2a = torch.from_numpy(np.loadtxt(f'{pth}/Region2_ph_2a.txt')).to(device)
    Tph_2b = torch.from_numpy(np.loadtxt(f'{pth}/Region2_ph_2b.txt')).to(device)
    Tph_2c = torch.from_numpy(np.loadtxt(f'{pth}/Region2_ph_2c.txt')).to(device)

    Tps_2a = torch.from_numpy(np.loadtxt(f'{pth}/Region2_ps_2a.txt')).to(device)
    Tps_2b = torch.from_numpy(np.loadtxt(f'{pth}/Region2_ps_2b.txt')).to(device)
    Tps_2c = torch.from_numpy(np.loadtxt(f'{pth}/Region2_ps_2c.txt')).to(device)

    # B2bc boundary coefficients (Table 19)
    B2bc_n = torch.tensor([
        0.90584278514723e3,
        -0.67955786399241,
        0.12809002730136e-1,
        0.26526571908428e4,
        0.45257578905948e1,
    ], dtype=torch.float64, device=device)

    @classmethod
    @iapws_input
    def _h_B2bc(cls, p):
        """Boundary enthalpy between 2b and 2c given p [MPa], Eq. (21)"""
        n = cls.B2bc_n
        pi = p / 1.0  # p* = 1 MPa
        eta = n[3] + ((pi - n[4]) / n[2])**0.5
        return eta * 1.0  # h* = 1 kJ/kg

    @classmethod
    @iapws_input
    def _eval_backward(cls, coeff, p, x, pstar, xstar, x_shift):
        """Generic backward equation evaluator"""
        I = coeff[:, 0:1]
        J = coeff[:, 1:2]
        n = coeff[:, 2:3]
        p = p[None, :]
        x = x[None, :]
        pi    = p / pstar
        sigma = x / xstar
        val   = n * pi**I * (sigma + x_shift)**J
        return val.sum(dim=0)

    @classmethod
    @iapws_input
    def Tph(cls, p, h):
        """T(p,h) for region 2, auto-selects subregion 2a/2b/2c
        p [MPa], h [kJ/kg] -> T [K]
        Subregions: 2a: p<=4, 2b: p>4 and h>=h_B2bc(p), 2c: p>4 and h<h_B2bc(p)
        """
        # pstar=1 MPa, hstar=2000 kJ/kg for all three subregions
        pstar = 1.0
        hstar = 2000.0

        # Build output tensor
        T_out = torch.zeros_like(p)

        mask_2a = p <= 4.0
        mask_hi = p > 4.0

        # Subregion 2a
        if mask_2a.any():
            p_ = p[mask_2a]
            h_ = h[mask_2a]
            T_out[mask_2a] = cls._eval_backward(
                cls.Tph_2a, p_, h_, pstar, hstar, x_shift=1.0)

        # Subregions 2b and 2c
        if mask_hi.any():
            p_hi = p[mask_hi]
            h_hi = h[mask_hi]
            h_bnd = cls._h_B2bc(p_hi)

            mask_2b = h_hi >= h_bnd
            mask_2c = h_hi <  h_bnd

            if mask_2b.any():
                T_out[mask_hi.nonzero(as_tuple=True)[0][mask_2b]] = \
                    cls._eval_backward(cls.Tph_2b, p_hi[mask_2b], h_hi[mask_2b],
                                       pstar, hstar, x_shift=1.0)
            if mask_2c.any():
                T_out[mask_hi.nonzero(as_tuple=True)[0][mask_2c]] = \
                    cls._eval_backward(cls.Tph_2c, p_hi[mask_2c], h_hi[mask_2c],
                                       pstar, hstar, x_shift=1.0)
        return T_out

    @classmethod
    @iapws_input
    def Tps(cls, p, s):
        """T(p,s) for region 2, auto-selects subregion 2a/2b/2c
        p [MPa], s [kJ/kg·K] -> T [K]
        Subregions: 2a: p<=4, 2b: p>4 and s>=5.85, 2c: p>4 and s<5.85
        """
        T_out = torch.zeros_like(p)

        mask_2a = p <= 4.0
        mask_hi = p > 4.0

        # Subregion 2a: pstar=1, sstar=2
        if mask_2a.any():
            T_out[mask_2a] = cls._eval_backward(
                cls.Tps_2a, p[mask_2a], s[mask_2a],
                pstar=1.0, xstar=2.0, x_shift=2.0)

        # Subregions 2b/2c split at s=5.85 kJ/kg·K
        if mask_hi.any():
            p_hi = p[mask_hi]
            s_hi = s[mask_hi]
            idx_hi = mask_hi.nonzero(as_tuple=True)[0]

            mask_2b = s_hi >= 5.85
            mask_2c = s_hi <  5.85

            # Subregion 2b: pstar=1, sstar=0.7853
            if mask_2b.any():
                T_out[idx_hi[mask_2b]] = cls._eval_backward(
                    cls.Tps_2b, p_hi[mask_2b], s_hi[mask_2b],
                    pstar=1.0, xstar=0.7853, x_shift=10.0)

            # Subregion 2c: pstar=1, sstar=2.9251
            if mask_2c.any():
                T_out[idx_hi[mask_2c]] = cls._eval_backward(
                    cls.Tps_2c, p_hi[mask_2c], s_hi[mask_2c],
                    pstar=1.0, xstar=2.9251, x_shift=2.0)

        return T_out

class RSAT():
    Coeff = np.loadtxt(f'{pth}/Region4.txt')
    C = torch.from_numpy(Coeff).to(device)
    pstar = 1
    tstar = 1
    @classmethod
    @iapws_input
    def p(cls, T):
        tstar = cls.tstar
        pstar = cls.pstar
        c = cls.C
        a = T/tstar
        thet = a + c[8]/(a - c[9])
        A = thet**2 + c[0]*thet + c[1]
        B = c[2]*thet**2 + c[3]*thet + c[4]
        C = c[5]*thet**2 + c[6]*thet + c[7]
        val = pstar*((2*C)/(-B+(B**2-4*A*C)**0.5))**4
        return val
    @classmethod
    @iapws_input
    def T(cls, p):
        tstar = cls.tstar
        pstar = cls.pstar
        n = cls.C
        beta = (p/pstar)**(0.25)
        G = n[1]*beta**2 + n[4]*beta+n[7]
        F = n[0]*beta**2 + n[3]*beta+n[6]
        E = beta**2 + n[2]*beta + n[5]
        D = 2*G/(-F -(F**2-4*E*G)**(0.5))
        numr = n[9]+D-((n[9]+D)**2-4*(n[8]+n[9]*D))**(0.5)
        val = tstar/2 * numr
        return val


T_star   = 647.096   # K
rho_star = 322.0     # kg/m³
mu_star  = 1.00e-6   # Pa.s

# ── Table 1: H_i for mu0 (dilute gas) ─────────────────────────────────────────
H = torch.tensor([1.67752, 2.20462, 0.6366564, -0.241605], dtype=torch.float64, device=device)

# ── Table 2: H_ij for mu1 (finite density) ────────────────────────────────────
# Eq 12: mu1 = exp( rho_bar * SUM_i (1/T_bar - 1)^i * SUM_j H_ij*(rho_bar-1)^j )
# i (rows, 0-5) -> temperature term
# j (cols, 0-6) -> density term
Hij = torch.zeros((6, 7), dtype=torch.float64)
Hij[0,0] =  5.20094e-1;  Hij[1,0] =  8.50895e-2
Hij[2,0] = -1.08374;     Hij[3,0] = -2.89555e-1
Hij[0,1] =  2.22531e-1;  Hij[1,1] =  9.99115e-1
Hij[2,1] =  1.88797;     Hij[3,1] =  1.26613
Hij[5,1] =  1.20573e-1;  Hij[0,2] = -2.81378e-1
Hij[1,2] = -9.06851e-1;  Hij[2,2] = -7.72479e-1
Hij[3,2] = -4.89837e-1;  Hij[4,2] = -2.57040e-1
Hij[0,3] =  1.61913e-1;  Hij[1,3] =  2.57399e-1
Hij[0,4] = -3.25372e-2;  Hij[3,4] =  6.98452e-2
Hij[4,5] =  8.72102e-3;  Hij[3,6] = -4.35673e-3
Hij[5,6] = -5.93264e-4
Hij = Hij.to(device)


class VISC:

    @classmethod
    @iapws_input
    def mu(cls, rho, T):
        """
        Dynamic viscosity of water, IAPWS 2008 (mu2=1, no critical enhancement).

        Args:
            rho : density      [kg/m³]   torch tensor, any shape
            T   : temperature  [K]       torch tensor, same shape as rho
        Returns:
            mu  : viscosity    [Pa.s]    torch tensor, same shape as rho
        """
        T_bar   = T   / T_star
        rho_bar = rho / rho_star

        # ── mu0: dilute-gas contribution, Eq (11) ─────────────────────────────
        # mu0_bar = 100 * sqrt(T_bar) / sum_k( H_k / T_bar^k )
        T_pows = torch.stack([T_bar**(-k) for k in range(4)], dim=0)  # (4, ...)
        denom  = (H.view(-1, *([1]*T_bar.dim())) * T_pows).sum(dim=0)
        mu0    = 100.0 * torch.sqrt(T_bar) / denom

        # ── mu1: finite-density contribution, Eq (12) ─────────────────────────
        # mu1_bar = exp( rho_bar * SUM_i (1/T_bar-1)^i * SUM_j H_ij*(rho_bar-1)^j )
        t_term   = 1.0 / T_bar - 1.0
        rho_term = rho_bar - 1.0

        t_pows   = torch.stack([t_term  **i for i in range(6)], dim=0)  # (6, ...)
        rho_pows = torch.stack([rho_term**j for j in range(7)], dim=0)  # (7, ...)

        # inner[i] = SUM_j H_ij * (rho_bar-1)^j
        inner = (Hij[:, :, *([None]*rho_bar.dim())] * rho_pows[None]).sum(dim=1)  # (6, ...)

        outer = (t_pows * inner).sum(dim=0)
        mu1   = torch.exp(rho_bar * outer)

        # ── mu2: critical enhancement = 1 outside critical region ─────────────
        mu2 = torch.ones_like(mu0)

        return mu_star * mu0 * mu1 * mu2


# =============================================================================
# COND — IAPWS 2011 Thermal Conductivity of Ordinary Water Substance
# Reference: IAPWS R15-11, "Release on the IAPWS Formulation 2011 for the
#            Thermal Conductivity of Ordinary Water Substance"
# Coefficient tables are loaded from external text files:
#   ThCond_Lk.txt  — Table 1: L_k  (dilute-gas,    Eq. 16)
#   ThCond_Lij.txt — Table 2: L_ij (finite-density, Eq. 17)
#   ThCond_Aij.txt — Table 6: A_ij (ref. compressibility at T_R, Eq. 25)
# =============================================================================

class COND:

    # ── Reference constants (Section 2.2) ────────────────────────────────────
    T_star   = 647.096    # K
    rho_star = 322.0      # kg/m³
    p_star   = 22.064     # MPa
    lam_star = 1.00e-3    # W/m/K
    mu_star  = 1.00e-6    # Pa·s
    R_gas    = 0.46151805 # kJ/kg/K  (specific gas constant, Eq. 6)

    # ── Critical-enhancement constants (Table 3) ──────────────────────────────
    Lambda  = 177.8514    # numerical prefactor, Eq. (18)
    qD_inv  = 0.40e-9     # m  — 1/q_D (reference wave-number inverse)
    xi0     = 0.13e-9     # m  — amplitude of correlation length, Eq. (22)
    Gamma0  = 0.06        # amplitude of dimensionless compressibility, Eq. (22)
    nu      = 0.630       # critical exponent
    gamma_c = 1.239       # critical exponent
    T_R     = 1.5         # dimensionless reference temperature, Eq. (23)

    # ── Density-range boundaries for A_ij column selection (Eq. 26) ──────────
    rho_bounds = torch.tensor(
        [0.310559006, 0.776397516, 1.242236025, 1.863354037],
        dtype=torch.float64, device=device)

    # ── Coefficient tables ────────────────────────────────────────────────────
    Lk  = torch.from_numpy(np.loadtxt(f'{pth}/ThCond_Lk.txt')).to(device)   # shape (5,)
    Lij = torch.from_numpy(np.loadtxt(f'{pth}/ThCond_Lij.txt')).to(device)  # shape (5, 6)
    Aij = torch.from_numpy(np.loadtxt(f'{pth}/ThCond_Aij.txt')).to(device)  # shape (6, 5)

    # ── lambda_0: dilute-gas term, Eq. (16) ───────────────────────────────────
    @classmethod
    @iapws_input
    def _lam0(cls, T_bar):
        T_pows = torch.stack([T_bar**(-k) for k in range(5)], dim=0)      # (5,...)
        Lk     = cls.Lk.view(-1, *([1]*T_bar.dim()))                       # (5,1,...)
        denom  = (Lk * T_pows).sum(dim=0)
        return torch.sqrt(T_bar) / denom

    # ── lambda_1: finite-density term, Eq. (17) ───────────────────────────────
    @classmethod
    @iapws_input
    def _lam1(cls, T_bar, rho_bar):
        t_pows   = torch.stack([(1.0/T_bar - 1.0)**i for i in range(5)], dim=0)  # (5,...)
        rho_pows = torch.stack([(rho_bar   - 1.0)**j for j in range(6)], dim=0)  # (6,...)
        # Lij: (5,6) -> broadcast to (5,6,...) against rho_pows (1,6,...)
        Lij   = cls.Lij[:, :, *([None]*rho_bar.dim())]   # (5, 6, ...)
        inner = (Lij * rho_pows[None]).sum(dim=1)         # (5, ...)
        outer = (t_pows * inner).sum(dim=0)               # (...)
        return torch.exp(rho_bar * outer)

    # ── zeta(T_R, rho_bar): reference isothermal compressibility, Eq. (25) ───
    @classmethod
    @iapws_input
    def _zeta_TR(cls, rho_bar):
        """Returns SUM_i A_ij(rho_bar) * rho_bar^i (the denominator of Eq. 25)."""
        rho_pows = torch.stack([rho_bar**i for i in range(6)], dim=0)  # (6,...)
        j = torch.zeros_like(rho_bar, dtype=torch.long)
        j = torch.where(rho_bar > cls.rho_bounds[0], torch.ones_like(j),     j)
        j = torch.where(rho_bar > cls.rho_bounds[1], torch.full_like(j, 2),  j)
        j = torch.where(rho_bar > cls.rho_bounds[2], torch.full_like(j, 3),  j)
        j = torch.where(rho_bar > cls.rho_bounds[3], torch.full_like(j, 4),  j)
        A_col = cls.Aij[:, j]                   # (6, ...)
        return (A_col * rho_pows).sum(dim=0)    # (...)

    # ── lambda_2: critical-enhancement term, Eqs. (18)–(25) ──────────────────
    @classmethod
    @iapws_input
    def _lam2(cls, rho_bar, T_bar, drhodp_T, cp, cv, mu):
        # Dimensionless isothermal compressibility: zeta = (d rho_bar / d p_bar)_T
        zeta_T  = drhodp_T * cls.p_star / cls.rho_star   # at temperature T, Eq. (24)
        zeta_TR = 1.0 / cls._zeta_TR(rho_bar)            # at T_R,          Eq. (25)

        # Delta chi_bar, Eq. (23); must be >= 0
        dchi = rho_bar * (zeta_T - cls.T_R / T_bar * zeta_TR)
        dchi = torch.clamp(dchi, min=0.0)

        # Correlation length xi [m], Eq. (22)
        xi = cls.xi0 * (dchi / cls.Gamma0) ** (cls.nu / cls.gamma_c)

        # Dimensionless argument y = q_D * xi, Eq. (20)
        y = xi / cls.qD_inv

        # Z(y), Eq. (19); set to 0 for y < 1.2e-7 to avoid truncation error
        kappa = cp / cv
        arg   = 1.0 / (1.0/y + y**2 / (3.0 * rho_bar**2))
        Zy = (2.0 / (torch.pi * y)) * (
                  (1.0 - 1.0/kappa) * torch.arctan(y) + y/kappa
                - (1.0 - torch.exp(-arg))
             )
        Zy = torch.where(y < 1.2e-7, torch.zeros_like(Zy), Zy)

        # lambda_2 bar, Eq. (18)
        mu_bar = mu / cls.mu_star
        cp_bar = cp / cls.R_gas      # c_bar_p = c_p / R  (dimensionless)
        return cls.Lambda * rho_bar * T_bar * cp_bar / mu_bar * Zy

    # ── Public interface ──────────────────────────────────────────────────────

    @classmethod
    @iapws_input
    def lam(cls, rho, T, drhodp_T, cp, cv, mu):
        """
        Thermal conductivity [W/m/K], IAPWS 2011.

        Args:
            rho      : density              [kg/m³]
            T        : temperature          [K]
            drhodp_T : (∂ρ/∂p)_T           [kg/m³/MPa]
            cp       : isobaric heat cap.   [kJ/kg/K]
            cv       : isochoric heat cap.  [kJ/kg/K]
            mu       : dynamic viscosity    [Pa·s]
        Returns:
            lam      : thermal conductivity [W/m/K]
        """
        T_bar   = T   / cls.T_star
        rho_bar = rho / cls.rho_star
        lam0 = cls._lam0(T_bar)
        lam1 = cls._lam1(T_bar, rho_bar)
        lam2 = cls._lam2(rho_bar, T_bar, drhodp_T, cp, cv, mu)
        return cls.lam_star * (lam0 * lam1 + lam2)   # W/m/K

    @classmethod
    @iapws_input
    def lam_R1(cls, p, T):
        """
        Thermal conductivity for Region 1 (subcooled liquid) [W/m/K].
        Args: p [MPa], T [K]  — torch tensors
        """
        g, g_p, g_pp, g_t, g_tt, g_pt = R1.Gamma(p, T)
        tau = R1.Tstar / T

        # Specific volume [m³/kg]: v = R * T * g_p / pstar * 1e-3
        # (R [kJ/kg/K], T [K], pstar [MPa] → kJ/kg/MPa = 1e-3 m³/kg)
        vol    = R * T * g_p / R1.pstar * 1e-3
        rho    = 1.0 / vol

        # cp, cv [kJ/kg/K]
        # Note: R1.Gamma returns g_pp = -γ_ππ (sign-flipped vs IF97 definition),
        # so the cv formula and dvdp must compensate with an extra minus sign.
        cp = -R * tau**2 * g_tt
        cv =  R * (-(g_p - tau*g_pt)**2 / g_pp - tau**2 * g_tt)

        # (∂ρ/∂p)_T [kg/m³/MPa]: γ_ππ = -g_pp, so dvdp = R*T*γ_ππ/pstar² * 1e-3
        dvdp   = -R * T * g_pp / R1.pstar**2 * 1e-3   # < 0 for liquid ✓
        drhodp = -rho**2 * dvdp                         # > 0 for liquid ✓

        mu = VISC.mu(rho, T)
        return cls.lam(rho, T, drhodp, cp, cv, mu)

    @classmethod
    @iapws_input
    def lam_R2(cls, p, T):
        """
        Thermal conductivity for Region 2 (superheated steam) [W/m/K].
        Args: p [MPa], T [K]  — torch tensors
        """
        g0, g0_p, g0_pp, g0_t, g0_tt = R2.Ideal(p, T)
        gr, gr_p, gr_pp, gr_t, gr_tt, gr_pt = R2.Residual(p, T)

        pstar2 = 1.0    # MPa
        Tstar2 = 540.0  # K
        pi  = p / pstar2
        tau = Tstar2 / T

        # Specific volume [m³/kg]: v = R * T * (g0_p + gr_p) / pstar2 * 1e-3
        vol    = R * T * (g0_p + gr_p) / pstar2 * 1e-3
        rho    = 1.0 / vol

        # cp, cv [kJ/kg/K]
        cp  = -R * tau**2 * (g0_tt + gr_tt)
        num =  (1.0 + pi*gr_p - tau*pi*gr_pt)**2
        den =  1.0 - pi**2 * gr_pp
        cv  = -R * tau**2 * (g0_tt + gr_tt) + R * num/den

        # (∂ρ/∂p)_T [kg/m³/MPa]: dv/dp = R*T*(g0_pp + gr_pp) / pstar2² * 1e-3
        dvdp   =  R * T * (g0_pp + gr_pp) / pstar2**2 * 1e-3
        drhodp = -rho**2 * dvdp

        mu = VISC.mu(rho, T)
        return cls.lam(rho, T, drhodp, cp, cv, mu)
    
    

class Sigma:
    @classmethod
    @iapws_input
    def sigma(cls,T):
        Tc = 647.096
        tau = 1 - T/Tc
        B = 235.8/1E3
        b = -0.625
        mu = 1.256
        sig = B*tau**mu *(1+b*tau)
        return sig
