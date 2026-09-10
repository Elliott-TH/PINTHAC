import numpy as np
import matplotlib.pyplot as plt
#import torch
import iapws
import Liquid_Metals as LM
import scipy
from scipy.optimize import fsolve
import os
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import brentq
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(script_dir, 'Props2.csv') #Property Table name

df = pd.read_csv(csv_file) #Imports property table
df.columns = df.columns.str.strip() # Remove leading/trailing whitespace
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce') #convert values to numeric floats

def Property(Prop, prop):# Property Lookup function
    f = interp1d(df[Prop[0]], df[prop], kind='linear', fill_value="extrapolate") #Utilize Scipy interpolation
    val = f(Prop[1])
    return val

sigma = scipy.constants.sigma

      

def Shen(T,G,D):
        props = LM.Props(LM.Lead,T)
        Re, Pr = LM.RePr(G,D,props)
        #print('Lead',Re,Pr)
        Pe = Re*Pr
        k = props['k']
        Nu = 10.287*Pe**(-0.1175)+(0.0599/2.5)*Pe**(0.7575)
        #print('Pe,Nu is',Pe,Nu)
        #print('k is',D)
        htc = Nu*(k/D)
        return htc

def Dittus(Tm,p,G,D):
        mu, cp, k = Property(['T',Tm],'mu'),Property(['T',Tm],'cp'),Property(['T',Tm],'k')#w.mu, w.cp*1000, w.k
        Pr = mu*cp/k
        Re = G*D/mu
        return (0.023*Re**(0.8)*Pr**(0.3))*k/D

def psi(pitch,D):
       q = pitch/D
       val = 0.9217 + 0.1478*(q) - 0.1130*np.exp(-7*(q-1))
       
       return val

def Swenson(Tb,Ts,p,G,D):
        w_b = iapws.IAPWS95(T=Tb,P=p)
        w_s = iapws.IAPWS95(T=Ts, P=p)
        rho_s, rho_b = w_s.rho, w_b.rho
        #props_b, props_s = scw.Props(Tb,rho_b), scw.Props(Ts,rho_s)
        h_s, cp_s, mu_s, k_s = w_s.h*1000, w_s.cp*1000, w_s.mu, w_s.k##1000*props_s['h'],1000*props_s['cp'],props_s['mu'],props_s['k']
        h_b, cp_b, mu_b, k_b = w_b.h*1000, w_b.cp*1000, w_b.mu, w_b.k
        cp_bar = (h_s-h_b)/(Ts-Tb)
        Re_s, Pr_s = G*D/mu_s, mu_s*cp_s/k_s
        #print(mu_s)
        A,B = cp_bar/cp_b, rho_s/rho_b
        Nu_s = 0.00459*Re_s**(0.92)*Pr_s**(0.61)*A**(0.61)*B**(0.23)
        htc = Nu_s*k_s/D
        return htc

def htc_scw(Tm,qp,p,G,D):
        guess = Tm + qp/(np.pi*D*Dittus(Tm,p,G,D))
        def res(Tco):
            Tco = float(Tco[0])
            htc = Swenson(Tm,Tco,p,G,D)
            residue = (Tco - Tm) - qp/(np.pi*D*htc)
            return residue
        Tco = float(fsolve(res,guess)[0])
        htc_val = Swenson(Tm,Tco,p,G,D)
        return htc_val

def gap(qp, delta, Tci,rci,rfo):
        def res(Tfo):
            Tave = (Tfo+Tci)/2
            kgas = 15.8E-4 * Tave**(-0.79)
            htc_cond = kgas/delta
            htc_rad = sigma * (Tfo**4-Tci**4)/(Tfo-Tci)#(Tfo+Tci)*(Tfo**2+Tci**2)#(Tfo**4-Tci**4)/(Tfo-Tci)
            htc_net = htc_cond + htc_rad
            residue = Tfo - (Tci + qp/(np.pi*(rci+rfo)*htc_net))
            return residue
        sol = fsolve(res,Tci+100)
        return sol[0]



def erf(x):
      return scipy.special.erf(x)

def Kint(T):
    # Clamp (not abs/fold): folding a negative excursion onto its positive
    # mirror creates a spurious second root at -T that fsolve/least_squares
    # can lock onto, since Kint(-T) == Kint(T) either way. Clamping to a
    # floor keeps the function monotonic so the solver gets pushed back
    # toward the feasible region instead of finding a false match.
    T = np.maximum(T, 1.0)
    tau = T/1000
    a=16.35
    Term1 = 7.00155*np.log((tau+0.471675)/(tau+4.42356))
    Term2 = 6400*(np.exp(-16.35/tau)/(a*np.sqrt(tau))-np.sqrt(np.pi/a)/(2*a**3)*erf(np.sqrt(a/tau)))
    return 1000*(Term1 + Term2)

def Kfo(T):
    T = np.maximum(T, 1.0)
    tau = T/1000
    a=16.35
    Term1 = (7.5408+17.692*tau+3.6142*tau**2)**(-1)
    Term2 = 6400*tau**(-5/2) * np.exp(-16.35/tau)
    return Term1 + Term2


def HeatEqn(qvol,A1,A2,r):
    Tfunc = -0.25*qvol*r**2 + A1*np.log(r) + A2
    return Tfunc

def qp(r,qvol, A1):
    qpval = 2*np.pi*r * (qvol/2*r - A1/r)
    return qpval

def T_at_r(q_ppp, A1, A2, r, guess):
    # HeatEqn gives K(T) (the conductivity-integral, not T itself) at r, so
    # recovering the actual temperature means inverting Kint.
    K_target = HeatEqn(q_ppp, A1, A2, r)
    def res(T):
        return Kint(T) - K_target
    return fsolve(res, guess)[0]

def peak_fuel_T(q_ppp, A1, A2, rfo_i, rfo_o, Tfo_i, Tfo_o):
    # The peak is either at whichever fuel surface is hotter, or at the
    # interior point where dT/dr=0 (equivalently dK/dr=0, since dK/dT=k(T)>0
    # always) if that point falls inside the fuel annulus: solving
    # -0.5*qvol*r + A1/r = 0 gives r_star^2 = 2*A1/qvol.
    Tmax = max(Tfo_i, Tfo_o)
    if q_ppp > 0:
        r_star_sq = 2*A1/q_ppp
        if r_star_sq > 0:
            r_star = np.sqrt(r_star_sq)
            if rfo_i < r_star < rfo_o:
                T_star = T_at_r(q_ppp, A1, A2, r_star, Tmax + 200)
                Tmax = max(Tmax, T_star)
    return Tmax




def qp_new(Tm_i,Tm_o,p,qp_i,qp_o,qp_total,C0_Guess,inputs):
    G_i, G_o =     inputs['G_i'], inputs['G_o']
    rco_i, rco_o = inputs['rco_i'], inputs['rco_o']
    tc_i,tc_o =    inputs['tc_i'], inputs['tc_o']
    kc_i,kc_o =    inputs['kc_i'], inputs['kc_o']
    delta =        inputs['delta']
    pitch =        inputs['pitch']

    Cir_o = 2*np.pi*rco_o
    d_i = 2*rco_i
    d_o = 2*rco_o
    rci_i = rco_i + tc_i
    rci_o = rco_o - tc_o
    rfo_i = rci_i + delta
    rfo_o = rci_o - delta
    t_fuel = rfo_o-rfo_i
    A_fuel = np.pi*(rfo_o**2-rfo_i**2)
    # Fixed by the actual axial power shape, NOT by the trial (qp_i+qp_o):
    # qp(r,qvol,A1) = 2*pi*(qvol/2*r^2 - A1), so qp(rfo_o,...)-qp(rfo_i,...)
    # = qvol*A_fuel identically, for *any* A1 - i.e. qp_i_new+qp_o_new always
    # equals q_ppp*A_fuel regardless of the boundary conditions. Deriving
    # q_ppp from the trial split instead of qp_total made that identity a
    # tautology against itself, so nothing in the system ever pinned the
    # split to the real total generation rate - it could float to whatever
    # the two sign-matching equations happened to settle on.
    q_ppp = qp_total/A_fuel

    A_o= pitch**2 - np.pi*rco_o**2
    Dh = 4*A_o/Cir_o

    Psi = psi(pitch,d_o)

    htc_conv_i = htc_scw(Tm_i,qp_i,p,G_i,d_i)
    htc_conv_o = Shen(Tm_o,G_o,Dh)*Psi

    # Wall temp from convection, then clad conduction using the linear heat
    # rate qp directly: dT = qp*ln(r_far/r_near)/(2*pi*k). Taking the ratio
    # the "right way around" (always > 1) keeps this positive regardless of
    # which radius is bigger, instead of the previous qpp/htc_clad form which
    # had its log ratio inverted (negative conductance) and was missing a
    # 1/r reference-radius term.
    Tco_i = Tm_i + qp_i/(2*np.pi*rco_i*htc_conv_i)
    j_i = np.log(rci_i/rco_i) if rci_i > rco_i else np.log(rco_i/rci_i)
    Tci_i = Tco_i + qp_i/(2*np.pi*kc_i)*j_i

    Tco_o = Tm_o + qp_o/(2*np.pi*rco_o*htc_conv_o)
    j_o = np.log(rco_o/rci_o) if rco_o > rci_o else np.log(rci_o/rco_o)
    Tci_o = Tco_o + qp_o/(2*np.pi*kc_o)*j_o

    Tfo_i = gap(qp_i,delta,Tci_i,rci_i,rfo_i)
    Tfo_o = gap(qp_o,delta,Tci_o,rci_o,rfo_o)

    def func(x):
          A1, A2 = x
          fi = HeatEqn(q_ppp,A1,A2,rfo_i)
          fo = HeatEqn(q_ppp,A1,A2,rfo_o)
          kint_i = Kint(Tfo_i)
          kint_o = Kint(Tfo_o)
          res1 = kint_i - fi
          res2 = kint_o - fo
          return [res1,res2]
    sol = fsolve(func,C0_Guess)
    A1sol, A2sol = sol[0], sol[1]
    Avec = [A1sol,A2sol]

    qp_i_new = qp(rfo_i,q_ppp,A1sol)
    qp_o_new = qp(rfo_o,q_ppp,A1sol)
    qpvec = [qp_i_new,qp_o_new]

    Tmax = peak_fuel_T(q_ppp, A1sol, A2sol, rfo_i, rfo_o, Tfo_i, Tfo_o)

    return qpvec, Avec, Tmax

c_guess = [1000,1000]

def LHGR(Tm_i, Tm_o, p, qp_total, Cguess, inputs):
    Tm_i = np.abs(float(Tm_i))
    Tm_o = np.abs(float(Tm_o))
    # Root-find the coupled (qp_i, qp_o) pair directly with fsolve instead of
    # fixed-point iterating qp_new() by hand. Cg_state carries the (A1, A2)
    # gap-conduction constants forward so each qp_new() call inside the
    # residual gets a warm-started guess, same as the old loop did.
    Cg_state = {'C': list(Cguess)}

    # Residuals normalized by the local generation scale so fsolve's
    # finite-difference Jacobian sees an O(1) system regardless of where we
    # are on the axial power profile, instead of a system whose magnitude
    # swings by orders of magnitude call to call.
    qp_scale = max(abs(qp_total), 1.0)

    def residual(x):
        qp_i, qp_o = x
        qpvec, Avec, _ = qp_new(Tm_i, Tm_o, p, qp_i, qp_o, qp_total, Cg_state['C'], inputs)
        Cg_state['C'] = Avec
        qp_i_new, qp_o_new = -qpvec[0], qpvec[1]
        return [(qp_i - qp_i_new)/qp_scale, (qp_o - qp_o_new)/qp_scale]

    x0 = [qp_total/2, qp_total/2]
    sol = fsolve(residual, x0)
    qp_i_sol, qp_o_sol = sol[0], sol[1]

    # One final evaluation at the converged point to get the matching (A1, A2)
    qpvec, Avec, Tmax = qp_new(Tm_i, Tm_o, p, qp_i_sol, qp_o_sol, qp_total, Cg_state['C'], inputs)
    Cg = Avec
    return qp_i_sol, qp_o_sol, Cg, Tmax


input = {
    "pitch":  0.06,             # P      : rod pitch (m)
    "rco_i":  0.002,            # r_co,i : inner-cladding radius at the SCW/clad interface (m)
    "rco_o":  0.025,            # r_co,o : outer-cladding radius at the clad/Pb interface (m)
    "tc_i":   0.002,            # t_c,i  : inner cladding thickness (m)
    "tc_o":   0.00055,          # t_c,o  : outer cladding thickness (m)
    "delta":  5e-4,             # delta  : gas gap thickness, both sides (m)
    "kc_i":   24,                # k_c,i  : inner cladding thermal conductivity (W/m-K)
    "kc_o":   24, 
    "G_i":  1000,              # SCW mass flux (kg/m2-s)
    "G_o":   1000, 
}

qp_i0, qp_o0, C0, Tmax0 = LHGR(300+273.15, 600, 25, 25000, [1000,1000], input)
print(qp_i0, qp_o0, C0, Tmax0)


def T_Pb(h):
        # Lead.h(T) is monotonic over its valid range, so a bracketed brentq
        # solve can't wander off to an absurd temperature the way an
        # unconstrained fsolve could; clip to the nearer bound if h itself
        # is slightly out of range rather than extrapolating into nonsense.
        lo, hi = LM.Lead.Tm - 20, 1300
        def res(T):
              return h - LM.Lead.h(T)
        try:
              return brentq(res, lo, hi)
        except ValueError:
              return lo if abs(res(lo)) < abs(res(hi)) else hi


# ---------------------------------------------------------------------------
# Axial marching solution: cosine power shape over a channel of length L,
# coupling the SCW (inner) and Pb (outer) coolant enthalpy balances to the
# LHGR solve above at each axial node.
# ---------------------------------------------------------------------------

pval = 25
Tscw_in = 300 + 273.15
TPb_in = 600
L = 3
n = 400
dz = L/n
Z = np.arange(-L/2+dz, L/2+dz, dz)
q0 = 25E3

def q_p(z):
       return q0*np.cos(np.pi*z/L)

# Flow areas / mass flow rates, built the same way qp_new() builds them
# internally so mdot_i, mdot_o stay consistent with the inputs dict.
A_flow_i = np.pi*input['rco_i']**2
A_flow_o = input['pitch']**2 - np.pi*input['rco_o']**2
mdot_i = input['G_i']*A_flow_i
mdot_o = input['G_o']*A_flow_o

hin_scw = Property(['T',Tscw_in],'h')
hin_Pb = LM.Lead.h(TPb_in)

qp0 = q_p(-L/2+dz/2)
qp0_i, qp0_o, C0_guess, Tmax0 = LHGR(Tscw_in, TPb_in, pval, qp0, [1000,1000], input)

h0_scw = hin_scw + qp0_i*dz/mdot_i
h0_pb = hin_Pb + qp0_o*dz/mdot_o
T0_i = Property(['h',h0_scw],'T')
T0_o = T_Pb(h0_pb)

qpI = [qp0_i]
qpO = [qp0_o]
h_i = [h0_scw]
h_o = [h0_pb]
T_i = [T0_i]
T_o = [T0_o]
T_fuel_max = [Tmax0]

for i in range(1,n):
       z = Z[i]
       qp_local = q_p(z-dz/2)
       Ti = T_i[i-1]
       To = T_o[i-1]
       qp_i, qp_o, C0_guess, Tmax = LHGR(Ti, To, pval, qp_local, C0_guess, input)

       h_scw = h_i[i-1] + qp_i*dz/mdot_i
       T_scw = Property(['h',h_scw],'T')

       h_pb = h_o[i-1] + qp_o*dz/mdot_o
       T_pb = T_Pb(h_pb)

       qpI.append(qp_i)
       qpO.append(qp_o)
       h_i.append(h_scw)
       h_o.append(h_pb)
       T_i.append(T_scw)
       T_o.append(T_pb)
       T_fuel_max.append(Tmax)

       print(f'Step {i}/{n-1} complete')

print('Run Complete')

rho = []
for t in T_i:
      t = float(t)
      rho_i = iapws.IAPWS95(T=t,P=pval).rho
      rho.append(rho_i)

plt.figure()
plt.plot(Z,T_i,label='SCW Temperature')
plt.plot(Z,T_o,label='Lead Temperature')
plt.xlabel('Axial position, z (m)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.show()

plt.figure()
plt.plot(Z,rho,label='Density')
plt.xlabel('Axial position, z (m)')
plt.ylabel('SCW density (kg/m^3)')
plt.legend()
plt.show()

plt.figure()
plt.plot(Z,qpI,label='SCW LHGR')
plt.plot(Z,qpO,label='Lead LHGR')
plt.xlabel('Axial position, z (m)')
plt.ylabel('Linear heat generation rate (W/m)')
plt.legend()
plt.show()

peak_idx = int(np.argmax(T_fuel_max))
print(f'Peak fuel temperature: {T_fuel_max[peak_idx]:.1f} K at z = {Z[peak_idx]:.3f} m')

plt.figure()
plt.plot(Z,T_fuel_max,label='Peak Fuel Temperature')
plt.xlabel('Axial position, z (m)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.show()