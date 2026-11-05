"""Dual-cooled annular SCW fuel: global fixed-point solve of the radial heat split,
gap and coolant heat transfer, and the axial enthalpy rise in both channels.

Layout:
    User inputs      geometry, materials, coolant, axial power, solver settings
    correlations     coolant htc, gap htc and fill gas all come from pinthac
    Geometry         derived radii, flow areas, bundle factor
    property tables  IAPWS-95 water (cached) and the UO2 Kirchhoff transform
    Radial model     coolant -> clad -> insulator -> gap -> fuel at every axial node
    Axial model      power shape and RK4 enthalpy integration
    solve            fixed-point iteration over all unknowns
    Plots
"""
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pinthac.properties.matmod as mat
import pinthac.properties.getprop as gp
from pinthac.correlations.bundle import Bundle
from pinthac.correlations.htc import SCW
from pinthac.pin.gap import htc_gap


# =============================================================================
# 1. User inputs
# =============================================================================
geom = {
    'tci':     0.00065,    # inner clad thickness [m]
    'tco':     0.00065,    # outer clad thickness [m]
    'rco_i':   0.020,     # inner clad, coolant-side radius (= inner channel radius) [m]
    'rco_o':   0.030,     # outer clad, coolant-side radius [m]
    'delta_i': 0.00006,   # inner gas gap [m]
    'delta_o': 0.00006,   # outer gas gap [m]
    't_ins':   0.002,     # inner insulator thickness [m]
    'pitch':   0.061,     # hexagonal lattice pin pitch [m]
}

materials = {
    'kc':    15,                         # clad conductivity [W/mK]
    'kins':  4,                          # inner insulator conductivity [W/mK]
    'eps_c': 1,                          # clad emissivity
    'eps_f': 1,                          # fuel emissivity
    'gas':   'He',                       # gap fill gas, any species pinthac's Gas.k knows (He, Ar, Kr, Xe, H2, N2, Air)
    'theta': mat.UO2.Theta_NFI,     # UO2 Kirchhoff transform Theta(T) [W/m]
}

coolant = {
    'P':    25,       # pressure mpa (both channels)
    'G_i':  700,      # inner channel mass flux
    'G_o':  1000,      # outer channel mass flux
    'T_in': 280 + 273.15,   # inlet temperature, both channels [K]
}

axial = {
    'L':    4.2,      # heated length
    'L_ex': 0.25,     # cosine extrapolation distance
    'Nz':   401,       # number axial nodes
    'qp0':  350E3,    # peak lhgr
}

solver = {
    'n_iter': 200,
    'tol':    1E-8,   # residual tol
    'omega':  0.6,    # under-relaxation for heat split
    'fq0':    0.25,   # initial guess for inner heat fraction
    'htcg0':  5000.0, # initial guess for gap htc
}

tables = {
    'T_min': 500.0, 'T_max': 1270.0, 'n_T': 7701,   # water table, 0.1 K spacing resolves the cp peak near T_pc
    # where the cached table is saved: next to this script (or the working directory in a notebook)
    'cache_dir': os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd(),
}


# =============================================================================
# 2. Correlations
# =============================================================================
def kgas(T):
    """Gap fill-gas conductivity [W/mK] at T [K], from pinthac (PNNL-35702)."""
    return mat.Gas.k(materials['gas'], T)


# =============================================================================
# Geometry
# =============================================================================
def GEOM(geo):
    '''
    Takes iinput geometry dict and makes a new dict with all other calculated 
    geometric params
    '''
    tci, tco = geo['tci'], geo['tco']
    rco_i, rco_o = geo['rco_i'], geo['rco_o']
    delta_i, delta_o = geo['delta_i'], geo['delta_o']
    t_ins = geo['t_ins']
    rci_i = rco_i + tci
    rci_o = rco_o - tco
    rins_i = rci_i+t_ins
    rfo_i = rins_i + delta_i
    rfo_o = rci_o - delta_o

    # Inner coolant us a tube
    A_i = np.pi*rco_i**2
    D_i = 2*rco_i

    # Hexaagonal unit cell for outer surface
    pitch = geo['pitch']
    A_o = np.sqrt(3)/2*pitch**2 - np.pi*rco_o**2
    Dh_o = 4*A_o/(2*np.pi*rco_o)
    #Presser rod bundler correction, applied to outer pin surface
    psi_o = float(Bundle.Presser(pitch, 2*rco_o))

    t_fuel = rfo_o - rfo_i

    #Check for unphysical fgeometery inputs
    if t_fuel <= 0:
        warnings.warn(f"Invalid geometry: fuel thickness rfo_o - rfo_i = {1000*t_fuel:.3f} mm. "
                      f"rco_o must exceed rco_i + tci + t_ins + delta_i + delta_o + tco "
                      f"= {rco_i + tci + t_ins + delta_i + delta_o + tco:.5f} m")
    if pitch <= 2*rco_o:
        warnings.warn(f"Invalid geometry: pitch = {pitch} m is not larger than the pin "
                      f"diameter 2*rco_o = {2*rco_o} m, so neighbouring pins overlap")

    #Build full geometry dict
    dic = {
        'tci': tci, 'tco': tco,
        'rco_i': rco_i, 'rco_o': rco_o,
        'delta_i': delta_i, 'delta_o': delta_o,
        'rci_i': rci_i, 'rci_o': rci_o,
        'rfo_i': rfo_i, 'rfo_o': rfo_o,
        'rins_i':rins_i, 't_ins':t_ins,
        'pitch':pitch, 'A_i':A_i, 'A_o':A_o, 'D_i':D_i, 'Dh_o':Dh_o, 'psi_o':psi_o
    }
    return dic


# =============================================================================
# 4. Property tables
# =============================================================================
'''
Checks if a property table already exists in the current directory, if it doesnt,
one is generated using IAPWS95 pinthac library.
'''
def load_cool_table(P, T_min, T_max, n_T, cache_dir):
    """IAPWS-95 water table at P [MPa] on np.linspace(T_min, T_max, n_T).
    Loaded from cache_dir if a matching table was saved before, otherwise built and saved."""
    T = np.linspace(T_min, T_max, n_T)
    path = os.path.join(cache_dir, f"scw_table_{P:g}MPa_{T_min:g}-{T_max:g}K_{n_T}.npz")
    if os.path.exists(path):
        print(f"Loading water property table: {path}")
        data = dict(np.load(path))
        return data.pop('T'), data

    print(f"Building water property table at {P:g} MPa (IAPWS-95) ...")
    data = gp._getprop('SCW', T, P)   # 'rho', 'h', 'cp', 'mu', 'k', 'beta'
    np.savez(path, T=T, **data)
    print(f"Saved to {path}")
    return T, data


# Reuse the table already in memory (e.g. re-running in an interactive session)
# when it was built for the same pressure and temperature grid
cool_key = (coolant['P'], tables['T_min'], tables['T_max'], tables['n_T'])
if globals().get('cool_table_key') != cool_key:
    T_cool_table, cool_table = load_cool_table(*cool_key, tables['cache_dir'])
    cool_table_key = cool_key

def props(T):
    """IAPWS-95 water properties at coolant['P'] and T [K], from the table."""
    return {k: np.interp(T, T_cool_table, v) for k, v in cool_table.items()}

def T_from_h(h):
    """Coolant temperature [K] from enthalpy [J/kg]."""
    return np.interp(h, cool_table['h'], T_cool_table)

def h_from_T(T):
    """Coolant enthalpy [J/kg] from temperature [K]."""
    return np.interp(T, T_cool_table, cool_table['h'])

theta = materials['theta']
T_fuel_table = np.linspace(300, 2800, 2501)
Theta_table = theta(T_fuel_table)

def T_from_theta(Th):
    """Fuel temperature [K] from the UO2 Kirchhoff transform [W/m]."""
    return np.interp(Th, Theta_table, T_fuel_table)


# =============================================================================
# Vectorized radial model, prolly the most important thing here
# =============================================================================
def radial_heat(Tb, fq,qp, HTCconv, HTCgap, eps_c, eps_f, kc, kins , Theta_f, geo):
    Tb_i, Tb_o = Tb[0], Tb[1]
    qp_i, qp_o = fq*qp, qp*(1-fq)

    mdot_i = HTCconv['G_i']*geo['A_i']
    mdot_o = HTCconv['G_o']*geo['A_o']

    #compute new LHGRs
    q_i = qp_i/(2*np.pi*geo['rco_i'])
    q_o = qp_o/(2*np.pi*geo['rco_o'])

    htc_i = SCW.Cheng_Q(props(Tb_i), HTCconv['G_i'], HTCconv['D_i'], q_i)
    htc_o = SCW.Cheng_Q(props(Tb_o), HTCconv['G_o'], HTCconv['D_o'], q_o)*geo['psi_o']

    Tco_i = Tb_i + q_i/htc_i
    Tco_o = Tb_o + q_o/htc_o

    logR_i = np.log(geo['rci_i']/geo['rco_i'])
    logR_o = np.log(geo['rco_o']/geo['rci_o'])

    Tci_i = Tco_i + qp_i/(2*np.pi*kc) * logR_i
    Tci_o = Tco_o + qp_o/(2*np.pi*kc) * logR_o

    logR_ins = np.log(geo['rins_i']/geo['rci_i'])
    Tins_i = Tci_i + qp_i/(2*np.pi*kins) * logR_ins

    htcg_i0, htcg_o0 = HTCgap[0], HTCgap[1]
    Tfo_i = Tins_i + qp_i/(np.pi*(geo['rfo_i']+geo['rins_i'])*htcg_i0)
    Tfo_o = Tci_o + qp_o/(np.pi*(geo['rfo_o']+geo['rci_o'])*htcg_o0)

    # gap htc
    htcg_i = htc_gap(Tfo_i, Tins_i, geo['delta_i'], kgas, eps_c, eps_f)
    htcg_o = htc_gap(Tfo_o, Tci_o, geo['delta_o'], kgas, eps_c, eps_f)

    #computing residual error 
    res_htcg_i = np.abs((htcg_i-htcg_i0)/htcg_i)
    res_htcg_o = np.abs((htcg_o-htcg_o0)/htcg_o)

    qpp_i = qp_i/(np.pi*(geo['rfo_i']+geo['rins_i']))
    qpp_o = qp_o/(np.pi*(geo['rfo_o']+geo['rci_o']))

    Tfo_i = (Tins_i + qpp_i/htcg_i)
    Tfo_o = (Tci_o + qpp_o/htcg_o)

    Th_i = Theta_f(Tfo_i)
    Th_o = Theta_f(Tfo_o)

    rfo_i, rfo_o = geo['rfo_i'], geo['rfo_o']
    Rlog = np.log(geo['rfo_o']/geo['rfo_i'])
    Af = np.pi*(geo['rfo_o']**2-geo['rfo_i']**2)
    qvol = qp/Af

    # Theta(r) = Th_i - qvol/4*(r^2 - rfo_i^2) + C1*ln(r/rfo_i)
    C1 = (Th_o-Th_i+0.25*qvol*(rfo_o**2-rfo_i**2))/Rlog

    #new qp calculations, inner negated so flow to coolant is positive
    new_qp_i = -2*np.pi*(0.5*qvol*rfo_i**2 - C1)
    new_qp_o = 2*np.pi*(0.5*qvol*rfo_o**2 - C1)

    # peak fuel temperature at the extrema
    r_max = np.clip(np.sqrt(np.maximum(2*C1/qvol, 0)), rfo_i, rfo_o) #grap rmax where max temp occurs
    Th_max = Th_i - 0.25*qvol*(r_max**2-rfo_i**2) + C1*np.log(r_max/rfo_i) #compute temp at rmax
    Tfo_max = T_from_theta(Th_max)

    # normalised by total qp (not new_qp, which can cross zero)
    res_qp_i = np.abs((new_qp_i-qp_i)/qp)
    res_qp_o = np.abs((new_qp_o-qp_o)/qp)

    out = {
        'mdot_i': mdot_i, 'mdot_o': mdot_o,
        'htc_i': htc_i, 'htc_o': htc_o,
        'Tco_i': Tco_i, 'Tco_o': Tco_o,
        'Tci_i': Tci_i, 'Tci_o': Tci_o,
        'Tins_i': Tins_i,
        'Tfo_i': Tfo_i, 'Tfo_o': Tfo_o, 'Tfo_max': Tfo_max,
        'Th_i': Th_i, 'Th_o': Th_o,
        'htcg_i': htcg_i, 'htcg_o': htcg_o,
        'new_qp_i': new_qp_i, 'new_qp_o': new_qp_o,
        'fq':new_qp_i/qp,
        'res_htcg': np.array([res_htcg_i, res_htcg_o]),
        'res_qp': np.array([res_qp_i, res_qp_o]),
    }
    return out


# =============================================================================
# 6. Axial model
# =============================================================================
L, Nz, qp0 = axial['L'], axial['Nz'], axial['qp0']
Le = L + 2*axial['L_ex']          # extrapolated length
z = np.linspace(0, L, Nz)

def qp_profile(zz):
    """extrapolated cosine profile"""
    return qp0*np.cos(np.pi*(zz - L/2)/Le)

def rk4(f, z, y0):
    """runge kutta for integrating heat to get enthalpy,
     Returns y with shape (len(y0), len(z))."""
    y = np.zeros((len(y0), len(z)))
    y[:, 0] = y0
    for k in range(len(z)-1):
        dz = z[k+1] - z[k]
        k1 = f(z[k],        y[:, k])
        k2 = f(z[k]+dz/2,   y[:, k] + dz/2*k1)
        k3 = f(z[k]+dz/2,   y[:, k] + dz/2*k2)
        k4 = f(z[k]+dz,     y[:, k] + dz*k3)
        y[:, k+1] = y[:, k] + dz/6*(k1 + 2*k2 + 2*k3 + k4)
    return y


# =============================================================================
# Solver Setup
# =============================================================================

#pull input values
geo = GEOM(geom)
qp = qp_profile(z)
h_in = h_from_T(coolant['T_in'])
eps_c, eps_f = materials['eps_c'], materials['eps_f']
kc, kins = materials['kc'], materials['kins']

# Inputs for Cheng htc corelation
HTCconv = {'G_i':coolant['G_i'], 'G_o':coolant['G_o'],
           'D_i':geo['D_i'], 'D_o':geo['Dh_o']}

#Just labels for the residual printing in the loop
LABELS = ['htc_i', 'htc_o', 'htcg_i', 'htcg_o', 'qp_i', 'qp_o', 'h_i', 'h_o']

# initial guess values for heat split, gas htc, and axial enthalpy
fq = np.full(Nz, solver['fq0'])
htcg = [np.full(Nz, solver['htcg0']), np.full(Nz, solver['htcg0'])]
h = np.full((2, Nz), h_in)   # row 0 = inner, row 1 = outer, guess is set to the input value
htc_prev = np.full((2, Nz), np.inf)   # coolant htc follows q, so track its change per iteration


# =============================================================================
# Solver Loop
# =============================================================================
omega = solver['omega'] #relaxation factor for heat split iteration
res = []      # worst axial node, per residual
for i in range(solver['n_iter']):

    Tb = T_from_h(h) #Compute axial Tb from enthalpy values

    #Compute radial heat step along the full axis
    val = radial_heat(Tb,fq,qp,HTCconv,htcg,eps_c,eps_f,kc,kins,theta,geo)

    #Set new guess values for heat split fraction and gap htc
    fq = fq + omega*(val['fq'] - fq)
    htcg = [val['htcg_i'], val['htcg_o']]

    # Integrate LHGRs to get new axial enthalpy distribution
    mdot = np.array([val['mdot_i'], val['mdot_o']])
    def dhdz(zz, hh):
        fq_z = np.interp(zz, z, fq)
        return qp_profile(zz)*np.array([fq_z, 1-fq_z])/mdot
    h_new = rk4(dhdz, z, [h_in, h_in])
    res_h = np.abs((h_new-h)/h_new) #compute enthalpy residual
    h = h_new

    #Calculate new convective htc and compute residual
    htc_now = np.array([val['htc_i'], val['htc_o']])
    res_htc = np.abs((htc_now-htc_prev)/htc_now)
    htc_prev = htc_now

    res_all = np.concatenate([res_htc, val['res_htcg'], val['res_qp'], res_h])
    res.append(res_all.max(axis=1))
    print(f"iter {i:3d}: " + "  ".join(f"{lab}={r:.4e}" for lab, r in zip(LABELS, res[-1])))
    if res[-1].max() < solver['tol']:
        break

Tb = T_from_h(h)
print(f"mdot_i = {val['mdot_i']:.4f} kg/s, mdot_o = {val['mdot_o']:.4f} kg/s")


# =============================================================================
# 8. Plots
# =============================================================================
plt.semilogy(np.array(res), 'o-', label=LABELS)
plt.xlabel('Iteration')
plt.ylabel('Max relative residual over z')
plt.title('Fixed-point iteration')
plt.legend()
plt.show()

# Converged axial profiles
C = 273.15
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax = ax.ravel()
ax[0].plot(z, h.T/1000, label=['inner', 'outer'])
ax[0].set(ylabel='h [kJ/kg]', title='Bulk enthalpy')
ax[1].plot(z, Tb.T - C, label=['inner', 'outer'])
ax[1].set(ylabel=r'T [$^\circ$C]', title=r'Bulk coolant $T_b$')
ax[2].plot(z, np.array([val['Tco_i'], val['Tco_o']]).T - C, label=['inner', 'outer'])
ax[2].set(ylabel=r'T [$^\circ$C]', title=r'Clad wetted surface $T_{co}$')
ax[3].plot(z, np.array([val['Tci_i'], val['Tci_o']]).T - C, label=['inner', 'outer'])
ax[3].set(ylabel=r'T [$^\circ$C]', title=r'Clad fuel-side surface $T_{ci}$')
ax[4].plot(z, np.array([val['Tfo_i'], val['Tfo_o']]).T - C, label=['inner surface', 'outer surface'])
ax[4].plot(z, val['Tfo_max'] - C, 'k--', label='max')
ax[4].set(ylabel=r'T [$^\circ$C]', title='Fuel temperature')
ax[5].plot(z, fq)
ax[5].set(ylabel='fq', title='Inner heat fraction')
for a in ax:
    a.set_xlabel('z [m]')
    if a.get_legend_handles_labels()[0]:
        a.legend()
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(z, np.array([val['htc_i'], val['htc_o']]).T/1000, label=['inner', 'outer'])
ax.set(xlabel='z [m]', ylabel=r'htc [kW/m$^2$K]', title='Coolant convective htc (Cheng 2019)')
ax.legend()
fig.tight_layout()
plt.show()
