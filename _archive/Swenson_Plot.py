import numpy as np
import torch
import HTC
import getprop as gp
import matplotlib.pyplot as plt
from Arr_Compat import compat
#Brief script to plot the swenson model and visualize the peaks
ri = 0.0035      # fuel inner radius, m
ro = 0.0055      # fuel outer radius, m
tci = 0.0006      # inner cladding thickness, m
tco = 0.0006      # outer cladding thickness, m
delta_i = 0.0001  # inner (fuel-ID-side) gas gap, m
delta_o = 0.0001
rfo_i = ri + tci + delta_i
rfo_o = ro - tco - delta_o
P_i = 2*np.pi*rfo_i  # wetted perimeter of the inner (cooled) surface, m
qpp_vals = P_i**(-1) * torch.tensor([5E3, 10E3, 15E3, 20E3, 25E3, 50E3])

def T_Pseudo(p):
    '''
    Determines the pseutofritical temperature/Widom line for Cp given a pressure.
    '''
    lib = compat(p)
    c, b, a = 554.56975-p, -1.91072, 0.00168
    Tval_rho = (-b + lib.sqrt(b**2-4*a*c))/(2*a)
    #c, b, a = 466.30814-p, -1.61866, 0.00144
    c, b, a = 339.66383-p, -1.23373, 0.00115
    Tval_cp = (-b + lib.sqrt(b**2-4*a*c))/(2*a)
    Tval = 0.5*(Tval_rho+Tval_cp)
    return Tval_rho, Tval_cp
Tps_r, Tps_cp = T_Pseudo(25)

Tm = 360 + 273.15
def sw(Ts):
    prop_m = gp._getprop("SCW",Tm,25)
    prop_s = gp._getprop("SCW",Ts,25)
    return HTC.SCW.Swenson_dT(Props_b=prop_m,Props_w=prop_s,G=1000,D=2*ri,Tw=Ts,Tb=Tm)
Ts_lin = torch.linspace(Tm+1,Tm+200,500)
for qpp in qpp_vals:
    # residual solved for Tw in HTC._solve_Tw_scw: h(Tw)*(Tw-Tb) - q
    res = sw(Ts_lin)*(Ts_lin - Tm) - qpp
    plt.plot(Ts_lin,res,label=f'LHGR={qpp*P_i/1e3:.0f} kW/m')

Tps_y = 1E6 * torch.linspace(-0.75,1,500)
Tpslinr = Tps_r * torch.ones_like(Tps_y)
Tpslincp = Tps_cp * torch.ones_like(Tps_y)
plt.plot(Tpslinr,Tps_y,color='black',linestyle='-',label=r'$T_{Pseudocritical}$ rho')
plt.plot(Tpslincp,Tps_y,color='black',linestyle='--',label=r'$T_{Pseudocritical}$, cp')
plt.axhline(0, color='k', linewidth=0.8)
plt.xlabel('Tw [K]')
plt.ylabel(r'Residual, $h(T_w)(T_w-T_b)-q$  [W/m$^2$]')
plt.legend()
plt.title(f'Swenson Residual Tm={Tm}')

plt.savefig('Swenson_Plot2.png')
plt.show()