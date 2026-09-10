from pinthac.properties import liqprops as lm
import torch
import numpy as np
import torchquad as tq
import matplotlib.pyplot as plt

L = 2
qp0 = 25E3
mdot = 0.32
Tin = 350 + 273.15
N = 20000  # MC trials

def qp(z):
    return qp0 * torch.cos(np.pi * z / L)

# integrate the LHGR over the channel to get total power (W)
simpson = tq.Simpson()
Q = simpson.integrate(qp, dim=1, N=999, integration_domain=[[-1, 1]]).item()

# inlet enthalpy + its uncertainty (Sobolev-reported, treat as 95% CI -> k=2)
Hin = lm.Lead.h(Tin)
sigma_h = np.mean(lm.Lead.uncert_h) / 2

z = np.random.randn(N)
Hin_samples = Hin * (1 + sigma_h * z)
Hout_samples = Hin_samples + Q / mdot

plt.hist(Hout_samples, bins=80, density=True, color='tab:blue', alpha=0.75)
plt.xlabel('Outlet enthalpy (J mol$^{-1}$)')
plt.ylabel('Probability density')
plt.title('Monte Carlo distribution of channel outlet enthalpy (Pb)')
plt.tight_layout()
plt.savefig('outlet_enthalpy_dist.png', dpi=180)
plt.tick_params(top=True, right=True, direction='in')
plt.show()