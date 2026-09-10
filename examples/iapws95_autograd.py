import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_optimizer.optimizer.soap import SOAP as Soap
from pinthac.properties.iapws95 import IAPWS95
import time

n = 1000
Tin = torch.linspace(350+273.15,500+273.15,n)
Tin = Tin.unsqueeze(1)
print(Tin)
Tin.requires_grad_(True)
p_in = 25 * torch.ones_like(Tin)

# rho_Tp() detaches its solved density, so autograd through it alone would
# drop the drho/dT|_p term that h(rho,T) depends on. Reattach rho to Tin via
# the implicit function theorem instead of unrolling the solver:
# F(rho,T) = p(rho,T) - p_in = 0 defines rho(T)|_p, so
# drho/dT|_p = -(dF/dT)_rho / (dF/drho)_T -- dF/drho comes from the
# analytic p_rho(), dF/dT comes from autograd through T at the converged
# (detached) rho0. F ~ 0 there, so this leaves rho_in numerically equal to
# rho0 while giving it the correct gradient w.r.t. Tin.

def Newton(T,p,iter = 3000):
    guess = IAPWS95.rho_Tp(T,p)
    #print('Guess Val is',guess)
    err = torch.ones_like(p)
    rho_old=guess
    i = 0
    maxiter=iter
    while err.all()>1E-8:
        i = i+1
        rho_new = (rho_old - (f(rho_old,T,p))/(grad_p(rho_old,T)))
        #print(xnew,xold)
        #print(rho_new)
        err = torch.abs(rho_new-rho_old)
        rho_old = rho_new

        if i>=maxiter:
            print('Max Iterations Exceeded')
            break
    print(f'Converged in {i} Iterations')
    return rho_new

rho0 = IAPWS95.rho_Tp(Tin, p_in)
d0 = IAPWS95.helmholtz(rho0, Tin)
F = IAPWS95.p(d0, units='MPa') - p_in.to(d0['delta'].device)
dp_drho = IAPWS95.p_rho(d0, units='MPa').detach()
rho_in = rho0 - F / dp_drho

d = IAPWS95.helmholtz(rho_in,Tin)
hin = IAPWS95.h(d)
plt.plot(Tin.detach().cpu().numpy(),hin.detach().cpu().numpy())
plt.show()
cpin = IAPWS95.cp(d)
h_T = torch.autograd.grad(hin,Tin,grad_outputs=torch.ones_like(hin))[0]
print(hin.size())
print(h_T.size())

T = Tin.detach().cpu().numpy()
dh_dT = h_T.detach().cpu().numpy()
cp = cpin.detach().cpu().numpy()

plt.plot(T,cp,label='Specific Heat')
plt.plot(T,dh_dT,label='dh_dT',linestyle='--')
plt.legend()
plt.show()


pval = 25
Tscw_in = 300 + 273.15
TPb_in = 600
L = 3
n = 400
dz = L/n
Z = np.arange(-L/2+dz, L/2+dz, dz)
q0 = 25E3
print(hin)
def q_p(z):
       return q0*np.cos(np.pi*z/L)

h = hin
mdot = 0.32
start_time = time.perf_counter()
for i in range(1,n):
       z = Z[i]
       qp_local = q_p(z-dz/2)

       h_scw = h[:,i-1] + q_p(z + dz/2)*dz/mdot
       #print(f'h_i-1 = {h[:,i-1]}')
       #print(f'h_i = {h_scw}')
       h = torch.cat([h,h_scw.unsqueeze(1)],dim=1)


       #print(f'Step {i}/{n-1} complete')
end_time = time.perf_counter()
print(f"Time Elapsed = {end_time-start_time}")
print(h.size())
h = h.detach().cpu().numpy()
plt.plot(Z,h[0,:])
plt.show()