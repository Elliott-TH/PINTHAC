"""
Parametric direct PINN for the annular fuel pin: same two-stream axial
energy balance Direct_PINN2.py solves, but with the geometry and flow
promoted from script constants to *network inputs*, and with the full
coolant -> cladding -> gas gap -> fuel-surface resistance chain
(Ann_SCA.closure's chain, in torch) instead of convection + a constant-kc
cladding log-drop alone.

Network:  (zstar, mdot_i, mdot_o, r_i, r_o, tc_i, tc_o, pitch,
           delta_i, delta_o)  ->  (h_i, h_o)

so a single trained model covers the whole parameter box in RANGE below,
rather than one fixed pin.  The model, its input normalization and the
config it was trained under are saved together to OUTDIR/model.pt; see
load_model()/predict() at the bottom for the reload path.

Radial layout (center to edge), matching Ann_SCA.geometry:

  inner coolant | Rci_ID | inner clad | Rci_OD | delta_i | r_i |
  fuel | r_o | delta_o | Rco_ID | outer clad | Rco_OD | outer coolant

Residuals (both dimensionless, normalized by q0):
  l1  two-stream energy balance, mdot_i*dh_i/dz + mdot_o*dh_o/dz = q'(z)
  l2  fuel conduction consistency: the network's own flux split must
      reproduce the split that PinHT.Ann_HT/Ann_qpp give from the fuel
      surface temperatures that same split implies through the
      resistance chain.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pytorch_optimizer.optimizer.soap import SOAP as Soap
from pinthac.properties import matmod as mat
from pinthac import pin as ht
from pinthac.correlations import htc as htc
from pinthac.properties import getprop as gp
from pinthac.properties import iapws95 as iapws

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# The tables below get built once on cpu and moved to device after. IAPWS_97's
# transport coefficients (VISC.H/Hij, COND.Lk/Lij/Aij) are cpu tensors with no
# .to(device), so mu/lam raise a device mismatch if IAPWS_95 runs on cuda.
iapws.device = torch.device("cpu")
for key,val in list(vars(iapws.IAPWS95).items()):
    if isinstance(val,torch.Tensor):
        setattr(iapws.IAPWS95,key,val.cpu())

# ---------------------------------------------------------------- config

# Fixed operating point -- not network inputs, same for every sampled case.
L = 2.0
q0 = 36E3
p = 25.0
T_in = 620.0
GAS = 'He'          # gap fill gas, MatMod.Gas.k
EPS_F = 1.0         # fuel-surface / clad-surface emissivities for the gap
EPS_C = 1.0         # radiation term; 1.0 matches Ann_SCA.closure's default
N_GAP = 4           # unrolled Picard passes for the implicit gap htc(Tfo)

# Network inputs, in this column order (column 0 is zstar).
PARAMS = ['mdot_i','mdot_o','r_i','r_o','tc_i','tc_o','pitch','delta_i','delta_o']

# The pin the model is centered on -- also the case reported/plotted at the
# end, and (with FORCE_NOMINAL) pinned into every training batch.
NOM = dict(mdot_i=0.5, mdot_o=0.4,
           r_i=5.0E-3, r_o=7.0E-3,
           tc_i=0.5E-3, tc_o=0.6E-3,
           pitch=16.3E-3,
           delta_i=1.0E-4, delta_o=1.0E-4)

# Input normalization box, [-1,1] on each column. Wide enough to contain
# everything sample_params() can produce -- r_o and pitch are sampled
# through derived quantities (see sample_params), so their boxes are the
# induced ranges rather than independent draws.
RANGE = dict(mdot_i=(0.30,0.70), mdot_o=(0.24,0.56),
             r_i=(4.0E-3,6.0E-3), r_o=(5.2E-3,9.0E-3),
             tc_i=(0.35E-3,0.70E-3), tc_o=(0.40E-3,0.85E-3),
             pitch=(1.16E-2,2.41E-2),
             delta_i=(4.0E-5,1.6E-4), delta_o=(4.0E-5,1.6E-4))

# Sampling controls for the two derived columns.
TFUEL_RNG = (1.2E-3, 3.0E-3)   # annulus thickness r_o - r_i
PD_RNG    = (1.03, 1.20)       # pitch / (2*Rco_OD), keeps A_o > 0 by construction

w = 64
Nc = 32             # parameter cases per training step
Nz = 48             # axial collocation points per case
Nit = 6000
lr = 2E-3
FORCE_NOMINAL = True
outdir = 'Direct_PINN3_out'

# ------------------------------------------------------- property tables

# Built once at fixed p. T_hp/rho_Tp are bisection solves (~10 s per call at
# this size), so they cannot sit inside the training loop. The range is wider
# than Direct_PINN2's because the flow rates are now sampled down as far as
# RANGE['mdot_i'][0], which raises the attainable enthalpy rise.
T_tab = torch.linspace(520.0,900.0,761)
P_tab = gp._getprop('SCW',T_tab,p)
P_tab = {key: val.to(device=device,dtype=dtype) for key,val in P_tab.items()}
T_tab = T_tab.to(device=device,dtype=dtype)
h_tab = P_tab['h']

Tf_tab = torch.linspace(500.0,3500.0,3001,device=device,dtype=dtype)
kf_tab = mat.UO2.k_Klimenko(Tf_tab)
Theta_tab = torch.cat([torch.zeros(1,device=device,dtype=dtype),
                        torch.cumsum(0.5*(kf_tab[1:]+kf_tab[:-1])*torch.diff(Tf_tab),0)])

def interp(x,xp,fp):
    xc = x.reshape(-1).clamp(xp[0],xp[-1]).contiguous()
    j = torch.searchsorted(xp,xc).clamp(1,len(xp)-1)
    x0, x1 = xp[j-1], xp[j]
    f0, f1 = fp[j-1], fp[j]
    return (f0 + (f1-f0)*(xc-x0)/(x1-x0)).reshape(x.shape)

def Props(h):
    return {key: interp(h,h_tab,val) for key,val in P_tab.items()}

def T_h(h):
    return interp(h,h_tab,T_tab)

def Theta(T):
    """Kirchhoff transform of UO2 conductivity, integral of k_Klimenko dT.
    Same role as PinHT.Ann_Theta, but as a torch interpolant so it stays in
    the autograd graph."""
    return interp(T,Tf_tab,Theta_tab)

def qp(z):
    return q0*torch.cos(np.pi*z/L)

h_in = interp(torch.tensor([T_in],device=device),T_tab,h_tab)
kgas = lambda T: mat.Gas.k(GAS,T)

# ------------------------------------------------------------- geometry

def geometry(P):
    """
    Per-case radii and channel hydraulics from the raw parameter dict, all
    columns broadcastable (N,1) tensors. Mirrors Ann_SCA.geometry: the
    coolant-wetted surfaces are the *cladding* ID/OD, not r_i/r_o, now
    that there is cladding and a gas gap in between.
    """
    r_i, r_o = P['r_i'], P['r_o']
    Rci_OD = r_i - P['delta_i']
    Rci_ID = Rci_OD - P['tc_i']
    Rco_ID = r_o + P['delta_o']
    Rco_OD = Rco_ID + P['tc_o']

    A_i = np.pi*Rci_ID**2
    A_o = P['pitch']**2 - np.pi*Rco_OD**2
    D_i = 2*Rci_ID
    D_o = 4*A_o/(2*np.pi*Rco_OD)
    return dict(Rci_ID=Rci_ID, Rci_OD=Rci_OD, Rco_ID=Rco_ID, Rco_OD=Rco_OD,
                A_f=np.pi*(r_o**2 - r_i**2),
                D_i=D_i, D_o=D_o,
                G_i=P['mdot_i']/A_i, G_o=P['mdot_o']/A_o)

def sample_params(n,generator=None):
    """
    Uniform draw over the parameter box, returned as an (n,9) tensor in
    PARAMS order.

    r_o and pitch are drawn through derived quantities rather than
    independently, so that every sample is geometrically admissible: the
    annulus thickness r_o - r_i is drawn from TFUEL_RNG (guaranteeing
    r_o > r_i with a workable fuel thickness), and the pitch from a
    pitch-to-outer-clad-diameter ratio (guaranteeing the outer subchannel
    flow area pitch**2 - pi*Rco_OD**2 stays positive). Independent draws
    would put a large fraction of the box in unphysical territory.
    """
    def U(lo,hi):
        return lo + (hi-lo)*torch.rand(n,1,device=device,generator=generator)

    mdot_i, mdot_o = U(*RANGE['mdot_i']), U(*RANGE['mdot_o'])
    r_i = U(*RANGE['r_i'])
    r_o = r_i + U(*TFUEL_RNG)
    tc_i, tc_o = U(*RANGE['tc_i']), U(*RANGE['tc_o'])
    delta_i, delta_o = U(*RANGE['delta_i']), U(*RANGE['delta_o'])
    pitch = U(*PD_RNG)*2*(r_o + delta_o + tc_o)
    return torch.cat([mdot_i,mdot_o,r_i,r_o,tc_i,tc_o,pitch,delta_i,delta_o],1)

def as_dict(P):
    return {k: P[:,j:j+1] for j,k in enumerate(PARAMS)}

def nominal(n=1):
    return torch.tensor([[NOM[k] for k in PARAMS]],device=device).repeat(n,1)

# ---------------------------------------------------------------- model

Nin = 1 + len(PARAMS)
Nout = 2

class PINN(nn.Module):
    """
    Same MLP as Direct_PINN2, widened for the parametric input, with the
    input normalization carried as buffers so a reloaded state_dict is
    self-contained (no need to re-derive RANGE at load time).
    """
    def __init__(self,lo,hi):
        super().__init__()
        self.register_buffer('lo',lo)
        self.register_buffer('hi',hi)

        self.Input = nn.Linear(Nin,w)
        self.Layer1 = nn.Linear(w,w)
        self.Layer2 = nn.Linear(w,w)
        self.Layer3 = nn.Linear(w,w)
        self.Layer4 = nn.Linear(w,w)
        self.Output = nn.Linear(w,Nout)
        self.Tanh = nn.Tanh()

    def norm(self,x):
        # column 0 (zstar) already lives on [-1,1]
        xp = 2*(x[:,1:] - self.lo)/(self.hi - self.lo) - 1.0
        return torch.cat([x[:,0:1],xp],1)

    def forward(self,x):
        x = self.norm(x)
        x = self.Tanh(self.Input(x))
        x = self.Tanh(self.Layer1(x))
        x = self.Tanh(self.Layer2(x))
        x = self.Tanh(self.Layer3(x))
        x = self.Tanh(self.Layer4(x))
        y = self.Output(x)
        return y

lo = torch.tensor([[RANGE[k][0] for k in PARAMS]],device=device)
hi = torch.tensor([[RANGE[k][1] for k in PARAMS]],device=device)
model = PINN(lo,hi).to(device)

def H(x):
    """
    Enthalpy field with the inlet condition hard-enforced: h(-L/2) = h_in
    and h > h_in everywhere, so neither needs a loss term. h_s is the
    scale of the attainable rise for *this* case's flow rates, which now
    vary across the batch.
    """
    mdot_i, mdot_o = x[:,1:2], x[:,2:3]
    h_s = q0*L/(np.pi*(mdot_i + mdot_o))
    return h_in + h_s*(x[:,0:1]+1.0)*nn.functional.softplus(model(x))

# ------------------------------------------------------- resistance chain

def clad_drop(T_wet,qp_lin,Rin,Rout):
    """
    Temperature rise across a cladding layer carrying linear heat rate
    qp_lin, from its coolant-wetted surface to its fuel-facing surface.
    Rin/Rout are the smaller/larger radius of the layer, so the log term
    is positive either way and the drop always points away from the
    coolant -- inner clad: wetted at Rci_ID, hot at Rci_OD; outer clad:
    wetted at Rco_OD, hot at Rco_ID.

    kc = MatMod.Zircalloy.k evaluated at the layer mean temperature; one
    corrector pass is enough since k varies by only a few percent over a
    typical few-tens-of-K wall drop.
    """
    lg = torch.log(Rout/Rin)
    dT = qp_lin*lg/(2*np.pi*mat.Zircalloy.k(T_wet))
    return qp_lin*lg/(2*np.pi*mat.Zircalloy.k(T_wet + 0.5*dT))

def gap_drop(T_clad,qp_lin,r_fuel,delta):
    """
    Fuel surface temperature across an open gas gap, via PinHT.htc_gap
    (conduction through the fill gas + surface-to-surface radiation).

    htc_gap depends on the fuel surface temperature it is being used to
    find -- both through kgas at the gap mean temperature and through the
    radiation term -- so this is a small unrolled Picard, started from the
    conduction-only drop. It converges in 2-3 passes because the radiation
    term is a modest correction at these temperatures, and being unrolled
    it stays differentiable end to end.
    """
    qpp = qp_lin/(2*np.pi*r_fuel)
    Tfo = T_clad + qpp*delta/kgas(T_clad)
    for _ in range(N_GAP):
        hg = ht.htc_gap(Tfo,T_clad,delta,kgas,eps_c=EPS_C,eps_f=EPS_F)
        Tfo = T_clad + qpp/hg
    return Tfo, hg

def chain(h_i,h_o,qp_i,qp_o,P,g):
    """
    Full series resistance chain on both sides: bulk coolant -> convective
    film -> cladding conduction -> gas gap -> fuel surface, given the
    enthalpies and the linear-heat-rate split.

    This is the piece Direct_PINN2 was missing: it stopped at convection
    plus a constant-kc cladding log-drop taken directly off the fuel
    surface, with no gas gap at all.
    """
    Tb_i, Tb_o = T_h(h_i), T_h(h_o)

    htc_i = htc.Water.Dittus(Props(h_i),g['G_i'],g['D_i'])
    htc_o = htc.Water.Dittus(Props(h_o),g['G_o'],g['D_o'])

    Tci_ID = Tb_i + qp_i/(2*np.pi*g['Rci_ID']*htc_i)
    Tco_OD = Tb_o + qp_o/(2*np.pi*g['Rco_OD']*htc_o)

    Tci_OD = Tci_ID + clad_drop(Tci_ID,qp_i,g['Rci_ID'],g['Rci_OD'])
    Tco_ID = Tco_OD + clad_drop(Tco_OD,qp_o,g['Rco_ID'],g['Rco_OD'])

    Tfo_i, hgap_i = gap_drop(Tci_OD,qp_i,P['r_i'],P['delta_i'])
    Tfo_o, hgap_o = gap_drop(Tco_ID,qp_o,P['r_o'],P['delta_o'])

    return dict(Tb_i=Tb_i, Tb_o=Tb_o, htc_i=htc_i, htc_o=htc_o,
                Tci_ID=Tci_ID, Tci_OD=Tci_OD, Tco_ID=Tco_ID, Tco_OD=Tco_OD,
                Tfo_i=Tfo_i, Tfo_o=Tfo_o, hgap_i=hgap_i, hgap_o=hgap_o)

# ------------------------------------------------------------- residuals

def state(x):
    """Everything the residuals and the post-processing both need."""
    P = as_dict(x[:,1:])
    g = geometry(P)
    z = x[:,0:1]*L/2

    Y = H(x)
    h_i, h_o = Y[:,0:1], Y[:,1:2]
    # Two separate grad calls rather than one on Y: grad_outputs=ones would
    # sum the outputs and hand back dh_i/dz + dh_o/dz in a single column,
    # and the two streams need their own derivative.
    hi_z = torch.autograd.grad(h_i,x,grad_outputs=torch.ones_like(h_i),
                               create_graph=True)[0][:,0:1]*2/L
    ho_z = torch.autograd.grad(h_o,x,grad_outputs=torch.ones_like(h_o),
                               create_graph=True)[0][:,0:1]*2/L

    qp_i, qp_o = P['mdot_i']*hi_z, P['mdot_o']*ho_z
    q3 = (qp_i + qp_o)/g['A_f']

    c = chain(h_i,h_o,qp_i,qp_o,P,g)

    # PinHT closes the annular conduction problem the other way round:
    # given the two surface temperatures it returns C1, and Ann_qpp turns
    # that into the surface fluxes. The inner surface carries a minus sign
    # because the coolant is on the -r side there (see Ann_HT's docstring).
    C1, C2 = ht.Ann_HT(P['r_i'],P['r_o'],q3,Theta(c['Tfo_i']),Theta(c['Tfo_o']))
    qp_i_ht = -ht.Ann_qpp(P['r_i'],q3,C1)*2*np.pi*P['r_i']

    l1 = (qp_i + qp_o - qp(z))/q0
    l2 = (qp_i - qp_i_ht)/q0

    c.update(P=P, g=g, z=z, h_i=h_i, h_o=h_o, qp_i=qp_i, qp_o=qp_o,
             q3=q3, C1=C1, C2=C2, l1=l1, l2=l2)
    return c

def loss(x):
    s = state(x)
    return torch.mean(s['l1']**2), torch.mean(s['l2']**2)

def make_batch(nc,nz,generator=None,params=None):
    """(nc*nz, Nin) collocation tensor: nz axial points for each of nc cases."""
    if params is None:
        params = sample_params(nc,generator)
        if FORCE_NOMINAL:
            params[0:1] = nominal()
    zs = torch.linspace(-1.0,1.0,nz,device=device).reshape(1,-1,1).expand(nc,nz,1)
    Pz = params.reshape(nc,1,-1).expand(nc,nz,params.shape[1])
    x = torch.cat([zs,Pz],2).reshape(nc*nz,-1)
    return x.detach().requires_grad_(True)

# ------------------------------------------------------------- training

if __name__ == "__main__":
    os.makedirs(outdir,exist_ok=True)
    gen = torch.Generator(device=device).manual_seed(0)
    x_val = make_batch(64,Nz,torch.Generator(device=device).manual_seed(12345))
    opt = Soap(model.parameters(),lr=lr)

    hist = []
    for it in range(Nit):
        x = make_batch(Nc,Nz,gen)
        opt.zero_grad()
        L1, L2 = loss(x)
        Ls = L1 + L2
        Ls.backward()
        opt.step()
        hist.append([Ls.item(),L1.item(),L2.item()])
        if it % 250 == 0 or it == Nit-1:
            V1, V2 = loss(x_val)
            print(f'{it:6d}  loss {Ls.item():.3e}  energy {L1.item():.3e}  '
                  f'cond {L2.item():.3e}  |  val {V1.item()+V2.item():.3e}')
    hist = np.array(hist)

    torch.save({'state_dict': model.state_dict(),
                'params': PARAMS, 'range': RANGE, 'nominal': NOM,
                'arch': dict(Nin=Nin,Nout=Nout,w=w),
                'config': dict(L=L,q0=q0,p=p,T_in=T_in,GAS=GAS,
                               EPS_F=EPS_F,EPS_C=EPS_C,N_GAP=N_GAP),
                'hist': hist},
               f'{outdir}/model.pt')
    print(f'saved model to {outdir}/model.pt')

    # ------------------------------------------------------ nominal case
    Npts = 201
    x = make_batch(1,Npts,params=nominal())
    s = state(x)
    out = {k: v.detach().squeeze(1).cpu().numpy() for k,v in s.items()
           if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == 1}
    z = out['z']

    qp_tot = out['qp_i'] + out['qp_o']
    print(f"inner power fraction = {np.trapezoid(out['qp_i'],z)/np.trapezoid(qp_tot,z):.3f}")
    print(f"energy residual  max |l1| = {np.abs(out['l1']).max():.2e}")
    print(f"conduction resid max |l2| = {np.abs(out['l2']).max():.2e}")
    print(f"h = {min(out['h_i'].min(),out['h_o'].min())/1E3:.1f} to "
          f"{max(out['h_i'].max(),out['h_o'].max())/1E3:.1f} kJ/kg "
          f"(table {h_tab[0].item()/1E3:.1f} to {h_tab[-1].item()/1E3:.1f})")
    print(f"peak Tfo_i = {out['Tfo_i'].max():.1f} K   peak Tfo_o = {out['Tfo_o'].max():.1f} K")
    print(f"gap dT inner = {(out['Tfo_i']-out['Tci_OD']).max():.1f} K   "
          f"outer = {(out['Tfo_o']-out['Tco_ID']).max():.1f} K")
    print(f"clad dT inner = {(out['Tci_OD']-out['Tci_ID']).max():.1f} K   "
          f"outer = {(out['Tco_ID']-out['Tco_OD']).max():.1f} K")

    np.savez(f'{outdir}/profiles_nominal.npz',hist=hist,**out)

    fig, ax = plt.subplots(figsize=(6.5,4.2))
    ax.semilogy(hist[:,0],label='total')
    ax.semilogy(hist[:,1],label='energy')
    ax.semilogy(hist[:,2],label='conduction')
    ax.set_xlabel('iteration'); ax.set_ylabel('MSE'); ax.legend()
    plt.tight_layout()
    plt.savefig(f'{outdir}/loss.png',dpi=150)

    fig, ax = plt.subplots(1,4,figsize=(19,4.2))
    ax[0].plot(z,out['h_i']/1E3,label='inner')
    ax[0].plot(z,out['h_o']/1E3,label='outer')
    ax[0].set_xlabel('z [m]'); ax[0].set_ylabel('h [kJ/kg]'); ax[0].legend()

    ax[1].plot(z,out['qp_i']/1E3,label="q'_i")
    ax[1].plot(z,out['qp_o']/1E3,label="q'_o")
    ax[1].plot(z,qp_tot/1E3,'--',label="q'_i+q'_o")
    ax[1].plot(z,qp(torch.as_tensor(z)).numpy()/1E3,':k',label="q'(z)")
    ax[1].set_xlabel('z [m]'); ax[1].set_ylabel("q' [kW/m]"); ax[1].legend()

    for key,lab in [('Tb_i','Tb inner'),('Tci_ID','clad_i ID'),('Tci_OD','clad_i OD'),
                    ('Tfo_i','fuel inner surf')]:
        ax[2].plot(z,out[key],label=lab)
    for key,lab in [('Tb_o','Tb outer'),('Tco_OD','clad_o OD'),('Tco_ID','clad_o ID'),
                    ('Tfo_o','fuel outer surf')]:
        ax[2].plot(z,out[key],'--',label=lab)
    ax[2].set_xlabel('z [m]'); ax[2].set_ylabel('T [K]'); ax[2].legend(fontsize=7,ncol=2)

    ax[3].plot(z,out['hgap_i']/1E3,label='gap inner')
    ax[3].plot(z,out['hgap_o']/1E3,label='gap outer')
    ax[3].plot(z,out['htc_i']/1E3,label='conv inner')
    ax[3].plot(z,out['htc_o']/1E3,label='conv outer')
    ax[3].set_xlabel('z [m]'); ax[3].set_ylabel('htc [kW/m^2-K]'); ax[3].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{outdir}/profiles_nominal.png',dpi=150)
    print(f'saved to {outdir}/')
    plt.show()

# ------------------------------------------------------------ reload API

def load_model(path=f'{outdir}/model.pt'):
    """Rebuild the trained network from a checkpoint written above."""
    ck = torch.load(path,map_location=device,weights_only=False)
    m = PINN(torch.tensor([[ck['range'][k][0] for k in ck['params']]],device=device),
             torch.tensor([[ck['range'][k][1] for k in ck['params']]],device=device)).to(device)
    m.load_state_dict(ck['state_dict'])
    m.eval()
    # state()/predict() go through the module-level `model` via H(), so point
    # that at the reloaded weights rather than the freshly-initialized ones.
    global model
    model = m
    return m, ck

def predict(nz=201,**kw):
    """
    Evaluate the trained model for one pin. Any of PARAMS may be given as
    a keyword; anything omitted falls back to NOM. Returns the same dict
    state() builds, as numpy arrays.

    Example:
        predict(mdot_i=0.45, delta_i=1.2E-4, pitch=16.0E-3)
    """
    P = dict(NOM); P.update(kw)
    unknown = set(kw) - set(PARAMS)
    if unknown:
        raise KeyError(f'not network inputs: {sorted(unknown)}')
    par = torch.tensor([[P[k] for k in PARAMS]],device=device)
    s = state(make_batch(1,nz,params=par))
    return {k: v.detach().squeeze(1).cpu().numpy() for k,v in s.items()
            if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == 1}
