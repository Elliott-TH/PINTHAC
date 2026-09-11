"""
Validates IAPWS95 autograd derivatives against an independent finite-difference
reference (docs/brief/PHASE67_BRIEF.md figure 2).

rho_Tp() finds the density at fixed (T,p) with a Newton solve and then detaches the
result (see its own docstring), so a naive torch.autograd.grad(rho_Tp(T,p), T) returns
nothing useful -- the graph the Newton loop built is already gone. examples/
iapws95_autograd.py works around this with the implicit function theorem: rho(T)|_p is
defined by F(rho,T) = p(rho,T) - p_target = 0, so

    drho/dT|_p = -(dF/dT)_rho / (dF/drho)_T

dF/drho is the analytic p_rho() (already differentiable, no solve inside), dF/dT comes
from ordinary autograd through T at the converged (but detached) rho0 -- F ~ 0 there
already, so re-attaching rho to T this way costs nothing in the returned *value* (rho0
itself is unchanged) while giving it the correct gradient.

That reattached rho then flows into h(rho,T) via the ordinary property formulas, so
dh/dT|_p comes out of the same autograd graph, and (by the definition of cp) should equal
cp(T,p) to within solver/floating-point precision -- an identity check, not a second
independent method.

The genuinely independent reference is central finite differences of the *black-box*
rho_Tp(T,p) function itself, evaluated at T+eps/T-eps: a completely different computation
path (two fresh Newton solves, not the analytic IFT expression) that should agree with
the IFT-based autograd derivative if both are actually computing drho/dT|_p correctly.
"""
import os

import numpy as np
import torch

from pinthac.properties.iapws95 import IAPWS95, device

import style

DTYPE = torch.float64
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def ift_drho_dT(T, p):
    """
    drho/dT|_p via the implicit function theorem, reattaching rho_Tp's detached Newton
    solution to T's autograd graph.

    Inputs:
        T : (n,) torch tensor, K, requires_grad=True
        p : (n,) torch tensor, MPa
    Returns:
        rho      : (n,) torch tensor, kg/m^3 -- numerically equal to rho_Tp(T,p)
        drho_dT  : (n,) torch tensor, kg/m^3/K
    """
    rho0 = IAPWS95.rho_Tp(T.detach(), p.detach())
    rho0 = rho0.clone().requires_grad_(True)
    d0 = IAPWS95.helmholtz(rho0, T)
    F = IAPWS95.p(d0, units="MPa") - p
    dF_dT = torch.autograd.grad(F, T, grad_outputs=torch.ones_like(F), create_graph=True)[0]
    dF_drho = IAPWS95.p_rho(d0, units="MPa")   # analytic, no solve inside
    drho_dT = -dF_dT / dF_drho
    return rho0.detach(), drho_dT


def finite_difference_drho_dT(T, p, eps=1e-3):
    """
    Independent reference: central finite difference of the black-box rho_Tp(T,p)
    itself (two fresh Newton solves per point), not the IFT's analytic expression.

    Inputs:
        T   : (n,) torch tensor, K
        p   : (n,) torch tensor, MPa
        eps : finite-difference step, K
    Returns:
        drho_dT : (n,) torch tensor, kg/m^3/K
    """
    rho_plus = IAPWS95.rho_Tp(T + eps, p)
    rho_minus = IAPWS95.rho_Tp(T - eps, p)
    return (rho_plus - rho_minus) / (2 * eps)


def main():
    style.apply()

    p_val = 25.0   # MPa, supercritical -- single-valued rho(T) everywhere on this sweep
    n = 200
    T = torch.linspace(300.0, 800.0, n, dtype=DTYPE, device=device, requires_grad=True)
    p = p_val * torch.ones_like(T)

    rho, drho_dT_autograd = ift_drho_dT(T, p)
    drho_dT_fd = finite_difference_drho_dT(T.detach(), p.detach(), eps=1e-3)

    T_np = T.detach().cpu().numpy()
    autograd_np = drho_dT_autograd.detach().cpu().numpy()
    fd_np = drho_dT_fd.detach().cpu().numpy()
    rel_err = np.abs(autograd_np - fd_np) / np.maximum(np.abs(fd_np), 1e-8)

    print(f"drho/dT|_p at p={p_val} MPa, T in [300,800] K, n={n} points, device={device}")
    print(f"  max |autograd - FD| relative error: {rel_err.max():.3e}")
    print(f"  mean |autograd - FD| relative error: {rel_err.mean():.3e}")

    fig, (ax1, ax2) = style.figure(figsize=(9.0, 4.3), ncols=2)

    ax1.plot(T_np, fd_np, color=style.MUTED, lw=2.5, alpha=0.6,
              label="finite difference (reference)")
    ax1.plot(T_np, autograd_np, color=style.ACCENT, lw=1.2, ls="--",
              label="autograd (implicit function theorem)")
    ax1.set_xlabel("Temperature [K]")
    ax1.set_ylabel(r"$\partial \rho/\partial T|_p$  [kg m$^{-3}$ K$^{-1}$]")
    ax1.legend(frameon=False, fontsize=8, loc="best")

    ax2.semilogy(T_np, np.maximum(rel_err, 1e-16), color=style.ACCENT, lw=1.3)
    ax2.set_xlabel("Temperature [K]")
    ax2.set_ylabel("relative error, autograd vs. finite difference")

    out_path = os.path.join(OUT_DIR, "iapws95-derivative-validation.svg")
    style.finish(fig, out_path)


if __name__ == "__main__":
    main()
