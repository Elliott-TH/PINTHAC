"""Batched scalar root finding, shared by every layer that has an implicit correlation.

Why this module exists: several models in this library are implicit and each of them
needs the same thing -- solve one scalar equation per batch element, on GPU, without
branching, and keep the answer differentiable. Swenson's wall temperature, Chen and
Bjorge's, Colebrook's friction factor, and the inversion of a conductivity integral back
to a temperature are all that same shape.

The `torchsolve` package alongside this repository does the same job with more care
(guaranteed bracketing, typed failure results, extremum search for non-monotone
residuals) and is what `correlations/htc.py` uses for the supercritical wall-temperature
solves, where the residual really is non-monotone. This module is the plain version for
the many cases where the residual is monotone and a fixed iteration count is enough.
"""
import torch

from pinthac import backend


def bisect_newton(residual, lo, hi, deriv=None, bisect_iters=40, newton_iters=6,
                  differentiable=True):
    """Solve residual(x) = 0 elementwise, by bisection then Newton polish.

    Formulation:
        Bisection maintains a sign-change bracket [lo, hi]:
            keep_lo = residual(mid)*residual(lo) > 0
            lo, hi  = where(keep_lo, mid, lo), where(keep_lo, hi, mid)
        then Newton refines:
            x <- x - residual(x)/residual'(x)

        The Newton phase runs on a detached x, so the iteration itself never builds a
        graph. That is deliberate: differentiating through the unrolled iterates would
        be both expensive and wrong-headed, since the converged root is a well-defined
        implicit function of the inputs regardless of how many steps were taken to find
        it.

        With `differentiable=True` the gradient is reattached afterwards by one final
        Newton step whose derivative is detached:

            x_out = x - residual(x) / residual'(x).detach()

        At convergence residual(x) is ~0, so the *value* is unchanged to rounding, while
        the derivative of that expression with respect to anything the residual closes
        over is exactly what the implicit function theorem gives:
        dx/dp = -(dF/dp)/(dF/dx). One extra residual evaluation buys a correct gradient
        through a solve that is otherwise opaque to autograd.

    Inputs:
        residual   : callable x -> F(x), elementwise over the batch
        lo, hi     : bracket endpoints, broadcastable; must straddle the root
        deriv      : optional callable x -> dF/dx. When None, autograd supplies it.
                     Supplying it saves a graph build per Newton step, and several
                     callers here have it in closed form.
        bisect_iters : bisection steps. 40 halvings shrink any bracket by 1e-12.
        newton_iters : Newton steps after bisection.
        differentiable : reattach the gradient as described above. Set False when the
                     caller is in plain numpy space and does not need it.

    Returns:
        x : the root, same shape as the broadcast bracket. A torch tensor.
    """
    lo_ = torch.as_tensor(lo, dtype=torch.float64)
    hi_ = torch.as_tensor(hi, dtype=torch.float64)
    lo_, hi_ = torch.broadcast_tensors(lo_, hi_)
    lo_, hi_ = lo_.clone(), hi_.clone()

    with torch.no_grad():
        f_lo = residual(lo_)
        for _ in range(bisect_iters):
            mid = 0.5 * (lo_ + hi_)
            f_mid = residual(mid)
            keep_lo = (f_mid * f_lo) > 0
            lo_ = torch.where(keep_lo, mid, lo_)
            f_lo = torch.where(keep_lo, f_mid, f_lo)
            hi_ = torch.where(keep_lo, hi_, mid)
        x = 0.5 * (lo_ + hi_)

        for _ in range(newton_iters):
            if deriv is None:
                with torch.enable_grad():
                    xg = x.detach().requires_grad_(True)
                    F = residual(xg)
                    # grad_outputs=ones_like rather than the implicit scalar default:
                    # the residual is elementwise, so summing before differentiating
                    # still gives the right per-element dF/dx with no cross-batch leak.
                    dF, = torch.autograd.grad(F, xg, grad_outputs=torch.ones_like(F))
                F = F.detach()
            else:
                F = residual(x)
                dF = deriv(x)
            x = x - F / dF

    if not differentiable:
        return x

    x = x.detach()
    F = residual(x)
    if deriv is None:
        xg = x.detach().requires_grad_(True)
        with torch.enable_grad():
            dF, = torch.autograd.grad(residual(xg), xg,
                                      grad_outputs=torch.ones_like(x))
    else:
        dF = deriv(x)
    return x - F / dF.detach()
