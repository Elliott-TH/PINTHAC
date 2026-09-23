"""Shared coolant-film interface for rod and annular channel solvers."""
import numpy as np
import torch

from pinthac.correlations import htc

EXPLICIT = {
    'dittus': lambda p, G, D, pitch, T: htc.Water.Dittus(p, G, D),
    'petukhov': lambda p, G, D, pitch, T: htc.Water.Petchukov(p, G, D),
    'gnielinski': lambda p, G, D, pitch, T: htc.Water.Gnielinski(p, G, D),
    'lyon': lambda p, G, D, pitch, T: htc.Sodium.Lyon(p, G, D),
    'seban': lambda p, G, D, pitch, T: htc.Sodium.SebanShimazaki(p, G, D),
    'mikityuk': lambda p, G, D, pitch, T: htc.Sodium.Mikityuk(p, G, D, pitch),
    'lead_shen': lambda p, G, D, pitch, T: htc.Lead.Shen(p, T, G, D),
}


def pseudocritical(props_at, lo=550., hi=750., n=401):
    """Locate the cp peak once per fixed-pressure property lookup."""
    T = np.linspace(lo, hi, n)
    cp = props_at(T)['cp']
    if torch.is_tensor(cp):
        cp = cp.detach().cpu().numpy()
    return float(T[np.argmax(cp)])


def solve(props_at, Tb, G, D, q, name='swenson', psi=1., pitch=None,
          anchor=None, hi=None, lo=None):
    """Return {Tw, htc, residual}; q is flux on the heated surface, W/m².

    The property callable must accept tensors for implicit correlations. Use
    numpy_adapter for a NumPy-only property provider.
    """
    if name in EXPLICIT:
        coefficient = psi * EXPLICIT[name](props_at(Tb), G, D, pitch, Tb)
        return dict(Tw=Tb + q/coefficient, htc=coefficient, residual=q*0)
    if name in ('chen_h2o', 'bjorge', 'schrock_grossman'):
        raise NotImplementedError('single-channel solvers do not support two-phase HTC')
    if name not in ('swenson', 'chen_scw'):
        raise ValueError(f'unknown htc correlation {name!r}; expected swenson, chen_scw or {sorted(EXPLICIT)}')
    function = htc.SCW.Swenson if name == 'swenson' else htc.SCW.Chen_SCW
    return function(props_at(Tb), props_at, G, D, q, Tb, psi=psi,
                    anchor=anchor, hi=hi, lo=lo, return_state=True)


def numpy_adapter(props_at):
    """Adapt a NumPy property provider to the tensor wall solver (no autograd)."""
    def properties(T):
        if not torch.is_tensor(T):
            return props_at(T)
        return {key: torch.as_tensor(value, dtype=T.dtype, device=T.device)
                for key, value in props_at(T.detach().cpu().numpy()).items()}
    return properties
