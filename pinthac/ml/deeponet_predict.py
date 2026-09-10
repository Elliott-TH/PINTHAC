"""
Pure-inference wrapper around the trained rod DeepONet-PINN
(SCA_PINN_Rod_DeepONet.py's model + sca_rod_deeponet_best.pth) -- the
surrogate counterpart to SCA_IAPWS95_Rod.run_SCA_batch(), deliberately
given the same calling convention (inputs_b dict + Tscw_in_b +
q_sensors_b) so the two are interchangeable in scripts that compare or
benchmark one against the other.

Only the architecture and input/output normalization are duplicated
here rather than imported from SCA_PINN_Rod_DeepONet.py -- importing
that module directly would also execute its full training-data load and
SOAP/SSBroyden setup as an import side effect, which a plain "give me a
prediction" caller shouldn't have to pay for. The normalization stats
(Xb_mean/std, z_mean/std, Y_mean/std) aren't stored in the .pth
checkpoint, so _load() recomputes them from the training split the
first time predict_rod() runs -- deterministic (fixed dataset + fixed
split seed), so this reproduces the exact stats the checkpoint was
trained under, it just needs the dataset file present alongside it.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from pinthac.paths import DATA_DIR, data_file

script_dir = DATA_DIR
DATA_FILE = data_file('sca_rod_deeponet_dataset.npz')
WEIGHTS_FILE = data_file('sca_rod_deeponet_best.pth')
DTYPE = torch.float64

GEOM_KEYS = ["pitch", "rco", "tc", "delta", "kc", "G"]
OUT_NAMES = ["T_i", "T_fuel_max"]


class DeepONet(nn.Module):
    """Identical architecture to SCA_PINN_Rod_DeepONet.DeepONet -- kept in
    sync by hand since it has to match the checkpoint's state_dict."""
    def __init__(self, n_branch_in, n_out=2, p=128, width=192):
        super().__init__()
        self.n_out, self.p = n_out, p
        self.branch = nn.Sequential(
            nn.Linear(n_branch_in, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, n_out*p),
        )
        self.trunk = nn.Sequential(
            nn.Linear(1, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, p), nn.Tanh(),
        )
        self.out_bias = nn.Parameter(torch.zeros(n_out, dtype=DTYPE))

    def branch_forward(self, xb_norm):
        return self.branch(xb_norm).view(-1, self.n_out, self.p)

    def trunk_forward(self, z_norm):
        return self.trunk(z_norm)

    def combine(self, branch_out, trunk_out):
        return (branch_out * trunk_out.unsqueeze(1)).sum(dim=-1) + self.out_bias


_cache = {}   # device -> (model, norm stats dict, sensor_z tensor, L)


def _load(device):
    if device in _cache:
        return _cache[device]

    raw = np.load(DATA_FILE)
    branch_geom = raw['branch_geom']
    branch_Tin = raw['branch_Tin']
    branch_q_sensors = raw['branch_q_sensors']
    sensor_z_np = raw['sensor_z']
    z_all = raw['z']
    outputs_all = raw['outputs']
    n_axial = int(raw['n_axial'])
    n_runs = branch_geom.shape[0]
    L = float(raw['L'])

    branch_input_all = np.concatenate([branch_geom, branch_Tin[:, None], branch_q_sensors], axis=1)
    n_branch_in = branch_input_all.shape[1]

    # Same split as SCA_PINN_Rod_DeepONet.py: fixed seed => the *train*
    # subset, and therefore these normalization stats, are bit-for-bit
    # what the checkpoint was actually trained against.
    rng = np.random.default_rng(0)
    run_perm = rng.permutation(n_runs)
    n_val_runs = max(1, int(0.15 * n_runs))
    train_runs = run_perm[n_val_runs:]
    row_idx = np.concatenate([np.arange(r*n_axial, (r+1)*n_axial) for r in train_runs])

    Xb_train = torch.tensor(branch_input_all[train_runs], dtype=DTYPE, device=device)
    z_train = torch.tensor(z_all[row_idx], dtype=DTYPE, device=device)
    Y_train = torch.tensor(outputs_all[row_idx], dtype=DTYPE, device=device)

    norm = dict(
        Xb_mean=Xb_train.mean(dim=0, keepdim=True), Xb_std=Xb_train.std(dim=0, keepdim=True) + 1e-12,
        z_mean=z_train.mean(dim=0, keepdim=True), z_std=z_train.std(dim=0, keepdim=True) + 1e-12,
        Y_mean=Y_train.mean(dim=0, keepdim=True), Y_std=Y_train.std(dim=0, keepdim=True) + 1e-12,
    )

    model = DeepONet(n_branch_in).to(device).to(DTYPE)
    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device, weights_only=True))
    model.eval()

    sensor_z = torch.tensor(sensor_z_np, dtype=DTYPE, device=device)
    _cache[device] = (model, norm, sensor_z, L)
    return _cache[device]


def sensor_grid(device=None):
    """The m=21 axial locations [m] this model's q_sensors_b columns must
    line up with (see SCA_Rod_DataGen.N_SENSORS/L_FIXED)."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, sensor_z, _ = _load(device)
    return sensor_z.cpu().numpy()


@torch.no_grad()
def predict_rod(inputs_b, Tscw_in_b, q_sensors_b, z_q, device=None):
    """DeepONet surrogate for SCA_IAPWS95_Rod.run_SCA_batch(): same
    inputs_b/Tscw_in_b/q_sensors_b convention, but z_q is an explicit query
    grid (run_SCA_batch instead marches its own n evenly-spaced nodes),
    since a trained trunk net can be queried anywhere in [-L/2, L/2], not
    just wherever the ground-truth axial march happened to step.

    inputs_b     : dict of (B,) array-likes -- pitch, rco, tc, delta, kc, G
    Tscw_in_b    : (B,) array-like, SCW inlet temperature [K]
    q_sensors_b  : (B, m) array-like, axial LHGR [W/m] sampled at
                   sensor_grid() (m=21 sensors over [-L/2, L/2], L=3m)
    z_q          : (n,) array-like of axial query positions [m], shared
                   across every row in the batch
    Returns dict with 'Z' (n,) and 'T_i', 'T_fuel_max' each (B, n) numpy
    arrays -- deliberately the same key names as run_SCA_batch's return.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, norm, _, _ = _load(device)

    geom = np.stack([np.asarray(inputs_b[k], dtype=np.float64).reshape(-1) for k in GEOM_KEYS], axis=1)
    Tin = np.asarray(Tscw_in_b, dtype=np.float64).reshape(-1)
    q_sensors = np.asarray(q_sensors_b, dtype=np.float64)
    B = geom.shape[0]
    z_q = np.asarray(z_q, dtype=np.float64).reshape(-1)
    n = z_q.shape[0]

    branch_phys = torch.tensor(np.concatenate([geom, Tin[:, None], q_sensors], axis=1), dtype=DTYPE, device=device)
    branch_phys = branch_phys.repeat_interleave(n, dim=0)                             # (B*n, n_branch_in)
    z_phys = torch.tensor(np.tile(z_q, B), dtype=DTYPE, device=device).reshape(-1, 1)  # (B*n, 1)

    xb_n = (branch_phys - norm['Xb_mean']) / norm['Xb_std']
    z_n = (z_phys - norm['z_mean']) / norm['z_std']
    y_n = model.combine(model.branch_forward(xb_n), model.trunk_forward(z_n))
    y = (y_n * norm['Y_std'] + norm['Y_mean']).cpu().numpy().reshape(B, n, 2)

    return {'Z': z_q, 'T_i': y[:, :, 0], 'T_fuel_max': y[:, :, 1]}


if __name__ == '__main__':
    # Smoke-test / usage example: one rod, the geometry from
    # SCA_IAPWS95_Rod.py's own __main__ sanity check, a flat-topped LHGR
    # shape, queried on a finer grid than the training set ever used.
    inputs_b = {"pitch": [0.0112], "rco": [0.00071+0.00878/2], "tc": [0.00071],
                "delta": [5e-4], "kc": [21.5], "G": [1500.0]}
    Tscw_in_b = [573.0]
    sz = sensor_grid()
    q_sensors_b = [45000.0 * np.cos(np.pi * sz / (3.0+1.5))]
    z_q = np.linspace(-1.5, 1.5, 300)

    out = predict_rod(inputs_b, Tscw_in_b, q_sensors_b, z_q)
    print(f"T_i range: {out['T_i'].min():.1f}-{out['T_i'].max():.1f} K")
    print(f"T_fuel_max range: {out['T_fuel_max'].min():.1f}-{out['T_fuel_max'].max():.1f} K")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(out['Z'], out['T_i'][0], 'b-')
    ax.set_xlabel('z [m]')
    ax.set_ylabel('Coolant temperature $T_i$ [K]')
    ax.set_title('Coolant temperature vs. axial position')
    fig.tight_layout()
    out_path = os.path.join(script_dir, 'sca_rod_deeponet_predict_coolant_temp.png')
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
