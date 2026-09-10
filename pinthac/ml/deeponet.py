"""
DeepONet-PINN surrogate for the rod SCA (SCA_IAPWS95_Rod.run_SCA_batch()).

SCA_PINN.py's plain MLP-PINN maps (z, *scalar params*) -> outputs, which
works because every training run there shares the *same* q0*cos(pi z/L)
axial power shape -- q0 is just one more scalar input alongside pitch, G,
etc. This module instead has to learn a genuine operator: given an
*arbitrary* axial LHGR curve q(z) (not just a member of the cosine family)
plus the rod's geometry and flow rate, produce the axial temperature
response. That's exactly a DeepONet's job --

    branch net:  [pitch, rco, tc, delta, kc, G, Tscw_in, q(sensor_1..m)]
                 -> a p-dim embedding per run (the geometry/flow/heat-shape
                    "instance" -- everything about a run except *where*
                    along it you're asking)
    trunk net:   z (query location) -> a p-dim embedding (a nonlinear
                    basis over axial position, shared by every run)
    output:      sum_p branch_p * trunk_p + bias, one such combination per
                 output channel (T_i, T_fuel_max)

Physics loss: the coolant energy balance mdot*cp(T_i)*dT_i/dz = q(z), the
same residual SCA_PINN.py checks for its inner channel, evaluated at fresh
random (geometry, shape) collocation draws every step -- reusing
SCA_Rod_DataGen's own sampler/shape-builder so the collocation
distribution matches the training-set distribution exactly, rather than a
second hand-tuned copy of it. No physics term is applied to T_fuel_max: as
in SCA_PINN.py, that quantity only enters the data loss (its own governing
relation runs through the fuel-conduction/gap/convection chain, not a
first-order ODE autograd can check as cheaply).

Run SCA_Rod_DataGen.py first to produce sca_rod_deeponet_dataset.npz.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorch_optimizer.optimizer.soap import SOAP as Soap
# SSBroyden (Urban, Stefanou & Pons 2025) is used only for the optional polish stage
# at the end of training. Its source is not in this repository -- see
# docs/OPEN_QUESTIONS.md Q24 -- so the import is guarded and training falls back to
# SOAP alone when it is absent, rather than making the whole module unimportable.
try:
    from ssbroyden import SSBroyden
    SSBROYDEN_AVAILABLE = True
except ImportError:
    SSBROYDEN_AVAILABLE = False

from pinthac.sca.rod import build_scw_table, make_Property, interp_sensors
from pinthac.ml.datagen import (PARAM_BOUNDS, SCALAR_NAMES, N_SHAPE_MODES,
                              build_shapes, legendre_basis, L_FIXED, PVAL_FIXED)
from pinthac.ml import losses

torch.manual_seed(3472)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
DTYPE = torch.float64

from pinthac.paths import DATA_DIR, data_file

script_dir = DATA_DIR
DATA_FILE = data_file('sca_rod_deeponet_dataset.npz')
WEIGHTS_FILE = data_file('sca_rod_deeponet_best.pth')

GEOM_NAMES = ["pitch", "rco", "tc", "delta", "kc", "G"]
OUT_NAMES = ['T_i', 'T_fuel_max']


# =====================================================================
# Data
# =====================================================================
raw = np.load(DATA_FILE)
branch_geom = raw['branch_geom']            # (Nrun, 6)
branch_Tin = raw['branch_Tin']              # (Nrun,)
branch_q_sensors = raw['branch_q_sensors']  # (Nrun, m)
sensor_z_np = raw['sensor_z']               # (m,)
z_all = raw['z']                            # (Nrun*n_axial, 1)
outputs_all = raw['outputs']                # (Nrun*n_axial, 2)
n_axial = int(raw['n_axial'])
m_sensors = int(raw['m_sensors'])
L = float(raw['L'])
pval = float(raw['pval'])
n_runs = branch_geom.shape[0]

branch_input_all = np.concatenate([branch_geom, branch_Tin[:, None], branch_q_sensors], axis=1)
n_branch_in = branch_input_all.shape[1]   # 6 + 1 + m

rng = np.random.default_rng(0)
run_perm = rng.permutation(n_runs)
n_val_runs = max(1, int(0.15*n_runs))
val_runs, train_runs = run_perm[:n_val_runs], run_perm[n_val_runs:]


def run_row_idx(run_ids):
    return np.concatenate([np.arange(r*n_axial, (r+1)*n_axial) for r in run_ids])


Xb_train_t = torch.tensor(branch_input_all[train_runs], dtype=DTYPE, device=device)
Xb_val_t = torch.tensor(branch_input_all[val_runs], dtype=DTYPE, device=device)
z_train_t = torch.tensor(z_all[run_row_idx(train_runs)], dtype=DTYPE, device=device)
z_val_t = torch.tensor(z_all[run_row_idx(val_runs)], dtype=DTYPE, device=device)
Y_train_t = torch.tensor(outputs_all[run_row_idx(train_runs)], dtype=DTYPE, device=device)
Y_val_t = torch.tensor(outputs_all[run_row_idx(val_runs)], dtype=DTYPE, device=device)

Xb_mean = Xb_train_t.mean(dim=0, keepdim=True)
Xb_std = Xb_train_t.std(dim=0, keepdim=True) + 1e-12
z_mean = z_train_t.mean(dim=0, keepdim=True)
z_std = z_train_t.std(dim=0, keepdim=True) + 1e-12
Y_mean = Y_train_t.mean(dim=0, keepdim=True)
Y_std = Y_train_t.std(dim=0, keepdim=True) + 1e-12

Xb_train_n = (Xb_train_t - Xb_mean) / Xb_std
Xb_val_n = (Xb_val_t - Xb_mean) / Xb_std
z_train_n = (z_train_t - z_mean) / z_std
z_val_n = (z_val_t - z_mean) / z_std
Y_train_n = (Y_train_t - Y_mean) / Y_std
Y_val_n = (Y_val_t - Y_mean) / Y_std

# row -> local-run-index maps, so a mini-batch of rows can look up its
# branch embedding by indexing the small (n_runs,28) table instead of ever
# materializing a (Nrow, n_out, p) tensor: with n_runs*n_axial potentially
# in the millions of rows, doing this batchwise (below) rather than
# full-batch (as SCA_PINN.py does, fine there since its whole dataset is
# only ~4e5 rows) is what keeps memory bounded regardless of dataset size.
row_run_train = torch.repeat_interleave(torch.arange(len(train_runs), device=device), n_axial)
row_run_val = torch.repeat_interleave(torch.arange(len(val_runs), device=device), n_axial)

# Fixed subset for periodic val-loss checks during training: cheap and
# stable (same rows every check), vs. re-sampling a fresh noisy val batch
# each time or eval-ing the full (possibly huge) val set every 25 epochs.
_val_check_n = min(20000, z_val_n.shape[0])
_val_check_idx = torch.randperm(z_val_n.shape[0], device=device)[:_val_check_n]


# =====================================================================
# SCW cp(T) table at the dataset's fixed pval, used inside the physics
# residual exactly the way SCA_IAPWS95_Rod.py's own Property() is used in
# run_SCA_batch -- a torch.searchsorted lookup, differentiable end to end.
# =====================================================================
scw_table = build_scw_table(pval, device=device)
Property = make_Property(scw_table)
sensor_z = torch.tensor(sensor_z_np, dtype=DTYPE, device=device)


# =====================================================================
# Model: branch net (geometry+flow+LHGR-shape -> p-dim embedding per
# output channel) and trunk net (z -> p-dim embedding), combined by an
# inner product plus a learned bias -- the standard DeepONet head.
# =====================================================================
class DeepONet(nn.Module):
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
        """branch_out: (batch, n_out, p), trunk_out: (batch, p) -- already
        row-aligned (one branch embedding per row, pre-expanded by the
        caller). Returns (batch, n_out) normalized predictions."""
        return (branch_out * trunk_out.unsqueeze(1)).sum(dim=-1) + self.out_bias

    def predict(self, branch_phys, z_phys):
        """branch_phys: (batch, n_branch_in), z_phys: (batch,1), both
        already row-aligned and in physical units. Returns (batch, n_out)
        in physical units."""
        xb_n = (branch_phys - Xb_mean) / Xb_std
        z_n = (z_phys - z_mean) / z_std
        y_n = self.combine(self.branch_forward(xb_n), self.trunk_forward(z_n))
        return y_n * Y_std + Y_mean


model = DeepONet(n_branch_in).to(device).to(DTYPE)


# =====================================================================
# Losses. Both draw a mini-batch of *rows* (not runs) each call -- one
# branch_forward per sampled row, looked up from the small per-run
# Xb_*_n table by row_run_train/row_run_val, rather than ever expanding
# branch embeddings out to the full (possibly multi-million-row) dataset.
# =====================================================================
def data_loss(batch_size=8192):
    idx = torch.randint(0, z_train_n.shape[0], (batch_size,), device=device)
    run_idx = row_run_train[idx]
    branch_out = model.branch_forward(Xb_train_n[run_idx])   # (batch, n_out, p)
    trunk_out = model.trunk_forward(z_train_n[idx])           # (batch, p)
    pred = model.combine(branch_out, trunk_out)
    return torch.mean((pred - Y_train_n[idx])**2)


@torch.no_grad()
def val_loss_fn():
    run_idx = row_run_val[_val_check_idx]
    branch_out = model.branch_forward(Xb_val_n[run_idx])
    trunk_out = model.trunk_forward(z_val_n[_val_check_idx])
    pred = model.combine(branch_out, trunk_out)
    return torch.mean((pred - Y_val_n[_val_check_idx])**2).item()


def sample_collocation(n):
    """Fresh random (geometry, LHGR shape, z) collocation draws, from the
    exact same distribution SCA_Rod_DataGen.py used to build the training
    set -- so the physics term regularizes the same input space the data
    loss covers, not some separately-tuned box."""
    lo = np.array([PARAM_BOUNDS[k][0] for k in SCALAR_NAMES])
    hi = np.array([PARAM_BOUNDS[k][1] for k in SCALAR_NAMES])
    scalars = lo + np.random.rand(n, len(SCALAR_NAMES))*(hi - lo)
    coeffs = np.random.uniform(-1.0, 1.0, size=(n, N_SHAPE_MODES))
    shapes = build_shapes(coeffs, sensor_z_np / (L/2))     # (n, m)
    q0 = scalars[:, SCALAR_NAMES.index("q0")]
    q_sensors = q0[:, None] * shapes

    tc = scalars[:, SCALAR_NAMES.index("tc_frac")] * scalars[:, SCALAR_NAMES.index("rco")]
    delta = scalars[:, SCALAR_NAMES.index("delta_frac")] * scalars[:, SCALAR_NAMES.index("rco")]
    pitch = scalars[:, SCALAR_NAMES.index("pitch_ratio")] * (2*scalars[:, SCALAR_NAMES.index("rco")])
    rco = scalars[:, SCALAR_NAMES.index("rco")]
    kc = scalars[:, SCALAR_NAMES.index("kc")]
    G = scalars[:, SCALAR_NAMES.index("G")]
    Tin = scalars[:, SCALAR_NAMES.index("Tscw_in")]

    geom = np.stack([pitch, rco, tc, delta, kc, G], axis=1)
    branch_phys = np.concatenate([geom, Tin[:, None], q_sensors], axis=1)

    branch_t = torch.tensor(branch_phys, dtype=DTYPE, device=device)
    q_sensors_t = torch.tensor(q_sensors, dtype=DTYPE, device=device)
    pitch_t = torch.tensor(pitch, dtype=DTYPE, device=device)
    rco_t = torch.tensor(rco, dtype=DTYPE, device=device)
    G_t = torch.tensor(G, dtype=DTYPE, device=device)

    z_c = (torch.rand(n, 1, dtype=DTYPE, device=device) - 0.5)*L
    z_c.requires_grad_(True)
    return branch_t, z_c, q_sensors_t, pitch_t, rco_t, G_t


def physics_loss(n_colloc=2048):
    branch_phys, z_c, q_sensors_t, pitch_t, rco_t, G_t = sample_collocation(n_colloc)

    xb_n = (branch_phys - Xb_mean) / Xb_std
    z_n = (z_c - z_mean) / z_std
    branch_out = model.branch_forward(xb_n)
    trunk_out = model.trunk_forward(z_n)
    y_n = model.combine(branch_out, trunk_out)
    y = y_n * Y_std + Y_mean
    Ti = y[:, 0:1]

    dTi_dz = torch.autograd.grad(Ti, z_c, grad_outputs=torch.ones_like(Ti), create_graph=True)[0]

    mdot = G_t * (pitch_t**2 - np.pi*rco_t**2)
    q_z = interp_sensors(q_sensors_t, sensor_z, z_c.detach().reshape(-1)).reshape(-1, 1)

    # cp looked up at the network's current T_i but not backpropped through
    # (a frozen coefficient, same role cp_i plays in SCA_PINN.py's
    # physics_loss) -- only dTi_dz needs to come from the network here.
    cp_i = Property(['T', Ti.detach().reshape(-1)], 'cp').reshape(-1, 1)

    res = losses.coolant_energy_residual_T(mdot.reshape(-1, 1), cp_i, dTi_dz, q_z)
    scale = torch.clamp(q_z.detach().abs(), min=1.0)
    return losses.normalized_residual_loss(res, scale)


@torch.no_grad()
def relative_errors(run_ids):
    idx = run_row_idx(run_ids)
    branch_out = model.branch_forward((torch.tensor(branch_input_all[run_ids], dtype=DTYPE, device=device) - Xb_mean)/Xb_std)
    branch_out = branch_out.repeat_interleave(n_axial, dim=0)
    z_phys = torch.tensor(z_all[idx], dtype=DTYPE, device=device)
    trunk_out = model.trunk_forward((z_phys - z_mean)/z_std)
    pred_n = model.combine(branch_out, trunk_out)
    pred = (pred_n*Y_std + Y_mean).cpu().numpy()
    true = outputs_all[idx]
    rel = np.abs(pred - true) / (np.abs(true) + 1e-6)
    return {name: (rel[:, i].mean(), rel[:, i].max()) for i, name in enumerate(OUT_NAMES)}


# =====================================================================
# Training: SOAP (mini-batch, noisy objective) gets most of the way
# there, same structure as SCA_PINN.py; SSBroyden -- Urban, Stefanou &
# Pons (2025), see ssbroyden.py -- then polishes the last stretch with
# second-order steps and a strong-Wolfe line search, the way that paper
# refines PINNs/DeepONets past SOAP/Adam. The loss-history plot from an
# earlier run of this script (sca_rod_deeponet_loss.png) plateaus in val
# loss well before its 200k SOAP epochs finish -- occasional raw-loss
# spikes after that from rare bad physics-residual collocation draws
# (clipped before they reach the optimizer, so harmless to the fit) are
# the only thing still moving -- hence cutting SOAP off around 25k and
# handing the rest to a method suited to fine convergence instead of
# grinding out another 175k noisy-batch epochs for no further gain.
# =====================================================================
if __name__ == '__main__':
    soap_epochs = 25000
    phys_warmup_epochs = 2000
    w_data, w_phys_max = 1.0, 0.05

    optimizer = Soap(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=300)

    loss_history = []
    best_val = float('inf')

    for epoch in range(soap_epochs + 1):
        optimizer.zero_grad()

        l_data = data_loss()
        w_phys = w_phys_max * min(1.0, epoch/phys_warmup_epochs)
        l_phys = physics_loss() if w_phys > 0 else torch.zeros((), device=device)
        loss = w_data*l_data + w_phys*l_phys

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        loss_history.append(loss.item())

        if epoch % 25 == 0:
            val_loss = val_loss_fn()
            scheduler.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), WEIGHTS_FILE)
            if epoch % 200 == 0:
                print(f"  epoch {epoch:5d}  loss {loss.item():.4e}  "
                      f"data {l_data.item():.4e}  phys {l_phys.item():.4e}  "
                      f"val {val_loss:.4e}  lr {optimizer.param_groups[0]['lr']:.1e}")

    # =================================================================
    # Polish: SSBroyden's curvature pairs assume a *fixed* objective
    # across the several closure() calls its line search makes per step,
    # so this draws one large batch (data rows + physics collocation)
    # once and holds it fixed for the whole polish phase, rather than
    # SOAP's fresh mini-batch every step -- the same full-batch
    # determinism SCA_PINN.py's own (much smaller) dataset gets for free,
    # capped here since the rod dataset's ~8e6 rows are too many to also
    # carry a physics-residual autograd graph through every line-search
    # evaluation.
    # =================================================================
    polish_iters = 3000
    polish_batch = 200_000
    polish_colloc = 8192

    idx_p = torch.randint(0, z_train_n.shape[0], (polish_batch,), device=device)
    run_idx_p = row_run_train[idx_p]
    branch_p, z_p, Y_p = Xb_train_n[run_idx_p], z_train_n[idx_p], Y_train_n[idx_p]

    branch_phys_c, z_c, q_sensors_c, pitch_c, rco_c, G_c = sample_collocation(polish_colloc)

    if not SSBROYDEN_AVAILABLE:
        # The import at the top of this module is already guarded, but the use site was
        # not -- so a full training run completed its SOAP phase and then died here.
        # Nothing is lost when this stage is skipped: the best checkpoint is saved during
        # the main loop, and this is a polish pass on an already-converged model.
        print('SSBroyden unavailable (see docs/OPEN_QUESTIONS.md Q24) -- '
              'skipping the polish stage; the SOAP checkpoint is already saved.')
        # SystemExit rather than return: this block is the script's __main__ body, not a
        # function, so `return` is a syntax error here. Exit status 0 because a run that
        # trained and saved successfully has not failed -- the polish pass is optional.
        raise SystemExit(0)
    optimizer2 = SSBroyden(model.parameters(), lr=1.0, history_size=40, method='ssbroyden')

    def polish_closure():
        # xb_n_c/z_n_c recomputed fresh each call (not hoisted) -- z_c is
        # reused across every line-search evaluation this whole polish
        # phase, and hoisting these would make every one of those calls
        # share one fixed (z_c -> z_n_c) autograd node, which the first
        # backward() frees and every later call then fails to re-traverse.
        optimizer2.zero_grad()
        pred = model.combine(model.branch_forward(branch_p), model.trunk_forward(z_p))
        l_data = torch.mean((pred - Y_p)**2)

        xb_n_c = (branch_phys_c - Xb_mean) / Xb_std
        z_n_c = (z_c - z_mean) / z_std
        y_n = model.combine(model.branch_forward(xb_n_c), model.trunk_forward(z_n_c))
        y = y_n * Y_std + Y_mean
        Ti = y[:, 0:1]
        dTi_dz = torch.autograd.grad(Ti, z_c, grad_outputs=torch.ones_like(Ti), create_graph=True)[0]
        mdot = G_c * (pitch_c**2 - np.pi*rco_c**2)
        q_z = interp_sensors(q_sensors_c, sensor_z, z_c.detach().reshape(-1)).reshape(-1, 1)
        cp_i = Property(['T', Ti.detach().reshape(-1)], 'cp').reshape(-1, 1)
        res = losses.coolant_energy_residual_T(mdot.reshape(-1, 1), cp_i, dTi_dz, q_z)
        scale = torch.clamp(q_z.detach().abs(), min=1.0)
        l_phys = losses.normalized_residual_loss(res, scale)

        loss = w_data*l_data + w_phys_max*l_phys
        loss.backward()
        return loss

    for it in range(polish_iters + 1):
        loss = optimizer2.step(polish_closure).item()
        loss_history.append(loss)
        if it % 10 == 0:
            val_loss = val_loss_fn()
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), WEIGHTS_FILE)
            print(f"  polish {it:5d}  loss {loss:.6e}  val {val_loss:.6e}")

    # Eval/plotting on CPU, same as SCA_PINN.py -- it's a few thousand
    # points through a small net either way, so there's no real cost to
    # not gambling on the GPU context still being alive by the time this
    # runs, and it sidesteps the batch-size-vs-VRAM tradeoff entirely for
    # this part. model.predict()/relative_errors() close over Xb_mean/
    # Xb_std/z_mean/z_std/Y_mean/Y_std, so those move too.
    device = torch.device('cpu')
    model.to(device)
    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device, weights_only=True))
    model.eval()
    Xb_mean, Xb_std = Xb_mean.to(device), Xb_std.to(device)
    z_mean, z_std = z_mean.to(device), z_std.to(device)
    Y_mean, Y_std = Y_mean.to(device), Y_std.to(device)

    # Cap how many val runs the metric touches on CPU -- a bounded random
    # the metric touches -- a bounded random subset is enough to report a
    # stable mean/max relative error without walking every held-out run.
    metric_runs = val_runs if len(val_runs) <= 2000 else rng.choice(val_runs, 2000, replace=False)
    metrics = relative_errors(metric_runs)
    print("\nValidation relative error (held-out geometry/flow/LHGR-shape combos):")
    for name, (mean_e, max_e) in metrics.items():
        print(f"  {name:12s} mean {mean_e:.3e}  max {max_e:.3e}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.semilogy(loss_history)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Total loss'); ax.set_title('Training history')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'sca_rod_deeponet_loss.png'), dpi=150)

    n_show = min(4, len(val_runs))
    fig, axes = plt.subplots(n_show, 2, figsize=(11, 3.3*n_show), squeeze=False)
    with torch.no_grad():
        for row, r in enumerate(val_runs[:n_show]):
            idx = np.arange(r*n_axial, (r+1)*n_axial)
            branch_phys = torch.tensor(np.tile(branch_input_all[r], (n_axial, 1)), dtype=DTYPE, device=device)
            z_phys = torch.tensor(z_all[idx], dtype=DTYPE, device=device)
            pred = model.predict(branch_phys, z_phys).numpy()
            zz = z_all[idx, 0]

            axes[row, 0].plot(zz, outputs_all[idx, 0], 'k-', label='SCA T_i')
            axes[row, 0].plot(zz, pred[:, 0], 'r--', label='DeepONet T_i')
            axes[row, 0].set_ylabel('T [K]'); axes[row, 0].legend(fontsize=7)

            axes[row, 1].plot(zz, outputs_all[idx, 1], 'k-', label='SCA T_fuel_max')
            axes[row, 1].plot(zz, pred[:, 1], 'r--', label='DeepONet T_fuel_max')
            axes[row, 1].set_ylabel('T_fuel_max [K]'); axes[row, 1].legend(fontsize=7)

    for a in axes[-1, :]:
        a.set_xlabel('z [m]')
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'sca_rod_deeponet_profiles.png'), dpi=150)
    print("Saved sca_rod_deeponet_loss.png and sca_rod_deeponet_profiles.png")