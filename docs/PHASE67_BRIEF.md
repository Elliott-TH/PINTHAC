# Brief — finish Phase 6, then Phase 7 (figures)

Read `CLAUDE.md` first; it outranks anything here. Then `docs/DECISIONS.md` and
`docs/OPEN_QUESTIONS.md`.

**Environment.** `/home/elliott/Codes/miniconda3/envs/GenEnv3.12/bin/python`.
The GPU is available again: **prefix every GPU run with `HIP_VISIBLE_DEVICES=0`** to pin to
the discrete RX 7800 XT. Device 1 is integrated graphics and benchmarking it would be
meaningless. Confirm `torch.cuda.get_device_name(0)` says "RX 7800 XT" before trusting a
timing.

## Already done, do not redo

- Dataset: `data/sca_rod_deeponet_dataset.npz`, 59,773 converged runs of 60,000, Legendre
  power profiles (6 modes, 21 sensors, 100 axial nodes).
- Trained checkpoint: `data/sca_rod_deeponet_best.pth`, best validation loss 1.069e-4.
  **Trained on CPU** -- retraining on GPU is optional and only worth it if you have time.
- Measured accuracy, 5,978 held-out runs (2,000 evaluated):

      T_i          MAE 0.113 K   mean rel 0.018 %   p99 rel 0.092 %   max  8.84 K
      T_fuel_max   MAE 5.029 K   mean rel 0.356 %   p99 rel 2.033 %   max 115.50 K

- Measured surrogate-vs-solver speed (CPU):

      N=100    surrogate  27.3 ms   FVM 1415.1 ms   51.8x
      N=1000   surrogate 426.6 ms   FVM 2343.3 ms    5.5x

  **The speedup collapses with batch size** because `run_SCA_batch` is itself vectorized.
  Re-measure on GPU. Whatever you find, the figure must not quote a single number without
  its batch size -- see Phase 7 figure 5 below.

## Part A -- finish Phase 6

### A1. `pinthac/ml/losses.py` (new)
Reusable physics residual terms, per the original brief:
- coolant energy balance, `mdot*dh/dz = q'(z)`
- the non-dimensionalized enthalpy equation from
  `docs/reference/Annular_Heat_Transfer_Final.pdf` section 2.1
- positivity constraints on network outputs

Plain functions taking tensors and returning scalar losses. `ml/deeponet.py` and
`ml/pinn.py` already compute their own residuals inline -- factor those out into this module
and have both import them, but **verify the refactor changes no number** before keeping it.

### A2. Fourier power-profile parameterization
`ml/datagen.py` has `legendre_basis`/`build_shapes`. Add the squared-Fourier-with-offset form
from the derivation PDF section 2.1 alongside it, selectable by name:

    S(x) = sum_n [a_n cos(pi n x) + b_n sin(pi n x)]
    Fq(x) = S(x)^2 + phi_q

Requirements the owner stated directly:
- **vary the order**, weighted toward low order (more 2nd than 3rd or 4th)
- **per-mode amplitude decay**, for smoothness and differentiability
- **the profile must never reach exactly zero** -- `phi_q > 0` guarantees this, the same
  role the existing Legendre version's 5 percent floor plays
- use the **analytic** mean `<Fq> = 0.5*sum(a_n^2 + b_n^2) + phi_q` for normalization, not
  numerical integration. Derived in the PDF from orthogonality.

Note the PDF defines `Fq` as a ratio `q'(z)/<q'>`, which requires `<Fq> = 1`, but then writes
`<q'> = <Fq>`. Normalize by the analytic mean so `Fq` really is a ratio, and say so in a
comment.

Test the analytic mean against numerical quadrature of the same profile.

### A3. `ml/pinn.py`
Bring to the `CLAUDE.md` docstring standard. **Do not restructure it** -- it is a working,
trained model (`data/Direct_PINN3_out/model.pt`). Docstrings, the backend contract where it
applies, and importing from `losses.py`. Nothing else.

## Part B -- Phase 7, the figures

The spec is the `<!-- PROMPT FOR CLAUDE CODE -->` comments in
`docs/reference/index.html`. Six SVGs into `figures/output/`, each from a committed script in
`figures/` that runs from a clean checkout.

**Style, matching the site:** background `#0a0d12`, accent `#6fd3f7`, muted grey `#8b93a1`,
light gridlines, **no title baked into the image** (captions live in the HTML), SVG output,
legible at panel width. Build one shared matplotlib style module in `figures/` and use it
everywhere. Look at the rendered output and check contrast and label size.

1. `iapws95-benchmark.svg` -- log-log GPU vs CPU property-lookup runtime, 10^2 to 10^7
   points. `figures/iapws95_benchmark.py` already exists and compares against the pip
   `iapws` package. Run it on the GPU. **Report the number you actually measure.** The CV
   says 36x and the owner recalls ~96x at 1e6 points; if neither is what you get, the
   measurement wins and the caption states it.
2. `iapws95-derivative-validation.svg` -- autograd derivatives vs finite-difference
   reference. `examples/iapws95_autograd.py` has the implicit-function-theorem approach for
   `rho_Tp`, which detaches its solve. Use it.
3. `liquid-metal-uncertainty.svg` -- thermal conductivity vs temperature for Na, Pb and LBE
   with Monte Carlo bands from `pinthac/uncertainty.py` and the `uncert_k` tables in
   `properties/liqprops.py`. Accent one series.
4. `sca-surrogate-validation.svg` -- DeepONet axial profile against ground-truth FVM for 2-3
   operating points, from held-out runs only.
5. `sca-inference-speed.svg` -- surrogate vs iterative solver wall clock. Given the batch-size
   dependence above, plot **speedup against batch size** rather than a single bar pair, or if
   you keep a bar chart, label the batch size on it. Do not quote a headline number that only
   holds at one batch size.
6. `pinthac-architecture.svg` -- module dependency diagram. PINTHAC modules in accent,
   external dependencies in muted grey. The real import graph, not an idealized one.

**Every number must come from a run in this repository.** State the hardware
(AMD RX 7800 XT, ROCm 7.2, torch 2.9.1) and the grid sizes in the caption text.

## Deliverables

- The six SVGs in `figures/output/`, each regenerable from its script.
- `docs/FIGURE_CAPTIONS.md`: for each figure, the caption text and the HTML snippet to paste
  into `index.html`, with the measured numbers and stated hardware.
- Any CV or page claim the measurements do not support, listed explicitly. The owner asked
  for this directly: "If the measured IAPWS-95 speedup is not 36x, report the number you
  actually measured."

## Ground rules

- Branch `cleanup`. Commit per logical unit.
- **Never invent a number.** Every figure value comes from a run.
- Never assert a test value produced by the code under test.
- Never `git reset --hard`, never force-push, never delete a file.
- `pytest tests/ -q` is green at 335 passed. Keep it green.
- End commit messages with:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` and the `Claude-Session:` line.

## Report back

1. The measured IAPWS-95 GPU speedup, with grid sizes and hardware.
2. Each figure, and anything that did not come out as the spec asked.
3. Every claim the measurements do not support.
4. What you could not do, and why.
5. Verbatim test output.
