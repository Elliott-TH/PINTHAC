"""
Permanent tests for pinthac.ml.datagen's Fourier power-profile parameterization
(Phase 6, docs/brief/PHASE67_BRIEF.md part A2).

The analytic mean <Fq> = 0.5*sum(a_n^2+b_n^2) + phi_q is checked against numerical
quadrature of the same S(x)^2 + phi_q profile -- an independent computation of the same
quantity, not a value obtained by running the code under test on itself. Positivity and
the peak/mean normalization conventions of the two shape families are checked as
structural consequences of the construction (squaring plus a positive offset can never
go negative), not as asserted numbers.
"""
import numpy as np
import pytest

from pinthac.ml import datagen


def test_fourier_mean_matches_quadrature():
    # Independent reference: integrate Fq(x) = S(x)^2 + phi_q over the axial length by
    # the trapezoid rule on a fine grid, and compare to the analytic closed form.
    rng = np.random.default_rng(7)
    n_samples, K_max = 12, 4
    a, b, phi_q = datagen.sample_fourier_coeffs(n_samples, K_max=K_max, seed=3)

    x_fine = np.linspace(-1.0, 1.0, 20001)   # z = x*L/2, so this spans the full [-L/2, L/2]
    cos_nx, sin_nx = datagen.fourier_basis(x_fine, K_max)
    S = a @ cos_nx.T + b @ sin_nx.T           # (n_samples, len(x_fine))
    Fq = S**2 + phi_q[:, None]

    # Average over z, not over x: dz = (L/2) dx, and the mean divides by the length L,
    # so the two L/2 factors cancel and integrating-then-dividing by the x-span (2.0)
    # gives the same z-averaged quantity the PDF's <Fq> = (1/L) integral dz does.
    quad_mean = np.trapezoid(Fq, x_fine, axis=1) / (x_fine[-1] - x_fine[0])
    analytic_mean = datagen.fourier_mean(a, b, phi_q)

    np.testing.assert_allclose(quad_mean, analytic_mean, rtol=1e-6)


def test_fourier_shape_is_strictly_positive_and_mean_normalized():
    a, b, phi_q = datagen.sample_fourier_coeffs(50, K_max=4, seed=11)
    x = np.linspace(-1.0, 1.0, 501)
    shape = datagen.build_fourier_shapes(a, b, phi_q, x)

    assert np.all(shape > 0.0)   # phi_q > 0 keeps Fq, and therefore shape, off zero
    # Mean-normalized (not peak-normalized like the Legendre family): the arithmetic
    # mean over a discrete grid is only an approximation of the analytic (integral)
    # mean that shape was actually normalized by, so this needs a slightly looser
    # tolerance than the quadrature check above (which integrates rather than averages).
    np.testing.assert_allclose(shape.mean(axis=1), 1.0, atol=5e-3)


def test_fourier_order_is_weighted_toward_low_order():
    # "vary the order, weighted toward low order (more 2nd than 3rd or 4th)" --
    # docs/brief/PHASE67_BRIEF.md A2. Check the *active* mode count (nonzero coefficients)
    # is monotonically less common as order increases, over a large draw.
    a, b, phi_q = datagen.sample_fourier_coeffs(20000, K_max=4, seed=5)
    active = (a != 0.0) | (b != 0.0)
    order = active.sum(axis=1)   # 1..4
    counts = np.array([(order == k).sum() for k in range(1, 5)])
    assert np.all(np.diff(counts) < 0), counts


def test_legendre_shape_still_peak_normalized():
    # build_shapes (Legendre) is unchanged by this phase -- confirm its own convention
    # (peak, not mean, normalized) so the two families are not confused with each other.
    rng = np.random.default_rng(0)
    coeffs = rng.uniform(-1.0, 1.0, size=(10, datagen.N_SHAPE_MODES))
    x = np.linspace(-1.0, 1.0, 201)
    shape = datagen.build_shapes(coeffs, x)
    assert np.all(shape > 0.0)
    np.testing.assert_allclose(shape.max(axis=1), 1.0, atol=1e-12)


def test_build_shapes_by_basis_selector():
    x = np.linspace(-1.0, 1.0, 101)
    rng = np.random.default_rng(1)
    coeffs = rng.uniform(-1.0, 1.0, size=(5, datagen.N_SHAPE_MODES))
    leg = datagen.build_shapes_by_basis("legendre", x, coeffs=coeffs)
    np.testing.assert_allclose(leg, datagen.build_shapes(coeffs, x))

    a, b, phi_q = datagen.sample_fourier_coeffs(5, K_max=3, seed=2)
    fou = datagen.build_shapes_by_basis("fourier", x, a=a, b=b, phi_q=phi_q)
    np.testing.assert_allclose(fou, datagen.build_fourier_shapes(a, b, phi_q, x))

    with pytest.raises(ValueError):
        datagen.build_shapes_by_basis("chebyshev", x)
