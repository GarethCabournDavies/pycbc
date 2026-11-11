"""Unit tests for autocorrelation, Gelman-Rubin and Geweke diagnostics."""

import numpy as np
import pytest

from pycbc.filter import autocorrelation
from pycbc.inference import gelman_rubin, geweke
from pycbc.types import TimeSeries

PHI = 0.9


def ar1(n, phi, rng):
    """First-order autoregressive series with ACF phi ** k"""
    noise = rng.normal(size=n)
    out = np.empty(n)
    out[0] = noise[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + noise[i]
    return out


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_acf_of_white_noise(rng):
    acf = autocorrelation.calculate_acf(rng.normal(size=10000))
    assert len(acf) == 10000
    assert acf[0] == pytest.approx(1.0)
    assert np.all(np.abs(acf[1:50]) < 0.05)


def test_acf_of_ar1_process(rng):
    acf = autocorrelation.calculate_acf(ar1(100000, PHI, rng))
    lags = np.arange(10)
    np.testing.assert_allclose(acf[:10], PHI**lags, atol=0.02)


def test_acf_unbiased_zero_lag(rng):
    acf = autocorrelation.calculate_acf(rng.normal(size=1000), unbiased=True)
    assert acf[0] == pytest.approx(1.0)


def test_acf_timeseries_input_keeps_delta_t(rng):
    data = TimeSeries(rng.normal(size=1000), delta_t=0.25)
    acf = autocorrelation.calculate_acf(data)
    assert isinstance(acf, TimeSeries)
    assert acf.delta_t == 0.25
    assert len(acf) == len(data)


def test_acl_white_noise_and_ar1(rng):
    assert autocorrelation.calculate_acl(rng.normal(size=10000)) == 1
    # The ACL of an AR(1) process is (1 + phi) / (1 - phi)
    acl = autocorrelation.calculate_acl(ar1(100000, PHI, rng), dtype=float)
    assert acl == pytest.approx((1 + PHI) / (1 - PHI), rel=0.15)


def test_acl_edge_cases(rng):
    assert autocorrelation.calculate_acl(np.ones(1)) == 1
    with pytest.raises(ValueError, match="dtype"):
        autocorrelation.calculate_acl(rng.normal(size=10), dtype=complex)


def test_gelman_rubin_converged_chains(rng):
    chains = rng.normal(size=(4, 2, 5000))
    psrf = gelman_rubin.gelman_rubin(chains)
    assert psrf.shape == (2,)
    np.testing.assert_allclose(psrf, 1.0, atol=0.01)


def test_gelman_rubin_unconverged_chains(rng):
    chains = rng.normal(size=(4, 1, 5000))
    chains[0] += 5.0
    psrf = gelman_rubin.gelman_rubin(chains, auto_burn_in=False)
    assert psrf.shape == (1,)
    assert psrf[0] > 1.5


def test_gelman_rubin_walk(rng):
    chains = rng.normal(size=(3, 2, 1000))
    # Each statistic uses samples from index 0 up to the corresponding end
    starts, ends, stats = gelman_rubin.walk(chains, 200, 1001, 200)
    np.testing.assert_array_equal(ends, [200, 400, 600, 800, 1000])
    np.testing.assert_array_equal(starts, np.full(5, 200))
    assert stats.shape == (2, 5)
    np.testing.assert_allclose(stats[:, -1], gelman_rubin.gelman_rubin(chains))


def test_geweke_stationary_chain(rng):
    x = rng.normal(size=10000)
    starts, ends, stats = geweke.geweke(
        x, seg_length=1000, seg_stride=1000, end_idx=5000, ref_start=5000
    )
    np.testing.assert_array_equal(starts, [0, 1000, 2000, 3000, 4000])
    np.testing.assert_array_equal(ends, starts + 1000)
    assert np.all(np.abs(stats) < 0.1)


def test_geweke_detects_burn_in(rng):
    x = rng.normal(size=10000)
    x[:1000] += 5.0
    _, _, stats = geweke.geweke(
        x, seg_length=1000, seg_stride=1000, end_idx=5000, ref_start=5000
    )
    assert stats[0] > 2.0
    assert np.all(np.abs(stats[1:]) < 0.1)
