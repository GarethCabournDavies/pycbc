"""Unit tests for pycbc.events.trigger_fits."""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from pycbc.events import trigger_fits as tf

THRESH = 5.0
ALPHA = {"exponential": 2.0, "rayleigh": 0.5, "power": 4.0}
NSAMP = 100000


def draw(distr, alpha, thresh, n, rng):
    """Draw samples above thresh from the distributions fitted by the module"""
    if distr == "exponential":
        return thresh + rng.exponential(1.0 / alpha, n)
    if distr == "rayleigh":
        # x**2 - thresh**2 is exponentially distributed with rate alpha / 2
        return np.sqrt(thresh**2 + rng.exponential(2.0 / alpha, n))
    # Pareto tail with survival function (x / thresh) ** (1 - alpha)
    return thresh * (1.0 + rng.pareto(alpha - 1.0, n))


@pytest.fixture(params=sorted(ALPHA))
def distr(request):
    return request.param


@pytest.fixture
def samples(distr):
    rng = np.random.default_rng(1234)
    return draw(distr, ALPHA[distr], THRESH, NSAMP, rng)


def test_fit_recovers_alpha(distr, samples):
    alpha, sigma = tf.fit_above_thresh(distr, samples, thresh=THRESH)
    assert sigma > 0
    assert abs(alpha - ALPHA[distr]) < 5 * sigma


def test_fit_discards_values_below_threshold(distr, samples):
    noisy = np.concatenate([samples, np.full(1000, THRESH - 1.0)])
    clean = tf.fit_above_thresh(distr, samples, thresh=THRESH)
    assert tf.fit_above_thresh(distr, noisy, thresh=THRESH) == pytest.approx(clean)


def test_fit_integer_weights_match_repeated_values(distr, samples):
    vals = samples[:100]
    weights = np.arange(1, 101)
    weighted = tf.fit_above_thresh(distr, vals, thresh=THRESH, weights=weights)
    repeated = tf.fit_above_thresh(distr, np.repeat(vals, weights), thresh=THRESH)
    assert weighted == pytest.approx(repeated)


def test_fit_default_threshold_is_minimum(distr, samples):
    explicit = tf.fit_above_thresh(distr, samples, thresh=samples.min())
    assert tf.fit_above_thresh(distr, samples) == pytest.approx(explicit)


def test_fit_no_values_above_threshold():
    assert tf.fit_above_thresh("exponential", [1.0, 2.0], thresh=10.0) == (-1.0, -1.0)


def test_fit_fn_is_normalised_and_zero_below_threshold(distr):
    alpha = ALPHA[distr]
    xvals = np.linspace(THRESH, THRESH + 200.0, 400001)
    pdf = tf.fit_fn(distr, xvals, alpha, THRESH)
    # Integral up to the upper limit plus the analytic tail above it
    total = trapezoid(pdf, xvals) + tf.cum_fit(distr, xvals[-1:], alpha, THRESH)[0]
    assert total == pytest.approx(1.0, rel=1e-4)
    assert tf.fit_fn(distr, [THRESH - 1.0], alpha, THRESH)[0] == 0.0


def test_cum_fit_is_reverse_cdf(distr):
    alpha = ALPHA[distr]
    xvals = np.linspace(THRESH, THRESH + 5.0, 10001)
    cum = tf.cum_fit(distr, xvals, alpha, THRESH)
    assert cum[0] == pytest.approx(1.0)
    assert np.all(np.diff(cum) < 0)
    # The derivative of the reverse CDF is minus the PDF
    deriv = -np.gradient(cum, xvals)
    pdf = tf.fit_fn(distr, xvals, alpha, THRESH)
    np.testing.assert_allclose(deriv[1:-1], pdf[1:-1], rtol=1e-3)
    assert tf.cum_fit(distr, [THRESH - 1.0], alpha, THRESH)[0] == 0.0


def test_ks_test_accepts_true_distribution(distr, samples):
    _, pval = tf.ks_test(distr, samples[:2000], ALPHA[distr], thresh=THRESH)
    assert pval > 0.01


def test_ks_test_rejects_wrong_alpha(distr, samples):
    _, pval = tf.ks_test(distr, samples[:2000], 2 * ALPHA[distr], thresh=THRESH)
    assert pval < 1e-6


def test_tail_threshold():
    vals = np.arange(100.0)
    assert tf.tail_threshold(vals, n=10) == 90.0
    with pytest.raises(RuntimeError):
        tf.tail_threshold(vals, n=101)


@pytest.mark.parametrize(
    ("args", "log", "expected"),
    [
        ((0.0, 0.0, 10.0, 5), False, 0),
        ((4.9, 0.0, 10.0, 5), False, 2),
        ((10.0, 0.0, 10.0, 5), False, 4),
        ((10.0, 1.0, 1000.0, 3), True, 1),
        ((1000.0, 1.0, 1000.0, 3), True, 2),
        ((3.0, 3.0, 3.0, 4), False, 3),
    ],
)
def test_which_bin(args, log, expected):
    assert tf.which_bin(*args, log=log) == expected


def test_which_bin_out_of_range():
    with pytest.raises(AssertionError):
        tf.which_bin(11.0, 0.0, 10.0, 5)
