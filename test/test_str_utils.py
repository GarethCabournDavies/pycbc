"""Unit tests for pycbc.results.str_utils."""

import pytest

from pycbc.results import str_utils


@pytest.mark.parametrize(
    ("num", "expected"),
    [(1.5, "1.5"), (2.0, "2"), (10.0, "10"), (0.125, "0.125"), (0.0, "0")],
)
def test_drop_trailing_zeros(num, expected):
    assert str_utils.drop_trailing_zeros(num) == expected


@pytest.mark.parametrize(
    ("val", "err", "kwargs", "expected"),
    [
        # Small error: decimal places follow the error's leading digit
        (3.14159, 0.01, {}, "3.142"),
        # max_sig caps the number of decimal places
        (3.14159, 1e-6, {"max_sig": 2}, "3.142"),
        # Large error: value is rounded to the tens/hundreds place
        (1234.3, 10.0, {}, "1234"),
        (1234.3, 100.0, {}, "1230"),
        # Error much larger than value: rounding is relaxed until non-zero
        (0.004, 100.0, {}, "0.004"),
    ],
)
def test_get_signum(val, err, kwargs, expected):
    assert str_utils.get_signum(val, err, **kwargs) == expected


def test_mathjax_header_contains_script():
    header = str_utils.mathjax_html_header()
    assert "MathJax" in header
    assert "<script" in header


def test_format_value_plain():
    assert str_utils.format_value(3.14159, 0.01) == r"3.142 \pm 0.010"


def test_format_value_negative_without_error():
    assert str_utils.format_value(-3.14159, 0.01, include_error=False) == "-3.142"


def test_format_value_scientific_notation():
    val, err = 3.9278372067613837e-22, 2.2351435286500487e-23
    assert str_utils.format_value(val, err) == r"3.93 \pm 0.22\times 10^{-22}"


def test_format_value_relative_error():
    val, err = 3.9278372067613837e-22, 2.2351435286500487e-23
    out = str_utils.format_value(val, err, use_relative_error=True)
    assert out == r"3.93 \times 10^{-22} \pm5.6\%"


def test_format_value_asymmetric_errors():
    val, err = 3.9278372067613837e-22, 2.2351435286500487e-23
    err_plus = 8.2700310560051804e-24
    out = str_utils.format_value(val, err, plus_error=err_plus)
    assert out == r"3.928^{+0.083}_{-0.224}\times 10^{-22}"


def test_format_value_zero():
    assert str_utils.format_value(0.0, 0.1, include_error=False) == "0.00"
