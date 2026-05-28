"""Tests for automatic beta calibration."""

import numpy as np

from anisotropy.ising_params import _load_sampling_params
from anisotropy.orientation_diagnostics import (
    calibrate_beta_auto,
    effective_sample_size_for_beta,
)


def test_load_sampling_beta_auto_string():
    p = _load_sampling_params({"beta": "auto", "n_samples": 100})
    assert p.beta_auto is True


def test_calibrate_beta_auto_ess_on_stiff_energies():
    rng = np.random.default_rng(0)
    E = rng.standard_normal(400) * 1e7
    E[0] -= 5e5
    beta, details = calibrate_beta_auto(E, target_ess=20.0, method="ess")
    assert beta > 0
    ess = effective_sample_size_for_beta(E, beta)
    assert 15 <= ess <= 25
    assert details["method"] == "ess"


def test_calibrate_beta_rank2():
    E = np.array([0.0, 38800.0, 50000.0])
    beta, details = calibrate_beta_auto(E, method="rank2")
    assert abs(beta - np.log(10) / 38800) < 1e-12
    assert details["method"] == "rank2"
