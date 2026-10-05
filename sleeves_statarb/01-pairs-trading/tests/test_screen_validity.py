"""The screen's own statistics, checked on cases with known answers.

The headline of screen_validity.py is a simulation, and a simulation that is
subtly wrong looks exactly like a simulation that is right. So the pieces
around it are pinned to things that can be verified without running it: the
Benjamini-Hochberg procedure against its definition, Storey's pi0 against
p-values drawn from a distribution whose null share is known by construction,
and the two cointegration tests against series that genuinely are, and
genuinely are not, cointegrated.
"""
from __future__ import annotations

import numpy as np
import pytest

from screen_validity import (ALPHA, benjamini_hochberg, engle_granger_pvalue,
                             storey_pi0)
from scan import beta_and_pvalue


def test_bh_matches_the_definition_on_a_worked_example():
    """Five p-values, q = 0.20, thresholds k*q/n = .04 .08 .12 .16 .20.
    Sorted p = .01 .03 .05 .30 .40 -> .01<=.04, .03<=.08, .05<=.12, then
    .30 > .16. The largest k that passes is 3, so three are rejected."""
    p = np.array([0.30, 0.01, 0.40, 0.03, 0.05])
    keep = benjamini_hochberg(p, 0.20)
    assert keep.sum() == 3
    assert set(np.flatnonzero(keep)) == {1, 3, 4}


def test_bh_is_a_step_up_procedure_not_a_simple_threshold():
    """The point of "step up": a large p-value that clears its own threshold
    drags the smaller ones in with it, so BH can reject more than the naive
    count of p < q."""
    p = np.array([0.001, 0.19, 0.19, 0.19, 0.19])
    assert benjamini_hochberg(p, 0.20).sum() == 5
    assert (p < 0.20 / len(p)).sum() == 1          # Bonferroni keeps one


def test_bh_is_never_stricter_than_bonferroni():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.uniform(size=200)
        assert benjamini_hochberg(p, 0.05).sum() >= (p < 0.05 / 200).sum()


def test_bh_rejects_nothing_when_nothing_is_small():
    p = np.linspace(0.3, 1.0, 100)
    assert benjamini_hochberg(p, 0.05).sum() == 0


def test_bh_gets_stricter_as_q_falls():
    rng = np.random.default_rng(1)
    p = np.concatenate([rng.uniform(0, 0.01, 30), rng.uniform(size=970)])
    counts = [benjamini_hochberg(p, q).sum() for q in (0.01, 0.05, 0.10, 0.20)]
    assert counts == sorted(counts)


def test_pi0_is_one_when_every_p_value_is_null():
    rng = np.random.default_rng(2)
    assert storey_pi0(rng.uniform(size=20_000)) == pytest.approx(1.0, abs=0.03)


def test_pi0_recovers_a_known_null_share():
    """70% uniform, 30% piled near zero. pi0 should come back near 0.7."""
    rng = np.random.default_rng(3)
    p = np.concatenate([rng.uniform(size=7_000),
                        rng.uniform(0, 0.01, 3_000)])
    assert storey_pi0(p) == pytest.approx(0.7, abs=0.05)


def test_pi0_is_capped_at_one():
    assert storey_pi0(np.full(100, 0.9)) == 1.0


def test_both_tests_find_a_genuinely_cointegrated_pair():
    """x is a random walk, y is x plus a stationary wobble. The spread is
    stationary by construction, so a working test must say so."""
    rng = np.random.default_rng(5)
    x = np.cumsum(rng.normal(0, 0.01, 1_500))
    y = 1.4 * x + rng.normal(0, 0.01, 1_500)
    assert beta_and_pvalue(y, x)[1] < 0.01
    assert engle_granger_pvalue(y, x) < 0.01


def test_the_naive_test_is_the_looser_of_the_two_on_unrelated_walks():
    """The whole finding in miniature: on independent random walks the
    adfuller-on-residual p-value is systematically smaller - more
    significant - than the Engle-Granger one on the same data."""
    rng = np.random.default_rng(6)
    naive, proper = [], []
    for _ in range(60):
        y = np.cumsum(rng.normal(0, 0.012, 800))
        x = np.cumsum(rng.normal(0, 0.012, 800))
        naive.append(beta_and_pvalue(y, x)[1])
        proper.append(engle_granger_pvalue(y, x))
    assert np.mean(naive) < np.mean(proper)
    assert np.mean(np.array(naive) < ALPHA) > np.mean(np.array(proper) < ALPHA)
