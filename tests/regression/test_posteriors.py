"""Tests for the Posteriors data model. Synthetic nested samples only."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from alderaan.data_models import Posteriors
from alderaan.utils.astro import ld_q_to_u
from alderaan.utils.stats import legendre_p1, weighted_percentile


class TestConstructor:
    def test_rejects_ref_epoch_per_planet_mismatch(self, make_posteriors):
        with pytest.raises(ValueError, match="ref_epoch"):
            make_posteriors(ref_epoch=[131.1])

    def test_rejects_sample_columns_inconsistent_with_npl(self, make_posteriors):
        with pytest.raises(ValueError, match="sample columns"):
            make_posteriors(samples=make_posteriors(npl=1).samples)

    def test_rejects_missing_sampler_key(self, make_posteriors):
        info = dict(make_posteriors().sampler_info)
        del info["logvol"]
        with pytest.raises(ValueError, match="sampler_info keys"):
            make_posteriors(sampler_info=info)

    def test_rejects_sampler_arrays_of_wrong_length(self, make_posteriors):
        info = dict(make_posteriors().sampler_info)
        info["logl"] = info["logl"][:-1]
        with pytest.raises(ValueError, match="logl"):
            make_posteriors(sampler_info=info)


class TestFromDynesty:
    def test_labels_samples_and_keeps_sampler_info(self, make_posteriors):
        ref = make_posteriors()
        samples = ref.samples.to_numpy()
        results = SimpleNamespace(samples=samples, **ref.sampler_info)
        tm = SimpleNamespace(
            npl=2,
            ref_period=ref.ref_period,
            ref_epoch=ref.ref_epoch,
            t_start=ref.t_start,
            t_end=ref.t_end,
        )
        post = Posteriors.from_dynesty("Kepler", "K00137", results, tm)
        assert list(post.samples.columns) == list(ref.samples.columns)
        assert_allclose(post.samples.to_numpy(), samples)
        assert post.sampler_info["niter"] == ref.sampler_info["niter"]

    def test_rejects_planet_count_mismatch(self, make_posteriors):
        ref = make_posteriors()
        results = SimpleNamespace(samples=ref.samples.to_numpy(), **ref.sampler_info)
        tm = SimpleNamespace(npl=3)
        with pytest.raises(ValueError, match="planets"):
            Posteriors.from_dynesty("Kepler", "K00137", results, tm)


class TestWeights:
    def test_weights_normalized_and_proportional_to_exp_logwt(self, make_posteriors):
        post = make_posteriors()
        wt = post.weights()
        assert np.isclose(wt.sum(), 1.0)
        expected = np.exp(post.sampler_info["logwt"])
        assert_allclose(wt, expected / expected.sum())

    def test_neff_is_kish_ess(self, make_posteriors):
        post = make_posteriors()
        w = np.exp(post.sampler_info["logwt"])
        assert np.isclose(post.neff(), w.sum() ** 2 / np.sum(w**2))

    def test_summary_weighted_mean_and_percentiles(self, make_posteriors):
        post = make_posteriors()
        summary = post.summary()
        assert list(summary.index) == list(post.samples.columns)
        wt = post.weights()
        x = post.samples["ROR_1"].to_numpy()
        mean = np.sum(wt * x)
        row = summary.loc["ROR_1"]
        assert np.isclose(row["mean"], mean)
        assert np.isclose(row["stdev"], np.sqrt(np.sum(wt * (x - mean) ** 2)))
        assert_allclose(
            row[["CI_2.5", "median", "CI_97.5"]].to_numpy(dtype=float),
            weighted_percentile(x, [2.5, 50, 97.5], w=wt),
        )


class TestPlanetSamples:
    def test_period_epoch_reproduce_sampled_transit_times(self, make_posteriors):
        """C0 + C1 * legx shifts the reference transit times; planet_samples must
        give the linear ephemeris through those shifted times"""
        post = make_posteriors()
        n = 1
        index = np.arange(30)
        ref_times = post.ref_epoch[n] + post.ref_period[n] * index
        legx = legendre_p1(ref_times, (post.t_start, post.t_end))
        C0 = post.samples[f"C0_{n}"].to_numpy() - 0.5
        C1 = post.samples[f"C1_{n}"].to_numpy() - 0.5
        post.samples[f"C0_{n}"] = C0
        post.samples[f"C1_{n}"] = C1

        ps = post.planet_samples(n)
        for i in [0, 17]:
            sampled = ref_times + C0[i] + C1[i] * legx
            period, epoch = np.polyfit(index, sampled, 1)
            assert np.isclose(ps["PERIOD"][i], period, rtol=0, atol=1e-9)
            assert np.isclose(ps["EPOCH"][i], epoch, rtol=0, atol=1e-9)

    def test_shape_and_limb_darkening_columns(self, make_posteriors):
        post = make_posteriors()
        ps = post.planet_samples(0)
        assert list(ps.columns) == [
            "PERIOD",
            "EPOCH",
            "ROR",
            "IMPACT",
            "DUR14",
            "LD_U1",
            "LD_U2",
            "LN_WT",
        ]
        assert_allclose(ps["ROR"], post.samples["ROR_0"])
        u1, u2 = ld_q_to_u(post.samples["LD_Q1"], post.samples["LD_Q2"])
        assert_allclose(ps["LD_U1"], u1)
        assert_allclose(ps["LD_U2"], u2)
        assert_allclose(ps["LN_WT"], post.sampler_info["logwt"])
