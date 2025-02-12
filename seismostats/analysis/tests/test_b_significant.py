import numpy as np
from numpy.testing import assert_almost_equal
from datetime import datetime
import pytest
import warnings

from seismostats.analysis.b_significant import (
    est_morans_i,
    bs_from_partitioning,
    cut_constant_idx,
    transform_n,
    mac_1D_constant_nm,
)


def test_est_morans_i():
    # 1 dimensional case
    values = np.array([1, 2, 3, 4, 5])
    w = np.array([[0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0]])
    ac, n, n_p = est_morans_i(values, w)
    assert_almost_equal(ac, 0.4)
    assert n == 5
    assert n_p == 4

    ac_no_w, n_no_w, n_p_no_w = est_morans_i(values)
    assert ac == ac_no_w
    assert n == n_no_w
    assert n_p == n_p_no_w

    # 2D case, test if weight matrix can be transposed with same result
    w = np.array([[0, 1, 0, 1, 1],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0]])
    ac, n, n_p = est_morans_i(values, w)
    ac_T, n_T, n_p_T = est_morans_i(values, w.T)
    assert ac == ac_T
    assert n == n_T
    assert n_p == n_p_T
    ac_symmetric, n_symmetric, n_p_symmetric = est_morans_i(values, w + w.T)
    assert ac == ac_symmetric
    assert n == n_symmetric
    assert n_p == n_p_symmetric

    # test that correct error is raised
    with pytest.raises(ValueError):
        # non-square matrix
        w = np.array([[0, 1, 0, 1, 1],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]])
        est_morans_i(values, w)
    with pytest.raises(ValueError):
        # trace is not zero
        w = np.array([[0, 1, 0, 1, 1],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 0],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0]])
        est_morans_i(values, w)
    with pytest.raises(ValueError):
        # other values than 0 and 1
        w = np.array([[0, 1, 0, 1, 1],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, -1],
                      [0, 0, 0, 0, 0]])
        est_morans_i(values, w)
    with pytest.raises(ValueError):
        # non symmetric matrix
        w = np.array([[0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0]])
        est_morans_i(values, w)


def test_transform_n():
    b_est = np.array([1, 1, 2, 0.5, 2])
    n1 = np.array([10, 100, 100, 100, 1000])
    n2 = 1000
    b_true = 1
    b_transformed = transform_n(b_est, b_true, n1, n2)
    correct_b_transformed = np.array(
        [1., 1., 1.1878091107778657, 0.7597469266479578, 2.])
    assert_almost_equal(b_transformed, correct_b_transformed)

    b_est = np.array([1, 1, 2, 0.5, 2])
    n1 = 1000
    n2 = 1000
    b_true = 1
    b_transformed = transform_n(b_est, b_true, n1, n2)
    assert_almost_equal(b_transformed, b_est)

    # check that sanity checks work
    with pytest.raises(AssertionError):
        transform_n(b_est, b_true, np.array([1, 2]), n2)
    with pytest.raises(ValueError):
        transform_n(b_est, b_true, 5, 1)
    with pytest.raises(ValueError):
        transform_n(b_est, b_true, n1, np.array([1, 2]))


def test_bs_from_partitioning():
    list_magnitudes = [np.array([1, 2, 3, 4, 5]),
                       np.array([1, 2, 3, 4, 5]),
                       np.array([1, 2, 3, 4, 5])]
    list_times = [np.array([datetime(2021, 1, 1), datetime(2021, 1, 2),
                            datetime(2021, 1, 3), datetime(2021, 1, 4),
                            datetime(2021, 1, 5)]),
                  np.array([datetime(2021, 1, 6), datetime(2021, 1, 7),
                            datetime(2021, 1, 8), datetime(2021, 1, 9),
                            datetime(2021, 1, 10)]),
                  np.array([datetime(2021, 1, 11), datetime(2021, 1, 12),
                            datetime(2021, 1, 13), datetime(2021, 1, 14),
                            datetime(2021, 1, 15)])]
    delta_m = 1
    mc = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b_values, std_b, n_ms = bs_from_partitioning(
            list_magnitudes, list_times, mc, delta_m)

    assert_almost_equal(b_values, np.array(
        [0.17609125905568124, 0.17609125905568124, 0.17609125905568124]))
    assert_almost_equal(std_b, np.array([0.05048661905780697,
                        0.05048661905780697, 0.05048661905780697]))
    assert_almost_equal(n_ms, np.array([5, 5, 5]))


def test_cut_constant_idx():
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                       12, 13, 14, 15, 16, 17, 18, 19, 20])
    idx, subsamples = cut_constant_idx(values, n=5)

    correct_idx = [5, 10, 15]
    correct_subsamples = [np.array([1, 2, 3, 4, 5]),
                          np.array([6, 7, 8, 9, 10]),
                          np.array([11, 12, 13, 14, 15]),
                          np.array([16, 17, 18, 19, 20])]
    assert all(idx == correct_idx)
    assert all(all(subsample == correct_subsample) for subsample,
               correct_subsample in zip(subsamples, correct_subsamples))

    # check that the split samples are the same as the subsamples
    values = np.random.rand(100)
    idx, subsamples = cut_constant_idx(values, 4, offset=2)
    split_samples = np.array_split(values, idx)
    assert all(all(subsample == split_sample) for subsample,
               split_sample in zip(subsamples, split_samples))


def test_mac_1D_constant_nm():
    mags = np.arange(0, 1000, 1)
    times = np.arange(0, 1000, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mac, mu_mac, std_mac = mac_1D_constant_nm(
            mags,
            mc=0,
            delta_m=1,
            times=times,
            n_m=20)

    assert_almost_equal(mac, 0.5184342563144473)
    assert_almost_equal(mu_mac, -0.020387359836901122)
    assert_almost_equal(std_mac, 0.14426245016127676)

    with pytest.raises(ValueError):
        # mags larger than mc present
        mac_1D_constant_nm(
            mags,
            mc=1,
            delta_m=1,
            times=times,
            n_m=20)
    with pytest.raises(ValueError):
        # min_num larger than n_m
        mac_1D_constant_nm(
            mags,
            mc=0,
            delta_m=1,
            times=times,
            n_m=10,
            min_num=11)
    with pytest.raises(ValueError):
        # n_m larger than len(mags)/3
        mac_1D_constant_nm(
            mags,
            mc=0,
            delta_m=1,
            times=times,
            n_m=500)
    with pytest.raises(ValueError):
        # mags are not an array
        mac_1D_constant_nm(
            list(mags),
            delta_m=1,
            mc=0,
            times=times,
            n_m=20)
    with pytest.raises(ValueError):
        # times and mags have different lengths
        mac_1D_constant_nm(
            mags,
            delta_m=1,
            mc=0,
            times=times[:-1],
            n_m=20)
