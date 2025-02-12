"""This module contains functions for the evaluation if a one dimensional
b-value series varies significantly, following Mirwald et al, SRL (2024)
"""

import numpy as np
import warnings

from seismostats.analysis.bvalue import ClassicBValueEstimator
from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.utils._config import get_option


def est_morans_i(values: np.ndarray,
                 w: np.ndarray | None = None,
                 mean_v: float = None) -> tuple:
    """
    Estimate the nearest neighbor auto correlation (Moran's I) of the values.

    Args:
        values:     Values for which the autocorrelation is estimated.
        w:          Weight matrix, indicating which of the values are
                neighbors to each other. It should be a square matrix of
                size `len(values) x len(values)`, with zeros on the diagonal.
                At places where the value is 1, the values are considered
                neighbors to each other. Values that are NaN are not considered
                neighbors to any other value. If w is None, it is assumed that
                the series of values is 1-dimensional, and the values are
                sorted along that dimension. Then, the ac that is returned
                corresponds to the  usual 1D autocorrelation with a lag of 1.
        mean_v:     Mean value of the series. If not provided, it is estimated
                from the non-nan values.

    Returns:
        ac:     Auto correlation of the values.
        n:      Number of values that are not NaN.
        n_p:    Sum of the weight matrix. In the limit of a large n (number of
            values), the upper limit of the standard deviation of the
            autocorrelation is `1/sqrt(n_p)`. This number is can be interpreted
            as the number of neighboring pairs.

    Examples:
        .. code-block:: python

        import numpy as np
        from seismostats.analysis.b_significant import est_morans_i

        values = np.array([2, 2, 2, 4, 4])
        w = np.array([[0, 1, 1, 0, 0],
                      [1, 0, 1, 0, 0],
                      [1, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 1, 0]])
        # now, the first three values are neighbors to each other, as well as
        # the last two values.

        ac, n, n_p = est_morans_i(values, w, mean_v=3)
    """
    # sanity checks
    if len(values) < 2:
        raise ValueError("At least 2 values are needed for the estimation")

    # Checks regardning the weight matrix. In case it is not provided, 1D case
    # is assumed
    if w is None:
        n_values = len(values)
        w = np.zeros((n_values, n_values))
        for ii in range(n_values):
            for jj in range(n_values):
                if jj == ii + 1:
                    w[ii, jj] = 1
    else:
        if w.shape[0] != w.shape[1]:
            raise ValueError("Weight matrix must be square.")
        if w.shape[0] != len(values):
            raise ValueError(
                "Weight matrix must have the same size as the values.")
        if sum(w.diagonal()) != 0:
            raise ValueError("Weight matrix must have zeros on the diagonal.")
        if np.sum(np.tril(w)) != 0 and np.sum(np.triu(w)) != 0:
            if np.all(w == w.T):
                w = np.triu(w)
            else:
                raise ValueError(
                    "Weight matrix must be triangular or at least symmetric.")
        elif np.sum(np.triu(w)) == 0:
            w = w.T
        # Check that there are only 0 and 1 in the weight matrix.
        if not np.all(np.isin(w, [0, 1])):
            raise ValueError("Weight matrix must only contain 0 and 1.")

    if mean_v is None:
        mean_v = np.mean(values[~np.isnan(values)])

    # estimate autocorrelation
    ac = 0
    ac_0 = 0
    n = len(values[~np.isnan(values)])
    for ii, v1 in enumerate(values):
        if np.isnan(v1):
            w[ii, :] = 0
            continue
        ac_0 += (v1 - mean_v) ** 2
        for jj in range(ii + 1, len(values)):
            v2 = values[jj]
            if np.isnan(v2):
                w[ii, jj] = 0
                continue
            if w[ii, jj] == 1:
                ac += (v1 - mean_v) * (v2 - mean_v)

    n_p = np.sum(w)
    ac = (n - 1) / n_p * ac / ac_0
    return ac, n, n_p


def transform_n(
    b_estimates: np.ndarray | float,
    b_true: float,
    n1: np.ndarray | int,
    n2: int,
) -> np.ndarray:
    """Transform a b-value estimated from n1 events to a b-value estimated from
    n2 events, such that the distribution of the transformed b-values is
    consistent with one that would be estimated from n2 events. The
    transformation is based on the assumption that the true b-value is known,
    and that the b-values estimated follow the reciprocaln inverse distribution
    (which is only true for a large enough n1, see Shi and Bolt, 1981, BSSA).

    Source:
        Mirwald et al, SRL (2024), supplementary material

    Args:
        b_estimates:    b-value estimates to be transformed
        b_true:         True b-value
        n1:             Number of events used for the the b-value estimates.
            Has to be an integer or an array of the same length as b_estimate.
        n2:             number of events to which the distribution is
            transformed. It is only possible to transform to a larger number of
            events, so n2 must be larger than n1. while n1 can be an array, n2
            must be an integer.

    Returns:
        b_transformed:  Transformed b-values.
    """

    # sanity checks
    if not isinstance(n2, int):
        raise ValueError("n2 must be an integer.")
    if np.any(n1 > n2):
        raise ValueError("n2 must be larger or equal than n1.")
    if not isinstance(n1, (int, np.ndarray)):
        raise ValueError("n1 must be an integer or an array.")
    elif isinstance(n1, int):
        n1 = np.ones(len(b_estimates)) * n1
    else:
        assert len(b_estimates) == len(
            n1), ("If n1 is an array, it must have tha same length as"
                  "b_estimate.")

    # transform the b-values
    b_transformed = b_true / (1 - np.sqrt(n1 / n2) * (1 - b_true / b_estimates))
    return b_transformed


def bs_from_partitioning(
    list_magnitudes: list[np.ndarray],
    list_times: list[np.ndarray[np.datetime64]],
    list_mc: float | np.ndarray,
    delta_m: float,
    b_method: BValueEstimator = ClassicBValueEstimator,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Estimate the series of b-values from a list of subsets of magnitudes and
    times.

    Args:
        list_mags:  List of arrays of magnitudes. From each array within the
            list, a b-value is estimated.
        list_times: List of arrays of times, in the same order as the magnitudes
        list_mc:    List of completeness magnitude corresponding to the
            magnitudes. If a single value is provided, it is used for all
            magnitudes.
        delta_m:    Discretization of magnitudes.
        b_method:   Method to estimate the b-value.
        **kwargs:   Additional arguments to the b-value estimation method.

    Returns:
        b_values:   Series of b-values, each one is estimated from the
            magnitudes contained in the corresponding element of
            ``list_magnitudes``.
        std_b:      Standard deviations corresponding to the b-values.
        n_ms:       Number of events used for the b-value estimates.

    Examples:
        .. code-block:: python

        import numpy as np
        from seismostats.analysis.b_significant import bs_from_partitioning

        mags = [np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
                np.array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
                np.array([31, 32, 33, 34, 35, 36, 37, 38, 39, 40])]
        times = [np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
                np.array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30])]
        delta_m = 1
        mc = 11

        b_values, std_bs, n_ms = bs_from_partitioning(mags, times, delta_m, mc)
    """

    b_values = np.zeros(len(list_magnitudes))
    std_bs = np.zeros(len(list_magnitudes))
    n_ms = np.zeros(len(list_magnitudes))
    if isinstance(list_mc, (float, int)):
        list_mc = np.ones(len(list_magnitudes)) * list_mc

    estimator = b_method()

    for ii, mags_loop in enumerate(list_magnitudes):
        # Sort the magnitudes of the subsets by time.
        times_loop = list_times[ii]
        idx_sorted = np.argsort(times_loop)
        mags_loop = mags_loop[idx_sorted]
        times_loop = times_loop[idx_sorted]

        estimator.calculate(
            mags_loop, mc=list_mc[ii], delta_m=delta_m, **kwargs)
        b_values[ii] = estimator.b_value
        std_bs[ii] = estimator.std
        n_ms[ii] = estimator.n

    return b_values, std_bs, n_ms.astype(int)


def cut_constant_idx(
    values: np.ndarray,
    n: int,
    offset: int = 0,
) -> tuple[list[int], list[np.ndarray]]:
    """
    find the indices to cut a series such that the subsamples have a constant
    number of events, n.

    the subsamples can then be obtained in the following way:
    subsamples = np.array_split(values, idx)

    Args:
        values:     original series to be cut
        n:          number of events in each subsample
        offset:     idx where to start cutting the series. This should be
                between 0 and n

    Returns:
        idx:            indices of the subsamples
        subsamples:     list of subsamples
    """
    # Check that the offset is not larger than n.
    if offset >= n:
        raise ValueError("Offset must be smaller than n.")

    idx = np.arange(offset, len(values), n)

    if offset == 0:
        idx = idx[1:]

    subsamples = np.array_split(values, idx)
    return idx, subsamples


def mac_1D_constant_nm(
        mags: np.ndarray,
        mc: float,
        delta_m: float,
        times: np.ndarray[np.timedelta64],
        n_m: int,
        min_num: int = 10,
        b_method: BValueEstimator = ClassicBValueEstimator,
        **kwargs,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    This function estimates the mean autocorrelation for the one-dimensional
    case (along the dimension of order). Additionally, it provides the mean
    a- and b-values for each grid-point. The partitioning method is based on
    voronoi tesselation (random area).

    With the mean and standard deviation of the autocorrelation under H0, the
    hypothesis that the b-values are constant can be tested. If the number of
    subsamples is large enough, the autocorrelation can be assumed to be normal.
    As a lower limit, no less than 25 subsamples (which can be estimated by
    len(mags) / n_m) should be used.

    In order to plot the corresponding series of b-values, use the function
    plot_b_constant_mn (from seismostats.plots.statistical  import
    plot_b_constant_mn) with the same parameters as used here.

    Args:
        mags:       Magnitudes of the events. They are assumed to be order
            along the dimension of interest (e.g. time or depth)
        mc:     Completeness magnitude. If a single value is provided, it is
            used for all magnitudes. Otherwise, the individual completeness of
            each magnitude can be provided.
        delta_m:    Magnitude bin width.
        times:  Times of the events.
        n_m:    Number of magnitudes in each partition.
        min_num:    Minimum number of events in a partition.
        b_method:   Method to estimate the b-values.
        **kwargs:   Additional arguments to the b-value estimation method.

    Returns:
        mac:        Mean autocorrelation.
        mu_mac:     Expected mean autocorrelation und H0.
        std_mac:    Standard deviation of the mean autocorrelation under H0.
            (i.e. constant b-value). Here, the conservatice estimate is used -
            in case the non-conservative estimate is needed, the standard
            deviation can be mulitplied by the factor gamma = 0.81 given by
            Mirwald et al, SRL (2024).
    """
    if isinstance(mc, (float, int)):
        if min(mags) < mc:
            raise ValueError("The completeness magnitude is larger than the "
                             "smallest magnitude.")
        mc = np.ones(len(mags)) * mc
    else:
        if any(mags < mc):
            raise ValueError("There are earthquakes below their respective "
                             "completeness magnitude.")

    if n_m < min_num:
        raise ValueError("n_m cannot be smaller than min_num.")

    if len(mags) / n_m < 3:
        raise ValueError(
            "n_m is too large - less than three subsamples are created.")
    elif len(mags) / n_m < 25:
        if get_option("warnings") is True:
            warnings.warn(
                "The number of subsamples is less than 25. The normality "
                "assumption of the autocorrelation might not be valid.")

    if not isinstance(mags, np.ndarray):
        raise ValueError("Magnitudes must be an array.")
    if not isinstance(times, np.ndarray):
        raise ValueError("Times must be an array.")
    if len(mags) != len(times):
        raise ValueError("Magnitudes and times must have the same length.")

    # Estimate a and b values for n_m realizations.
    ac_1D = np.zeros(n_m)
    n = np.zeros(n_m)
    n_p = np.zeros(n_m)
    n_ms = np.zeros(n_m)

    for ii in range(n_m):
        # partition data
        idx, list_magnitudes = cut_constant_idx(
            mags, n_m, offset=ii
        )
        list_times = np.array_split(times, idx)
        list_mc = np.array_split(mc, idx)
        for jj, mc_loop in enumerate(list_mc):
            list_mc[jj] = float(max(mc_loop))

        # Make sure that data at the edges is not included if not enough
        # samples.
        if len(list_magnitudes[-1]) < n_m:
            list_magnitudes.pop(-1)
            list_times.pop(-1)
            list_mc.pop(-1)
        if len(list_magnitudes[0]) < n_m:
            list_magnitudes.pop(0)
            list_times.pop(0)
            list_mc.pop(0)

        # estimate b-values
        b_vec, _, n_m_loop = bs_from_partitioning(
            list_magnitudes, list_times, list_mc,
            delta_m, b_method=b_method, **kwargs)
        b_vec[n_m_loop < min_num] = np.nan

        # Estimate average events per b-value estimate.
        n_ms[ii] = np.mean(n_m_loop[n_m_loop >= min_num])
        # estimate autocorrelation (1D)
        ac_1D[ii], n[ii], n_p[ii], = est_morans_i(b_vec)

    # Estimate mean and (conservative) standard deviation of the
    # autocorrelation under H0.
    mac = np.nanmean(ac_1D)
    mean_n = np.nanmean(n)
    mean_np = np.nanmean(n_p)
    mu_mac = -1 / mean_n
    std_mac = 1 / np.sqrt(mean_np)

    return mac, mu_mac, std_mac
