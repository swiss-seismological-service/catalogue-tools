# standard
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal

# statistical
from scipy.stats import norm

# Own functions
from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.classic import ClassicBValueEstimator
from seismostats.analysis.b_significant import mac_1D_constant_nm


def plot_mc_vs_b(
    magnitudes: np.ndarray,
    mcs: np.ndarray,
    delta_m: float,
    b_method: BValueEstimator = ClassicBValueEstimator,
    confidence_intvl: float = 0.95,
    ax: plt.Axes | None = None,
    color: str = "blue",
    label: str | None = None,
    **kwargs,
) -> plt.Axes:
    """Plots the estimated b-value in dependence of the completeness magnitude.

    Args:
        magnitudes: magnitudes of the catalog
        mcs:        completeness magnitudes (list or numpy array)
        delta_m:    discretization of the magnitudes
        method:     method used for b-value estimation
        confidence_intvl:   confidence interval that should be plotted
        ax:         axis where figure should be plotted
        color:      color of the data
        label:      label of the data that will be put in the legend
        **kwargs:   Additional keyword arguments for the b-value
                estimator.

    Returns:
        ax that was plotted on
    """

    b_values = []
    b_errors = []
    estimator = b_method()

    for mc in mcs:
        estimator.calculate(
            magnitudes, mc=mc, delta_m=delta_m, **kwargs)
        b_values.append(estimator.b_value)
        b_errors.append(estimator.std)

    b_values = np.array(b_values)
    b_errors = np.array(b_errors)

    if ax is None:
        _, ax = plt.subplots()

    error_factor = norm.ppf((1 + confidence_intvl) / 2)
    ax.plot(mcs, b_values, "-o", color=color, label=label)
    ax.fill_between(
        mcs,
        b_values - error_factor * b_errors,
        b_values + error_factor * b_errors,
        alpha=0.2,
        color=color,
        linewidth=0,
    )
    ax.set_xlabel("Completeness magnitude $m_c$")
    ax.set_ylabel("b-value")
    ax.grid(True)
    if label is not None:
        ax.legend()

    return ax


def plot_b_series_constant_nm(
        mags: np.ndarray,
        delta_m: float,
        mc: np.ndarray,
        times: np.ndarray,
        n_m: int,
        min_num: float = 2,
        b_method: BValueEstimator = ClassicBValueEstimator,
        plot_technique: Literal['left', 'midpoint', 'right'] = 'right',
        x_variable: np.ndarray | None = None,
        confidence: float = 0.95,
        ax: plt.Axes | None = None,
        color: str = "blue",
        label: str | None = None,
        **kwargs,
) -> plt.Axes:
    """
    Plots the b-values estimated from a running window of n_m magnitudes.

    Args:
        mags:   magnitudes of the events. If x_variable is None, the magnitudes
            are assumed to be sorted in the dimension of interest.
        delta_m:    magnitude bin width
        mc:     completeness magnitude. If a single value is provided, it is
            used for all magnitudes. Otherwise, the individual completeness of
            each magnitude can be provided.
        times:  times of the events
        n_m:   number of magnitudes in each partition
        min_num:    minimum number of events from which a b-value is estimated.
            If the number of events is smaller, the b-value is set to np.nan
        b_method:   method to estimate the b-values
        plot_technique:    technique where to plot the b-values with respect to
            the x-variable. Options are 'left', 'midpoint', 'right'. If set to
            'right' (default), the b-value is plotted at the right edge. For
            time series, this is the most common choice, as it avoids optical
            illusions of the b-value predicting future seismicity.
        x_variable: values of the dimension of interest, along which the
            b-values should be plotted. It should be a 1D array with the same
            length as the magnitudes, e.g., the time of the events. If None,
            the b-values are plotted against the event index.
        confidence:    confidence interval that should be plotted. Default
            is 0.95 (i.e., the 95% confidence interval is plotted)
        ax:    axis where the plot should be plotted
        color: color of the data

    Returns:
        ax that was plotted on
    """

    if isinstance(mc, (float, int)):
        if min(mags) < mc:
            raise ValueError("The completeness magnitude is larger than the "
                             "smallest magnitude")
        mc = np.ones(len(mags)) * mc
    else:
        if any(mags < mc):
            raise ValueError("There are earthquakes below their respective "
                             "completeness magnitude")

    if n_m < min_num:
        raise ValueError("n_m cannot be smaller than min_num")

    if not isinstance(mags, np.ndarray):
        raise ValueError("mags must be an array")
    if not isinstance(times, np.ndarray):
        raise ValueError("times must be an array")
    if len(mags) != len(times):
        raise ValueError("mags and times must have the same length")

    if x_variable is None:
        x_variable = np.arange(len(mags))
    elif len(x_variable) != len(mags):
        raise ValueError(
            "x_variable must have the same length as magnitudes")
    else:
        idx_sort = np.argsort(x_variable)
        mags = mags[idx_sort]
        times = times[idx_sort]
        mc = mc[idx_sort]
        x_variable = x_variable[idx_sort]

    if ax is None:
        _, ax = plt.subplots()

    if plot_technique == 'left':
        idx_start = 0
    elif plot_technique == 'midpoint':
        idx_start = n_m // 2
    elif plot_technique == 'right':
        idx_start = n_m - 1
    else:
        raise ValueError(
            "plot_technique must be 'left', 'midpoint', or 'right'")

    # estimation
    estimator = b_method()
    b_values = np.ones(len(mags)) * np.nan
    std_bs = np.ones(len(mags)) * np.nan
    for ii in range(len(mags) - n_m + 1):
        # runnning window of n_m magnitudes
        mags_window = mags[ii:ii + n_m]
        times_window = times[ii:ii + n_m]

        # sort the magnitudes and times
        idx = np.argsort(times_window)
        mags_window = mags_window[idx]
        times_window = times_window[idx]

        # estimate the b-value
        estimator.calculate(mags_window, mc=mc[ii], delta_m=delta_m, **kwargs)
        if estimator.n < min_num:
            b_values[idx_start + ii] = np.nan
            std_bs[idx_start + ii] = np.nan
        else:
            b_values[idx_start + ii] = estimator.b_value
            std_bs[idx_start + ii] = estimator.std

    # plotting
    ax.plot(x_variable, b_values, color=color, label=label)
    error_factor = norm.ppf((1 + confidence) / 2)
    ax.fill_between(
        x_variable,
        b_values - error_factor * std_bs,
        b_values + error_factor * std_bs,
        alpha=0.2,
        color=adjust_color_brightness(color, 0.7),
        linewidth=0,
    )

    ax.set_ylabel("b-value")
    if label is not None:
        ax.legend()
    return ax


def plot_nm_vs_mac1D(
        magnitudes: np.ndarray,
        times: np.ndarray,
        mc: np.ndarray,
        delta_m: float,
        n_ms: np.ndarray | None = None,
        min_num: float = 2,
        b_method: BValueEstimator = ClassicBValueEstimator,
        x_variable: np.ndarray | None = None,
        p_threshold: float = 0.05,
        ax: plt.Axes | None = None,
        color: str = "blue",
        label: str | None = None,
        **kwargs,
):
    """
    Plots the mean autocorrelation (MAC) vs the number of magnitudes uased per
    sample, together with the chosen tresold. If the MAC is outside of the
    confidence interval, the null hypothesis of a constant b-value can be
    rejected.

    Source:
        Mirwald et. al. 2024, SRL, How to b-signicant when analysing b-value
        variations

    Args:
        magnitudes: Magnitudes of the events.
        times:      Times of the events.
        mc:         Completeness magnitude. If a single value is provided, it is
            used for all magnitudes. Otherwise, the individual completeness of
            each magnitude can be provided.
        delta_m:    Magnitude descretization.
        n_ms:       List of number of magnitudes used per sample. If None,
            the function will use an array of values that are increasing by
            10 within a range of reasonable values.
        min_num:    Minimum number of events from which a b-value is estimated.
        b_method:   Method to estimate the b-values.
        x_variable: values of the dimension of interest, along which the
            b-values should be plotted. It should be a 1D array with the same
            length as the magnitudes, e.g., the time of the events. If None,
            the b-values are plotted against the event index.
        p_threshold:    Threshold above which the null hypothesis of a constant
            b-value can be rejected.
        ax:         Axis where the plot should be plotted.
        color:      Color of the data.
        label:      Label of the data that will be put in the legend.
        **kwargs:   Additional keyword arguments for the b-value estimator.

    """
    if n_ms is None:
        n_ms = np.arange(20, len(magnitudes) / 25, 5).astype(int)
    if isinstance(mc, (float, int)):
        mc = np.ones(len(magnitudes)) * mc
    elif not isinstance(mc, np.ndarray):
        raise ValueError("mc must be a float, or numpy array.")
    if x_variable is None:
        x_variable = np.arange(len(magnitudes))
    else:
        if len(x_variable) != len(magnitudes):
            raise ValueError(
                "x_variable must have the same length as magnitudes.")
        # sort in the dimension of interest (x-variable)
        srt = np.argsort(x_variable)
        magnitudes = magnitudes[srt]
        times = times[srt]
        x_variable = x_variable[srt]
    if ax is None:
        _, ax = plt.subplots()

    # filter magnitudes
    idx = magnitudes >= mc - delta_m / 2
    magnitudes = magnitudes[idx]
    times = times[idx]
    mc = mc[idx]

    # estimate the MAC for each n_m
    mac = np.zeros(len(n_ms))
    mu_mac = np.zeros(len(n_ms))
    std_mac = np.zeros(len(n_ms))
    for ii, n_m in enumerate(n_ms):
        mac[ii], mu_mac[ii], std_mac[ii] = mac_1D_constant_nm(
            magnitudes, mc, delta_m, times, n_m, min_num=min_num,
            b_method=b_method, **kwargs)

    # plot the results
    plt.plot(n_ms, mac, color=color, marker='o', label=label)
    plt.fill_between(n_ms, mu_mac - 1.96 * std_mac, mu_mac + 1.96 * std_mac,
                     color=adjust_color_brightness(color, 0.7), alpha=0.2,
                     linewidth=0)
    plt.plot(n_ms, mu_mac, color=adjust_color_brightness(
        color, 1.3), linestyle='--')

    plt.xlabel('$n_m$')
    plt.ylabel('MAC')
    if label is not None:
        ax.legend()
    return ax


def adjust_color_brightness(color, factor=1.2):
    """
    Adjusts the brightness of a given Matplotlib color.

    Args:
        color: A valid Matplotlib color string (e.g., "blue", "#ff5733", "C1").
        factor: A float value that adjusts the brightness of the color. if < 1,
            the color is lightened; if > 1, the color is darkened.

    Returns:
        str: An adjusted hex color string.

    Examples:
        .. code-block:: python
        from seismostats.plots.statistical import adjust_color_brightness
        original_color = "blue"
        lighter_color = adjust_color_brightness(original_color, factor=0.7)
    """

    # Convert color name or hex to RGB (values between 0 and 1)
    rgb = mcolors.to_rgb(color)

    if factor < 1:
        # Lighten: Move RGB values closer to 1 (white)
        adjusted_rgb = tuple(c + (1 - c) * (1 - factor) for c in rgb)
    else:
        # Darken: Scale down RGB values
        adjusted_rgb = tuple(c / factor for c in rgb)

    # Convert back to hex
    return mcolors.to_hex(adjusted_rgb)
