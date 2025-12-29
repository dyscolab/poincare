from __future__ import annotations

import sys
import warnings
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.tsa.stattools
from matplotlib.patches import Rectangle
from scipy.fft import fft, fftfreq
from scipy.optimize import OptimizeResult, minimize_scalar
from scipy.signal import periodogram
from scipy.stats import linregress

from .._utils import logger
from ..types import Array1d


class LinregressResult(Protocol):
    slope: float
    intercept: float


def number_peaks(data: Array1d, n: int) -> int:
    """Determines the period size based on the number of peaks. This method is based on
    tsfresh's implementation of the same name:
    :func:`~tsfresh.feature_extraction.feature_calculators.number_peaks`.

    Calculates the number of peaks of at least support :math:`n` in the time series.
    A peak of support :math:`n` is defined as a subsequence where a value occurs, which
    is bigger than its :math:`n` neighbours to the left and to the right. The time
    series length divided by the number of peaks defines the period size.

    Parameters
    ----------
    data : array_like
        Time series to calculate the number of peaks of.
    n : int
        The required support for the peaks.

    Returns
    -------
    period_size : float
        The estimated period size.

    Examples
    --------

    Estimate the period length of a simple sine curve:

    See Also
    --------
    tsfresh.feature_extraction.number_peaks :
        tsfresh's implementation, on which this method is based on.
    """
    x_reduced = data[n:-n]

    res: np.ndarray | None = None
    for i in range(1, n + 1):
        result_first = x_reduced > _roll(data, i)[n:-n]

        if res is None:
            res = result_first
        else:
            res &= result_first

        res &= x_reduced > _roll(data, -i)[n:-n]
    n_peaks = np.sum(res)  # type: ignore
    if n_peaks < 1:
        return 1
    return data.shape[0] // n_peaks


def _roll(a: Array1d, shift: int) -> Array1d:
    """Exact copy of tsfresh's ``_roll``-implementation:
    https://github.com/blue-yonder/tsfresh/blob/611e04fb6f7b24f745b4421bbfb7e986b1ec0ba1/tsfresh/feature_extraction/feature_calculators.py#L49  # noqa: E501

    This roll is for 1D arrays and significantly faster than ``np.roll()``.

    Parameters
    ----------
    a : array_like
        input array
    shift : int
        the number of places by which elements are shifted

    Returns
    -------
    array : array_like
        shifted array with the same shape as the input array ``a``

    See Also
    --------
    https://github.com/blue-yonder/tsfresh/blob/611e04fb6f7b24f745b4421bbfb7e986b1ec0ba1/tsfresh/feature_extraction/feature_calculators.py#L49 :  # noqa: E501
        Implementation in tsfresh.
    """
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    idx = shift % len(a)
    return np.concatenate([a[-idx:], a[:-idx]])


def autoperiod(
    data: Array1d,
    timestep: float,
    *,
    pt_n_iter: int = 100,
    random_state: Any = None,
    detrend: bool = False,
    use_number_peaks_fallback: bool = False,
    number_peaks_n: int = 100,
    acf_hill_steepness: float = 0.0,
) -> tuple[float, bool]:
    """AUTOPERIOD method calculates the period in a two-step process. First, it
    extracts candidate periods from the periodogram (using an automatically
    determined power threshold, see ``pt_n_iter`` parameter). Then, it uses the circular
    autocorrelation to validate the candidate periods. Periods on a hill of the ACF
    with sufficient steepness are considered valid. The candidate period with the
    highest power is returned.

    Changes compared to the paper:

    - Potential detrending of the time series before estimating the period.
    - Potentially returns multiple detected periodicities.
    - Option to use the number of peaks method as a fallback if no periods are found.
    - Potentially exclude periods, whose ACF hill is not steep enough.

    Parameters
    ----------
    data : np.ndarray
        Array containing the data of a univariate, equidistant time series.
    pt_n_iter : int
        Number of shuffling iterations to determine the power threshold. The higher the
        number, the tighter the confidence interval. The percentile is calculated using
        :math:`percentile = 1 - 1 / pt\\_n\\_iter`.
    random_state : Any
        Seed for the random number generator. Used for determining the power threshold
        (data shuffling).
    detrend : bool
        Removes linear trend from the time series before calculating the candidate
        periods. (Addition to original method).
    use_number_peaks_fallback : bool
        If ``True`` and no periods are found, the number of peaks method is used as a
        fallback. (Addition to original method).
    number_peaks_n: int
        Number of peaks to return when using the number of peaks method as a fallback.
    acf_hill_steepness : float
        Minimum steepness of the ACF hill to consider a period valid. The higher the
        value, the steeper the hill must be. A value of ``0`` means that any hill is
        considered valid. The threshold is applied to the sum of the absolute slopes of
        the two fitted lines left and right of the candidate period.
    --------
    `<https://epubs.siam.org/doi/epdf/10.1137/1.9781611972757.40>`_ : Paper reference
    """
    result, verified = Autoperiod(
        pt_n_iter=pt_n_iter,
        random_state=random_state,
        detrend=detrend,
        use_number_peaks_fallback=use_number_peaks_fallback,
        number_peaks_n=number_peaks_n,
        acf_hill_steepness=acf_hill_steepness,
        plot=False,
        verbose=0,
        return_multi=1,
    )(data)

    return (result * timestep if result > 0 else -1, verified)


class Autoperiod:
    """AUTOPERIOD method to calculate the most dominant periods in a time series using
    the periodogram and the autocorrelation function (ACF).

    For more details, please see :func:`periodicity_detection.autoperiod`!

    Parameters
    ----------
    pt_n_iter : int
        Number of shuffling iterations to determine the power threshold. The higher the
        number, the tighter the confidence interval. The percentile is calculated using
        :math:`percentile = 1 - 1 / pt_n_iter`.
    random_state : Any
        Seed for the random number generator. Used for determining the power threshold
        (data shuffling).
    plot : bool
        Show the periodogram and ACF plots.
    verbose : int
        Controls the log output verbosity. If set to ``0``, no messages are printed;
        when ``>=3``, all messages are printed.
    detrend : bool
        Removes linear trend from the time series before calculating the candidate
        periods. (Addition to original method).
    use_number_peaks_fallback : bool
        If ``True`` and no periods are found, the number of peaks method is used as a
        fallback. (Addition to original method).
    number_peaks_n: int
        Number of peaks to return when using the number of peaks method as a fallback.
    return_multi : int
        Maximum number of periods to return.
    acf_hill_steepness : float
        Minimum steepness of the ACF hill to consider a period valid. The higher the
        value, the steeper the hill must be. A value of ``0`` means that any hill is
        considered valid. The threshold is applied to the sum of the absolute slopes of
        the two fitted lines left and right of the candidate period.
    """

    # potential improvement:
    # https://link.springer.com/chapter/10.1007/978-3-030-39098-3_4
    def __init__(
        self,
        *,
        pt_n_iter: int = 100,
        random_state: Any = None,
        plot: bool = False,
        verbose: int = 0,
        detrend: bool = False,
        use_number_peaks_fallback: bool = False,
        number_peaks_n: int = 100,
        return_multi: int = 1,
        acf_hill_steepness: float = 0.0,
    ):
        self._pt_n_iter = pt_n_iter
        self._rng: np.random.Generator = np.random.default_rng(random_state)
        self._plot = plot
        self._verbosity = verbose
        self._detrend = detrend
        self._use_np_fb = use_number_peaks_fallback
        self._np_n = number_peaks_n
        self._trend: Array1d | None = None
        self._orig_data: Array1d | None = None
        self._return_multi = return_multi
        self._acf_hill_steepness = acf_hill_steepness

    def __call__(self, data: Array1d) -> tuple[int, bool]:
        """Estimate the period length of a time series.

        Parameters
        ----------
        data : np.ndarray
            Array containing the data of a univariate, equidistant time series.

        Returns
        -------
        periods : Union[List[int], int]
            List of periods sorted by their power. If ``return_multi`` is set to ``1``,
            only the most dominant period is returned.
        """
        if self._detrend:
            logger.debug("Detrending")
            index = np.arange(data.shape[0])
            trend_fit: LinregressResult = linregress(index, data)  # type: ignore
            if trend_fit.slope > 1e-4:
                trend = trend_fit.intercept + index * trend_fit.slope
                if self._plot:
                    self._trend = trend
                    self._orig_data = data
                data = data - trend
                logger.debug(
                    f"removed trend with slope {trend_fit.slope:.6f} "
                    f"and intercept {trend_fit.intercept:.4f}",
                )
            else:
                logger.debug(
                    f"skipping detrending because slope ({trend_fit.slope:.6f}) "
                    f"is too shallow (< 1e-4)",
                )
                logger.debug(f"removing remaining mean ({data.mean():.4f})")
                data = data - data.mean()

        if self._verbosity > 1:
            logger.debug("Determining power threshold")
        p_threshold = self._power_threshold(data)
        logger.debug(f"Power threshold: {p_threshold:.6f}")

        if self._verbosity > 1:
            logger.debug("\nDiscovering candidate periods (hints) from periodogram")
        period_hints = self._candidate_periods(data, p_threshold)
        logger.debug(f"{len(period_hints)} candidate periods (hints)")

        if self._verbosity > 1:
            logger.debug("\nVerifying hints using ACF")
        periods, verified = self._verify(data, period_hints)

        if len(periods) < 1 or periods[0] <= 1 and self._use_np_fb:
            logger.debug(
                f"\nDetected invalid period ({periods}), "
                f"falling back to number_peaks method"
            )
            periods = [number_peaks(data, n=self._np_n)]
        logger.debug(f"Periods are {periods}")
        if self._return_multi > 1:
            return periods[: self._return_multi], verified
        else:
            return int(periods[0]), verified

    def _print(self, msg: str, level: int = 1) -> None:
        if self._verbosity >= level:
            print("  " * (level - 1) + msg, file=sys.stderr)

    def _power_threshold(self, data: Array1d) -> float:
        n_iter = self._pt_n_iter
        percentile = 1 - 1 / n_iter
        logger.debug(
            f"determined confidence interval as {percentile} "
            f"(using {n_iter} iterations)"
        )
        max_powers = []
        values = data.copy()
        for i in range(n_iter):
            self._rng.shuffle(values)
            _, p_den = periodogram(values)
            # p_den = np.abs(fft(values)) ** 2
            # print(p_den)
            max_powers.append(np.max(p_den))
        max_powers.sort()
        return max_powers[-1]

    def _candidate_periods(
        self, data: Array1d, p_threshold: float
    ) -> list[tuple[int, float, float]]:
        N = data.shape[0]
        f, p_den = periodogram(data)
        # f = fftfreq(len(data))
        # p_den = np.abs(fft(data)) ** 2
        # k are the DFT bin indices (see paper)
        k = np.array(f * N, dtype=np.int_)
        # print("k:", k)
        # print("frequency:", f)  # between 0 and 0.5
        # print("period:", N/k)

        logger.debug(
            f"inspecting periodogram between 2 and {N // 2} (frequencies 0 and 0.5)"
        )
        period_candidates: dict[int, tuple[int, float, float]] = {}
        removed_hints = 0
        for i in np.arange(2, N // 2):
            if p_den[i] > p_threshold:
                period = N // k[i]
                if period not in period_candidates:
                    period_candidates[period] = (k[i], f[i], p_den[i])
                    logger.debug(
                        f"detected hint at bin k={k[i]} (f={f[i]:.4f}, "
                        f"power={p_den[i]:.2f})"
                    )
                else:
                    removed_hints += 1
                    logger.debug(
                        f"detected hint at bin k={k[i]} (f={f[i]:.4f}, "
                        f"power={p_den[i]:.2f}) - skipped due to duplicated "
                        f"period ({period})"
                    )
        period_hints = list(period_candidates.values())

        # start with the highest power frequency:
        logger.debug("sorting hints by highest power first")
        period_hints = sorted(period_hints, key=lambda x: x[-1], reverse=True)

        if self._plot:
            plt.figure()
            plt.title("Periodogram")
            plt.semilogy(f, p_den, color="blue", label="PSD")
            plt.hlines(
                [p_threshold], f[0], f[-1], color="orange", label="power threshold"
            )
            plt.plot(
                [p[1] for p in period_hints],
                [p[2] for p in period_hints],
                "*",
                color="red",
                label="period hints",
            )
            plt.xlabel("frequency")
            plt.ylabel("PSD")
            plt.legend()
            # plt.show()

        return period_hints

    def _verify(
        self, data: Array1d, period_hints: list[tuple[int, float, float]]
    ) -> tuple[list[int], bool]:
        # produces wrong acf:
        # acf = fftconvolve(data, data[::-1], 'full')[data.shape[0]:]
        # acf = acf / np.max(acf)

        # Using statsmodels because circular autocorrelation function is needed,
        # scipy's and numpy's were tested and didn't work as well.
        acf = statsmodels.tsa.stattools.acf(data, fft=True, nlags=data.shape[0])

        assert isinstance(acf, np.ndarray)
        index = np.arange(acf.shape[0])
        N = data.shape[0]
        ranges = []

        warnings.filterwarnings(
            action="ignore",
            category=RuntimeWarning,
            message=r".*invalid value encountered.*",
        )
        for k, f, power in period_hints:
            if k < 2:
                logger.debug(f"processing hint at {N // k}: k={k}, f={f}")
                logger.debug("k < 2 --> INVALID")
                continue

            # determine search interval
            begin = int((N / (k + 1) + N / k) / 2) - 1
            end = int((N / k + N / (k - 1)) / 2) + 1
            while end - begin < 4:
                if begin > 0:
                    begin -= 1
                if end < N - 1:
                    end += 1
            logger.debug(
                f"processing hint at {N // k}, k={k}: begin={begin}, end={end + 1}"
            )
            slopes = {}

            # Compute error of approximating acf in vicinity of period hints by two segments
            # to be determined by linear regression
            def two_segment(t: float, args: list[np.ndarray]) -> float:
                x, y = args
                t = int(np.round(t))
                left_slope: LinregressResult = linregress(x[:t], y[:t])  # type: ignore
                right_slope: LinregressResult = linregress(x[t:], y[t:])  # type: ignore
                slopes[t] = (left_slope, right_slope)
                error = np.sum(
                    np.abs(y[:t] - (left_slope.intercept + left_slope.slope * x[:t]))
                ) + np.sum(
                    np.abs(y[t:] - (right_slope.intercept + right_slope.slope * x[t:]))
                )
                return error

            # print("outer indices", begin, end+1)
            # print("inner indices", 0, end - begin + 1)
            # print("bounds", 2, end - begin - 2)
            res = minimize_scalar(
                two_segment,
                args=[index[begin : end + 1], acf[begin : end + 1]],
                method="bounded",
                bounds=(2, end - begin - 2),
                options={
                    "disp": 1 if self._verbosity > 2 else 0,
                    "xatol": 1e-8,
                    "maxiter": 500,
                },
            )

            assert isinstance(res, OptimizeResult)
            if not res.success:
                # logger.debug(f"curve fitting failed ({res.message}) --> INVALID")
                # continue
                raise ValueError(
                    "Failed to find optimal midway-point for slope-fitting "
                    f"(hint: k={k}, f={f}, power={power})!"
                )

            t = int(np.round(res.x))
            optimal_t = begin + t
            slope = slopes[t]
            logger.debug(f"found optimal t: {optimal_t} (t={t})")

            # change from paper: we require a certain hill size to prevent noise
            # influencing our results:
            lslope = slope[0].slope
            rslope = slope[1].slope
            steepness = np.abs(lslope) + np.abs(rslope)
            if lslope < 0 < rslope:
                logger.debug("valley detected --> INVALID")

            elif steepness < self._acf_hill_steepness:
                logger.debug(
                    f"insufficient steepness ({np.abs(slope[0].slope):.4f} and "
                    f"{np.abs(slope[1].slope):.4f}) --> INVALID"
                )

            elif lslope > 0 > rslope:
                logger.debug(f"hill detected (steepness={steepness:.4f}) --> VALID")
                period = begin + np.argmax(acf[begin : end + 1])
                logger.debug(f"corrected period (from {N // k}): {period}")
                ranges.append((begin, end, optimal_t, period, slope))
                if self._return_multi <= 1:
                    break

            else:
                logger.debug("not a hill, but also not a valley --> INVALID")

        warnings.filterwarnings(
            action="default",
            category=RuntimeWarning,
            message=r".*invalid value encountered.*",
        )

        if self._plot:
            n_plots = 3 if self._detrend and self._trend is not None else 2
            fig, axs = plt.subplots(n_plots, 1, sharex="col")
            axs[0].set_title("Original time series")
            axs[0].set_xlabel("time")

            if self._trend is not None and self._orig_data is not None:
                axs[0].plot(self._orig_data, label="time series", color="blue")
                axs[0].plot(self._trend, label="linear trend", color="black")
                axs[1].set_title("Detrended time series")
                axs[1].plot(data, label="time series", color="black")
                axs[1].set_xlabel("time")
            else:
                axs[0].plot(data, label="time series", color="black")

            axs[-1].set_title(f"Circular autocorrelation ({len(ranges)} valid periods)")
            axs[-1].plot(acf, label="ACF", color="blue")
            data_min = acf.min()
            data_max = acf.max()
            for i, (b, e, t, period, (slope1, slope2)) in enumerate(ranges):
                axs[-1].add_patch(
                    Rectangle(
                        (b, data_min),
                        e - b,
                        data_max - data_min,
                        color="yellow",
                        alpha=0.5,
                    )
                )
                axs[-1].plot(period, acf[period], "*", color="red", label="period")
                axs[-1].plot(
                    index[b:t],
                    slope1.intercept + slope1.slope * index[b:t],
                    color="orange",
                    label="fitted slope1",
                )
                axs[-1].plot(
                    index[t : e + 1],
                    slope2.intercept + slope2.slope * index[t : e + 1],
                    color="orange",
                    label="fitted slope2",
                )

            axs[-1].set_xlabel("lag (period size)")
            axs[-1].set_ylabel("magnitude")
            axs[-1].legend()
            # plt.show()

        periods = list(x[3] for x in ranges)
        if len(periods) > 0:
            verified = True
            return list(
                np.unique(periods)[::-1][: self._return_multi].astype(int)
            ), verified
        elif len(period_hints) > 0:
            verified = False
            powers = np.transpose(np.array(period_hints))[2]
            peak = np.argmax(powers)
            return [np.round(1 / period_hints[peak][1])], verified
        else:
            verified = False
            return [-1], verified


# Function mainly for debugging purposes, simplest possible period finder
# to test if problem is with caller or with method
def fft_peak(data: Array1d, timestep: float) -> tuple[Any, bool]:
    amplitudes = np.abs(fft(data))
    freqs = fftfreq(len(data), d=timestep)

    # Boolean indicates if the period is verified, expected by caller
    return 1 / freqs[np.argmax(amplitudes)], False
