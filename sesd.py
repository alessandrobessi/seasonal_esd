from typing import Tuple
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import STL

import matplotlib.pyplot as plt

def compute_test_statistic_one_sided(
    ts: np.array, hybrid: bool, one_sided_type: str
) -> Tuple[float, float, int]:
    """
    Calculate the test statistic (one-sided).
    Args:
        ts: The time series to compute the test statistic on.
        hybrid: A boolean that determines the type of z-score.
    Returns:
        The top z-score, the index of the top z-score, and the value related to the top z-score.
    """

    if one_sided_type == "min":
        if hybrid:
            median = np.ma.median(ts)
            mad = np.ma.median(np.abs(ts - median))
            score = (median - np.min(ts)) / mad
        else:
            score = (np.ma.mean(ts) - np.min(ts)) / np.ma.std(ts, ddof=1)

        idx = np.argmin(ts)
        return score, np.min(ts), idx

    if one_sided_type == "max":
        if hybrid:
            median = np.ma.median(ts)
            mad = np.ma.median(np.abs(ts - median))
            score = (np.max(ts) - median) / mad
        else:
            score = (np.max(ts) - np.ma.mean(ts)) / np.ma.std(ts, ddof=1)

        idx = np.argmax(ts)
        return score, np.max(ts), idx


def compute_test_statistic_two_sided(
    ts: np.array, hybrid: bool
) -> Tuple[float, float, int]:
    """
    Calculate the test statistic (two-sided).
    Args:
        ts: The time series to compute the test statistic on.
        hybrid: A boolean that determines the type of z-score.
    Returns:
        The top z-score, the index of the top z-score, and the value related to the top z-score.
    """

    if hybrid:
        median = np.ma.median(ts)
        mad = np.ma.median(np.abs(ts - median))
        scores = np.abs((ts - median) / mad)
    else:
        scores = np.abs((ts - np.ma.mean(ts)) / np.ma.std(ts, ddof=1))

    idx = np.argmax(scores)
    return scores[idx], ts[idx], idx


def compute_critical_value(size: int, alpha: float, two_sided: bool) -> float:
    """
    Calculate the critical value with the formula given for example in
    https://en.wikipedia.org/wiki/Grubbs%27_test_for_outliers#Definition
    Args:
        size: The current size of the time series.
        alpha: The significance level.
        two_sided: A boolean that determines whether the test is two-sided or not.
    Returns:
        The critical value for this test.
    """

    if two_sided:
        p = 1 - (alpha / (2 * (size + 1)))
    else:
        p = 1 - (alpha / (size + 1))

    df = size - 1
    icf = stats.t.ppf(p, df)

    numerator = size * icf
    denominator = np.sqrt((df + icf**2) * (size + 1))
    critical_value = numerator / denominator

    return critical_value


def generalized_esd(
    ts: np.array,
    alpha: float,
    max_anomalies: int,
    hybrid: bool,
    two_sided: bool,
    one_sided_type: str = "min",
) -> np.array:

    """
    Compute the Extreme Studentized Deviate of a time series.
    A Grubbs Test is performed max_anomalies times with the caveat
       that each time the top value is removed. For more details visit
       http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm
    Args:
        ts: The time series to compute the ESD on.
        max_anomalies: The number of times the Grubbs' Test will be applied to the ts.
        alpha: The significance level.
        hybrid: A boolean that determines the type of z-score.
        two_sided: A boolean that determines whether the test is two-sided or not.
        one_sided_type: In case two_sided is False, this boolean determines whether to consider min or max anomalies.
    Returns:
        The indices of the anomalies in the time series, the values of the anomalies in the time series.
    """

    test_type = "two-sided" if two_sided else "one-sided (" + one_sided_type + ")"
    is_hybrid = "(hybrid)" if hybrid else ""
    print(
        f"""
    Repeated {test_type} Grubbs test {is_hybrid}

    H0:  there are no anomalies in the data
    Ha:  there are up to {max_anomalies} anomalies in the data

    Significance level:  α = {alpha}
    Critical region:  Reject H0 if R > C (critical value)
    """
    )
    # A masked array is needed to ignore outliers in subsequent ESD tests.
    ts = np.ma.array(ts)

    n = len(ts)
    num_anomalies = 0
    anomalies_indices = []

    for i in range(1, max_anomalies + 1):
        if two_sided:
            test_statistic, value, idx = compute_test_statistic_two_sided(ts, hybrid)
        else:
            test_statistic, value, idx = compute_test_statistic_one_sided(
                ts, hybrid, one_sided_type
            )

        anomalies_indices.append(idx)
        ts[idx] = np.ma.masked

        critical_value = compute_critical_value(n - i, alpha, two_sided)

        print(
            f"exact number of anomalies={i}\tR={test_statistic:.3f}\tC={critical_value:.3f}",
            end=" ",
        )

        if test_statistic > critical_value:
            print("*")
            num_anomalies = i
        else:
            print()

    return np.array(anomalies_indices[:num_anomalies])


def seasonal_esd(
    ts: np.array,
    periodicity: int = None,
    hybrid: bool = False,
    max_anomalies: int = 10,
    alpha: float = 0.05,
    two_sided: bool = True,
    one_sided_type: str = "min",
) -> np.array:

    """
    Compute the Seasonal Extreme Studentized Deviate of a time series.
    The steps taken are first to to decompose the time series into STL
    decomposition (trend, seasonality, residual). Then, calculate
    the the median and perform a regular ESD test on the residual, which
    we calculate as:
                    R = ts - seasonality - median
    Note: The statsmodel library requires a seasonality to compute the STL
    decomposition, hence the parameter seasonality. If none is given,
    then it will automatically be calculated to be 20% of the total
    timeseries.

    Args:
        ts: The timeseries to compute the ESD on.
        periodicity: Number of time points for a season.
        hybrid: A boolean that determines the type of z-score.
        max_anomalies: The number of times the Grubbs' Test will be applied to the time series.
        alpha: The significance level.
        two_sided: A boolean that determines whether the test is two-sided or not.
        one_sided_type: In case two_sided is False, this boolean determines whether to consider min or max anomalies.
    Returns:
        The indices of the anomalies in the timeseries, the values of the anomalies in the time series.
    """

    if max_anomalies >= len(ts) / 2:
        ValueError(
            "The maximum number of anomalies must be less than half the size of the time series."
        )

    if alpha < 0 or alpha > 1:
        ValueError("alpha should be a float between 0 and 1.")

    if one_sided_type not in ("min", "max"):
        ValueError("one_sided_type should be either max or min.")

    # Seasonality is 20% of the ts if not given.
    period = periodicity or int(0.2 * len(ts))
    stl = STL(ts, period=period, robust=True)
    decomposition = stl.fit()
    residual = ts - decomposition.seasonal - np.median(ts)

    anomalies_indices = generalized_esd(
        residual,
        max_anomalies=max_anomalies,
        alpha=alpha,
        hybrid=hybrid,
        two_sided=two_sided,
        one_sided_type=one_sided_type,
    )

    return anomalies_indices
