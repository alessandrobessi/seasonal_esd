# Anomaly Detection: Seasonal ESD

<h4><p align="center">Disclaimer</p></h4>

> This code is built upon the one that you can find in Nacho Navarro's [repository](https://github.com/nachonavarro/seasonal-esd-anomaly-detection). I have extended that code to:
> -  add the possibility to perform one-sided tests, both for positive and negative anomalies;
> - use sample standard deviations (`np.std(x, ddof=1)`).

### Introduction
Seasonal ESD is an anomaly detection algorithm implemented at Twitter: https://arxiv.org/pdf/1704.07706.pdf.

The algorithm uses the Extreme Studentized Deviate test (also known as [Grubbs Test](https://en.wikipedia.org/wiki/Grubbs%27s_test#Definition)) to calculate the anomalies. In fact, the novelty doesn't come in the fact that ESD is used, but rather on what it is tested.

The problem with the ESD test on its own is that it assumes a normal data distribution, while real world data can have a multimodal distribution. To circumvent this, STL decomposition is used. Any time series can be decomposed with STL decomposition into a seasonal, trend, and residual component. The key is that the residual has a unimodal distribution that ESD can test.

However, there is still the problem that extreme, spurious anomalies can corrupt the residual component. To fix it, the paper proposes to use the median to represent the "stable" trend, instead of the trend found by means of STL decomposition.

Finally, for data sets that have a high percentage of anomalies, the research papers proposes to use the median and Median Absolute Deviate (MAD) instead of the mean and standard deviation to compute the z-score. Using MAD enables a more consistent measure of central tendency of a time series with a high percentage of anomalies.

### Grubbs Test
#### Two-sided case
Grubbs's test is defined for the hypothesis:

$H_{0}$: There are no outliers in the data set
$H_{a}$: There is exactly one outlier in the data set

The Grubbs test statistic is defined as:
$$G = \frac{\max_{i-1,\dots,N}{|Y_{i} - \bar{Y}|}}{s}$$

where $\bar{Y}$ and $s$ denoting the sample mean and the sample standard deviation, respectively. The Grubbs test statistic is the largest absolute deviation from the sample mean in units of the sample standard deviation.

This is the two-sided test, for which the hypothesis of no outliers is rejected at significance level $\alpha$ if

$$ G > \frac{N - 1}{\sqrt{N}}\sqrt\frac{t^{2}_{\alpha / (2N), N-2}}{N - 2 + t^{2}_{\alpha/(2N),N-2} } $$

with $t^{2}_{\alpha / (2N), N-2}$ denoting the upper critical value of the t-distribution with $N-2$ degrees of freedom and a significance level of $\alpha/(2N)$.

#### One-sided case
The Grubbs test can also be defined as a one-sided test, replacing $\alpha/(2N)$ with $\alpha/N$. To test whether the minimum value is an outlier, the test statistic is

$$G = \frac{\bar{Y} - Y_{\min}}{s}$$

with $Y_{\min}$ denoting the minimum value. To test whether the maximum value is an outlier, the test statistic is

$$G = \frac{ Y_{\max} - \bar{Y}}{s}$$

with $Y_{\max}$ denoting the maximum value.

