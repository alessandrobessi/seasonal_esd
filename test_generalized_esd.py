import numpy as np
from sesd import generalized_esd

"""
    H0:  there are no outliers in the data
    Ha:  there are up to 10 outliers in the data

    Significance level:  α = 0.05
    Critical region:  Reject H0 if Ri > critical value
 
    Summary Table for Two-Tailed Test
    ---------------------------------------
            Exact           Test     Critical  
        Number of      Statistic    Value, λi  
      Outliers, i      Value, Ri          5 %  
    ---------------------------------------
              1          3.118          3.158  
              2          2.942          3.151  
              3          3.179          3.143 * 
              4          2.810          3.136  
              5          2.815          3.128  
              6          2.848          3.120  
              7          2.279          3.111  
              8          2.310          3.103  
              9          2.101          3.094  
             10          2.067          3.085  

Reference: 
https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm

"""

ts = np.array([-0.25, 0.68, 0.94, 1.15, 1.20, 1.26, 1.26,
              1.34, 1.38, 1.43, 1.49, 1.49, 1.55, 1.56,
              1.58, 1.65, 1.69, 1.70, 1.76, 1.77, 1.81,
              1.91, 1.94, 1.96, 1.99, 2.06, 2.09, 2.10,
              2.14, 2.15, 2.23, 2.24, 2.26, 2.35, 2.37,
              2.40, 2.47, 2.54, 2.62, 2.64, 2.90, 2.92,
              2.92, 2.93, 3.21, 3.26, 3.30, 3.59, 3.68,
              4.30, 4.64, 5.34, 5.42, 6.01])

generalized_esd(ts, alpha=0.05, max_anomalies=10, hybrid=False, two_sided=True)
