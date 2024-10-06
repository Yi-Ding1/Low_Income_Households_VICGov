"""
This program computes the mean and standard deviation for
individual features and summarize them into a table.
Author: Yi Ding
Version: 1.0
"""

import pandas as pd

DECIMALS = 3  # the number of decimal places for all results

df = pd.read_csv('communities_modified.csv')

# list of features to investigate
cols = [
    'Requires assistance with core activities, %',
    'Did not complete year 12, %',
    'Holds degree or higher, %',
    'ARIA+ (avg)',
    '2012 ERP age 70+, %',
    'Equivalent household income <$600/week, %'
]

for col in cols:
    series = df[col]

    # find descriptive statistics
    median = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]

    # output all the results
    print(f"Feature '{col}' descriptive statistics:")
    print(f"Median: {round(median, DECIMALS)}")
    print(f"Q1: {round(q1, DECIMALS)}")
    print(f"Q3: {round(q3, DECIMALS)}")
    print(f"IQR: {round(iqr, DECIMALS)}")
    print(f"Number of Outliers: {len(outliers)}")
    print("-" * 50)

