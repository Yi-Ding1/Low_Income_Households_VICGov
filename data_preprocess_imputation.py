'''
This program handles the imputation part of data preprocessing.
It takes the entire communities dataset, sums up the proportion
of elderly aged population and does imputation using the median
value.
Author: Yi (David) Ding
Version: 1.0
'''

import pandas as pd

df = pd.read_csv('communities.csv').drop_duplicates()

# sum up the proportion of the elderly population
elder_age_groups = [
    '2012 ERP age 70-74, %',
    '2012 ERP age 75-79, %',
    '2012 ERP age 80-84, %',
    '2012 ERP age 85+, %'
]
df['2012 ERP age 70+, %'] = df[elder_age_groups].sum(axis=1)

# features that we are trying to investigation
X_COLS = [
    'Community Name',
    'Requires assistance with core activities, %',
    'Did not complete year 12, %',
    'Holds degree or higher, %',
    'ARIA+ (avg)',
    'ABS remoteness category',
    '2012 ERP age 70+, %'
]
Y_COL = 'Equivalent household income <$600/week, %'
all_cols = X_COLS + [Y_COL]

# imputation with median value for columns with NaN
for col in all_cols:
    if df[col].isna().any():
        print(f"Column '{col}' has missing values.")
        median = df[col].median()
        df[col] = df[col].fillna(median)

# export the modified dataset
modified_df = df[all_cols]
modified_df.to_csv('communities_modified.csv', index=False)
