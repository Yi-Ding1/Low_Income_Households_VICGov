"""
This program produces heatmap, scatterplots, and computes mutual information
for the purpose of analyzing the relationship between the features.
Author: Yi Ding
Version: 1.0
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import normalized_mutual_info_score

def plot_heatmap(df, title, fname):
    """
    A function that produces heatmap based on a given dataframe
    """
    plt.rcParams['figure.figsize'] = (25, 20)
    plt.rc('font', size=20)

    # plot heatmap using Pearson correlation
    sns.heatmap(df[features].corr(method='pearson'), annot=True)
    plt.title(title)
    plt.yticks(rotation=0)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def plot_scatter(x, y, title, x_label, y_label, fname):
    """
    A function that produces scatterplot between two data series
    """
    plt.rcParams['figure.figsize'] = (18, 14)
    plt.rc('font', size=20)

    # plot scatter
    plt.scatter(x, y, color='blue', marker='o')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.savefig(f"{fname}.png")
    plt.close()


df = pd.read_csv('communities_modified.csv')
features = [
    'Requires assistance with core activities, %',
    'Did not complete year 12, %',
    'Holds degree or higher, %',
    '2012 ERP age 70+, %',
    'Equivalent household income <$600/week, %'
]

# produce graphs
plot_scatter(df['Did not complete year 12, %'],
             df['Equivalent household income <$600/week, %'],
             title='Low-income Households versus Population of People Who did not Complete Y12, %',
             x_label='Did not complete year 12, %',
             y_label='Equivalent household income <$600/week, %',
             fname="Scatter_for_Did_Not_Complete_Y12.png")

plot_heatmap(df[features],
             title="Pearson Correlation for Investigated Features of Victorian Communities",
             fname="Heatmap_for_Pearson_Correlation.png")

# discretization for the response variable
equal_width = KBinsDiscretizer(n_bins=3,
                               encode='ordinal', 
                               strategy='uniform')

# compute the normalized mutual information between remoteness and households in poverty
df['binned_low_income_households'] = equal_width.fit_transform(df[['Equivalent household income <$600/week, %']]).astype(int)
MI = normalized_mutual_info_score(df['binned_low_income_households'],
                                  df['ABS remoteness category'],
                                  average_method='min')
print("The mutual information between 'Equivalent household income <$600/week, %' and 'ARIA+ (avg)' is approximately:", MI)


