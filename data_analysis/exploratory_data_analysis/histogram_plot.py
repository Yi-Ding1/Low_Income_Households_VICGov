"""
This program produces histograms for individual features.
Author: Yi Ding
Version: 1.0
"""

import pandas as pd
import matplotlib.pyplot as plt

def histogram_for_series(series, title, bins=15):
    """
    Creates a histogram for a given pandas series.
    """
    # Set the figure size
    plt.rcParams['figure.figsize'] = (18, 14)
    plt.rc('font', size=20)
    
    # Create a histogram
    plt.hist(series, bins=bins, edgecolor='black')
    
    # Add title and labels
    plt.title(f"Histogram for {title}")
    plt.xlabel(series.name)
    plt.ylabel('Frequency')
    
    # Save the plot and clear the figure
    plt.savefig(f"{title}_histogram.png")
    plt.clf()

df = pd.read_csv('communities_modified.csv')

# list of features to investigate
X_COLS = [
    'Requires assistance with core activities, %',
    'Did not complete year 12, %',
    'Holds degree or higher, %',
    'ARIA+ (avg)',
    '2012 ERP age 70+, %',
]
Y_COL = 'Equivalent household income <$600/week, %'
all_cols = X_COLS + [Y_COL]

# titles for the graph
titles = [
    'people requiring assistance with core activites',
    'people who did not complete year 12',
    'people who holds degree or higher',
    'ARIA+ remoteness rating',
    'people above the age of 70+',
    'households with income of less than $600 per week'
]

# iterate through all features and plot them
for i in range(len(all_cols)):
    col = all_cols[i]
    histogram_for_series(df[col], f"Victorian communities based on {titles[i]}")

