"""
This program trains linear regression models that aim to
predict the percentage of low income household.
Author: Yi Ding
Version: 1.0
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

N = 10      # the number of subsets to be obtained from partition
R = 5       # the number of repetitions for repeated N-fold cross validation

def normalize(data):
    """
    This function normalizes the given data into the range [0,1]
    where input is expected to be a pandas series.
    Formula: (x - minimum) / range
    """
    highest = max(data)
    lowest = min(data)
    
    normalized_data = (data - lowest) / (highest - lowest)
    return normalized_data


df = pd.read_csv('communities_modified.csv')

X_COLS = [
    'Requires assistance with core activities, %',
    'Did not complete year 12, %',
    'Holds degree or higher, %',
    'ARIA+ (avg)',
    '2012 ERP age 70+, %'
]
Y_COL = 'Equivalent household income <$600/week, %'
random_states = [i for i in range(1449054, 1449054 + R)]
print(f"The standard deviation for {Y_COL}: {df[Y_COL].std()}")

# normalization to prevent any feature from overweighting
df[X_COLS] = df[X_COLS].apply(normalize)

MSE_total = 0
# repeat R times
for i in range(R):

    # randomly partition into 10 subsets
    kf = KFold(n_splits=N, shuffle=True, random_state=random_states[i])

    # split data into training and testing set
    for _, (train_index, test_index) in enumerate(kf.split(df)):
        train = df.iloc[train_index]
        test = df.iloc[test_index]

        # split into features and labels
        # for both training and testing set
        X_train = train[X_COLS]
        y_train = train[Y_COL]
        X_test = test[X_COLS]
        y_test = test[Y_COL]

        # create and fit linear regression model
        lm = LinearRegression()
        lm.fit(X_train, y_train)

        # make predictions with the testing set
        y_pred = lm.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        MSE_total += mse    # sum up mean squared error

# obtain a conservative measure for the
# overall MSE of linear regression models trained
print('Average MSE for all models:', MSE_total / (N * R))

# train a final model using all the data as training data
lm = LinearRegression()
lm.fit(df[X_COLS], df[Y_COL])

# report the intercepts and coefficients of this final model
intercept = lm.intercept_
coefficients = lm.coef_
print('Final model:')
print('Intercept', intercept)
print('Coefficients', coefficients)

# train a model using only 1 feature for visualization purpose
train_index, test_index = next(kf.split(df))
train = df.iloc[train_index]
test = df.iloc[test_index]

X_train_raca = train[X_COLS[0]].values.reshape(-1, 1)
y_train = train[Y_COL]
X_test_raca = test[X_COLS[0]].values.reshape(-1, 1)
y_test = test[Y_COL]

lm_raca = LinearRegression()
lm_raca.fit(X_train_raca, y_train)

y_pred_raca = lm_raca.predict(X_test_raca)

# plot the scatter plot with line of regression
plt.scatter(X_test_raca, y_test)
plt.plot(X_test_raca, y_pred_raca, color='red')
plt.title('Test set scatter plot')
plt.xlabel('Require assistance with core activities, %')
plt.ylabel('Households with equivalent income < $600/week, %')
plt.savefig("scatter_linear_regression.png")

