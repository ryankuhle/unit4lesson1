import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(414) # create seed based on 414 starting point

# create 1000 test data sets
X = np.linspace(0, 15, 1000)
y = 3 * np.sin(X) + np.random.normal(1 + X, .2, 1000)

# create training and testing data sets
train_X, train_y = X[:700], y[:700]
test_X, test_y = X[700:], y[700:]

# create training and testing DataFrame
train_df = pd.DataFrame({'X': train_X, 'y': train_y})
test_df = pd.DataFrame({'X': test_X, 'y': test_y})

# plot training data with linear regression
m, b = np.polyfit(train_df['X'], train_df['y'], 1)
plt.plot(train_df['X'], train_df['y'], '.')
plt.plot(train_df['X'], m*train_df['X'] + b, '-')
plt.show()

# plot testing data with linear regression
m, b = np.polyfit(test_df['X'], test_df['y'], 1)
plt.plot(test_df['X'], test_df['y'], '.')
plt.plot(test_df['X'], m*test_df['X'] + b, '-')
plt.show()

# linear fit
poly_1 = smf.ols(formula='y ~ 1 + X', data=train_df).fit()


# quadratic fit
poly_1 = smf.ols(formula='y ~ 1 + X + I(X**2)', data=train_df).fit()


'''
Using mean squared error as a metric, compare the performance of different polynomial curves in the training set and in the testing set. Submit your project as overfitting.py.
'''
