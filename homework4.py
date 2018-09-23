# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 06:48:43 2018
IE-598 Machine Learning Assignment_4
Raschka Chapter 10
@author: Haichao Bo
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures


#read data
df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-2nd-edition'
                 '/master/code/ch10/housing.data.txt',
                 header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print(df.head())


#part 1 EDA
summary = df.describe()
print(summary)
print("Number of Rows of Data = " + str(len(df)) + '\n')

cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2.5)
#sns.pairplot(df[df.columns], size=2.5)
plt.tight_layout()
plt.show()

#cm = np.corrcoef(df[cols].values.T)
cm = np.corrcoef(df[df.columns].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8},
                 yticklabels=df.columns, xticklabels=df.columns)
plt.show()

#data split 
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#part 2 
# Linear regression with one feature(not in the word file)
class LinearRegressionGD(object):
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def predict(self, X):
        return self.net_input(X)
  
X = df[['RM']].values
y = df['MEDV'].values
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

# fit with standardized data
lr = LinearRegressionGD()
lr.fit(X_std, y_std)

sns.reset_orig() # resets matplotlib style
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    return None

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')
plt.show()

num_rooms_std = sc_x.transform(5.0)
price_std = lr.predict(num_rooms_std)
print("Price in $1000s: %.3f" % sc_y.inverse_transform(price_std))

print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])


#fit with unstandardized data
slr = LinearRegression()
slr.fit(X, y)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)

lin_regplot(X, y, slr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')
plt.show()

'''
ransac = RANSACRegressor(LinearRegression(), max_trials=100, 
                         min_samples=50,loss='absolute_loss', residual_threshold=5.0,random_state=0)
ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask], c='steelblue', edgecolor='white', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], c='limegreen', edgecolor='white', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='black', lw=2)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper left')
plt.show()

print('Slope: %.3f' % ransac.estimator_.coef_[0])
print('Intercept: %.3f' % ransac.estimator_.intercept_)
'''
#part 2 linear regression(in the word file)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)

plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()


print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                                       mean_squared_error(y_test, y_test_pred)))

print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))



#part 3 Ridge Lasso ElasticNet
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=1.0)
elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)

alpha_space = np.logspace(-2, 0, 10)
ridge_coef = []
ridge_intercept = []
ridge_MSE_train = []
ridge_MSE_test = []
ridge_R2_train = []
ridge_R2_test = []
# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge = Ridge(alpha=alpha, normalize = True)
    ridge.fit(X_train, y_train)
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)
    ridge_coef.append(ridge.coef_[0])
    ridge_intercept.append(ridge.intercept_)
    ridge_MSE_train.append(mean_squared_error(y_train, y_train_pred))
    ridge_MSE_test.append(mean_squared_error(y_test, y_test_pred))
    ridge_R2_train.append(r2_score(y_train, y_train_pred))
    ridge_R2_test.append(r2_score(y_test, y_test_pred))

    plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
    plt.xlim([-10, 50])
    plt.show()
    
print("alpha    coef   intercept MSE_train  MSE_test  R2_train  R2_test")
for i in range(10):
    print('%.3f,   %.3f,  %.3f,   %.3f,   %.3f,    %.3f,   %.3f' % (alpha_space[i], ridge_coef[i], 
                                                                    ridge_intercept[i], ridge_MSE_train[i], 
                                                                    ridge_MSE_test[i], ridge_R2_train[i], 
                                                                    ridge_R2_test[i]))

alpha_space = np.logspace(-2, 0, 10)
lasso_coef = []
lasso_intercept = []
lasso_MSE_train = []
lasso_MSE_test = []
lasso_R2_train = []
lasso_R2_test = []
# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    lasso = Lasso(alpha=alpha, normalize = True)
    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)
    lasso_coef.append(lasso.coef_[0])
    lasso_intercept.append(lasso.intercept_)
    lasso_MSE_train.append(mean_squared_error(y_train, y_train_pred))
    lasso_MSE_test.append(mean_squared_error(y_test, y_test_pred))
    lasso_R2_train.append(r2_score(y_train, y_train_pred))
    lasso_R2_test.append(r2_score(y_test, y_test_pred))

    plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
    plt.xlim([-10, 50])
    plt.show()
    
print("alpha    coef   intercept MSE_train  MSE_test  R2_train  R2_test")
for i in range(10):
    print('%.3f,   %.3f,  %.3f,   %.3f,   %.3f,    %.3f,   %.3f' % (alpha_space[i], lasso_coef[i], 
                                                                    lasso_intercept[i], lasso_MSE_train[i], 
                                                                    lasso_MSE_test[i], lasso_R2_train[i], 
                                                                    lasso_R2_test[i]))
    
    
ratio_space = np.logspace(-2, 0, 10)
EN_coef = []
EN_intercept = []
EN_MSE_train = []
EN_MSE_test = []
EN_R2_train = []
EN_R2_test = []
# Compute scores over range of alphas
for ratio in ratio_space:

    # Specify the alpha value to use: ridge.alpha
    EN = ElasticNet(alpha=1.0, l1_ratio=ratio)
    EN.fit(X_train, y_train)
    y_train_pred = EN.predict(X_train)
    y_test_pred = EN.predict(X_test)
    EN_coef.append(EN.coef_[0])
    EN_intercept.append(EN.intercept_)
    EN_MSE_train.append(mean_squared_error(y_train, y_train_pred))
    EN_MSE_test.append(mean_squared_error(y_test, y_test_pred))
    EN_R2_train.append(r2_score(y_train, y_train_pred))
    EN_R2_test.append(r2_score(y_test, y_test_pred))
    
    plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
    plt.xlim([-10, 50])
    plt.show()

print("ratio    coef   intercept MSE_train  MSE_test  R2_train  R2_test")
for i in range(10):
    print('%.3f,   %.3f,  %.3f,   %.3f,   %.3f,    %.3f,   %.3f' % (ratio_space[i], EN_coef[i], 
                                                                    EN_intercept[i], EN_MSE_train[i], 
                                                                    EN_MSE_test[i], EN_R2_train[i], 
                                                                    EN_R2_test[i]))



print("alpha    coef   intercept MSE_train  MSE_test  R2_train  R2_test")
for i in range(10):
    print('%.3f,   %.3f,  %.3f,   %.3f,   %.3f,    %.3f,   %.3f' % (alpha_space[i], lasso_coef[i], 
                                                                    lasso_intercept[i], lasso_MSE_train[i], 
                                                                    lasso_MSE_test[i], lasso_R2_train[i], 
                                                                    lasso_R2_test[i]))

print("My name is Haichao Bo")
print("My NetID is: hbo2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
'''
X = np.array([ 258.0, 270.0, 294.0, 320.0, 342.0, 368.0, 396.0, 446.0, 480.0, 586.0])[:, np.newaxis]
y = np.array([ 236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 360.8, 368.0, 391.2, 390.8])
lr = LinearRegression()
pr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)

lr.fit(X, y)
X_fit = np.arange(250,600,10)[:, np.newaxis]
y_lin_fit = lr.predict(X_fit)

pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

plt.scatter(X, y, label='training points')
plt.plot(X_fit, y_lin_fit, label='linear fit', linestyle='--')
plt.plot(X_fit, y_quad_fit, label='quadratic fit')
plt.legend(loc='upper left')
plt.show()

y_lin_pred = lr.predict(X)
y_quad_pred = pr.predict(X_quad)
print('Training MSE linear: %.3f, quadratic: %.3f' % (mean_squared_error(y, y_lin_pred),
                                                      mean_squared_error(y, y_quad_pred)))
print('Training R^2 linear: %.3f, quadratic: %.3f' % (r2_score(y, y_lin_pred),
                                                      r2_score(y, y_quad_pred)))

X = df[['LSTAT']].values
y = df['MEDV'].values
regr = LinearRegression()
# create quadratic features
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)
# fit features
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))
regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))
regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))
# plot results
plt.scatter(X, y, label='training points', color='lightgray')
plt.plot(X_fit, y_lin_fit, label='linear (d=1), $R^2=%.2f$' % linear_r2, 
         color='blue', lw=2, linestyle=':')
plt.plot(X_fit, y_quad_fit, label='quadratic (d=2), $R^2=%.2f$' % quadratic_r2,
         color='red', lw=2, linestyle='-')
plt.plot(X_fit, y_cubic_fit, label='cubic (d=3), $R^2=%.2f$' % cubic_r2, 
         color='green', lw=2, linestyle='--')
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper right')
plt.show()

# transform features
X_log = np.log(X)
y_sqrt = np.sqrt(y)
# fit features
X_fit = np.arange(X_log.min()-1,
                  X_log.max()+1, 1)[:, np.newaxis]
regr = regr.fit(X_log, y_sqrt)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y_sqrt, regr.predict(X_log))
# plot results
plt.scatter(X_log, y_sqrt, label='training points', color='lightgray')
plt.plot(X_fit, y_lin_fit, label='linear (d=1), $R^2=%.2f$' % linear_r2,
         color='blue',lw=2)
plt.xlabel('log(% lower status of the population [LSTAT])')
plt.ylabel('$\sqrt{Price \; in \; \$1000s \; [MEDV]}$')
plt.legend(loc='lower left')
plt.show()
'''
