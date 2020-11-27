from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def linear_regression():
    global y_pred
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(test_x)
    print(regressor.score(X_train, y_train))


def lasso_regression():
    global y_pred
    regressor = Lasso(alpha=0.01, fit_intercept=False, normalize=True) #1173
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(test_x)


def ridge_regression():
    global y_pred
    regressor = Ridge(alpha=0.01, fit_intercept=False, normalize=True)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(test_x)
    print(regressor.score(X_train, y_train))


dataset = pd.read_csv('D:\Study\ML\Final_Project\Sources\Datasets\Train_data.csv')
test_data = pd.read_csv('D:\Study\ML\Final_Project\Sources\Datasets\Test_data.csv')
# test = dataset.iloc[21:25]
test_x = test_data.iloc[:, :-5]
test_y = test_data.iloc[:, 2]
print(test_x, test_y, dataset)
# dataset = dataset.iloc[0:10000]
X_train = dataset.iloc[:, :-5]
y_train = dataset.iloc[:, 2]
# pca = PCA()
# pca.fit(X_train)
# print(pca.components_)
# lasso_regression()
linear_regression()
# ridge_regression()
actual_values = test_y.to_numpy()
success = 0
for i, pred in enumerate(y_pred):
    actual_class = None
    for index, val in enumerate(actual_values[i]):
        if val == 1:
            actual_class = index
    # if (actual_values[i] >= 0.5 and pred >= 0.5) or (actual_values[i] < 0.5 and pred < 0.5):
    #     success += 1
    if np.argmax(pred) == actual_class:
        success += 1

print(round(success * 100 / len(actual_values), 2), '% accuracy')
