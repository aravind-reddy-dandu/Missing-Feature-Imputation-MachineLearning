import pandas as pd
import numpy as np
import matplotlib as plt


class LinearRegression:
    def __init__(self, X_matrix, Y_matrix, learning_rate=0.01, tot_iterations=1500):
        self.X = X_matrix
        self.Y = Y_matrix
        self.learning_rate = learning_rate
        self.iterations = tot_iterations

        self.num_samples = len(Y_matrix)
        self.num_features = X_matrix.shape[1]
        self.X = self.mean_center_normalize(self.X, self.num_samples)
        self.Y = Y_matrix[:, np.newaxis]
        self.weights = np.zeros((self.num_features + 1, 1))

    def mean_center_normalize(self, X_matrix, len_input):
        X_matrix = (X_matrix - np.mean(X_matrix, 0)) / np.std(X_matrix, 0)
        X_matrix = np.hstack((np.ones((len_input, 1)), X_matrix))
        return X_matrix

    def fit(self):
        for _ in range(self.iterations):
            d = self.X.T @ (self.X @ self.weights - self.Y)
            self.weights = self.weights - (self.learning_rate / self.num_samples) * d

        return self

    def get_error(self, X_matrix=None, Y_matrix=None):
        if X_matrix is None:
            X_matrix = self.X
        else:
            X_matrix = self.mean_center_normalize(X_matrix, X_matrix.shape[0])

        if Y_matrix is None:
            Y_matrix = self.Y
        else:
            Y_matrix = Y_matrix[:, np.newaxis]

        y_pred = X_matrix @ self.weights
        score = 1 - (((Y_matrix - y_pred) ** 2).sum() / ((Y_matrix - Y_matrix.mean()) ** 2).sum())

        return score

    def predict(self, X):
        return self.mean_center_normalize(X, X.shape[0]) @ self.weights

    def get_weights(self):
        return self.weights


dataset = pd.read_csv('D:\Study\ML\Final_Project\Sources\Datasets\Train_data.csv')
test_data = pd.read_csv('D:\Study\ML\Final_Project\Sources\Datasets\Test_data.csv')
test_x = test_data.iloc[:, :-5].drop(['weight'], axis=1)
test_y = test_data.iloc[:, 2]
print(test_x, test_y, dataset)
X_train = dataset.iloc[:, :-5].drop(['weight'], axis=1)
y_train = dataset.iloc[:, 2]
regressor = LinearRegression(X_train, np.asarray(y_train), learning_rate=0.1, tot_iterations=1000).fit()
print(regressor.get_error())
print(regressor.get_error(test_x, test_y))
y_pred = regressor.predict(test_x)
# missing_weight_df = pd.read_csv('D:\Study\ML\Final_Project\Sources\Datasets\Train_data_weights_missing.csv')
# missing_weight_df.insert(2, 'weight', y_pred)
# missing_weight_df.to_csv("D:\\Study\\ML\\Final_Project\\Sources\\Datasets\\Train_data_weights_imputed.csv", header=missing_weight_df.columns,
#                                 index=False)
actual_values = test_y.to_numpy()
success = 0
# for i, pred in enumerate(y_pred):
#     actual_class = None
#     for index, val in enumerate(actual_values[i]):
#         if val == 1:
#             actual_class = index
#     # if (actual_values[i] >= 0.5 and pred >= 0.5) or (actual_values[i] < 0.5 and pred < 0.5):
#     #     success += 1
#     if np.argmax(pred) == actual_class:
#         success += 1
# print(round(success * 100 / len(actual_values), 2), '% accuracy')
