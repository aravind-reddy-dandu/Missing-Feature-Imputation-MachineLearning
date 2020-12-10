import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


class NeuralNet():
    '''
    A two layer neural network
    '''

    def __init__(self, layers=None, learning_rate=0.001, iterations=100):
        if layers is None:
            layers = [13, 8, 1]
        self.params = {}
        self.layers = layers
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        self.X = None
        self.y = None

    def init_weights(self):
        '''
        Initialize the weights from a random normal distribution
        '''
        np.random.seed(1)  # Seed the random number generator
        self.params["W1"] = np.random.randn(self.layers[0], self.layers[1])
        self.params['b1'] = np.random.randn(self.layers[1], )
        self.params['W2'] = np.random.randn(self.layers[1], self.layers[2])
        self.params['b2'] = np.random.randn(self.layers[2], )

    def relu(self, Z):
        '''
        The ReLufunction performs a threshold
        operation to each input element where values less
        than zero are set to zero.
        '''
        return np.maximum(0, Z)

    def sigmoid(self, Z):
        '''
        The sigmoid function takes in real numbers in any range and
        squashes it to a real-valued output between 0 and 1.
        '''
        Z = np.clip(Z, -500, 500)
        return 1.0 / (1.0 + np.exp(-Z))

    def entropy_loss(self, y, yhat):
        nsample = len(y)
        loss = -1 / nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((1 - y), np.log(1 - yhat))))
        return loss

    def forward_propagation(self):
        '''
        Performs the forward propagation
        '''

        Z1 = self.X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        yhat = self.sigmoid(Z2)
        loss = self.entropy_loss(self.y, yhat)

        # save calculated parameters
        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['A1'] = A1

        return yhat, loss

    def back_propagation(self, yhat):
        '''
        Computes the derivatives and update weights and bias according.
        '''

        def dRelu(x):
            x[x <= 0] = 0
            x[x > 0] = 1
            return x

        dl_wrt_yhat = -(np.divide(self.y, yhat) - np.divide((1 - self.y), (1 - yhat)))
        dl_wrt_sig = yhat * (1 - yhat)
        dl_wrt_z2 = dl_wrt_yhat * dl_wrt_sig

        dl_wrt_A1 = dl_wrt_z2.dot(self.params['W2'].T)
        dl_wrt_w2 = self.params['A1'].T.dot(dl_wrt_z2)
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0)

        dl_wrt_z1 = dl_wrt_A1 * dRelu(self.params['Z1'])
        dl_wrt_w1 = self.X.T.dot(dl_wrt_z1)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0)

        # update the weights and bias
        self.params['W1'] = self.params['W1'] - self.learning_rate * dl_wrt_w1
        self.params['W2'] = self.params['W2'] - self.learning_rate * dl_wrt_w2
        self.params['b1'] = self.params['b1'] - self.learning_rate * dl_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * dl_wrt_b2

    def fit(self, X, y):
        '''
        Trains the neural network using the specified data and labels
        '''
        self.X = X
        self.y = y
        self.init_weights()  # initialize weights and bias

        for i in range(self.iterations):
            yhat, loss = self.forward_propagation()
            self.back_propagation(yhat)
            self.loss.append(loss)
            print(loss)

    def predict(self, X):
        '''
        Predicts on a test data
        '''
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        pred = self.sigmoid(Z2)
        return np.round(pred)

    def acc(self, y, yhat):
        '''
        Calculates the accutacy between the predicted valuea and the truth labels
        '''
        acc = int(sum(y == yhat) / len(y) * 100)
        return acc

    def plot_loss(self):
        '''
        Plots the loss curve
        '''
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("logloss")
        plt.title("Loss curve for training")
        plt.show()


dataset = pd.read_csv('D:\\Study\\ML\\Final_Project\\Sources\\Datasets\\Train_data.csv')
test_data = pd.read_csv('D:\\Study\\ML\\Final_Project\\Sources\\Datasets\\Test_data.csv')
test_x = test_data.iloc[:, :4]
test_y = test_data.iloc[:, [-6, -7]]
# print(test_x, test_y, dataset)
X_train = dataset.iloc[:, :4]
y_train = dataset.iloc[:, [-6, -7]]

feature_set = np.array(X_train)
one_hot_labels = np.array(y_train)
sc = StandardScaler()
sc.fit(feature_set)
feature_set = sc.transform(feature_set)

nn = NeuralNet(layers= [4, 8, 2])
nn.fit(feature_set, one_hot_labels)

nn.plot_loss()