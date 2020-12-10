import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

cat_images = np.random.randn(700, 2) + np.array([0, -3])
mouse_images = np.random.randn(700, 2) + np.array([3, 3])
dog_images = np.random.randn(700, 2) + np.array([-3, 3])

feature_set_old = np.vstack([cat_images, mouse_images, dog_images])

labels = np.array([0] * 700 + [1] * 700 + [2] * 700)

one_hot_labels_old = np.zeros((2100, 3))

for i in range(2100):
    one_hot_labels_old[i, labels[i]] = 1

# plt.figure(figsize=(10, 7))
# plt.scatter(feature_set_old[:, 0], feature_set_old[:, 1], c=labels, cmap='plasma', s=100, alpha=0.5)
# plt.show()

# ------------------------------------
dataset = pd.read_csv('D:\\Study\\ML\\Final_Project\\Sources\\Datasets\\Train_data.csv')
test_data = pd.read_csv('D:\\Study\\ML\\Final_Project\\Sources\\Datasets\\Test_data.csv')
test_x = test_data.iloc[:, :-5]
test_y = test_data.iloc[:, -5:]
# print(test_x, test_y, dataset)
X_train = dataset.iloc[:, :-5]
y_train = dataset.iloc[:, -5:]

feature_set = np.array(X_train)
one_hot_labels = np.array(y_train)

sc = StandardScaler()
sc.fit(feature_set)
feature_set = sc.transform(feature_set)


# -----------------------------------------

def sigmoid(signal):
    # Prevent overflow.
    signal = np.clip(signal, -500, 500)

    # Calculate activation signal
    signal = 1.0 / (1 + np.exp(-signal))

    return signal


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(z):
    a = np.maximum(0, z)
    return a


def relu_der(z):
    return np.greater(z, 0).astype(int)


def softmax(A):
    A = np.clip(A, -500, 500)
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


instances = feature_set.shape[0]
attributes = feature_set.shape[1]
hidden_nodes = 10
output_labels = 5

wh = np.random.randn(attributes, hidden_nodes)
bh = np.random.randn(hidden_nodes)

wo = np.random.randn(hidden_nodes, output_labels)
bo = np.random.randn(output_labels)
lr = 10e-4

error_cost = []

for epoch in range(1000):
    ############# feedforward

    # Phase 1
    zh = np.dot(feature_set, wh) + bh
    ah = sigmoid(zh)

    # Phase 2
    zo = np.dot(ah, wo) + bo
    ao = softmax(zo)

    ########## Back Propagation

    ########## Phase 1

    dcost_dzo = ao - one_hot_labels
    dzo_dwo = ah

    dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

    dcost_bo = dcost_dzo

    ########## Phases 2

    dzo_dah = wo
    dcost_dah = np.dot(dcost_dzo, dzo_dah.T)
    dah_dzh = sigmoid_der(zh)
    dzh_dwh = feature_set
    dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

    dcost_bh = dcost_dah * dah_dzh

    # Update Weights ================

    wh -= lr * dcost_wh
    bh -= lr * dcost_bh.sum(axis=0)

    wo -= lr * dcost_wo
    bo -= lr * dcost_bo.sum(axis=0)

    if epoch % 10 == 0:
        loss = np.sum(-one_hot_labels * np.log(ao))
        print('Loss function value: ', loss)
        error_cost.append(loss)
