# In[48]:
import PIL
import numpy as np
import pandas as pd
import matplotlib.image as mpimg


def rgb2gray(rgb):
    # return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])
    return np.dot(rgb[..., :3], [0.21, 0.72, 0.07])


# X = (hours sleeping, hours studying), y = test score of the student
X = np.array(([2, 9, 4, 5, 8], [1, 5, 6, 9, 6], [3, 6, 7, 10, 1]), dtype=float)
y = np.array(([92, 87], [86, 34], [89, 45]), dtype=float)
# X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
# y = np.array(([92], [86], [89]), dtype=float)
# data=pd.read_csv("celsiustofahrenheit.csv")

# X=data[:,0]


# scale units
X = X / np.amax(X, axis=0)  # maximum of X array
y = y / 100  # maximum test score is 100

path = 'D:\\Study\\Intro_AI\\Coloring_Assignment\\Images\\reduced.png'
rgba_image = PIL.Image.open(path)
rgb_image = rgba_image.convert('RGB')
img = np.array(rgb_image)
gray = rgb2gray(img)
left_actual = img[:, :int(img.shape[1] / 2), :]
left_half = gray[:, :int(img.shape[1] / 2)]
right_half = gray[:, int(img.shape[1] / 2):]
left_half = left_half.reshape(-1, 1)
left_actual = left_actual.reshape(-1, 3)

X = left_half/ 255
y = left_actual[:, 0][:, np.newaxis]/255
print(y.shape)


class NeuralNetwork(object):
    def __init__(self, input_size, output_size, hidden_size=None):
        # parameters
        self.inputSize = input_size
        self.outputSize = output_size
        self.hiddenSize = hidden_size
        if hidden_size is None:
            self.hiddenSize = self.inputSize

        # weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)  # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)  # (3x1) weight matrix from hidden to output layer

    def feedForward(self, X):
        # forward propogation through the network
        self.z = np.dot(X, self.W1)  # dot product of X (input) and first set of weights (3x2)
        self.z2 = self.sigmoid(self.z)  # activation function
        self.z3 = np.dot(self.z2, self.W2)  # dot product of hidden layer (z2) and second set of weights (3x1)
        output = self.sigmoid(self.z3)
        # output = self.z3
        return output

    def sigmoid(self, s, deriv=False):
        if (deriv == True):
            return s * (1 - s)
        return 1 / (1 + np.exp(-s))

    def backward(self, X, y, output):
        # backward propogate through the network
        self.output_error = y - output  # error in output
        self.output_delta = self.output_error * self.sigmoid(output, deriv=True)

        self.z2_error = self.output_delta.dot(
            self.W2.T)  # z2 error: how much our hidden layer weights contribute to output error
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True)  # applying derivative of sigmoid to z2 error

        self.W1 += X.T.dot(self.z2_delta)  # adjusting first set (input -> hidden) weights
        self.W2 += self.z2.T.dot(self.output_delta)  # adjusting second set (hidden -> output) weights

    def train(self, X, y):
        output = self.feedForward(X)
        self.backward(X, y, output)


NN = NeuralNetwork(1, 1, 5)

for i in range(1000):  # trains the NN 1000 times
    if (i % 100 == 0):
        print("Loss: " + str(np.mean(np.square(y - NN.feedForward(X)))))
    NN.train(X, y)

print("Input: " + str(X))
print("Actual Output: " + str(y))
print("Loss: " + str(np.mean(np.square(y - NN.feedForward(X)))))
print("\n")
print("Predicted Output: " + str(NN.feedForward(X)))

# In[51]:


p = NN.feedForward(X) * 255

# In[61]:


y = y * 255

# In[62]:


df = pd.DataFrame(data=p, columns=['p1'])

# In[63]:


print(df.head(1))

# In[64]:


df = pd.concat([df, pd.DataFrame(y)], axis=1)
print(df)
