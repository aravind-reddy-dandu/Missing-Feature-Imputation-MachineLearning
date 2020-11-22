import numpy as np
import pandas as pd

df = pd.read_csv('D:\\Study\\ML\\Final_Project\\dataset-har-PUC-Rio-ugulino\\Full_Data.csv', delimiter=';')
array = df.to_numpy()
np.random.shuffle(array)
train_data = array[:int(len(array)*0.8)]
test_data = array[int(len(array)*0.8):]
pd.DataFrame(train_data).to_csv("D:\\Study\\ML\\Final_Project\\dataset-har-PUC-Rio-ugulino\\Train_data.csv", index=False)
pd.DataFrame(test_data).to_csv("D:\\Study\\ML\\Final_Project\\dataset-har-PUC-Rio-ugulino\\Test_data.csv", index=False)

