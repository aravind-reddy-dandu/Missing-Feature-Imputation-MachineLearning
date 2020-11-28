import pandas as pd
import numpy as np

test_data = pd.read_csv('D:\Study\ML\Final_Project\Sources\Datasets\Test_data.csv')
test_data = test_data.drop('weight', axis= 1)
pd.DataFrame(test_data).to_csv("D:\\Study\\ML\\Final_Project\\Sources\\Datasets\\Train_data_weights_missing.csv", header=test_data.columns,
                                index=False)
print(test_data)