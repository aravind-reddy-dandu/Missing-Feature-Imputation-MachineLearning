import pandas as pd
import numpy as np

# test_data = pd.read_csv('D:\Study\ML\Final_Project\Sources\Datasets\Test_data.csv')
# test_data = test_data.drop('weight', axis= 1)
# pd.DataFrame(test_data).to_csv("D:\\Study\\ML\\Final_Project\\Sources\\Datasets\\Train_data_weights_missing.csv", header=test_data.columns,
#                                 index=False)
# print(test_data)

train_data = pd.read_csv('D:\Study\ML\Final_Project\Sources\Datasets\Train_data.csv')
test_data = pd.read_csv('D:\Study\ML\Final_Project\Sources\Datasets\Test_data.csv')

without_x1 = train_data.drop(['z3'], axis=1)
without_x1['z3'] = train_data['z3']
without_x1.to_csv("D:\\Study\\ML\\Final_Project\\Sources\\Datasets\\Train_data_z3.csv",
                  header=without_x1.columns, index=False)
