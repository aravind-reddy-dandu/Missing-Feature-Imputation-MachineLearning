import numpy as np
import pandas as pd
from pandas import CategoricalDtype

df = pd.read_csv('D:\\Study\\ML\\Final_Project\\dataset-har-PUC-Rio-ugulino\\Full_Data.csv', delimiter=';')
df['how_tall_in_meters'] = df['how_tall_in_meters'].apply(lambda x: int(x.replace(',', '')))
df['body_mass_index'] = df['body_mass_index'].apply(lambda x: float(x.replace(',', '.')))

df["user"] = df["user"].astype(CategoricalDtype(['debora', 'katia', 'wallace', 'jose_carlos']))
df = pd.concat([df, pd.get_dummies(df['user'], prefix='user')], axis=1)

df["gender"] = df["gender"].astype(CategoricalDtype(['Woman', 'Man']))
df = pd.concat([df, pd.get_dummies(df['gender'], prefix='gender')], axis=1)

df["class"] = df["class"].astype(CategoricalDtype(['sitting', 'sittingdown', 'standing', 'standingup', 'walking']))
df = pd.concat([df, pd.get_dummies(df['class'], prefix='class')], axis=1)

df.drop(['user'], axis=1, inplace=True)
df.drop(['gender'], axis=1, inplace=True)
df.drop(['class'], axis=1, inplace=True)

array = df.to_numpy()
np.random.shuffle(array)
train_data = array[:int(len(array) * 0.8)]
test_data = array[int(len(array) * 0.8):]
pd.DataFrame(train_data).to_csv("D:\\Study\\ML\\Final_Project\\Sources\\Datasets\\Train_data.csv", header=df.columns,
                                index=False)
pd.DataFrame(test_data).to_csv("D:\\Study\\ML\\Final_Project\\Sources\\Datasets\\Test_data.csv", header=df.columns,
                               index=False)
