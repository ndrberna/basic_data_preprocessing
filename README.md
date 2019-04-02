# basic_data_preprocessing

How to use Python, Pandas, Numpy, and Scikit-Learn to do some basic data cleaning and preprocessing
Source:  *The complete beginner's guide to data cleaning and preprocessing* [](https://towardsdatascience.com/the-complete-beginners-guide-to-data-cleaning-and-preprocessing-2070b7d4c6d)

# Import the libraries
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

# Import the dataset
dataset = pd.read_csv('my_data.csv')

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 3].values


# Take care of missing data
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)

imputer = imputer.fit(X[:, 1:3])

X[:, 1:3] = imputer.transform(X[:, 1:3])


# Encode categorical data
# Encode the independent variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features = [0])

X = onehotencoder.fit_transform(X).toarray()

# Encode the dependent variable

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)

print(y)

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)'''


