# basic_data_preprocessing

How to use Python, Pandas, Numpy, and Scikit-Learn to do some basic data cleaning and preprocessing
Source:  *The complete beginner's guide to data cleaning and preprocessing* [](https://towardsdatascience.com/the-complete-beginners-guide-to-data-cleaning-and-preprocessing-2070b7d4c6d)

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Basic data cleaning and preprocessing\n",
    "\n",
    "Here we'll use Numpy, Pandas, and Scikit-Learn to do some necessary basic cleaning and preprocessing of our data  so that we can use it in a machine learning model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 1 0 0 1 1 0 1 0 1]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'from sklearn.preprocessing import StandardScaler\\nsc_X = StandardScaler()\\nX_train = sc_X.fit_transform(X_train)\\nX_test = sc_X.transform(X_test)\\nsc_y = StandardScaler()\\ny_train = sc_y.fit_transform(y_train)'"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Import the libraries\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "\n",
    "# Import the dataset\n",
    "dataset = pd.read_csv('my_data.csv')\n",
    "X = dataset.iloc[:, :-1].values\n",
    "y = dataset.iloc[:, 3].values\n",
    "\n",
    "\n",
    "# Take care of missing data\n",
    "from sklearn.preprocessing import Imputer\n",
    "imputer = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)\n",
    "imputer = imputer.fit(X[:, 1:3])\n",
    "X[:, 1:3] = imputer.transform(X[:, 1:3])\n",
    "\n",
    "\n",
    "# Encode categorical data\n",
    "# Encode the independent variable\n",
    "from sklearn.preprocessing import LabelEncoder, OneHotEncoder\n",
    "labelencoder_X = LabelEncoder()\n",
    "X[:, 0] = labelencoder_X.fit_transform(X[:, 0])\n",
    "onehotencoder = OneHotEncoder(categorical_features = [0])\n",
    "X = onehotencoder.fit_transform(X).toarray()\n",
    "# Encode the dependent variable\n",
    "labelencoder_y = LabelEncoder()\n",
    "y = labelencoder_y.fit_transform(y)\n",
    "print(y)\n",
    "\n",
    "# Splitting the dataset into the Training set and Test set\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)\n",
    "\n",
    "# Feature Scaling\n",
    "'''from sklearn.preprocessing import StandardScaler\n",
    "sc_X = StandardScaler()\n",
    "X_train = sc_X.fit_transform(X_train)\n",
    "X_test = sc_X.transform(X_test)\n",
    "sc_y = StandardScaler()\n",
    "y_train = sc_y.fit_transform(y_train)'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
