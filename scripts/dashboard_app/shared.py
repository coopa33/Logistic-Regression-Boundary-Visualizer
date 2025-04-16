import numpy as np
import pandas as pd
import os
import sys

rootpath = os.path.join(os.getcwd(), "..")
sys.path.append(rootpath)
from gradient_descent import gradient_descent
# df = pd.read_csv("../../data/titanic.csv")

# Testing array
df = np.array([[1, 2, 5, 2], 
              [0.6, 4, 2, 4],
              [7, 1.5, -2, 3]])

####################################
# Data Structuring Functionalities #
####################################
def dummies(feature):
    # Assuming that no feature contains multiple datatypes
    if feature.dtype == "object":
        structured_feature = pd.get_dummies(feature)
        return structured_feature
    elif feature.dtype == "int" and set(feature.unique()) != {0, 1}:
        # This assumes that discrete numericals are NOT ordinal!
        structured_feature = pd.get_dummies(feature)
        return structured_feature
    else:
        return None


def drop_features(df, features):
    df = df.copy(deep=True)
    df = df.drop(columns = features)
    return df





def read_structure_data(df, y_name, columns_to_drop, columns_to_one_hot):

    # Drop unwanted columns
    df = df.drop(columns = columns_to_drop)

    # Drop Na rows (must be completed before feature selection, on the complete dataframe)
    df = df.dropna(how="any")

    # Transform categorical features into one-hot
    col_names = df.columns
    for col in columns_to_one_hot:
        dummies = pd.get_dummies(df[col], prefix = col + "_")    
        df = df.drop(columns = col)
        df = pd.concat((df, dummies), axis = 1)

    # z-score normalize all columns except y
    y = df.pop(y_name)
    df = df.apply(lambda x: (x - np.mean(x))/np.std(x), axis = 0)

    # Transform into numpy arrays
    n_obs = len(y)
    y = np.array(y).reshape((n_obs, 1))
    X = np.array(df)

    # Reassemble Dataframe
    y_series = pd.Series(y.flatten())
    y_series.index = df.index
    df[y_name] = y_series

    return df




columns_to_drop = ["PassengerId", "Name", "Cabin", "Ticket", "Cabin"]
columns_to_one_hot = ["Pclass", "Sex", "Embarked"]
# Shared datastructures
df = pd.read_csv("../../data/titanic.csv")
print(df)

# Extract available features into a dictionary

def extract_feature_names(df):
    feature_names = df.columns
    features_dict = {}
    features_type = []
    for name in feature_names:
        features_dict[name] = name
    return features_dict

def transform_array(df, y_idx):
    """
    Transforms a dataframe into numpy arrays X and y
    """
    X_array = np.array(df)
    y_array = X_array[:, y_idx].reshape((X_array.shape[0], 1))
    X_array = np.delete(X_array, y_idx, axis = 1)
    
    return X_array, y_array

def product_remainder(X, w, feature_idx):
    """
    Compute dot product of remaining features and coefficients, that are not to be plotted
    """
    features = np.delete(X, feature_idx, axis = 1)
    weights = np.delete(w, feature_idx)
    av = np.mean(features, axis = 0).flatten()
    product = np.dot(av, weights)

    return product



