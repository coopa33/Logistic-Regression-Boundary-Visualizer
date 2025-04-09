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



def read_structure_data(pathname):
    df = pd.read_csv(pathname)
    df = df.copy(deep=True)

    # Drop unwanted columns
    df = df.drop(columns = ["PassengerId", "Name", "Cabin", "Ticket", "Cabin", "Embarked"])

    # Drop Na rows 
    df = df.dropna(how="any")

    # Convert Sex to binary numbers, 1=male, 0=female
    df["Sex"] = df["Sex"].apply(lambda x: 1 if x == "male" else 0)

    # z-score normalize all columns except y
    y = df.pop("Survived")
    df = df.apply(lambda x: (x - np.mean(x))/np.std(x), axis = 0)

    # Transform into numpy arrays
    n_obs = len(y)
    y = np.array(y).reshape((n_obs, 1))
    X = np.array(df)

    # Reassemble Dataframe
    df["Survived"] = pd.Series(y.flatten())

    return X, y, df


# Shared datastructures
X, y, df = read_structure_data("../../data/titanic.csv")
w, b = gradient_descent(X, y, np.zeros(X.shape[1]), 0)


