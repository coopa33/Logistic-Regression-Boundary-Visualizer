import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def z_value(X, w, b):
    """
    Calculate z-value for linear term

    Parameters
    ----------
    X: ndarray
        2-D ndarray of datapoints (observations, features)
    w: ndarray
        1-D ndarray of feature coefficients
    b: float
        Intercept 

    Returns
    -------
    z: ndarray
        2-D column vector of z-values for each observation (observations, 1)
    """
    
    # Reshape input
    n_features = len(w)
    w = w.reshape((n_features, 1))

    # Compute z-value
    z = np.matmul(X, w) + b

    return z


def logistic_activation(X, w, b):
    """
    Runs z-values through a logistic activation function

    Parameters
    ----------
    
    X: ndarray
        2-D ndarray of datapoints (observations, features)
    w: ndarray
        1-D ndarray of feature coefficients
    b: float
        Intercept 

    Returns
    -------
    y_hat: ndarray
        2-D column vector, prediction of logistic regression (observations, 1)
    """
    # Calculate z-values
    z = z_value(X, w, b)

    # Define logistic activation and compute y_hat
    y_hat = 1/(1 + np.exp(-z))

    # Clip allowable ranges of y_hat, so that floating point rounding to 1 doesn't occur
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)

    return y_hat


def logistic_cost(X, y, w, b):
    """
    Calculates cost of logistic regression model

    Parameters
    ----------
    X: ndarray
        2-D ndarray of datapoints (observations, features)
    y: ndarray
        2-D column vector, ground truth (observations, 1)
    w: ndarray
        1-D ndarray of feature coefficients
    b: float
        Intercept 

    Returns
    -------
    cost: float
        Logistic cost
    """
    # Compute predictions
    y_hat = logistic_activation(X, w, b)

    # Unravel arrays for dot product
    y = y.flatten()
    y_hat = y_hat.flatten()

    # Compute cost
    n_obs = len(y)
    cost = (-1/n_obs) * (np.dot(y, np.log(y_hat)) + np.dot((1-y), np.log(1-y_hat)))

    return cost

def gradient_w(X, y, y_hat):
    """
    Calculate the gradient by logistic loss of feature coefficients

    Parameters
    ----------
    X: ndarray
        2-D ndarray, datapoints (observations, features)
    y: ndarray
        2-D column vector, ground truth (observations, 1)
    y_hat: ndarray
        2-D column vector, predictions (observations, 1)

    Returns
    -------
    w_gradient: ndarray
        2-D column vector, gradient for coefficients (features, 1)
    """
    n_obs = y.shape[0]
    diff = y_hat - y
    w_gradient = (1/n_obs) * np.matmul(X.T, diff)
    
    return w_gradient

    
   
def gradient_b(y, y_hat):
    """
    Calculate the gradient by logistic loss of feature coefficients

    Parameters
    ----------
    y: ndarray
        2-D column vector, ground truth (observations, 1)
    y_hat: ndarray
        2-D column vector, predictions (observations, 1)

    Returns
    -------
    b_gradient: float
        Gradient for intercept
    """
    n_obs = y.shape[0]
    diff = y_hat - y
    b_gradient = (1/n_obs) * np.sum(diff)
    
    return b_gradient


def gradient_descent(X, y, w, b, alpha = 10**-2, thres = 10**-5):
    """
    Runs gradiend descent algorithm

    Parameters
    ----------
    X: ndarray
        2-D ndarray, datapoints (observations, features)
    y: ndarray
        2-D column vector, ground truth (observations, 1)
    w: ndarray
        1-D vector, initial coefficients
    b: float
        Initial intercept

    Returns
    -------
    w_hat: ndarray
        1-D vector, estimated coefficients
    b_hat: float
        Estimated intercept
    """

    converged = False
    cost_history = [logistic_cost(X, y, w, b)]
    while not converged:
        # Make predictions with current parameters
        y_hat = logistic_activation(X, w, b)

        # Calculate gradients and update parameters
        w = w - (alpha * gradient_w(X, y, y_hat)).flatten()
        b = b - (alpha * gradient_b(y, y_hat))
         
        # Calculate cost 
        cost = logistic_cost(X, y, w, b)
        cost_history.append(cost)
        if cost_history[-2] < cost_history[-1]:
            print("Diverged")
            return w, b
        elif cost_history[-2] - cost_history[-1] < thres:
            converged = True
    
    return (w, b)

def plot_logistic_boundary(X, y, feature_idx, w, b):
    """
    Creates a plot of two features of a set of data, their labels, and the decision boundary given the features selected and the parameters

    Parameters
    ----------
    X: ndarray
        2-D ndarray, dataset (observations, features)
    y: ndarray
        2-D column vector, ground truth (observations, 1)
    feature_idx: tuple
        Tuple of integers, column indices to select features from dataset and select feature coefficients
    w: ndarray
        1-D vector, feature coefficients
    b: float
        Intercept
    """
    # Plot datapoints
    feature_x = X[:, feature_idx[0]]
    feature_y = X[:, feature_idx[1]]
    plt.scatter(feature_x, feature_y, c = y)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # Plot decision boundary
    x_1 = np.linspace(min(feature_x), max(feature_x), 100)
    x_2 = (-x_1 * w[feature_idx[0]] - b) / w[feature_idx[1]]
    plt.plot(x_1, x_2, color = "blue")
    plt.show()

    return None


def z_score_scaling(X):
    """
    Normalize features of a dataframe with z-score scaling
    """
    for i in range(X.shape[1]):
        feature = X[:, i]
        normalized_feature = (feature - np.mean(feature)) / np.std(feature)
        X[:, i] = normalized_feature

    return X

if __name__=="__main__":
    
    ### Testing
    X_test = np.array([
        [1, 2],
        [4, 5],
        [7, 8],
        [11, 22],
        [0.5, 3]])
    y_test = np.array([
        [1], 
        [1], 
        [0],
        [0],
        [1]])

    # Normalize X
    X_test = z_score_scaling(X_test)
    
    # Compute parameters, and plot decision boundary
    w = np.array([1, 2])
    b = 7.65
    w_hat, b_hat = gradient_descent(X_test, y_test, w, b)
    preds = logistic_activation(X_test, w_hat, b_hat)
    plot_logistic_boundary(X_test, y_test, (0, 1), w_hat, b_hat)
