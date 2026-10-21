import numpy as np


def Adaline(x, y, learning_rate=0.01, n_iter=50, mse_threshold=0.01, use_bias=True):
    """
    Adaline (Adaptive Linear Neuron) algorithm using MSE.

    Parameters:
    x : array, shape = [n_samples, 2]
        Training vectors (two features).
    y : array, shape = [n_samples]
        Target values (+1 / -1).
    learning_rate : float
        Learning rate (eta).
    n_iter : int
        Max number of epochs (m).
    mse_threshold : float
        Stop training when the MSE is less than this value.
    use_bias : bool
        Add a bias or not.

    Returns:
    weights : array, shape = [n_features]
        Weights after training.
    bias : float
        Bias after training (0 if use_bias is False).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples = x.shape[0]

    # Initialize weights and bias with small random numbers
    weights = np.random.uniform(-0.1, 0.1, x.shape[1])
    bias = np.random.uniform(-0.1, 0.1) if use_bias else 0.0

    for _ in range(n_iter):
        for i in range(n_samples):
            output = np.dot(weights, x[i]) + bias
            error = y[i] - output
            weights = weights + learning_rate * error * x[i]
            if use_bias:
                bias = bias + learning_rate * error

        # After the epoch: calculate MSE using the updated weights
        errors = y - (np.dot(x, weights) + bias)
        mse = np.sum(0.5 * errors ** 2) / n_samples

        # Stopping criteria
        if mse < mse_threshold:
            break

    return weights, bias


def predict(x, weights, bias):
    """Returns +1 or -1 for each row in x."""
    return np.where(np.dot(np.atleast_2d(x), weights) + bias >= 0, 1, -1)