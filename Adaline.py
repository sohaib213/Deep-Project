import numpy as np


def Adaline(x, y, learning_rate=0.01, n_iter=50, mse_threshold=0.01, use_bias=True):
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples = x.shape[0]

    weights = np.random.uniform(-0.1, 0.1, x.shape[1])
    bias = np.random.uniform(-0.1, 0.1) if use_bias else 0.0

    for _ in range(n_iter):
        for i in range(n_samples):
            output = np.dot(weights, x[i]) + bias
            error = y[i] - output
            weights = weights + learning_rate * error * x[i]
            if use_bias:
                bias = bias + learning_rate * error

        errors = y - (np.dot(x, weights) + bias)
        mse = np.sum(0.5 * errors ** 2) / n_samples

        if mse < mse_threshold:
            break

    return weights, bias


def predict(x, weights, bias):
    return np.where(np.dot(np.atleast_2d(x), weights) + bias >= 0, 1, -1)