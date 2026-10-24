import numpy as np

def perceptron_train(x, y, learning_rate=0.01, epochs=50, use_bias=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    np.random.seed(42)
    weights = np.random.uniform(-0.1, 0.1, x.shape[1])
    bias = np.random.uniform(-0.1, 0.1) if use_bias else 0.0

    for _ in range(epochs):
        errors_in_epoch = 0
        for i in range(x.shape[0]):
            net_value = np.dot(weights, x[i]) + bias
            y_pred = 1 if net_value >= 0 else -1 
            
            if y_pred != y[i]:
                error = y[i] - y_pred
                weights += learning_rate * error * x[i]
                if use_bias:
                    bias += learning_rate * error
                errors_in_epoch += 1
                
        if errors_in_epoch == 0:
            break

    return weights, bias

def predict_perceptron(x, weights, bias):
    return np.where(np.dot(np.atleast_2d(x), weights) + bias >= 0, 1, -1)