import numpy as np

def Mlp_train(x, y, n_of_neurons_per_hidden_layer=[5],n_hidden_layers=1,activation_sigmoid=True,learning_rate=0.01, epochs=50, over_all_error_threshold=0.01, use_bias=True):
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    layer_sizes=[x.shape[1]]+n_of_neurons_per_hidden_layer+[y.shape[1]]
    np.random.seed(42)
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        w = np.random.uniform(-0.1, 0.1, (layer_sizes[i], layer_sizes[i+1]))
        b = np.random.uniform(-0.1, 0.1, (1, layer_sizes[i+1])) if use_bias else np.zeros((1, layer_sizes[i+1]))
        weights.append(w)
        biases.append(b)

    for m in range(epochs) :
        over_all_error=0.0
        layer_outputs=[]
        #forward step
        for i in range(x.shape[0]): #loop on the input instances 
            current_input = np.atleast_2d(x[i])
            layer_outputs = [np.atleast_2d(x[i])]

            for w,b in zip(weights,biases):
                net_value= np.dot(current_input, w) + b
                if activation_sigmoid :
                    net_value = 1.0 / (1.0 + np.exp(-np.clip(net_value, -500, 500)))
                else :
                    net_value =np.tanh(net_value)
                layer_outputs.append(net_value)
                current_input =net_value

            final_output = layer_outputs[-1]
            error = np.atleast_2d(y[i]) - final_output
            over_all_error += 0.5 * np.sum(error ** 2)

            #backward step
            deltas = []

            delta_output =None
            if activation_sigmoid :
                delta_output = error*(final_output*(1-final_output))
            else :
                delta_output= error *(1-(final_output**2))
            deltas.append(delta_output)

            for j in range(len(weights) - 1, 0, -1):
                old_weights = weights[j]
                errors_in_front=deltas[-1]

                delta_in_current_layer=None
                if activation_sigmoid :
                    delta_in_current_layer=np.dot(errors_in_front,old_weights.T) *(layer_outputs[j]*(1-layer_outputs[j]))
                else :
                    delta_in_current_layer=np.dot(errors_in_front,old_weights.T) *(1-(layer_outputs[j]**2))
                deltas.append(delta_in_current_layer)
            deltas.reverse()

            #foward step again(updates)
            for j in range(len(weights)):
                weights[j]=weights[j]+learning_rate*np.dot(layer_outputs[j].T,deltas[j])
                if use_bias :
                    biases[j]=biases[j]+learning_rate*deltas[j]

        if over_all_error <= over_all_error_threshold:
            break

    return weights, biases


def Mlp_predict(x, weights, biases, activation_sigmoid=True):
    current_input = np.atleast_2d(x)
    for w,b in zip(weights,biases):
        net_value= np.dot(current_input, w) + b
        if activation_sigmoid :
            net_value = 1.0 / (1.0 + np.exp(-np.clip(net_value, -500, 500)))
        else :
            net_value =np.tanh(net_value)
        current_input =net_value
    
    final_out = current_input.flatten()
    #one-hot encode it 
    prediction = np.zeros_like(final_out)
    prediction[np.argmax(final_out)] = 1.0  
    return prediction  
