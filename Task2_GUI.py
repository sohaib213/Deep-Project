import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from dataPreprossing2 import load_and_preprocess_task2, ORIGIN_MAP
from Models.Mlp import Mlp_train, Mlp_predict
from metrics import evaluate_metrics_multi

trained_state = {}


def run_mlp():
    try:
        neurons_input = neurons_entry.get().split(',')
        hidden_neurons = [int(n.strip()) for n in neurons_input if n.strip()]
        num_layers = int(layers_entry.get())

        if len(hidden_neurons) != num_layers:
            messagebox.showerror("Error", f"Please enter exactly {num_layers} comma-separated values for neurons.")
            return

        lr = float(lr_entry.get())
        epochs = int(epochs_entry.get())
        use_bias = bias_var.get()
        use_sigmoid = (activation_var.get() == "Sigmoid")

        X_train, y_train, X_test, y_test, s_mean, s_std, classes = load_and_preprocess_task2()

        weights, biases = Mlp_train(
            x=X_train, y=y_train,
            n_of_neurons_per_hidden_layer=hidden_neurons,
            n_hidden_layers=num_layers,
            activation_sigmoid=use_sigmoid,
            learning_rate=lr, epochs=epochs,
            over_all_error_threshold=-1.0,
            use_bias=use_bias
        )

        # Test the model
        y_pred = [Mlp_predict(x, weights, biases, use_sigmoid) for x in X_test]
        acc, conf_mat = evaluate_metrics_multi(y_test, y_pred)

        trained_state.update({
            'weights': weights, 'biases': biases, 'use_sigmoid': use_sigmoid,
            'mean': s_mean, 'std': s_std, 'classes': classes
        })

        res = (f"Overall Accuracy: {acc * 100:.2f}%\n\n"
               f"Confusion Matrix (Rows: True, Cols: Pred):\n"
               f"{conf_mat}")
        messagebox.showinfo("Task 2 Results", res)

    except Exception as e:
        messagebox.showerror("Error", str(e))


def classify_sample():
    if not trained_state:
        messagebox.showwarning("Warning", "Train the model first.")
        return
    try:
        raw_vals = [s1_entry.get(), s2_entry.get(), s3_entry.get(), s4_entry.get(), s5_entry.get()]

        # Handle OriginLocation categorical text
        if raw_vals[4].strip().title() in ORIGIN_MAP:
            raw_vals[4] = ORIGIN_MAP[raw_vals[4].strip().title()]

        sample = np.array([float(v) for v in raw_vals])
        sample_scaled = (sample - trained_state['mean']) / trained_state['std']

        pred = Mlp_predict(sample_scaled, trained_state['weights'], trained_state['biases'],
                           trained_state['use_sigmoid'])
        class_idx = np.argmax(pred)
        predicted_class = trained_state['classes'][class_idx]

        messagebox.showinfo("Prediction", f"Output Vector: {pred}\nPredicted Class: {predicted_class}")
    except Exception as e:
        messagebox.showerror("Error", "Invalid sample inputs. Ensure 5 valid values.")


# Basic GUI Setup
root = tk.Tk()
root.title("MLP Backpropagation - Task 2")

tk.Label(root, text="Hidden Layers:").grid(row=0, column=0)
layers_entry = tk.Entry(root)
layers_entry.insert(0, "1")
layers_entry.grid(row=0, column=1)

tk.Label(root, text="Neurons (comma-separated):").grid(row=1, column=0)
neurons_entry = tk.Entry(root)
neurons_entry.insert(0, "5")
neurons_entry.grid(row=1, column=1)

tk.Label(root, text="Learning Rate (eta):").grid(row=2, column=0)
lr_entry = tk.Entry(root)
lr_entry.insert(0, "0.01")
lr_entry.grid(row=2, column=1)

tk.Label(root, text="Epochs (m):").grid(row=3, column=0)
epochs_entry = tk.Entry(root)
epochs_entry.insert(0, "100")
epochs_entry.grid(row=3, column=1)

activation_var = tk.StringVar(value="Sigmoid")
tk.Radiobutton(root, text="Sigmoid", variable=activation_var, value="Sigmoid").grid(row=4, column=0)
tk.Radiobutton(root, text="Tanh", variable=activation_var, value="Tanh").grid(row=4, column=1)

bias_var = tk.BooleanVar(value=True)
tk.Checkbutton(root, text="Add Bias", variable=bias_var).grid(row=5, column=0, columnspan=2)

tk.Button(root, text="Train & Test MLP", command=run_mlp).grid(row=6, column=0, columnspan=2, pady=10)

# Single Sample Frame
sample_frame = tk.LabelFrame(root, text="Classify Single Sample (5 Features)")
sample_frame.grid(row=7, column=0, columnspan=2, pady=10)

s1_entry = tk.Entry(sample_frame, width=8);
s1_entry.grid(row=0, column=0)  # CulmenLength
s2_entry = tk.Entry(sample_frame, width=8);
s2_entry.grid(row=0, column=1)  # CulmenDepth
s3_entry = tk.Entry(sample_frame, width=8);
s3_entry.grid(row=0, column=2)  # FlipperLength
s4_entry = tk.Entry(sample_frame, width=8);
s4_entry.grid(row=0, column=3)  # BodyMass
s5_entry = tk.Entry(sample_frame, width=8);
s5_entry.grid(row=0, column=4)  # OriginLocation

tk.Button(sample_frame, text="Classify", command=classify_sample).grid(row=1, column=0, columnspan=5)

root.mainloop()