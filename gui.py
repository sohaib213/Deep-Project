import sys
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from dataPreprocessing import load_and_preprocess, ORIGIN_MAP
from perceptron import perceptron_train, predict_perceptron
from metrics import evaluate_metrics
from Adaline import Adaline, predict

BG_COLOR = "#080616"
FRAME_COLOR = "#1A1953"
BTN_HOVER = "#162E93"
BTN_COLOR = "#2F2FE4"
TEXT_COLOR = "#FFFFFF"

trained_state = {}

def plot_decision_boundary(X_train, y_train, weights, bias, feature1, feature2, c1, c2, frame):
    for widget in frame.winfo_children():
        widget.destroy()
        
    plt.close('all')
    fig, ax = plt.subplots(figsize=(5, 3.8))
    
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(FRAME_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(TEXT_COLOR)
        
    ax.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='#2F2FE4', label=c1)
    ax.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='red', label=c2)
    
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min_data, y_max_data = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    
    if not np.isnan(weights).any():
        if abs(weights[1]) > 1e-6:
            y_min = -(weights[0] * x_min + bias) / weights[1]
            y_max = -(weights[0] * x_max + bias) / weights[1]
            ax.plot([x_min, x_max], [y_min, y_max], color='#00FF66', linewidth=2, label='Boundary')
        elif abs(weights[0]) > 1e-6:
            x_vert = -bias / weights[0]
            ax.axvline(x=x_vert, color='#00FF66', linewidth=2, label='Boundary')
            
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min_data, y_max_data)
    ax.set_xlabel(f"{feature1} (Scaled)")
    ax.set_ylabel(f"{feature2} (Scaled)")
    ax.legend(facecolor=BG_COLOR, edgecolor=TEXT_COLOR, labelcolor=TEXT_COLOR)
    fig.tight_layout()
    
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

def run_model():
    f1, f2 = feature1_var.get(), feature2_var.get()
    if f1 == f2:
        messagebox.showwarning("Warning", "Please select two different features")
        return
        
    c1, c2 = classes_var.get().split(" & ")
    try:
        lr = float(lr_entry.get())
        epochs = int(epochs_entry.get())
        mse_th = float(mse_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values")
        return
        
    use_bias, algo = bias_var.get(), algo_var.get()
    
    try:
        X_train, y_train, X_test, y_test, s_mean, s_std = load_and_preprocess(f1, f2, c1, c2)
        
        if algo == "Perceptron":
            weights, bias = perceptron_train(X_train, y_train, lr, epochs, use_bias)
            y_pred = predict_perceptron(X_test, weights, bias)
        else:
            weights, bias = Adaline(X_train, y_train, lr, epochs, mse_th, use_bias)
            y_pred = predict(X_test, weights, bias)
            
        acc, conf_mat = evaluate_metrics(y_test, y_pred)
        plot_decision_boundary(X_train, y_train, weights, bias, f1, f2, c1, c2, plot_frame)
        
        trained_state.update({
            'weights': weights, 'bias': bias, 'algo': algo,
            'f1': f1, 'f2': f2, 'c1': c1, 'c2': c2,
            'mean': s_mean, 'std': s_std
        })
        
        res = (f"Accuracy: {acc*100:.2f}%\n"
               f"Confusion Matrix:\n"
               f"TP ({c1}): {conf_mat[0][0]} | FP: {conf_mat[0][1]}\n"
               f"FN: {conf_mat[1][0]} | TN ({c2}): {conf_mat[1][1]}")
        messagebox.showinfo("Results", res)
    except Exception as e:
        messagebox.showerror("Error", str(e))

def classify_single_sample():
    if not trained_state:
        messagebox.showwarning("Warning", "Please train a model first by clicking 'Run Classification'.")
        return
        
    try:
        raw_v1 = sample_f1_entry.get().strip().title()
        raw_v2 = sample_f2_entry.get().strip().title()
        
        if trained_state['f1'] == 'OriginLocation' and raw_v1 in ORIGIN_MAP:
            v1 = ORIGIN_MAP[raw_v1]
        else:
            v1 = float(raw_v1)
            
        if trained_state['f2'] == 'OriginLocation' and raw_v2 in ORIGIN_MAP:
            v2 = ORIGIN_MAP[raw_v2]
        else:
            v2 = float(raw_v2)
        
        sample = np.array([v1, v2], dtype=float)
        sample_scaled = (sample - trained_state['mean']) / trained_state['std']
        
        if trained_state['algo'] == "Perceptron":
            pred = predict_perceptron(sample_scaled, trained_state['weights'], trained_state['bias'])[0]
        else:
            pred = predict(sample_scaled, trained_state['weights'], trained_state['bias'])[0]
            
        predicted_class = trained_state['c1'] if pred == 1 else trained_state['c2']
        messagebox.showinfo("Sample Prediction", f"Predicted Class ID: {predicted_class} ({pred:+d})")
        
    except ValueError:
        messagebox.showerror("Error", "Enter valid numbers (or Torgersen/Biscoe/Dream for OriginLocation).")
        
def on_closing():
    plt.close('all')
    root.quit()     
    root.destroy()  
    sys.exit()       

root = tk.Tk()
root.title("Neural Network Task 1 GUI")
root.configure(bg=BG_COLOR)

style = ttk.Style()
style.theme_use('clam')
style.configure("TCombobox", fieldbackground=FRAME_COLOR, background=BG_COLOR, foreground=TEXT_COLOR, arrowcolor=TEXT_COLOR)
style.map('TCombobox', fieldbackground=[('readonly', FRAME_COLOR)], selectbackground=[('readonly', BTN_HOVER)], selectforeground=[('readonly', TEXT_COLOR)])

lbl_style = {"bg": BG_COLOR, "fg": TEXT_COLOR, "font": ("Arial", 10, "bold")}
entry_style = {"bg": FRAME_COLOR, "fg": TEXT_COLOR, "insertbackground": TEXT_COLOR, "relief": "flat"}
check_radio_style = {"bg": BG_COLOR, "fg": TEXT_COLOR, "selectcolor": FRAME_COLOR, "activebackground": BG_COLOR, "activeforeground": TEXT_COLOR}

features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'OriginLocation', 'BodyMass']

tk.Label(root, text="Feature 1:", **lbl_style).grid(row=0, column=0, pady=4, padx=5, sticky="e")
feature1_var = ttk.Combobox(root, values=features, state="readonly")
feature1_var.grid(row=0, column=1, pady=4, padx=5)
feature1_var.current(0)

tk.Label(root, text="Feature 2:", **lbl_style).grid(row=1, column=0, pady=4, padx=5, sticky="e")
feature2_var = ttk.Combobox(root, values=features, state="readonly")
feature2_var.grid(row=1, column=1, pady=4, padx=5)
feature2_var.current(1)

tk.Label(root, text="Classes:", **lbl_style).grid(row=2, column=0, pady=4, padx=5, sticky="e")
classes_var = ttk.Combobox(root, values=["Adelie & Chinstrap", "Adelie & Gentoo", "Chinstrap & Gentoo"], state="readonly")
classes_var.grid(row=2, column=1, pady=4, padx=5)
classes_var.current(0)

tk.Label(root, text="Learning Rate:", **lbl_style).grid(row=3, column=0, pady=4, padx=5, sticky="e")
lr_entry = tk.Entry(root, **entry_style)
lr_entry.insert(0, "0.01")
lr_entry.grid(row=3, column=1, pady=4, padx=5)

tk.Label(root, text="Epochs:", **lbl_style).grid(row=4, column=0, pady=4, padx=5, sticky="e")
epochs_entry = tk.Entry(root, **entry_style)
epochs_entry.insert(0, "50")
epochs_entry.grid(row=4, column=1, pady=4, padx=5)

tk.Label(root, text="MSE Threshold:", **lbl_style).grid(row=5, column=0, pady=4, padx=5, sticky="e")
mse_entry = tk.Entry(root, **entry_style)
mse_entry.insert(0, "0.01")
mse_entry.grid(row=5, column=1, pady=4, padx=5)

bias_var = tk.BooleanVar(value=True)
tk.Checkbutton(root, text="Add Bias", variable=bias_var, **check_radio_style).grid(row=6, column=0, columnspan=2, pady=4)

algo_var = tk.StringVar(value="Perceptron")
tk.Radiobutton(root, text="Perceptron", variable=algo_var, value="Perceptron", **check_radio_style).grid(row=7, column=0)
tk.Radiobutton(root, text="Adaline", variable=algo_var, value="Adaline", **check_radio_style).grid(row=7, column=1)

btn = tk.Button(root, text="Run Classification", bg=BTN_COLOR, fg=TEXT_COLOR, activebackground=BTN_HOVER, activeforeground=TEXT_COLOR, relief="flat", font=("Arial", 10, "bold"), command=run_model)
btn.grid(row=8, column=0, columnspan=2, pady=8, ipadx=10, ipady=3)

sample_frame = tk.Frame(root, bg=BG_COLOR)
sample_frame.grid(row=9, column=0, columnspan=2, pady=4)

tk.Label(sample_frame, text="Sample X1:", **lbl_style).grid(row=0, column=0, padx=3)
sample_f1_entry = tk.Entry(sample_frame, width=10, **entry_style)
sample_f1_entry.grid(row=0, column=1, padx=3)

tk.Label(sample_frame, text="X2:", **lbl_style).grid(row=0, column=2, padx=3)
sample_f2_entry = tk.Entry(sample_frame, width=10, **entry_style)
sample_f2_entry.grid(row=0, column=3, padx=3)

sample_btn = tk.Button(sample_frame, text="Classify Sample", bg=BTN_HOVER, fg=TEXT_COLOR, activebackground=BTN_COLOR, activeforeground=TEXT_COLOR, relief="flat", font=("Arial", 9, "bold"), command=classify_single_sample)
sample_btn.grid(row=0, column=4, padx=5, ipadx=5, ipady=2)

plot_frame = tk.Frame(root, bg=BG_COLOR)
plot_frame.grid(row=10, column=0, columnspan=2, pady=5)

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()