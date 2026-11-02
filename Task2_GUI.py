import queue
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np

from dataPreprossing2 import load_and_preprocess_task2, ORIGIN_MAP
from Models.Mlp import Mlp_train, Mlp_predict
from metrics import evaluate_metrics_multi

# ---------------------------------------------------------------- settings
DEFAULTS = {"layers": "1", "neurons": "5", "lr": "0.01", "epochs": "100"}

FEATURES = [
    ("Culmen length (mm)", "entry"),
    ("Culmen depth (mm)", "entry"),
    ("Flipper length (mm)", "entry"),
    ("Body mass (g)", "entry"),
    ("Origin location", "combo"),
]

BG = "#f5f6f8"
CARD = "#ffffff"
ACCENT = "#1f6f8b"
ACCENT_DARK = "#17566b"
TEXT = "#1d2733"
MUTED = "#6b7785"
ERROR = "#b3261e"
OK = "#2e7d32"


class MlpApp:
    def __init__(self, root):
        self.root = root
        self.state = {}               # trained model lives here
        self.results_queue = queue.Queue()
        self.training = False

        root.title("MLP Backpropagation - Task 2")
        root.configure(bg=BG)
        root.minsize(900, 600)

        self._setup_style()
        self._build_layout()
        self._set_status("Set the options on the left, then press Train & Test.")

    # ------------------------------------------------------------- styling
    def _setup_style(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure(".", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        s.configure("Card.TLabelframe", background=CARD, bordercolor="#d8dde3")
        s.configure("Card.TLabelframe.Label", background=CARD,
                    foreground=TEXT, font=("Segoe UI", 10, "bold"))
        s.configure("Card.TFrame", background=CARD)
        s.configure("Card.TLabel", background=CARD)
        s.configure("Muted.TLabel", background=CARD, foreground=MUTED,
                    font=("Segoe UI", 9))
        s.configure("Card.TRadiobutton", background=CARD)
        s.configure("Card.TCheckbutton", background=CARD)
        s.configure("Accent.TButton", background=ACCENT, foreground="white",
                    font=("Segoe UI", 10, "bold"), padding=8, borderwidth=0)
        s.map("Accent.TButton",
              background=[("disabled", "#9fb9c4"), ("active", ACCENT_DARK)])
        s.configure("TButton", padding=6)
        s.configure("Big.TLabel", background=CARD, foreground=ACCENT,
                    font=("Segoe UI", 16, "bold"))
        s.configure("Status.TLabel", background="#e9ecf0", padding=(10, 5))
        s.configure("Treeview", rowheight=24, background=CARD,
                    fieldbackground=CARD)
        s.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        s.configure("Bar.Horizontal.TProgressbar", troughcolor="#e3e7ec",
                    background=ACCENT)

    # -------------------------------------------------------------- layout
    def _build_layout(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root)
        left.grid(row=0, column=0, sticky="ns", padx=(12, 6), pady=12)
        right = ttk.Frame(self.root)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self._build_settings(left)
        self._build_classifier(left)
        self._build_results(right)

        self.status = ttk.Label(self.root, text="", style="Status.TLabel",
                                anchor="w")
        self.status.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _build_settings(self, parent):
        box = ttk.LabelFrame(parent, text="Network settings",
                             style="Card.TLabelframe", padding=12)
        box.grid(row=0, column=0, sticky="ew")
        box.columnconfigure(1, weight=1)

        self.layers_var = tk.StringVar(value=DEFAULTS["layers"])
        self.neurons_var = tk.StringVar(value=DEFAULTS["neurons"])
        self.lr_var = tk.StringVar(value=DEFAULTS["lr"])
        self.epochs_var = tk.StringVar(value=DEFAULTS["epochs"])
        self.activation_var = tk.StringVar(value="Sigmoid")
        self.bias_var = tk.BooleanVar(value=True)

        def row(r, label, widget, hint=None):
            ttk.Label(box, text=label, style="Card.TLabel").grid(
                row=r * 2, column=0, sticky="w", pady=(6, 0))
            widget.grid(row=r * 2, column=1, sticky="ew", padx=(10, 0),
                        pady=(6, 0))
            if hint:
                ttk.Label(box, text=hint, style="Muted.TLabel").grid(
                    row=r * 2 + 1, column=1, sticky="w", padx=(10, 0))

        layers = ttk.Spinbox(box, from_=1, to=10, width=8,
                             textvariable=self.layers_var,
                             command=self._sync_neurons)
        layers.bind("<FocusOut>", lambda e: self._sync_neurons())
        row(0, "Hidden layers", layers)

        row(1, "Neurons per layer", ttk.Entry(box, textvariable=self.neurons_var),
            "One number per layer, e.g. 8, 4")
        row(2, "Learning rate", ttk.Entry(box, textvariable=self.lr_var))
        row(3, "Epochs", ttk.Entry(box, textvariable=self.epochs_var))

        act = ttk.Frame(box, style="Card.TFrame")
        ttk.Radiobutton(act, text="Sigmoid", variable=self.activation_var,
                        value="Sigmoid", style="Card.TRadiobutton").pack(
            side="left")
        ttk.Radiobutton(act, text="Tanh", variable=self.activation_var,
                        value="Tanh", style="Card.TRadiobutton").pack(
            side="left", padx=(12, 0))
        row(4, "Activation", act)

        ttk.Checkbutton(box, text="Use bias", variable=self.bias_var,
                        style="Card.TCheckbutton").grid(
            row=10, column=0, columnspan=2, sticky="w", pady=(10, 0))

        btns = ttk.Frame(box, style="Card.TFrame")
        btns.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        btns.columnconfigure(0, weight=1)
        self.train_btn = ttk.Button(btns, text="Train & Test",
                                    style="Accent.TButton",
                                    command=self.start_training)
        self.train_btn.grid(row=0, column=0, sticky="ew")
        ttk.Button(btns, text="Reset", command=self.reset_defaults).grid(
            row=0, column=1, padx=(8, 0))

        self.progress = ttk.Progressbar(box, mode="indeterminate",
                                        style="Bar.Horizontal.TProgressbar")
        self.progress.grid(row=12, column=0, columnspan=2, sticky="ew",
                           pady=(10, 0))

        self.root.bind("<Return>", lambda e: self.start_training())

    def _build_classifier(self, parent):
        box = ttk.LabelFrame(parent, text="Classify one sample",
                             style="Card.TLabelframe", padding=12)
        box.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        box.columnconfigure(1, weight=1)

        self.sample_widgets = []
        for i, (label, kind) in enumerate(FEATURES):
            ttk.Label(box, text=label, style="Card.TLabel").grid(
                row=i, column=0, sticky="w", pady=2)
            if kind == "combo":
                w = ttk.Combobox(box, values=list(ORIGIN_MAP.keys()), width=14)
            else:
                w = ttk.Entry(box, width=16)
            w.grid(row=i, column=1, sticky="ew", padx=(10, 0), pady=2)
            self.sample_widgets.append(w)

        self.classify_btn = ttk.Button(box, text="Classify",
                                       command=self.classify_sample,
                                       state="disabled")
        self.classify_btn.grid(row=len(FEATURES), column=0, columnspan=2,
                               sticky="ew", pady=(10, 0))
        ttk.Label(box, text="Train the model first to enable this.",
                  style="Muted.TLabel").grid(
            row=len(FEATURES) + 1, column=0, columnspan=2, sticky="w")

    def _build_results(self, parent):
        nb = ttk.Notebook(parent)
        nb.grid(row=0, column=0, sticky="nsew")

        # --- tab 1: test results
        tab1 = ttk.Frame(nb, padding=12, style="Card.TFrame")
        nb.add(tab1, text="Test results")
        tab1.columnconfigure(0, weight=1)
        tab1.rowconfigure(1, weight=1)

        self.acc_label = ttk.Label(tab1, text="No results yet",
                                   style="Big.TLabel")
        self.acc_label.grid(row=0, column=0, sticky="w")

        self.cm_canvas = tk.Canvas(tab1, bg=CARD, highlightthickness=0)
        self.cm_canvas.grid(row=1, column=0, sticky="nsew", pady=10)
        self.cm_canvas.bind("<Configure>", lambda e: self._draw_confusion())
        self.last_conf = None
        self.last_labels = None

        cols = ("class", "precision", "recall", "support")
        self.tree = ttk.Treeview(tab1, columns=cols, show="headings", height=4)
        for c, w in zip(cols, (140, 90, 90, 90)):
            self.tree.heading(c, text=c.capitalize())
            self.tree.column(c, width=w, anchor="center" if c != "class" else "w")
        self.tree.grid(row=2, column=0, sticky="ew")

        # --- tab 2: single prediction
        tab2 = ttk.Frame(nb, padding=12, style="Card.TFrame")
        nb.add(tab2, text="Prediction")
        tab2.columnconfigure(0, weight=1)
        self.pred_title = ttk.Label(tab2, text="No prediction yet",
                                    style="Big.TLabel")
        self.pred_title.grid(row=0, column=0, sticky="w")
        self.pred_frame = ttk.Frame(tab2, style="Card.TFrame")
        self.pred_frame.grid(row=1, column=0, sticky="ew", pady=10)
        self.pred_frame.columnconfigure(1, weight=1)
        self.notebook = nb
        self.pred_tab = tab2

    # ------------------------------------------------------------- helpers
    def _set_status(self, msg, kind="info"):
        color = {"info": TEXT, "error": ERROR, "ok": OK}[kind]
        self.status.configure(text=msg, foreground=color)

    def _sync_neurons(self):
        """Keep the neuron list the same length as the layer count."""
        try:
            n = int(self.layers_var.get())
        except ValueError:
            return
        n = max(1, min(10, n))
        vals = [v.strip() for v in self.neurons_var.get().split(",") if v.strip()]
        last = vals[-1] if vals else "5"
        vals = (vals + [last] * n)[:n]
        self.neurons_var.set(", ".join(vals))

    def reset_defaults(self):
        self.layers_var.set(DEFAULTS["layers"])
        self.neurons_var.set(DEFAULTS["neurons"])
        self.lr_var.set(DEFAULTS["lr"])
        self.epochs_var.set(DEFAULTS["epochs"])
        self.activation_var.set("Sigmoid")
        self.bias_var.set(True)
        self._set_status("Settings reset to defaults.")

    def _read_settings(self):
        """Return validated settings or raise ValueError with a clear message."""
        try:
            n_layers = int(self.layers_var.get())
        except ValueError:
            raise ValueError("Hidden layers must be a whole number.")
        if n_layers < 1:
            raise ValueError("Use at least 1 hidden layer.")

        try:
            neurons = [int(n) for n in self.neurons_var.get().split(",")
                       if n.strip()]
        except ValueError:
            raise ValueError("Neurons must be whole numbers separated by commas.")
        if len(neurons) != n_layers:
            raise ValueError(f"You chose {n_layers} layer(s), so enter exactly "
                             f"{n_layers} neuron value(s).")
        if any(n < 1 for n in neurons):
            raise ValueError("Each layer needs at least 1 neuron.")

        try:
            lr = float(self.lr_var.get())
        except ValueError:
            raise ValueError("Learning rate must be a number, like 0.01.")
        if lr <= 0:
            raise ValueError("Learning rate must be bigger than 0.")

        try:
            epochs = int(self.epochs_var.get())
        except ValueError:
            raise ValueError("Epochs must be a whole number.")
        if epochs < 1:
            raise ValueError("Epochs must be at least 1.")

        return dict(hidden=neurons, layers=n_layers, lr=lr, epochs=epochs,
                    sigmoid=self.activation_var.get() == "Sigmoid",
                    bias=self.bias_var.get())

    # ------------------------------------------------------------ training
    def start_training(self):
        if self.training:
            return
        try:
            cfg = self._read_settings()
        except ValueError as e:
            self._set_status(str(e), "error")
            return

        self.training = True
        self.train_btn.configure(state="disabled")
        self.classify_btn.configure(state="disabled")
        self.progress.start(12)
        self._set_status("Training... this can take a moment.")

        threading.Thread(target=self._train_worker, args=(cfg,),
                         daemon=True).start()
        self.root.after(100, self._poll_results)

    def _train_worker(self, cfg):
        """Runs in a background thread so the window never freezes."""
        try:
            X_train, y_train, X_test, y_test, s_mean, s_std, classes = \
                load_and_preprocess_task2()

            weights, biases = Mlp_train(
                x=X_train, y=y_train,
                n_of_neurons_per_hidden_layer=cfg["hidden"],
                n_hidden_layers=cfg["layers"],
                activation_sigmoid=cfg["sigmoid"],
                learning_rate=cfg["lr"], epochs=cfg["epochs"],
                over_all_error_threshold=-1.0,
                use_bias=cfg["bias"],
            )

            y_pred = [Mlp_predict(x, weights, biases, cfg["sigmoid"])
                      for x in X_test]
            acc, conf_mat = evaluate_metrics_multi(y_test, y_pred)

            self.results_queue.put(("ok", dict(
                weights=weights, biases=biases, use_sigmoid=cfg["sigmoid"],
                mean=s_mean, std=s_std, classes=classes,
                acc=acc, conf=conf_mat)))
        except Exception as e:  # report any error back to the window
            self.results_queue.put(("error", str(e)))

    def _poll_results(self):
        try:
            kind, payload = self.results_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_results)
            return

        self.progress.stop()
        self.training = False
        self.train_btn.configure(state="normal")

        if kind == "error":
            self._set_status(f"Training failed: {payload}", "error")
            return

        self.state = {k: payload[k] for k in
                      ("weights", "biases", "use_sigmoid", "mean", "std",
                       "classes")}
        self.classify_btn.configure(state="normal")
        self._show_results(payload["acc"], payload["conf"], payload["classes"])
        self._set_status(f"Done. Test accuracy: {payload['acc'] * 100:.2f}%",
                         "ok")

    # ------------------------------------------------------------- results
    def _show_results(self, acc, conf, classes):
        conf = np.asarray(conf)
        n = conf.shape[0]
        labels = [str(classes[i]) if i < len(classes) else f"Class {i}"
                  for i in range(n)]
        self.last_conf, self.last_labels = conf, labels

        self.acc_label.configure(text=f"Overall accuracy: {acc * 100:.2f}%")
        self._draw_confusion()

        self.tree.delete(*self.tree.get_children())
        for i, name in enumerate(labels):
            tp = conf[i, i]
            col, row = conf[:, i].sum(), conf[i, :].sum()
            prec = tp / col if col else 0.0
            rec = tp / row if row else 0.0
            self.tree.insert("", "end", values=(
                name, f"{prec * 100:.1f}%", f"{rec * 100:.1f}%", int(row)))

        self.notebook.select(0)

    def _draw_confusion(self):
        c = self.cm_canvas
        c.delete("all")
        if self.last_conf is None:
            return
        conf, labels = self.last_conf, self.last_labels
        n = conf.shape[0]
        w, h = c.winfo_width(), c.winfo_height()
        left, top = 90, 40
        size = max(30, min((w - left - 10) // n, (h - top - 10) // n))
        peak = conf.max() or 1

        c.create_text(left + size * n / 2, 12, text="Predicted",
                      fill=MUTED, font=("Segoe UI", 9))
        c.create_text(14, top + size * n / 2, text="True", angle=90,
                      fill=MUTED, font=("Segoe UI", 9))

        for i in range(n):
            c.create_text(left + i * size + size / 2, top - 12,
                          text=labels[i], fill=TEXT, font=("Segoe UI", 9))
            c.create_text(left - 8, top + i * size + size / 2,
                          text=labels[i], anchor="e", fill=TEXT,
                          font=("Segoe UI", 9))
            for j in range(n):
                v = int(conf[i, j])
                t = v / peak
                # white -> accent blue
                r = int(255 - t * (255 - 0x1f))
                g = int(255 - t * (255 - 0x6f))
                b = int(255 - t * (255 - 0x8b))
                x0, y0 = left + j * size, top + i * size
                c.create_rectangle(x0, y0, x0 + size, y0 + size,
                                   fill=f"#{r:02x}{g:02x}{b:02x}",
                                   outline="#d8dde3")
                c.create_text(x0 + size / 2, y0 + size / 2, text=str(v),
                              fill="white" if t > 0.55 else TEXT,
                              font=("Segoe UI", 11, "bold"))

    # -------------------------------------------------------- single sample
    def classify_sample(self):
        if not self.state:
            self._set_status("Train the model first.", "error")
            return

        values = []
        for (label, kind), widget in zip(FEATURES, self.sample_widgets):
            text = widget.get().strip()
            if not text:
                self._set_status(f"'{label}' is empty.", "error")
                return
            if kind == "combo" and text.title() in ORIGIN_MAP:
                values.append(float(ORIGIN_MAP[text.title()]))
                continue
            try:
                values.append(float(text))
            except ValueError:
                hint = (" Pick a location from the list or type a number."
                        if kind == "combo" else "")
                self._set_status(f"'{label}' is not a valid number.{hint}",
                                 "error")
                return

        try:
            sample = (np.array(values) - self.state["mean"]) / self.state["std"]
            pred = np.asarray(Mlp_predict(
                sample, self.state["weights"], self.state["biases"],
                self.state["use_sigmoid"])).flatten()
        except Exception as e:
            self._set_status(f"Could not classify: {e}", "error")
            return

        classes = self.state["classes"]
        idx = int(np.argmax(pred))
        self._show_prediction(pred, classes, idx)
        self._set_status(f"Predicted class: {classes[idx]}", "ok")

    def _show_prediction(self, pred, classes, best):
        self.pred_title.configure(text=f"Predicted class: {classes[best]}")
        for child in self.pred_frame.winfo_children():
            child.destroy()

        # Tanh outputs are in [-1, 1], sigmoid in [0, 1]: map both to 0-100
        lo = -1.0 if not self.state["use_sigmoid"] else 0.0
        for i, v in enumerate(pred):
            name = str(classes[i]) if i < len(classes) else f"Class {i}"
            font = ("Segoe UI", 10, "bold") if i == best else ("Segoe UI", 10)
            ttk.Label(self.pred_frame, text=name, style="Card.TLabel",
                      font=font).grid(row=i, column=0, sticky="w", pady=4)
            pct = float(np.clip((v - lo) / (1.0 - lo), 0, 1)) * 100
            ttk.Progressbar(self.pred_frame, value=pct, maximum=100,
                            style="Bar.Horizontal.TProgressbar").grid(
                row=i, column=1, sticky="ew", padx=10)
            ttk.Label(self.pred_frame, text=f"{v:.4f}",
                      style="Card.TLabel").grid(row=i, column=2, sticky="e")
        self.notebook.select(self.pred_tab)


if __name__ == "__main__":
    root = tk.Tk()
    MlpApp(root)
    root.mainloop()