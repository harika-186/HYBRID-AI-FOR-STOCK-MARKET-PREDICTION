# stock_gui_final_ready.py
"""
Stock GUI — Transformer (hybrid) + QINN (stock-only)
- Combines fixes requested (no crashes, plots displayed) and a light modernization
- No background threads (UI will block while training/predicting)
- Requires PyTorch for training; transformers optional for GPT features
- Save as stock_gui_final_ready.py and run: python stock_gui_final_ready.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Optional nicer theme
try:
    import ttkbootstrap as tb
    TB_AVAILABLE = True
except Exception:
    tb = None
    TB_AVAILABLE = False

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# ML libs
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# transformers optional (GPT2 embeddings)
try:
    from transformers import GPT2Tokenizer, GPT2Model
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu") if TORCH_AVAILABLE else None

# -------------------------
# Models & Datasets
# -------------------------
if TORCH_AVAILABLE:
    class TimeSeriesDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    class TimeSeriesTransformer(nn.Module):
        def __init__(self, input_size, sequence_length, num_layers=2, num_heads=2, d_model=64, d_ff=128, dropout=0.1):
            super().__init__()
            self.input_proj = nn.Linear(input_size, d_model)
            self.pos_enc = self._positional_encoding(sequence_length, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.out = nn.Linear(d_model, 1)
        def _positional_encoding(self, seq_len, d_model):
            pe = torch.zeros(seq_len, d_model)
            position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe.unsqueeze(0)
        def forward(self, x):
            x = self.input_proj(x)
            x = x + self.pos_enc.to(x.device)
            x = self.transformer(x)
            x = x.mean(dim=1)
            return self.out(x).squeeze(-1)

    class QINNLayer(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.weights = nn.Parameter(torch.randn(input_size, output_size) * 0.01)
        def forward(self, x):
            return torch.sin(torch.matmul(x, self.weights))

    class QINN(nn.Module):
        def __init__(self, input_size, hidden_size=64, output_size=32):
            super().__init__()
            self.q1 = QINNLayer(input_size, hidden_size)
            self.q2 = QINNLayer(hidden_size, output_size)
            self.out = nn.Linear(output_size, 1)
        def forward(self, x):
            if x.dim() == 3:
                x = x[:, -1, :]
            x = self.q1(x)
            x = self.q2(x)
            return self.out(x).squeeze(-1)

# -------------------------
# Helpers
# -------------------------
def prepare_data(features, label, sequence_length):
    """
    features: (N, feature_dim)
    label: (N,1)
    returns X (samples, seq_len, feature_dim), y (samples,1), scalerX, scalery
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    scaler_y = MinMaxScaler()
    label_scaled = scaler_y.fit_transform(label)
    X, y = [], []
    for i in range(len(scaled) - sequence_length):
        X.append(scaled[i:i+sequence_length])
        y.append(label_scaled[i+sequence_length])
    X = np.array(X)
    y = np.array(y).reshape(-1,1)
    return X, y, scaler, scaler_y

def get_gpt_features(headlines, log_fn=None):
    """
    Compute or load GPT2 mean embeddings. Saved to model/news.npy
    """
    os.makedirs("model", exist_ok=True)
    fn = "model/news.npy"
    if os.path.exists(fn):
        arr = np.load(fn)
        if log_fn: log_fn(f"Loaded existing {fn}, shape={arr.shape}")
        return arr
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not installed; install or provide model/news.npy")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2").to(device)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    batch = 16
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(headlines), batch):
            batch_texts = headlines[i:i+batch]
            toks = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
            toks = {k: v.to(device) for k,v in toks.items()}
            out = model(**toks)
            vecs = out.last_hidden_state.mean(dim=1).cpu().numpy()
            all_feats.append(vecs)
    all_feats = np.vstack(all_feats)
    np.save(fn, all_feats)
    return all_feats

# -------------------------
# GUI App
# -------------------------
class StockApp:
    def __init__(self, root):
        # root could be tb.Window or tk.Tk
        self.root = root
        self.root.title("Hybrid AI for Stock Markets Transformers & QINN")
        self.root.geometry("1180x740")

        # State variables
        self.train_df = None
        self.test_df = None
        self.news = None
        self.sequence_length = 8

        # scalers & models
        self.scaler_transformer_X = None
        self.scaler_transformer_y = None
        # QINN input scaler (used in training & prediction)
        self.qinn_input_scaler = None
        # y scaler used for inverse transform (labels)
        self.scaler_y = None

        self.transformer_model = None
        self.qinn_model = None

        # store test outputs & metrics for comparison
        self.transformer_test = None
        self.transformer_metrics = None
        self.qinn_test = None
        self.qinn_metrics = None

        self.device = device

        # Build UI (modern but similar layout)
        self.setup_ui()

        # Matplotlib figure area
        self.fig, self.ax = plt.subplots(figsize=(8,5))
        plt.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        self.log("App ready. Device: {}".format(self.device))

    def setup_ui(self):
        # sidebar + main
        sidebar = ttk.Frame(self.root, padding=8)
        sidebar.grid(row=0, column=0, sticky="ns")
        main = ttk.Frame(self.root, padding=8)
        main.grid(row=0, column=1, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Sidebar buttons
        ttk.Label(sidebar, text="Actions", font=("TkDefaultFont", 14)).pack(pady=(0,8))
        ttk.Button(sidebar, text="Load Train CSV", command=self.load_train_csv).pack(fill='x', pady=4)
        ttk.Button(sidebar, text="Load Test CSV (QINN only)", command=self.load_test_csv).pack(fill='x', pady=4)
        ttk.Button(sidebar, text="Compute/Load GPT features", command=self.compute_gpt).pack(fill='x', pady=4)
        ttk.Separator(sidebar).pack(fill='x', pady=6)
        ttk.Button(sidebar, text="Train Transformer (Hybrid)", command=self.train_transformer).pack(fill='x', pady=4)
        ttk.Button(sidebar, text="Train QINN (Stock-only)", command=self.train_qinn).pack(fill='x', pady=4)
        ttk.Separator(sidebar).pack(fill='x', pady=6)
        ttk.Button(sidebar, text="Predict Test Data (QINN ONLY)", command=self.predict_test_qinn).pack(fill='x', pady=4)
        ttk.Button(sidebar, text="Performance Evaluation", command=self.compare_models).pack(fill='x', pady=4)
        ttk.Separator(sidebar).pack(fill='x', pady=6)
        ttk.Button(sidebar, text="Save Models", command=self.save_models).pack(fill='x', pady=4)
        ttk.Button(sidebar, text="Load Models", command=self.load_models).pack(fill='x', pady=4)

        # Controls row
        controls = ttk.Frame(main)
        controls.pack(fill='x', pady=(0,8))
        ttk.Label(controls, text="Sequence length:").pack(side='left')
        self.seq_spin = ttk.Spinbox(controls, from_=2, to=30, width=5)
        self.seq_spin.set(self.sequence_length)
        self.seq_spin.pack(side='left', padx=6)
        ttk.Label(controls, text="Epochs:").pack(side='left', padx=(10,0))
        self.epochs_spin = ttk.Spinbox(controls, from_=1, to=500, width=6)
        self.epochs_spin.set(50)
        self.epochs_spin.pack(side='left', padx=6)
        ttk.Label(controls, text="Batch size:").pack(side='left', padx=(10,0))
        self.batch_spin = ttk.Spinbox(controls, from_=1, to=512, width=6)
        self.batch_spin.set(16)
        self.batch_spin.pack(side='left', padx=6)
        ttk.Label(controls, text="Learning rate:").pack(side='left', padx=(10,0))
        self.lr_entry = ttk.Entry(controls, width=8)
        self.lr_entry.insert(0, "0.0001")
        self.lr_entry.pack(side='left', padx=6)

        # Plot frame
        self.plot_frame = ttk.Frame(main)
        self.plot_frame.pack(fill='both', expand=True)

        # Bottom: logs + data preview
        bottom = ttk.Frame(main)
        bottom.pack(fill='x', pady=(8,0))
        self.log_text = tk.Text(bottom, height=10, wrap='word')
        self.log_text.pack(side='left', fill='both', expand=True, padx=(0,8))
        self.data_preview = ttk.Treeview(bottom, columns=("c1","c2","c3","c4","c5","c6","c7","c8"), show='headings', height=6)
        for i,h in enumerate(["Date","Top1","Open","High","Low","Close","Volume","Adj Close"]):
            self.data_preview.heading(f"c{i+1}", text=h)
        self.data_preview.pack(side='right', fill='y')

    def log(self, msg):
        try:
            self.log_text.insert('end', msg + "\n")
            self.log_text.see('end')
        except Exception:
            print(msg)

    # -------------------------
    # File / Data helpers
    # -------------------------
    def load_train_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if not path:
            self.log("Train CSV load cancelled.")
            return
        try:
            self.train_df = pd.read_csv(path)
            self.log(f"Loaded train CSV: {path} shape={self.train_df.shape}")
            self.preview_df(self.train_df)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            self.log("Error loading train CSV: " + str(e))

    def load_test_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if not path:
            self.log("Test CSV load cancelled.")
            return
        try:
            self.test_df = pd.read_csv(path)
            self.log(f"Loaded test CSV: {path} shape={self.test_df.shape}")
            self.preview_df(self.test_df)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            self.log("Error loading test CSV: " + str(e))

    def preview_df(self, df):
        if df is None: return
        try:
            self.data_preview.delete(*self.data_preview.get_children())
            head = df.head(10)
            for _, row in head.iterrows():
                vals = []
                for col in ["Date","Top1","Open","High","Low","Close","Volume","Adj Close"]:
                    vals.append(row.get(col, ""))
                self.data_preview.insert('', 'end', values=vals)
        except Exception:
            pass

    # -------------------------
    # GPT features
    # -------------------------
    def compute_gpt(self):
        if self.train_df is None:
            messagebox.showwarning("No train CSV", "Load training CSV first.")
            return
        try:
            headlines = self.train_df["Top1"].astype(str).tolist()
            self.log("Computing/loading GPT features (may be slow)...")
            feats = get_gpt_features(headlines, log_fn=self.log)
            self.news = feats
            self.log(f"News features shape: {feats.shape}")
        except Exception as e:
            messagebox.showerror("GPT error", str(e))
            self.log("GPT error: " + str(e))

    # -------------------------
    # Train Transformer (hybrid)
    # -------------------------
    def train_transformer(self):
        if not TORCH_AVAILABLE:
            messagebox.showerror("Torch missing", "PyTorch required.")
            return
        if self.train_df is None:
            messagebox.showwarning("No train CSV", "Load training CSV first.")
            return
        if self.news is None:
            messagebox.showwarning("No GPT features", "Compute/load GPT features first.")
            return

        try:
            seq = int(self.seq_spin.get())
            epochs = int(self.epochs_spin.get())
            batch = int(self.batch_spin.get())
            lr = float(self.lr_entry.get() or 1e-4)

            df = self.train_df.copy()
            for c in ["Open","High","Low","Close","Volume","Adj Close"]:
                if c not in df.columns:
                    messagebox.showerror("Format error", f"Train CSV missing column {c}")
                    return

            numeric = df[["Open","High","Low","Close"]].values
            news = self.news[:len(df)]
            hybrid = np.concatenate([numeric, news], axis=1)
            X, y, scaler_X, scaler_y = prepare_data(hybrid, df[["Adj Close"]].values, seq)

            self.scaler_transformer_X = scaler_X
            self.scaler_transformer_y = scaler_y

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            self.log(f"Transformer training data prepared: X_train={X_train.shape} X_test={X_test.shape}")

            model = TimeSeriesTransformer(input_size=X.shape[2], sequence_length=seq).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()

            train_ds = TimeSeriesDataset(X_train, y_train)
            train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)

            best_loss = float("inf")
            rmse = np.nan; r2 = np.nan; preds_inv = None; y_test_inv = None
            for epoch in range(1, epochs+1):
                model.train()
                running = 0.0
                for xb, yb in train_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device).squeeze(-1)
                    optimizer.zero_grad()
                    preds = model(xb)
                    loss = loss_fn(preds, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    running += loss.item() * xb.size(0)
                train_loss = running / len(train_loader.dataset)

                model.eval()
                with torch.no_grad():
                    preds_test = model(torch.tensor(X_test, dtype=torch.float32).to(self.device)).detach().cpu().numpy().reshape(-1,1)
                    y_test_inv = scaler_y.inverse_transform(y_test).ravel()
                    preds_inv = scaler_y.inverse_transform(preds_test).ravel()
                    rmse = mean_squared_error(y_test_inv, preds_inv, squared=False)
                    r2 = r2_score(y_test_inv, preds_inv)
                self.log(f"[Transformer] Epoch {epoch}/{epochs} train_loss={train_loss:.6f} test_rmse={rmse:.4f} r2={r2:.4f}")

                if train_loss < best_loss:
                    best_loss = train_loss
                    os.makedirs("model", exist_ok=True)
                    torch.save(model.state_dict(), "model/gpt_transformer.pth")

            self.transformer_model = model
            self.transformer_test = (y_test_inv, preds_inv)
            self.transformer_metrics = {"rmse": float(rmse), "r2": float(r2)}
            self.log("Transformer training complete and saved.")

            # Auto-plot on main thread
            self.plot_transformer_test()
        except Exception as e:
            messagebox.showerror("Transformer error", str(e))
            self.log("Transformer training error: " + str(e))

    def plot_transformer_test(self):
        if self.transformer_test is None:
            self.log("No transformer test to plot.")
            return
        y_true, y_pred = self.transformer_test
        try:
            self.fig.clf()
            ax = self.fig.add_subplot(111)
            ax.plot(y_true, label='True (Transformer test)', color='black')
            ax.plot(y_pred, label='Transformer Pred', color='red')
            ax.set_title("Transformer: True vs Pred (test)")
            ax.legend()
            self.canvas.draw()
        except Exception as e:
            self.log("Transformer plot error: " + str(e))

    # -------------------------
    # Train QINN (stock-only)
    # -------------------------
    def train_qinn(self):
        if not TORCH_AVAILABLE:
            messagebox.showerror("Torch missing", "PyTorch required.")
            return
        if self.train_df is None:
            messagebox.showwarning("No train CSV", "Load training CSV first.")
            return

        try:
            seq = int(self.seq_spin.get())
            epochs = int(self.epochs_spin.get())
            batch = int(self.batch_spin.get())
            lr = float(self.lr_entry.get() or 1e-4)

            df = self.train_df.copy()
            for c in ["Open","High","Low","Close","Volume","Adj Close"]:
                if c not in df.columns:
                    messagebox.showerror("Format error", f"Train CSV missing column {c}")
                    return

            numeric = df[["Open","High","Low","Close"]].values
            label = df[["Adj Close"]].values

            # prepare QINN training data (stock-only)
            X, y, scaler_X, scaler_y = prepare_data(numeric, label, seq)

            # IMPORTANT: store QINN scalers with the same names used by prediction
            # qinn_input_scaler (for flattened inputs) and scaler_y (for label)
            # For QINN we flatten sequences before scaling in the earlier code; to be consistent,
            # fit a scaler on flattened sequences:
            X_flat = X.reshape(len(X), -1)
            scaler_h = MinMaxScaler()
            X_flat_scaled = scaler_h.fit_transform(X_flat)
            self.qinn_input_scaler = scaler_h
            self.scaler_y = scaler_y  # labels scaler

            # split (time-series)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            self.log(f"QINN training data prepared: X_train={X_train.shape} X_test={X_test.shape}")

            # flatten for model input
            X_train_flat = X_train.reshape(len(X_train), -1)
            X_test_flat = X_test.reshape(len(X_test), -1)
            # scale flattened
            X_train_flat_s = self.qinn_input_scaler.transform(X_train_flat)
            X_test_flat_s = self.qinn_input_scaler.transform(X_test_flat)

            # reshape for QINN: (samples,1,input_dim)
            input_dim = X_train_flat_s.shape[1]
            train_tensor = X_train_flat_s.reshape(len(X_train_flat_s), 1, -1)
            test_tensor = X_test_flat_s.reshape(len(X_test_flat_s), 1, -1)

            model = QINN(input_size=input_dim).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()

            train_ds = TimeSeriesDataset(train_tensor, y_train)
            train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)

            best_loss = float("inf")
            rmse_q = np.nan; r2_q = np.nan; preds_q_inv = None; y_test_inv = None
            for epoch in range(1, epochs+1):
                model.train()
                running = 0.0
                for xb, yb in train_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device).squeeze(-1)
                    optimizer.zero_grad()
                    preds = model(xb)
                    loss = loss_fn(preds, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    running += loss.item() * xb.size(0)
                train_loss = running / len(train_loader.dataset)

                model.eval()
                with torch.no_grad():
                    Xq_test_t = torch.tensor(test_tensor, dtype=torch.float32).to(self.device)
                    preds_q = model(Xq_test_t).detach().cpu().numpy().reshape(-1,1)
                    preds_q_inv = self.scaler_y.inverse_transform(preds_q).ravel()
                    y_test_inv = self.scaler_y.inverse_transform(y_test).ravel()
                    rmse_q = mean_squared_error(y_test_inv, preds_q_inv, squared=False)
                    r2_q = r2_score(y_test_inv, preds_q_inv)
                self.log(f"[QINN] Epoch {epoch}/{epochs} train_loss={train_loss:.6f} test_rmse={rmse_q:.4f} r2={r2_q:.4f}")

                if train_loss < best_loss:
                    best_loss = train_loss
                    os.makedirs("model", exist_ok=True)
                    torch.save(model.state_dict(), "model/qinn.pth")

            self.qinn_model = model
            self.qinn_test = (y_test_inv, preds_q_inv)
            self.qinn_metrics = {"rmse": float(rmse_q), "r2": float(r2_q)}
            self.log("QINN training finished and saved.")
            # Auto-plot qall
            self.plot_qinn_test()
        except Exception as e:
            messagebox.showerror("QINN error", str(e))
            self.log("QINN training error: " + str(e))

    def plot_qinn_test(self):
        if self.qinn_test is None:
            self.log("No QINN test to plot.")
            return
        y_true, y_pred = self.qinn_test
        try:
            self.fig.clf()
            ax = self.fig.add_subplot(111)
            ax.plot(y_true, label='True (QINN test)', color='black')
            ax.plot(y_pred, label='QINN Pred', color='green')
            ax.set_title("QINN: True vs Pred (validation)")
            ax.legend()
            self.canvas.draw()
        except Exception as e:
            self.log("QINN plot error: " + str(e))

    # -------------------------
    # Predict test data using QINN (first 50 rows) — MATCHES QINN TRAINING
    # -------------------------
    def predict_test_qinn(self):
        if not TORCH_AVAILABLE:
            messagebox.showerror("Torch missing", "PyTorch required.")
            return
        if self.test_df is None:
            messagebox.showwarning("No test CSV", "Load test CSV first.")
            return
        if self.qinn_model is None:
            messagebox.showwarning("No QINN model", "Train QINN first or load model.")
            return

        try:
            df = self.test_df.copy().head(50)

            for c in ["Open","High","Low","Close","Volume","Adj Close"]:
                if c not in df.columns:
                    messagebox.showerror("Format error", f"Test CSV missing column {c}")
                    return

            numeric = df[["Open","High","Low","Close"]].values
            label = df[["Adj Close"]].values
            seq = int(self.seq_spin.get())

            X_test_all, y_test_all, scaler_X_test, scaler_y_test = prepare_data(numeric, label, seq)

            if len(X_test_all) == 0:
                messagebox.showerror("Sequence Error", f"Not enough test rows (need > {seq})")
                return

            # Use qinn_input_scaler used during training if available, else fallback
            scaler_X_used = self.qinn_input_scaler if (hasattr(self, "qinn_input_scaler") and self.qinn_input_scaler is not None) else scaler_X_test
            # Use label scaler from training if available
            scaler_y_used = self.scaler_y if (hasattr(self, "scaler_y") and self.scaler_y is not None) else scaler_y_test

            # flatten like in training
            X_test_flat = X_test_all.reshape(len(X_test_all), -1)
            # If the saved qinn_input_scaler expects flattened dims, transform accordingly
            Xq_s = scaler_X_used.transform(X_test_flat)
            Xq_t = torch.tensor(Xq_s.reshape(len(Xq_s), 1, -1), dtype=torch.float32).to(self.device)

            self.qinn_model.eval()
            with torch.no_grad():
                preds = self.qinn_model(Xq_t).detach().cpu().numpy().reshape(-1,1)

            preds_inv = scaler_y_used.inverse_transform(preds).ravel()
            y_true_inv = scaler_y_used.inverse_transform(y_test_all).ravel()

            rmse = mean_squared_error(y_true_inv, preds_inv, squared=False)
            r2 = r2_score(y_true_inv, preds_inv)

            self.log(f"QINN Test Predictions (first 50 rows) — Samples: {len(preds_inv)}  RMSE: {rmse:.4f}  R2: {r2:.4f}")

            # plot
            self.fig.clf()
            ax = self.fig.add_subplot(111)
            ax.plot(y_true_inv, label='True (Test)', color='black')
            ax.plot(preds_inv, label='QINN Pred', color='green')
            ax.set_title("QINN Test Prediction (First 50 Rows)")
            ax.legend()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Prediction error", str(e))
            self.log("Predict test QINN error: " + str(e))

    # -------------------------
    # Compare / Performance Evaluation
    # -------------------------
    def compare_models(self):
        try:
            # Recompute metrics if missing
            if self.transformer_metrics is None and self.transformer_model is not None and self.train_df is not None and self.news is not None:
                seq = int(self.seq_spin.get())
                numeric = self.train_df[["Open","High","Low","Close"]].values
                hybrid = np.concatenate([numeric[:len(self.news)], self.news], axis=1)
                X, y, sX, sY = prepare_data(hybrid, self.train_df[["Adj Close"]].values, seq)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                with torch.no_grad():
                    preds_test = self.transformer_model(torch.tensor(X_test, dtype=torch.float32).to(self.device)).detach().cpu().numpy().reshape(-1,1)
                preds_inv = sY.inverse_transform(preds_test).ravel()
                y_test_inv = sY.inverse_transform(y_test).ravel()
                self.transformer_test = (y_test_inv, preds_inv)
                self.transformer_metrics = {"rmse": float(mean_squared_error(y_test_inv, preds_inv, squared=False)),
                                            "r2": float(r2_score(y_test_inv, preds_inv))}
            if self.qinn_metrics is None and self.qinn_model is not None and self.train_df is not None:
                seq = int(self.seq_spin.get())
                numeric = self.train_df[["Open","High","Low","Close"]].values
                X1, y1, sX1, sY1 = prepare_data(numeric, self.train_df[["Adj Close"]].values, seq)
                X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, shuffle=False)
                X_test_flat = X_test1.reshape(len(X_test1), -1)
                # use qinn_input_scaler if available
                scaler_used = self.qinn_input_scaler if (hasattr(self, "qinn_input_scaler") and self.qinn_input_scaler is not None) else MinMaxScaler().fit(X_test_flat)
                X_test_flat_s = scaler_used.transform(X_test_flat)
                with torch.no_grad():
                    preds_q = self.qinn_model(torch.tensor(X_test_flat_s.reshape(len(X_test_flat_s),1,-1), dtype=torch.float32).to(self.device)).detach().cpu().numpy().reshape(-1,1)
                preds_q_inv = sY1.inverse_transform(preds_q).ravel()
                y1_test_inv = sY1.inverse_transform(y_test1).ravel()
                self.qinn_test = (y1_test_inv, preds_q_inv)
                self.qinn_metrics = {"rmse": float(mean_squared_error(y1_test_inv, preds_q_inv, squared=False)),
                                     "r2": float(r2_score(y1_test_inv, preds_q_inv))}

            if self.transformer_metrics is None or self.qinn_metrics is None:
                messagebox.showwarning("Insufficient data", "Need both models and/or training data to compare.")
                return

            # Draw combined figure: bar + dual-line
            self.fig.clf()
            ax_bar = self.fig.add_subplot(211)
            ax_lines = self.fig.add_subplot(212)

            algs = ['Transformer','QINN']
            rmse_vals = [self.transformer_metrics['rmse'], self.qinn_metrics['rmse']]
            r2_vals = [self.transformer_metrics['r2']*100.0, self.qinn_metrics['r2']*100.0]
            x = np.arange(len(algs))
            width = 0.35
            ax_bar.bar(x - width/2, rmse_vals, width, label='RMSE')
            ax_bar.bar(x + width/2, r2_vals, width, label='R2*100')
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(algs)
            ax_bar.set_title("Comparison: RMSE and R2*100")
            ax_bar.legend()

            if self.transformer_test is not None:
                yt, yp = self.transformer_test
                ax_lines.plot(yt, label='True (Transformer test)', color='black', alpha=0.8)
                ax_lines.plot(yp, label='Transformer Pred', color='red', alpha=0.8)
            if self.qinn_test is not None:
                yq_t, yq_p = self.qinn_test
                ax_lines.plot(yq_p, label='QINN Pred', color='green', alpha=0.8)
                if self.transformer_test is None:
                    ax_lines.plot(yq_t, label='True (QINN test)', color='black', alpha=0.8)

            ax_lines.set_title("True vs Predicted (Transformer & QINN)")
            ax_lines.legend()
            self.canvas.draw()

            self.log("=== Model Comparison ===")
            self.log(f"Transformer RMSE: {self.transformer_metrics['rmse']:.4f}  R2: {self.transformer_metrics['r2']:.4f}")
            self.log(f"QINN RMSE: {self.qinn_metrics['rmse']:.4f}  R2: {self.qinn_metrics['r2']:.4f}")
        except Exception as e:
            messagebox.showerror("Compare error", str(e))
            self.log("Compare error: " + str(e))

    # -------------------------
    # Save / Load models and scalers
    # -------------------------
    def save_models(self):
        try:
            os.makedirs("model", exist_ok=True)
            if self.transformer_model is not None:
                torch.save(self.transformer_model.state_dict(), "model/gpt_transformer.pth")
            if self.qinn_model is not None:
                torch.save(self.qinn_model.state_dict(), "model/qinn.pth")
            with open("model/scalers.pkl", "wb") as f:
                pickle.dump({
                    "scaler_transformer_X": self.scaler_transformer_X,
                    "scaler_transformer_y": self.scaler_transformer_y,
                    "qinn_input_scaler": self.qinn_input_scaler,
                    "scaler_y": self.scaler_y,
                    "transformer_test": self.transformer_test,
                    "transformer_metrics": self.transformer_metrics,
                    "qinn_test": self.qinn_test,
                    "qinn_metrics": self.qinn_metrics
                }, f)
            self.log("Saved models & scalers to model/")
        except Exception as e:
            messagebox.showerror("Save error", str(e))
            self.log("Save error: " + str(e))

    def load_models(self):
        try:
            if os.path.exists("model/gpt_transformer.pth"):
                if self.train_df is None or self.news is None:
                    self.log("Load transformer: please load train CSV and GPT features first.")
                else:
                    numeric = self.train_df[["Open","High","Low","Close"]].values
                    hybrid = np.concatenate([numeric[:len(self.news)], self.news], axis=1)
                    seq = int(self.seq_spin.get())
                    X, y, _, _ = prepare_data(hybrid, self.train_df[["Adj Close"]].values, seq)
                    model = TimeSeriesTransformer(input_size=X.shape[2], sequence_length=seq).to(self.device)
                    model.load_state_dict(torch.load("model/gpt_transformer.pth", map_location=self.device))
                    self.transformer_model = model
                    self.log("Loaded transformer model.")
            if os.path.exists("model/qinn.pth"):
                if self.train_df is None:
                    self.log("Load QINN: please load train CSV first.")
                else:
                    numeric = self.train_df[["Open","High","Low","Close"]].values
                    seq = int(self.seq_spin.get())
                    X1, y1, sX1, sY1 = prepare_data(numeric, self.train_df[["Adj Close"]].values, seq)
                    input_dim = X1.reshape(len(X1), -1).shape[1]
                    model = QINN(input_size=input_dim).to(self.device)
                    model.load_state_dict(torch.load("model/qinn.pth", map_location=self.device))
                    self.qinn_model = model
                    self.log("Loaded QINN model.")
            if os.path.exists("model/scalers.pkl"):
                with open("model/scalers.pkl", "rb") as f:
                    d = pickle.load(f)
                    self.scaler_transformer_X = d.get("scaler_transformer_X")
                    self.scaler_transformer_y = d.get("scaler_transformer_y")
                    self.qinn_input_scaler = d.get("qinn_input_scaler")
                    self.scaler_y = d.get("scaler_y")
                    self.transformer_test = d.get("transformer_test")
                    self.transformer_metrics = d.get("transformer_metrics")
                    self.qinn_test = d.get("qinn_test")
                    self.qinn_metrics = d.get("qinn_metrics")
                    self.log("Loaded scalers & metrics.")
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            self.log("Load error: " + str(e))

# -------------------------
# Entry point
# -------------------------
def main():
    if TB_AVAILABLE:
        root = tb.Window(themename="litera")
    else:
        root = tk.Tk()
    app = StockApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
