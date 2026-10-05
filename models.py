# -*- coding: utf-8 -*-
"""
模型定義與訓練/評分工具：
- LSTMAutoEncoder：時序重建模型（支援變長序列）
- SlidingWindowLSTMAutoEncoder：以因果式 trailing window 做線上重建
- DenseAutoEncoder：無時序結構的全連接 AE（baseline，固定長度輸入）
- 變長資料以「長度分桶」方式訓練與推論
- 有 CUDA 時自動使用 GPU（checkpoint 一律存成 CPU tensor，存檔可攜）
"""
import numpy as np
import torch
import torch.nn as nn

# 訓練/推論裝置：有 GPU 用 GPU，沒有就 CPU（所有存檔皆為 CPU tensor）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMAutoEncoder(nn.Module):
    """
    Encoder LSTM 將整段序列壓縮成 latent 向量，
    Decoder LSTM 從 latent 重建整段序列（長度跟隨輸入，天然支援變長）。
    只用正常資料訓練 → 異常波形的重建誤差會明顯偏高。
    """

    def __init__(self, n_features: int, hidden_size: int = 64, latent_size: int = 16):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, n_features)

    def forward(self, x):                       # x: (B, T, F)
        T = x.shape[1]
        _, (h, _) = self.encoder(x)
        z = self.to_latent(h[-1])
        dec_in = self.from_latent(z).unsqueeze(1).repeat(1, T, 1)
        dec_out, _ = self.decoder(dec_in)
        return self.output(dec_out)


class SlidingWindowLSTMAutoEncoder(LSTMAutoEncoder):
    """Deployment label for the complete-cycle-trained LSTM-AE architecture.

    V2 reuses weights trained by Step 3 on complete normal process cycles; it
    does not retrain this subclass on sampled windows. The class intentionally
    has no architectural differences from ``LSTMAutoEncoder``. Causal behavior
    is introduced only at inference, where sample ``t`` is scored from
    ``x[t-window+1:t+1]`` without future observations.
    """


class DenseAutoEncoder(nn.Module):
    """把 (T, F) 攤平成向量的全連接 AE — 沒有時序歸納偏置的對照組"""

    def __init__(self, seq_len: int, n_features: int, latent_size: int = 16):
        super().__init__()
        d = seq_len * n_features
        self.seq_len, self.n_features = seq_len, n_features
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(),
            nn.Linear(128, latent_size), nn.ReLU(),
            nn.Linear(latent_size, 128), nn.ReLU(),
            nn.Linear(128, d),
        )

    def forward(self, x):                       # x: (B, T, F)
        B = x.shape[0]
        return self.net(x.reshape(B, -1)).reshape(B, self.seq_len, self.n_features)


class LSTMForecaster(nn.Module):
    """Causal one-step forecaster for online anomaly detection.

    The output at index ``t`` predicts input index ``t + 1`` and therefore only
    depends on samples ``0..t``. Unlike the autoencoder, this model can retain
    its recurrent state and emit a new prediction as each sensor sample arrives.
    """

    def __init__(self, n_features: int, hidden_size: int = 64,
                 num_layers: int = 1):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers,
                            batch_first=True)
        self.output = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        encoded, _ = self.lstm(x)
        return self.output(encoded)


class LSTMForecasterStep(nn.Module):
    """State-explicit wrapper used for TorchScript edge inference."""

    def __init__(self, forecaster: LSTMForecaster):
        super().__init__()
        self.lstm = forecaster.lstm
        self.output = forecaster.output

    def forward(self, x_t, hidden, cell):
        encoded, (next_hidden, next_cell) = self.lstm(x_t, (hidden, cell))
        return self.output(encoded), next_hidden, next_cell


# ---------- 變長資料工具 ----------

def _buckets_by_length(X_list):
    """依序列長度分桶：{T: [indices]}（LSTM 一個 batch 內長度需一致）"""
    buckets = {}
    for i, x in enumerate(X_list):
        buckets.setdefault(len(x), []).append(i)
    return buckets


def _cpu_state(model):
    """取 model.state_dict() 的 CPU 深拷貝（GPU 訓練時存檔仍可在任何機器載入）"""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_collect_checkpoints(model, Xtr, Xva, epochs=400, batch_size=32,
                              lr=1e-3, ckpt_every=20, seed=42, verbose=True):
    """
    只用正常資料訓練 AE，每 ckpt_every 個 epoch 保存一份權重。
    X 可為變長 list（長度分桶）或固定長度 list/array。

    動機：AE 訓練越久重建能力越強，連異常波形都能重建，偵測力反而下降；
    且不同異常型態偏好不同收斂程度。因此保留整條訓練軌跡的 checkpoints，
    再由呼叫端以「驗證異常集 F1」挑選（測試集只在最終評估使用一次）。

    回傳 (hist, checkpoints)；checkpoints = [(epoch, cpu_state_dict), ...]
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    tr_buckets = {T: torch.as_tensor(np.stack([Xtr[i] for i in idx]),
                                     dtype=torch.float32).to(DEVICE)
                  for T, idx in _buckets_by_length(Xtr).items()}
    va_buckets = {T: torch.as_tensor(np.stack([Xva[i] for i in idx]),
                                     dtype=torch.float32).to(DEVICE)
                  for T, idx in _buckets_by_length(Xva).items()}
    n_val = sum(len(v) for v in va_buckets.values())

    hist = {"train": [], "val": []}
    checkpoints = []

    for epoch in range(1, epochs + 1):
        model.train()
        batches = []
        for T, tens in tr_buckets.items():
            perm = rng.permutation(len(tens))
            for i in range(0, len(perm), batch_size):
                batches.append(tens[perm[i:i + batch_size]])
        rng.shuffle(batches)

        tr = []
        for xb in batches:
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            tr.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = sum(loss_fn(model(t), t).item() * len(t)
                           for t in va_buckets.values()) / n_val
        hist["train"].append(float(np.mean(tr)))
        hist["val"].append(val_loss)

        if epoch % ckpt_every == 0:
            checkpoints.append((epoch, _cpu_state(model)))
            if verbose and epoch % (ckpt_every * 5) == 0:
                print(f"epoch {epoch:3d}  train={hist['train'][-1]:.4f}  "
                      f"val={val_loss:.4f}")

    return hist, checkpoints


def train_forecaster_collect_checkpoints(model, Xtr, Xva, epochs=200,
                                         batch_size=32, lr=1e-3,
                                         ckpt_every=20, seed=42,
                                         verbose=True):
    """Train a one-step forecaster on normal sequences and retain checkpoints."""
    if min(map(len, Xtr)) < 2 or min(map(len, Xva)) < 2:
        raise ValueError("Forecaster sequences must contain at least two samples")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    tr_buckets = {T: torch.as_tensor(np.stack([Xtr[i] for i in idx]),
                                     dtype=torch.float32).to(DEVICE)
                  for T, idx in _buckets_by_length(Xtr).items()}
    va_buckets = {T: torch.as_tensor(np.stack([Xva[i] for i in idx]),
                                     dtype=torch.float32).to(DEVICE)
                  for T, idx in _buckets_by_length(Xva).items()}
    n_val = sum(len(v) for v in va_buckets.values())

    hist = {"train": [], "val": []}
    checkpoints = []
    for epoch in range(1, epochs + 1):
        model.train()
        batches = []
        for tens in tr_buckets.values():
            perm = rng.permutation(len(tens))
            for i in range(0, len(perm), batch_size):
                batches.append(tens[perm[i:i + batch_size]])
        rng.shuffle(batches)

        train_losses = []
        for xb in batches:
            opt.zero_grad()
            loss = loss_fn(model(xb[:, :-1]), xb[:, 1:])
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = sum(
                loss_fn(model(t[:, :-1]), t[:, 1:]).item() * len(t)
                for t in va_buckets.values()) / n_val
        hist["train"].append(float(np.mean(train_losses)))
        hist["val"].append(float(val_loss))

        if epoch % ckpt_every == 0:
            checkpoints.append((epoch, _cpu_state(model)))
            if verbose and epoch % (ckpt_every * 5) == 0:
                print(f"epoch {epoch:3d}  train={hist['train'][-1]:.4f}  "
                      f"val={val_loss:.4f}")

    return hist, checkpoints


@torch.no_grad()
def pointwise_errors(model, X_list, batch_size=64):
    """逐時間步、逐 sensor 的平方誤差；支援變長，回傳與輸入同序的 list[(T,F)]"""
    model.eval()
    dev = next(model.parameters()).device
    out = [None] * len(X_list)
    for T, idx in _buckets_by_length(X_list).items():
        tens = torch.as_tensor(np.stack([X_list[i] for i in idx]),
                               dtype=torch.float32).to(dev)
        for i in range(0, len(idx), batch_size):
            xb = tens[i:i + batch_size]
            err = ((model(xb) - xb) ** 2).cpu().numpy()
            for k, j in enumerate(idx[i:i + batch_size]):
                out[j] = err[k]
    return out


@torch.no_grad()
def sliding_window_error_summaries(model, X_list, window_size,
                                   batch_size=1024):
    """Return causal reconstruction-error summaries for trailing windows.

    The returned mapping contains value-error and first-difference-error
    reductions. Each item has shape ``(T - window_size + 1, F)``. Output row
    ``i`` maps to source sample ``i + window_size - 1`` and cannot depend on
    later samples.
    """
    window_size = int(window_size)
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    model.eval()
    dev = next(model.parameters()).device
    modes = ("last", "mean", "max", "delta_mean", "delta_max")
    out = {name: [None] * len(X_list) for name in modes}
    for length, indices in _buckets_by_length(X_list).items():
        if length < window_size:
            raise ValueError(
                f"window_size={window_size} exceeds sequence length {length}")
        sequences = torch.as_tensor(
            np.stack([X_list[index] for index in indices]),
            dtype=torch.float32,
            device=dev,
        )
        # unfold shape: (N, number_of_windows, F, W)
        windows = sequences.unfold(1, window_size, 1).permute(0, 1, 3, 2)
        n_sequences, n_windows, _, n_features = windows.shape
        flat = windows.reshape(-1, window_size, n_features)
        summaries = {name: [] for name in out}
        for start in range(0, len(flat), batch_size):
            batch = flat[start:start + batch_size]
            reconstruction = model(batch)
            squared = (reconstruction - batch) ** 2
            delta_squared = (
                (reconstruction[:, 1:] - reconstruction[:, :-1]) -
                (batch[:, 1:] - batch[:, :-1])
            ) ** 2
            summaries["last"].append(squared[:, -1].cpu())
            summaries["mean"].append(squared.mean(dim=1).cpu())
            summaries["max"].append(squared.amax(dim=1).cpu())
            summaries["delta_mean"].append(delta_squared.mean(dim=1).cpu())
            summaries["delta_max"].append(delta_squared.amax(dim=1).cpu())
        for name, chunks in summaries.items():
            errors = torch.cat(chunks).numpy().reshape(
                n_sequences, n_windows, n_features)
            for local_index, source_index in enumerate(indices):
                out[name][source_index] = errors[local_index]
    return out


def sliding_window_errors(model, X_list, window_size, reduction="last",
                          batch_size=1024):
    """Return one configured error reduction for every causal trailing window."""
    if reduction not in ("last", "mean", "max", "delta_mean", "delta_max"):
        raise ValueError(
            "reduction must be one of: last, mean, max, delta_mean, delta_max")
    return sliding_window_error_summaries(
        model, X_list, window_size, batch_size)[reduction]


def sliding_window_last_errors(model, X_list, window_size, batch_size=1024):
    """Backward-compatible final-sample reduction helper."""
    return sliding_window_errors(
        model, X_list, window_size, "last", batch_size)


@torch.no_grad()
def forecaster_pointwise_errors(model, X_list, batch_size=64):
    """Causal one-step squared errors; output item shape is ``(T - 1, F)``."""
    model.eval()
    dev = next(model.parameters()).device
    out = [None] * len(X_list)
    for T, idx in _buckets_by_length(X_list).items():
        if T < 2:
            raise ValueError("Forecaster sequences must contain at least two samples")
        tens = torch.as_tensor(np.stack([X_list[i] for i in idx]),
                               dtype=torch.float32).to(dev)
        for i in range(0, len(idx), batch_size):
            xb = tens[i:i + batch_size]
            err = ((model(xb[:, :-1]) - xb[:, 1:]) ** 2).cpu().numpy()
            for k, j in enumerate(idx[i:i + batch_size]):
                out[j] = err[k]
    return out


def sensor_peak_scores(err_list, window=5):
    """
    每片 wafer、每個 sensor 的「平滑誤差峰值」，shape (N, F)。
    沿時間軸做移動平均平滑後取最大值——局部異常不被整片平均稀釋。
    實作以 cumsum 一次算完該片所有 sensors（與 np.convolve mode="valid" 等價）。
    """
    N = len(err_list)
    F = err_list[0].shape[1]
    peaks = np.empty((N, F))
    for n in range(N):
        e = err_list[n]                      # (T, F)
        cs = np.cumsum(e, axis=0, dtype=np.float64)
        cs = np.vstack([np.zeros((1, F)), cs])
        ma = (cs[window:] - cs[:-window]) / window   # (T-window+1, F) 移動平均
        peaks[n] = ma.max(axis=0)
    return peaks


def streaming_score_curves(err_list, window=5, calib=None):
    """Convert causal point errors to trailing-window online score curves.

    Each returned score can be emitted immediately when its trailing window is
    complete. No centered smoothing or future observation is used.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    denom = None
    if calib is not None:
        denom = np.asarray(calib, dtype=float)
        denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)

    curves = []
    for err in err_list:
        if len(err) < window:
            raise ValueError(
                f"window={window} exceeds an error sequence of length {len(err)}")
        cs = np.cumsum(err, axis=0, dtype=np.float64)
        cs = np.vstack([np.zeros((1, err.shape[1])), cs])
        moving = (cs[window:] - cs[:-window]) / window
        if denom is not None:
            moving = moving / denom
        curves.append(moving.max(axis=1))
    return curves


def combine_peaks(peaks, calib=None):
    """合併 per-sensor 峰值成單一異常分數：max over sensors（可選校正）"""
    if calib is not None:
        peaks = peaks / calib
    return peaks.max(axis=1)


def make_threshold(val_scores, rule):
    """閾值規則：max 型分數右偏，p99 比 mean+3σ 更穩健，兩者都納入網格"""
    if rule == "p99":
        return float(np.percentile(val_scores, 99))
    if rule == "mean3sigma":
        return float(val_scores.mean() + 3 * val_scores.std())
    raise ValueError(f"未知的閾值規則：{rule!r}（可用：'mean3sigma'、'p99'）")


def grid_select(model, checkpoints, Xva, Xva_anom, y_va_anom,
                windows=(5, 9), thr_rules=("mean3sigma", "p99"), verbose=True):
    """
    在驗證集上掃「checkpoint × 平滑窗口 × 峰值校正 × 閾值規則」網格，
    以 F1 選出最佳組合（不同異常型態偏好不同收斂程度，單一早停點兩頭不討好）。

    回傳 dict(epoch, window, use_calib, thr_rule, f1, state)；state 為 CPU tensor。
    """
    from sklearn.metrics import f1_score

    y_true = np.concatenate([np.zeros(len(Xva), dtype=int),
                             (np.asarray(y_va_anom) > 0).astype(int)])
    best = None
    for epoch, state in checkpoints:
        model.load_state_dict(state)
        va_pw = pointwise_errors(model, Xva)
        an_pw = pointwise_errors(model, Xva_anom)
        for w in windows:
            va_peaks = sensor_peak_scores(va_pw, w)
            an_peaks = sensor_peak_scores(an_pw, w)
            for use_calib in (False, True):
                calib = va_peaks.mean(axis=0) if use_calib else None
                va_s = combine_peaks(va_peaks, calib)
                an_s = combine_peaks(an_peaks, calib)
                for rule in thr_rules:
                    thr = make_threshold(va_s, rule)
                    y_pred = (np.concatenate([va_s, an_s]) > thr).astype(int)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    if best is None or f1 > best["f1"] + 1e-9:
                        best = {"epoch": epoch, "window": w, "use_calib": use_calib,
                                "thr_rule": rule, "f1": float(f1),
                                "state": {k: v.clone() for k, v in state.items()}}
    if verbose:
        print(f"網格選擇：epoch={best['epoch']}, window={best['window']}, "
              f"校正={best['use_calib']}, 閾值={best['thr_rule']}, "
              f"驗證 F1={best['f1']:.3f}")
    return best


def forecaster_grid_select(model, checkpoints, Xva, Xva_anom, y_va_anom,
                           windows=(3, 5, 9),
                           thr_rules=("mean3sigma", "p99"),
                           anom_onset_indices=None, verbose=True):
    """Select a causal forecaster checkpoint and event-level score settings."""
    from sklearn.metrics import f1_score

    if (anom_onset_indices is not None and
            len(anom_onset_indices) != len(Xva_anom)):
        raise ValueError("anom_onset_indices must align with Xva_anom")
    y_true = np.concatenate([np.zeros(len(Xva), dtype=int),
                             (np.asarray(y_va_anom) > 0).astype(int)])
    best = None
    for epoch, state in checkpoints:
        model.load_state_dict(state)
        va_pw = forecaster_pointwise_errors(model, Xva)
        an_pw = forecaster_pointwise_errors(model, Xva_anom)
        for window in windows:
            if any(len(e) < window for e in va_pw + an_pw):
                continue
            va_peaks = sensor_peak_scores(va_pw, window)
            for use_calib in (False, True):
                calib = va_peaks.mean(axis=0) if use_calib else None
                va_curves = streaming_score_curves(va_pw, window, calib)
                an_curves = streaming_score_curves(an_pw, window, calib)
                va_scores = np.asarray([curve.max() for curve in va_curves])
                if anom_onset_indices is None:
                    an_scores = np.asarray([curve.max() for curve in an_curves])
                else:
                    an_scores = np.asarray([
                        curve[max(int(onset) - window, 0):].max()
                        for curve, onset in zip(an_curves, anom_onset_indices)
                    ])
                for rule in thr_rules:
                    threshold = make_threshold(va_scores, rule)
                    pred = (np.concatenate([va_scores, an_scores]) >
                            threshold).astype(int)
                    f1 = f1_score(y_true, pred, zero_division=0)
                    if best is None or f1 > best["f1"] + 1e-9:
                        best = {
                            "epoch": epoch,
                            "window": window,
                            "use_calib": use_calib,
                            "thr_rule": rule,
                            "f1": float(f1),
                            "state": {k: v.clone() for k, v in state.items()},
                        }
    if best is None:
        raise ValueError("No valid forecaster score window for these sequences")
    if verbose:
        print(f"Forecaster grid: epoch={best['epoch']}, "
              f"window={best['window']}, calibrated={best['use_calib']}, "
              f"threshold={best['thr_rule']}, validation F1={best['f1']:.3f}")
    return best
