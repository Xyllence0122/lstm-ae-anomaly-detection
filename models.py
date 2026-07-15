# -*- coding: utf-8 -*-
"""
模型定義與訓練/評分工具：
- LSTMAutoEncoder：時序重建模型（支援變長序列）
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
