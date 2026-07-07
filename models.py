# -*- coding: utf-8 -*-
"""
模型定義：
- LSTMAutoEncoder：時序重建模型（本專案主角）
- DenseAutoEncoder：無時序結構的全連接 AE（baseline，用於對照）
"""
import torch
import torch.nn as nn


class LSTMAutoEncoder(nn.Module):
    """
    Encoder LSTM 將整段序列壓縮成 latent 向量，
    Decoder LSTM 從 latent 重建整段序列。
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
        _, (h, _) = self.encoder(x)             # h: (1, B, hidden)
        z = self.to_latent(h[-1])               # (B, latent)
        dec_in = self.from_latent(z)            # (B, hidden)
        dec_in = dec_in.unsqueeze(1).repeat(1, T, 1)  # 沿時間軸展開
        dec_out, _ = self.decoder(dec_in)
        return self.output(dec_out)             # (B, T, F)


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
        out = self.net(x.reshape(B, -1))
        return out.reshape(B, self.seq_len, self.n_features)


@torch.no_grad()
def reconstruction_errors(model, X, batch_size=64, device="cpu"):
    """每片 wafer 的重建 MSE（時間 × sensor 平均），回傳 shape (N,)"""
    model.eval()
    errs = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
        recon = model(xb)
        errs.append(((recon - xb) ** 2).mean(dim=(1, 2)).cpu())
    return torch.cat(errs).numpy()


def train_collect_checkpoints(model, Xtr, Xva, epochs=400, batch_size=32,
                              lr=1e-3, ckpt_every=10, seed=42, verbose=True):
    """
    只用正常資料訓練 AE，每 ckpt_every 個 epoch 保存一份權重。

    動機：AE 訓練越久重建能力越強，連異常波形都能重建，偵測力反而下降；
    且不同異常型態偏好不同收斂程度。因此保留整條訓練軌跡的 checkpoints，
    再由呼叫端以「驗證異常集 F1」挑選（測試集只在最終評估使用一次）。

    回傳 (hist, checkpoints)；checkpoints = [(epoch, state_dict), ...]
    """
    import numpy as np

    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float32)

    hist = {"train": [], "val": []}
    checkpoints = []

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        tr = []
        for i in range(0, len(perm), batch_size):
            xb = Xtr_t[perm[i:i + batch_size]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            tr.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva_t), Xva_t).item()
        hist["train"].append(float(np.mean(tr)))
        hist["val"].append(val_loss)

        if epoch % ckpt_every == 0:
            checkpoints.append(
                (epoch, {k: v.clone() for k, v in model.state_dict().items()}))
            if verbose and epoch % (ckpt_every * 5) == 0:
                print(f"epoch {epoch:3d}  train={hist['train'][-1]:.4f}  "
                      f"val={val_loss:.4f}")

    return hist, checkpoints


def sensor_peak_scores(pw_err, window=5):
    """
    每片 wafer、每個 sensor 的「平滑誤差峰值」，shape (N, F)。
    沿時間軸做移動平均平滑後取最大值——局部異常不被整片平均稀釋。
    """
    import numpy as np
    kernel = np.ones(window) / window
    N, T, F = pw_err.shape
    peaks = np.empty((N, F))
    for n in range(N):
        for f in range(F):
            peaks[n, f] = np.convolve(pw_err[n, :, f], kernel, mode="valid").max()
    return peaks


def combine_peaks(peaks, calib=None):
    """
    合併 per-sensor 峰值成單一異常分數：max over sensors。
    calib（shape (F,)）：以驗證集正常峰值的平均做校正，
    讓「正常峰值水準」不同的 sensors 可比較後再取 max。
    """
    if calib is not None:
        peaks = peaks / calib
    return peaks.max(axis=1)


@torch.no_grad()
def pointwise_errors(model, X, batch_size=64, device="cpu"):
    """逐時間步、逐 sensor 的平方誤差，回傳 shape (N, T, F)"""
    model.eval()
    errs = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
        recon = model(xb)
        errs.append(((recon - xb) ** 2).cpu())
    return torch.cat(errs).numpy()


def grid_select(model, checkpoints, Xva, Xva_anom, y_va_anom, windows=(5, 9),
                verbose=True):
    """
    在驗證集上掃「checkpoint × 平滑窗口 × 是否峰值校正」網格，
    以 F1 選出最佳組合（不同異常型態偏好不同收斂程度，單一早停點兩頭不討好）。

    回傳 dict(epoch, window, use_calib, f1, state)
    """
    import numpy as np
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
                thr = va_s.mean() + 3 * va_s.std()
                y_pred = (np.concatenate([va_s, an_s]) > thr).astype(int)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                if best is None or f1 > best["f1"] + 1e-9:
                    best = {"epoch": epoch, "window": w, "use_calib": use_calib,
                            "f1": float(f1),
                            "state": {k: v.clone() for k, v in state.items()}}
    if verbose:
        print(f"網格選擇結果：epoch={best['epoch']}, window={best['window']}, "
              f"峰值校正={best['use_calib']}, 驗證 F1={best['f1']:.3f}")
    return best
