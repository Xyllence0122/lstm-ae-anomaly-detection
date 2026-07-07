# -*- coding: utf-8 -*-
"""
Step 4：四種方法同場比較。

1. SPC X-bar 管制圖：只看每片 wafer 的最終值（傳統做法），±3σ 管制界限
2. Dense AE（無 LSTM）：與 LSTM-AE 完全相同的訓練/選擇/分數協議，公平比較
3. Isolation Forest：攤平向量 + 樹模型
4. LSTM-AE：讀取 Step 3 選出的模型與分數設定

所有方法都只用正常資料（+驗證異常做模型選擇）建立基準，
記錄 Precision / Recall / F1，並製作「SPC 看不到、LSTM-AE 看得到」對比圖。

輸出：
- outputs/comparison_metrics.json / comparison_metrics.csv
- figures/04_method_comparison.png（四方法指標長條圖）
- figures/04_spc_blind_spot.png（SPC 盲點示範：緩慢漂移 wafer）
"""
import json

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support

from config import (OUTPUT_DIR, FIGURE_DIR, RANDOM_SEED, SENSOR_NAMES,
                    COLORS, set_plot_style)
from models import (LSTMAutoEncoder, DenseAutoEncoder,
                    train_collect_checkpoints, grid_select,
                    pointwise_errors, sensor_peak_scores, combine_peaks)

ANOMALY_LABELS = {0: "Normal", 1: "A: 升溫過快", 2: "B: 過程震盪", 3: "C: 緩慢漂移"}


def load_all():
    d = np.load(OUTPUT_DIR / "synthetic_data.npz", allow_pickle=True)
    ckpt = torch.load(OUTPUT_DIR / "lstm_ae.pt", weights_only=False)
    return d, ckpt


def evaluate(name, y_test, y_pred):
    y_true = (y_test > 0).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    per_type = {ANOMALY_LABELS[k]: float(y_pred[y_test == k].mean()) for k in (1, 2, 3)}
    fpr = float(y_pred[y_test == 0].mean())
    row = {"method": name, "precision": float(prec), "recall": float(rec),
           "f1": float(f1), "fpr": fpr, **per_type}
    print(f"{name:18s} P={prec:.3f} R={rec:.3f} F1={f1:.3f} FPR={fpr:.3f} "
          f"| A={per_type['A: 升溫過快']:.2f} B={per_type['B: 過程震盪']:.2f} "
          f"C={per_type['C: 緩慢漂移']:.2f}")
    return row


# ---------- 方法 1：SPC X-bar（只看最終值） ----------
def spc_xbar(X_train, X_test):
    """每個 sensor 用訓練集最終值建立 ±3σ 管制界限，任一 sensor 超限即判異常"""
    finals_train = X_train[:, -1, :]                # (N, F)
    mu = finals_train.mean(axis=0)
    sd = finals_train.std(axis=0)
    ucl, lcl = mu + 3 * sd, mu - 3 * sd
    finals_test = X_test[:, -1, :]
    out = (finals_test > ucl) | (finals_test < lcl)
    return out.any(axis=1).astype(int), (mu, sd, ucl, lcl)


def ae_scores(model, Xva, Xte, window, use_calib):
    """依選定設定計算驗證閾值與測試分數"""
    va_peaks = sensor_peak_scores(pointwise_errors(model, Xva), window)
    calib = va_peaks.mean(axis=0) if use_calib else None
    va_s = combine_peaks(va_peaks, calib)
    thr = float(va_s.mean() + 3 * va_s.std())
    te_s = combine_peaks(sensor_peak_scores(pointwise_errors(model, Xte), window),
                         calib)
    return te_s, thr, calib


def main():
    set_plot_style()
    d, ckpt = load_all()
    X_train, X_val = d["X_train"], d["X_val"]
    X_val_anom, y_val_anom = d["X_val_anom"], d["y_val_anom"]
    X_test, y_test = d["X_test"], d["y_test"]
    mu, sd = ckpt["mu"], ckpt["sd"]
    Xtr, Xva, Xte = (X_train - mu) / sd, (X_val - mu) / sd, (X_test - mu) / sd
    Xva_an = (X_val_anom - mu) / sd
    seq_len, n_feat = Xtr.shape[1], Xtr.shape[2]
    rows = []

    print("=== 四方法比較（測試集：normal 200 + A/B/C 各 100）===")

    # 1. SPC X-bar
    spc_pred, spc_params = spc_xbar(X_train, X_test)
    rows.append(evaluate("SPC X-bar", y_test, spc_pred))

    # 2. Dense AE（與 LSTM-AE 相同協議：checkpoint 網格選擇）
    dense = DenseAutoEncoder(seq_len, n_feat)
    _, d_ckpts = train_collect_checkpoints(dense, Xtr, Xva, epochs=400,
                                           seed=RANDOM_SEED, verbose=False)
    d_best = grid_select(dense, d_ckpts, Xva, Xva_an, y_val_anom, verbose=False)
    dense.load_state_dict(d_best["state"])
    print(f"Dense AE 選中 epoch={d_best['epoch']}, window={d_best['window']}, "
          f"校正={d_best['use_calib']}, 驗證 F1={d_best['f1']:.3f}")
    dense_scores, thr_dense, _ = ae_scores(dense, Xva, Xte,
                                           d_best["window"], d_best["use_calib"])
    rows.append(evaluate("Dense AE", y_test, (dense_scores > thr_dense).astype(int)))

    # 3. Isolation Forest
    iso = IsolationForest(n_estimators=300, random_state=RANDOM_SEED)
    iso.fit(Xtr.reshape(len(Xtr), -1))
    val_if = -iso.decision_function(Xva.reshape(len(Xva), -1))
    thr_if = val_if.mean() + 3 * val_if.std()
    if_scores = -iso.decision_function(Xte.reshape(len(Xte), -1))
    rows.append(evaluate("Isolation Forest", y_test, (if_scores > thr_if).astype(int)))

    # 4. LSTM-AE（載入 Step 3 選出的模型與分數設定）
    lstm = LSTMAutoEncoder(n_feat, ckpt["hidden_size"], ckpt["latent_size"])
    lstm.load_state_dict(ckpt["state_dict"])
    window = int(ckpt["window"])
    calib = ckpt["calib"]
    pw_err = pointwise_errors(lstm, Xte)
    te_peaks = sensor_peak_scores(pw_err, window)
    lstm_scores = combine_peaks(te_peaks, calib)
    lstm_pred = (lstm_scores > ckpt["threshold"]).astype(int)
    rows.append(evaluate("LSTM-AE", y_test, lstm_pred))

    # ---------- 存指標 ----------
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "comparison_metrics.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "comparison_metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n指標已存檔：{OUTPUT_DIR / 'comparison_metrics.csv'}")

    # ---------- 圖 1：四方法指標長條圖 ----------
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    metrics = [("precision", "Precision", COLORS["normal"]),
               ("recall", "Recall", "#1baf7a"),
               ("f1", "F1", COLORS["series3"])]
    x = np.arange(len(df))
    w = 0.26
    for k, (col, label, color) in enumerate(metrics):
        bars = ax.bar(x + (k - 1) * w, df[col], width=w - 0.03, color=color, label=label)
        for b, v in zip(bars, df[col]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", fontsize=8, color=COLORS["ink2"])
    ax.set_xticks(x, df["method"])
    ax.set_ylim(0, 1.12)
    ax.set_title("異常偵測方法比較（合成測試集，閾值皆為驗證集 mean+3σ）")
    ax.legend(frameon=False, loc="upper left", ncols=3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_method_comparison.png", bbox_inches="tight")

    # ---------- 圖 2：SPC 盲點示範（緩慢漂移 wafer） ----------
    # 挑一片：C 型、LSTM-AE 抓到、SPC 沒抓到，取異常分數最高者；
    # 再從 per-sensor 峰值找出漂移發生在哪個 sensor
    kernel = np.ones(window) / window
    calib_vec = calib if calib is not None else np.ones(n_feat)
    cand = np.where((y_test == 3) & (lstm_pred == 1) & (spc_pred == 0))[0]
    demo = cand[np.argmax(lstm_scores[cand])]
    p_idx = int(np.argmax(te_peaks[demo] / calib_vec))
    p_name = SENSOR_NAMES[p_idx]
    print(f"示範 wafer index={demo}，漂移 sensor：{p_name}")

    fig, axes = plt.subplots(3, 1, figsize=(9, 10))

    # (a) 波形：漂移 wafer vs 正常範圍
    ax = axes[0]
    normals = X_test[y_test == 0][:, :, p_idx]
    ax.fill_between(range(seq_len), normals.min(axis=0), normals.max(axis=0),
                    color=COLORS["normal"], alpha=0.15, label="正常 wafer 範圍")
    ax.plot(normals.mean(axis=0), color=COLORS["normal"], linewidth=1.5,
            label="正常平均")
    ax.plot(X_test[demo, :, p_idx], color=COLORS["faulty"], linewidth=2,
            label="緩慢漂移 wafer")
    ax.set_ylabel(p_name)
    ax.set_title("(a) 緩慢漂移異常的波形：漂移量小、最終值仍在正常範圍")
    ax.legend(frameon=False, fontsize=9)

    # (b) SPC X-bar 管制圖：只看最終值 → 完全在管制界限內
    ax = axes[1]
    spc_mu, spc_sd, ucl, lcl = spc_params
    normal_idx = np.where(y_test == 0)[0][:60]
    finals = X_test[normal_idx, -1, p_idx]
    ax.plot(range(len(finals)), finals, "o-", color=COLORS["normal"], linewidth=1,
            markersize=4, alpha=0.7, label="正常 wafer 最終值")
    ax.plot(len(finals), X_test[demo, -1, p_idx], "o", color=COLORS["faulty"],
            markersize=9, label="漂移 wafer 最終值")
    for y_line, txt in [(ucl[p_idx], "UCL (+3σ)"), (spc_mu[p_idx], "CL"),
                        (lcl[p_idx], "LCL (-3σ)")]:
        ax.axhline(y_line, color=COLORS["muted"], linewidth=1.2, linestyle="--")
        ax.text(len(finals) + 1.5, y_line, txt, fontsize=8, color=COLORS["muted"],
                va="center")
    ax.set_ylabel(f"最終 {p_name}")
    ax.set_title("(b) SPC X-bar 管制圖（只看最終值）→ 在管制界限內，看不到異常")
    ax.legend(frameon=False, fontsize=9, loc="lower left")

    # (c) LSTM-AE 逐時間步重建誤差 → 超過閾值
    ax = axes[2]
    err_curve = np.convolve(pw_err[demo, :, p_idx], kernel, mode="valid") / calib_vec[p_idx]
    t = np.arange(len(err_curve)) + window // 2
    ax.plot(t, err_curve, color=COLORS["faulty"], linewidth=2,
            label="漂移 wafer 重建誤差（平滑）")
    normal_pool = np.where(y_test == 0)[0]
    err_n = np.stack([np.convolve(pw_err[i, :, p_idx], kernel, mode="valid")
                      for i in normal_pool]) / calib_vec[p_idx]
    ax.fill_between(t, err_n.min(axis=0), err_n.max(axis=0),
                    color=COLORS["normal"], alpha=0.15, label="正常 wafer 誤差範圍")
    ax.axhline(ckpt["threshold"], color=COLORS["ink"], linewidth=1.4, linestyle="--")
    ax.text(2, ckpt["threshold"] * 1.06, "異常閾值", fontsize=9, color=COLORS["ink"])
    over = err_curve > ckpt["threshold"]
    ax.fill_between(t, ckpt["threshold"], err_curve, where=over,
                    color=COLORS["faulty"], alpha=0.25)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Reconstruction error")
    ax.set_title("(c) LSTM-AE 逐時間步重建誤差 → 漂移段明顯超過閾值，看得到")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_spc_blind_spot.png", bbox_inches="tight")
    print(f"圖已存檔：{FIGURE_DIR}")


if __name__ == "__main__":
    main()
