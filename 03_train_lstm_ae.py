# -*- coding: utf-8 -*-
"""
Step 3：訓練 LSTM AutoEncoder（只用正常資料）。

模型選擇協議：
- AE 訓練越久重建能力越強，連異常都能重建，偵測力反而下降；
  且不同異常型態偏好不同收斂程度（升溫過快偏好早期、震盪偏好後期）
- 因此每 10 epochs 存 checkpoint，在驗證集（正常 + 驗證異常）上
  掃「checkpoint × 平滑窗口 × 峰值校正」網格，以 F1 選最佳組合
- 驗證異常集與測試集完全分開；測試集只在最終評估使用一次

輸出：
- outputs/lstm_ae.pt           模型權重 + 標準化參數 + 分數設定 + 閾值
- outputs/lstm_ae_metrics.json 測試集 Precision / Recall / F1 與各異常型態偵測率
- figures/03_training_curve.png
- figures/03_error_distribution.png
"""
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

from config import OUTPUT_DIR, FIGURE_DIR, RANDOM_SEED, COLORS, set_plot_style
from models import (LSTMAutoEncoder, train_collect_checkpoints, grid_select,
                    pointwise_errors, sensor_peak_scores, combine_peaks)

EPOCHS = 400
CKPT_EVERY = 10
HIDDEN_SIZE = 64
LATENT_SIZE = 16

ANOMALY_LABELS = {1: "A: 升溫過快", 2: "B: 過程震盪", 3: "C: 緩慢漂移"}


def load_data():
    d = np.load(OUTPUT_DIR / "synthetic_data.npz", allow_pickle=True)
    return (d["X_train"], d["X_val"], d["X_val_anom"], d["y_val_anom"],
            d["X_test"], d["y_test"], list(d["sensor_names"]))


def main():
    set_plot_style()
    np.random.seed(RANDOM_SEED)

    X_train, X_val, X_val_anom, y_val_anom, X_test, y_test, names = load_data()

    # ---------- z-score 標準化（只用訓練集統計） ----------
    mu = X_train.mean(axis=(0, 1))
    sd = X_train.std(axis=(0, 1))
    Xtr = (X_train - mu) / sd
    Xva = (X_val - mu) / sd
    Xva_an = (X_val_anom - mu) / sd
    Xte = (X_test - mu) / sd

    # ---------- 訓練並保存 checkpoints ----------
    model = LSTMAutoEncoder(n_features=Xtr.shape[2],
                            hidden_size=HIDDEN_SIZE, latent_size=LATENT_SIZE)
    hist, checkpoints = train_collect_checkpoints(
        model, Xtr, Xva, epochs=EPOCHS, ckpt_every=CKPT_EVERY, seed=RANDOM_SEED)

    # ---------- 驗證集網格選擇 ----------
    best = grid_select(model, checkpoints, Xva, Xva_an, y_val_anom)
    model.load_state_dict(best["state"])
    window, use_calib = best["window"], best["use_calib"]

    # ---------- 校正與閾值（皆由驗證正常集決定） ----------
    va_peaks = sensor_peak_scores(pointwise_errors(model, Xva), window)
    calib = va_peaks.mean(axis=0) if use_calib else None
    val_scores = combine_peaks(va_peaks, calib)
    threshold = float(val_scores.mean() + 3 * val_scores.std())
    print(f"異常判斷閾值 (mean+3σ) = {threshold:.4f}")

    # ---------- 測試集評估（只在此使用一次） ----------
    te_peaks = sensor_peak_scores(pointwise_errors(model, Xte), window)
    test_err = combine_peaks(te_peaks, calib)
    y_true = (y_test > 0).astype(int)
    y_pred = (test_err > threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)

    per_type = {}
    for k, label in ANOMALY_LABELS.items():
        per_type[label] = float(y_pred[y_test == k].mean())
    fpr = float(y_pred[y_test == 0].mean())

    print(f"\nLSTM-AE 測試結果：Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")
    print(f"誤報率（正常判成異常）={fpr:.3f}")
    for label, r in per_type.items():
        print(f"  {label} 偵測率 = {r:.3f}")

    # ---------- 存檔 ----------
    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                "threshold": threshold, "window": window,
                "calib": calib, "use_calib": use_calib,
                "hidden_size": HIDDEN_SIZE, "latent_size": LATENT_SIZE},
               OUTPUT_DIR / "lstm_ae.pt")
    metrics = {"precision": float(prec), "recall": float(rec), "f1": float(f1),
               "false_positive_rate": fpr, "threshold": threshold,
               "best_epoch": best["epoch"], "window": window,
               "use_calib": use_calib, "val_selection_f1": best["f1"],
               "per_type_recall": per_type}
    (OUTPUT_DIR / "lstm_ae_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(OUTPUT_DIR / "lstm_ae_scores.npz", test_err=test_err, y_test=y_test,
             threshold=threshold)
    print(f"\n模型與指標已存檔：{OUTPUT_DIR}")

    # ---------- 圖 1：訓練曲線 ----------
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(hist["train"], color=COLORS["normal"], label="Train loss")
    ax.plot(hist["val"], color=COLORS["series3"], label="Validation loss")
    ax.axvline(best["epoch"], color=COLORS["muted"], linestyle="--", linewidth=1.2)
    ax.text(best["epoch"] + 4, ax.get_ylim()[1] * 0.97,
            f"選中 epoch {best['epoch']}\n（驗證異常 F1 最高）",
            fontsize=9, color=COLORS["ink2"], va="top")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("LSTM-AE 訓練曲線（checkpoint 由驗證異常 F1 挑選，非 val loss）")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_training_curve.png", bbox_inches="tight")

    # ---------- 圖 2：異常分數分布（正常 vs 三種異常） ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    groups = [("Normal", y_test == 0, COLORS["normal"]),
              ("A: 升溫過快", y_test == 1, COLORS["series3"]),
              ("B: 過程震盪", y_test == 2, COLORS["series4"]),
              ("C: 緩慢漂移", y_test == 3, COLORS["series5"])]
    rng = np.random.default_rng(0)
    for gi, (label, mask, color) in enumerate(groups):
        x = gi + rng.uniform(-0.18, 0.18, mask.sum())
        ax.scatter(x, test_err[mask], s=14, color=color, alpha=0.6, edgecolors="none",
                   label=label)
    ax.axhline(threshold, color=COLORS["faulty"], linewidth=1.5, linestyle="--")
    ax.text(3.45, threshold * 1.05, "閾值 (mean+3σ)", color=COLORS["faulty"],
            fontsize=9, ha="right", va="bottom")
    ax.set_xticks(range(4), [g[0] for g in groups])
    ax.set_ylabel("Anomaly score（平滑誤差峰值）")
    ax.set_yscale("log")
    ax.set_title("LSTM-AE 異常分數：正常 vs 三種異常")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_error_distribution.png", bbox_inches="tight")
    print(f"圖已存檔：{FIGURE_DIR}")


if __name__ == "__main__":
    main()
