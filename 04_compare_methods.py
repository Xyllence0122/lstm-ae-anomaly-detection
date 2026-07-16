# -*- coding: utf-8 -*-
"""
Step 4：四種方法同場比較。

1. SPC X-bar 管制圖：只看每片 wafer 的最終值（傳統做法），±3σ 管制界限
2. Dense AE（無 LSTM）：與 LSTM-AE 相同的選擇協議；輸入為重採樣固定長度版
3. Isolation Forest：攤平固定長度向量 + 樹模型；閾值規則同樣以
   「驗證異常 F1」挑選（與 AE 方法一致的協議，避免對比不公平）
4. LSTM-AE：讀取 Step 3 發佈的模型與分數設定（變長輸入）

輸出：
- outputs/comparison_metrics.json / comparison_metrics.csv
- figures/04_method_comparison.png（四方法指標長條圖）
- figures/04_spc_blind_spot.png（SPC 盲點示範：緩慢漂移 wafer）
"""
import json

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, f1_score

from config import (OUTPUT_DIR, FIGURE_DIR, RANDOM_SEED, SENSOR_NAMES,
                    COLORS, set_plot_style)
import matplotlib.pyplot as plt
from models import (DEVICE, LSTMAutoEncoder, DenseAutoEncoder,
                    train_collect_checkpoints, grid_select,
                    pointwise_errors, sensor_peak_scores, combine_peaks,
                    make_threshold)

ANOMALY_LABELS = {
    0: "Normal",
    1: "A: 暫態到位過快",
    2: "B: 過程震盪",
    3: "C: 緩慢漂移",
}


def evaluate(name, y_test, y_pred):
    y_true = (y_test > 0).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    per_type = {ANOMALY_LABELS[k]: float(y_pred[y_test == k].mean()) for k in (1, 2, 3)}
    fpr = float(y_pred[y_test == 0].mean())
    row = {"method": name, "precision": float(prec), "recall": float(rec),
           "f1": float(f1), "fpr": fpr, **per_type}
    print(f"{name:18s} P={prec:.3f} R={rec:.3f} F1={f1:.3f} FPR={fpr:.3f} "
          f"| A={per_type['A: 暫態到位過快']:.2f} B={per_type['B: 過程震盪']:.2f} "
          f"C={per_type['C: 緩慢漂移']:.2f}")
    return row


def spc_xbar(X_train_list, X_test_list):
    """每個 sensor 用訓練集最終值建立 ±3σ 管制界限，任一 sensor 超限即判異常"""
    finals_train = np.stack([w[-1] for w in X_train_list])
    mu = finals_train.mean(axis=0)
    sd = finals_train.std(axis=0)
    ucl, lcl = mu + 3 * sd, mu - 3 * sd
    finals_test = np.stack([w[-1] for w in X_test_list])
    out = (finals_test > ucl) | (finals_test < lcl)
    return out.any(axis=1).astype(int), (mu, sd, ucl, lcl), finals_test


def select_threshold_by_val_f1(val_normal_scores, val_anom_scores,
                               rules=("mean3sigma", "p99")):
    """
    與 AE 方法一致的閾值選擇協議：在驗證集（正常 + 異常）上以 F1 挑閾值規則。
    回傳 (threshold, rule, val_f1)。
    """
    y_val = np.concatenate([np.zeros(len(val_normal_scores), dtype=int),
                            np.ones(len(val_anom_scores), dtype=int)])
    s_val = np.concatenate([val_normal_scores, val_anom_scores])
    best = None
    for rule in rules:
        thr = make_threshold(val_normal_scores, rule)
        f1 = f1_score(y_val, (s_val > thr).astype(int), zero_division=0)
        if best is None or f1 > best[2]:
            best = (thr, rule, f1)
    return best


def main():
    set_plot_style()
    d = np.load(OUTPUT_DIR / "synthetic_data.npz", allow_pickle=True)
    ckpt = torch.load(OUTPUT_DIR / "lstm_ae.pt", weights_only=False)
    mu, sd = ckpt["mu"], ckpt["sd"]

    X_train_raw = list(d["X_train"])
    X_test_raw = list(d["X_test"])
    y_test = d["y_test"]
    y_val_anom = d["y_val_anom"]

    # 變長版（LSTM 用）
    Xte = [(x - mu) / sd for x in X_test_raw]
    # 固定長度版（Dense AE / IF 用）
    Xtr_f = (d["X_train_fixed"] - mu) / sd
    Xva_f = (d["X_val_fixed"] - mu) / sd
    Xan_f = (d["X_val_anom_fixed"] - mu) / sd
    Xte_f = (d["X_test_fixed"] - mu) / sd
    n_feat = Xte[0].shape[1]
    rows = []

    print("=== 四方法比較（測試集：normal 200 + A/B/C 各 100）===")

    # 1. SPC X-bar
    spc_pred, spc_params, finals_test = spc_xbar(X_train_raw, X_test_raw)
    rows.append(evaluate("SPC X-bar", y_test, spc_pred))

    # 2. Dense AE（相同選擇協議，固定長度輸入）
    torch.manual_seed(RANDOM_SEED)
    dense = DenseAutoEncoder(Xtr_f.shape[1], n_feat)
    _, d_ckpts = train_collect_checkpoints(dense, list(Xtr_f), list(Xva_f),
                                           epochs=400, seed=RANDOM_SEED,
                                           verbose=False)
    d_best = grid_select(dense, d_ckpts, list(Xva_f), list(Xan_f), y_val_anom,
                         verbose=False)
    dense.load_state_dict(d_best["state"])
    print(f"Dense AE 選中 epoch={d_best['epoch']}, window={d_best['window']}, "
          f"校正={d_best['use_calib']}, 閾值={d_best['thr_rule']}, "
          f"驗證 F1={d_best['f1']:.3f}")
    d_peaks_va = sensor_peak_scores(pointwise_errors(dense, list(Xva_f)),
                                    d_best["window"])
    d_calib = d_peaks_va.mean(axis=0) if d_best["use_calib"] else None
    thr_dense = make_threshold(combine_peaks(d_peaks_va, d_calib),
                               d_best["thr_rule"])
    dense_scores = combine_peaks(
        sensor_peak_scores(pointwise_errors(dense, list(Xte_f)),
                           d_best["window"]), d_calib)
    rows.append(evaluate("Dense AE", y_test, (dense_scores > thr_dense).astype(int)))
    # 存檔供 Step 5 真實資料驗證（與 LSTM-AE 相同待遇）
    torch.save({"state_dict": {k: v.detach().cpu().clone()
                               for k, v in dense.state_dict().items()},
                "window": d_best["window"],
                "use_calib": d_best["use_calib"], "thr_rule": d_best["thr_rule"],
                "threshold": thr_dense, "calib": d_calib,
                "seq_len": Xtr_f.shape[1]}, OUTPUT_DIR / "dense_ae.pt")

    # 3. Isolation Forest（固定長度攤平；閾值規則同樣由驗證異常 F1 挑選）
    iso = IsolationForest(n_estimators=300, random_state=RANDOM_SEED)
    iso.fit(Xtr_f.reshape(len(Xtr_f), -1))
    val_if_n = -iso.decision_function(Xva_f.reshape(len(Xva_f), -1))
    val_if_a = -iso.decision_function(Xan_f.reshape(len(Xan_f), -1))
    thr_if, if_rule, if_val_f1 = select_threshold_by_val_f1(val_if_n, val_if_a)
    print(f"Isolation Forest 閾值規則={if_rule}（驗證 F1={if_val_f1:.3f}）")
    if_scores = -iso.decision_function(Xte_f.reshape(len(Xte_f), -1))
    rows.append(evaluate("Isolation Forest", y_test, (if_scores > thr_if).astype(int)))

    # 4. LSTM-AE（發佈模型，變長輸入）
    lstm = LSTMAutoEncoder(n_feat, ckpt["hidden_size"], ckpt["latent_size"])
    lstm.load_state_dict(ckpt["state_dict"])
    lstm.to(DEVICE)
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
        bars = ax.bar(x + (k - 1) * w, df[col], width=w - 0.03, color=color,
                      label=label)
        for b, v in zip(bars, df[col]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", fontsize=8, color=COLORS["ink2"])
    ax.set_xticks(x, df["method"])
    ax.set_ylim(0, 1.12)
    ax.set_title("異常偵測方法比較（合成測試集 v2.1：變長 + 量化）")
    ax.legend(frameon=False, loc="upper left", ncols=3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_method_comparison.png", bbox_inches="tight")

    # ---------- 圖 2：SPC 盲點示範（緩慢漂移 wafer） ----------
    kernel = np.ones(window) / window
    calib_vec = calib if calib is not None else np.ones(n_feat)
    # 首選：LSTM-AE 抓到而 SPC 沒抓到的漂移 wafer；若無則逐步放寬（防呆）
    cand = np.where((y_test == 3) & (lstm_pred == 1) & (spc_pred == 0))[0]
    if len(cand) == 0:
        cand = np.where((y_test == 3) & (lstm_pred == 1))[0]
        print("警告：沒有「LSTM-AE 抓到且 SPC 漏掉」的漂移 wafer，改用 LSTM-AE 抓到的")
    if len(cand) == 0:
        cand = np.where(y_test == 3)[0]
        print("警告：LSTM-AE 沒抓到任何漂移 wafer，示範圖僅供參考")
    demo = cand[np.argmax(lstm_scores[cand])]
    p_idx = int(np.argmax(te_peaks[demo] / calib_vec))
    p_name = SENSOR_NAMES[p_idx]
    print(f"示範 wafer index={demo}，漂移 sensor：{p_name}")

    fig, axes = plt.subplots(3, 1, figsize=(9, 10))
    # 變長序列 → 以正規化時間軸 (0~1) 呈現
    demo_T = len(X_test_raw[demo])
    u_demo = np.linspace(0, 1, demo_T)
    u_fixed = np.linspace(0, 1, d["X_test_fixed"].shape[1])

    # (a) 波形：漂移 wafer vs 正常範圍（固定長度版當背景帶）
    ax = axes[0]
    normals_f = d["X_test_fixed"][y_test == 0][:, :, p_idx]
    ax.fill_between(u_fixed, normals_f.min(axis=0), normals_f.max(axis=0),
                    color=COLORS["normal"], alpha=0.15, label="正常 wafer 範圍")
    ax.plot(u_fixed, normals_f.mean(axis=0), color=COLORS["normal"],
            linewidth=1.5, label="正常平均")
    ax.plot(u_demo, X_test_raw[demo][:, p_idx], color=COLORS["faulty"],
            linewidth=2, label="緩慢漂移 wafer")
    ax.set_ylabel(p_name)
    ax.set_title("(a) 緩慢漂移異常的波形：漂移量小、最終值仍在正常範圍")
    ax.legend(frameon=False, fontsize=9)

    # (b) SPC X-bar 管制圖：只看最終值 → 完全在管制界限內
    ax = axes[1]
    spc_mu, spc_sd, ucl, lcl = spc_params
    normal_idx = np.where(y_test == 0)[0][:60]
    finals = finals_test[normal_idx, p_idx]
    ax.plot(range(len(finals)), finals, "o-", color=COLORS["normal"], linewidth=1,
            markersize=4, alpha=0.7, label="正常 wafer 最終值")
    ax.plot(len(finals), finals_test[demo, p_idx], "o", color=COLORS["faulty"],
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
    err_curve = np.convolve(pw_err[demo][:, p_idx], kernel,
                            mode="valid") / calib_vec[p_idx]
    u_err = np.linspace(0, 1, len(err_curve))
    ax.plot(u_err, err_curve, color=COLORS["faulty"], linewidth=2,
            label="漂移 wafer 重建誤差（平滑）")
    # 正常誤差帶：各片長度不同 → 內插到共同格點
    grid = np.linspace(0, 1, 100)
    err_n = np.stack([
        np.interp(grid, np.linspace(0, 1, len(c)), c) for c in
        (np.convolve(pw_err[i][:, p_idx], kernel, mode="valid") / calib_vec[p_idx]
         for i in np.where(y_test == 0)[0])])
    ax.fill_between(grid, err_n.min(axis=0), err_n.max(axis=0),
                    color=COLORS["normal"], alpha=0.15, label="正常 wafer 誤差範圍")
    ax.axhline(ckpt["threshold"], color=COLORS["ink"], linewidth=1.4,
               linestyle="--")
    ax.text(0.02, ckpt["threshold"] * 1.06, "異常閾值", fontsize=9,
            color=COLORS["ink"])
    over = err_curve > ckpt["threshold"]
    ax.fill_between(u_err, ckpt["threshold"], err_curve, where=over,
                    color=COLORS["faulty"], alpha=0.25)
    ax.set_xlabel("正規化時間（0=製程開始，1=結束）")
    ax.set_ylabel("Reconstruction error")
    ax.set_title("(c) LSTM-AE 逐時間步重建誤差 → 漂移段明顯超過閾值，看得到")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_spc_blind_spot.png", bbox_inches="tight")
    print(f"圖已存檔：{FIGURE_DIR}")


if __name__ == "__main__":
    main()
