# -*- coding: utf-8 -*-
"""
Step 5：真實資料驗證（最重要的 sanity check）。

合成資料是自己設計的——如果設計本身有錯，模型在合成測試集上表現再好
也不會暴露問題。因此把訓練好的 LSTM-AE 直接套用到 LAM 9600 的
真實 wafer 上驗證：
- 107 片真實正常 wafer（其中一半用來重新校準閾值，另一半當測試負樣本）
- 20 片真實 faulty wafer（設定點偏移型異常：TCP +50、Pr +3、Cl2 -10 …）

評估兩種情境：
1. 直接遷移：閾值沿用合成驗證集的設定（考驗合成→真實的 gap）
2. 重新校準：用一半真實正常 wafer 重算 calib 與閾值（實務部署做法）

同時跑 SPC X-bar 對照。注意：真實 faults 是設定點偏移型（會持續到製程結束），
本來就是 SPC 擅長的型態，可觀察兩方法互補性。

輸出：
- outputs/real_validation.json
- figures/05_real_validation.png
"""
import json

import numpy as np
import scipy.io
import torch
import matplotlib.pyplot as plt

from config import (DATA_MAT, OUTPUT_DIR, FIGURE_DIR, SENSOR_IDX, STEP_COL,
                    PROCESS_STEPS, MIN_WAFER_LEN, COLORS, set_plot_style)
from models import LSTMAutoEncoder, sensor_peak_scores, combine_peaks


def load_real_wafers():
    """回傳 (正常 wafer list, 異常 wafer list, 異常名稱 list)，只留所選 sensors"""
    mat = scipy.io.loadmat(DATA_MAT)
    lam = mat["LAMDATA"][0, 0]
    fault_names = [str(n).strip() for n in lam["fault_names"]]

    def prep(w):
        w = w[np.isin(w[:, STEP_COL], PROCESS_STEPS)]
        return w[:, SENSOR_IDX]

    normal = [prep(lam["calibration"][i, 0])
              for i in range(lam["calibration"].shape[0])
              if lam["calibration"][i, 0].shape[0] >= MIN_WAFER_LEN]
    faulty, fnames = [], []
    for i in range(lam["test"].shape[0]):
        w = lam["test"][i, 0]
        if w.shape[0] >= MIN_WAFER_LEN:
            faulty.append(prep(w))
            fnames.append(fault_names[i])
    return normal, faulty, fnames


@torch.no_grad()
def wafer_peaks(model, wafers, mu, sd, window):
    """逐片計算 per-sensor 平滑誤差峰值（真實 wafer 長度不一，逐片 forward）"""
    model.eval()
    peaks = []
    for w in wafers:
        x = (w - mu) / sd
        xt = torch.as_tensor(x[None], dtype=torch.float32)
        err = ((model(xt) - xt) ** 2)[0].numpy()          # (T, F)
        peaks.append(sensor_peak_scores(err[None], window)[0])
    return np.stack(peaks)                                 # (N, F)


def spc_on_real(normal_calib, normal_eval, faulty):
    """SPC X-bar：用校準組正常 wafer 的最終值建管制界限"""
    finals_c = np.stack([w[-1] for w in normal_calib])
    mu, sd = finals_c.mean(axis=0), finals_c.std(axis=0)
    ucl, lcl = mu + 3 * sd, mu - 3 * sd

    def flag(wafers):
        finals = np.stack([w[-1] for w in wafers])
        return ((finals > ucl) | (finals < lcl)).any(axis=1).astype(int)

    return flag(normal_eval), flag(faulty)


def main():
    set_plot_style()
    ckpt = torch.load(OUTPUT_DIR / "lstm_ae.pt", weights_only=False)
    model = LSTMAutoEncoder(len(SENSOR_IDX), ckpt["hidden_size"], ckpt["latent_size"])
    model.load_state_dict(ckpt["state_dict"])
    mu, sd, window = ckpt["mu"], ckpt["sd"], int(ckpt["window"])

    normal, faulty, fnames = load_real_wafers()
    rng = np.random.default_rng(0)
    order = rng.permutation(len(normal))
    calib_idx, eval_idx = order[:len(order) // 2], order[len(order) // 2:]
    normal_calib = [normal[i] for i in calib_idx]
    normal_eval = [normal[i] for i in eval_idx]
    print(f"真實正常 wafer：{len(normal_calib)} 片校準 + {len(normal_eval)} 片評估；"
          f"真實 faulty：{len(faulty)} 片")

    pk_calib = wafer_peaks(model, normal_calib, mu, sd, window)
    pk_eval = wafer_peaks(model, normal_eval, mu, sd, window)
    pk_fault = wafer_peaks(model, faulty, mu, sd, window)

    results = {}

    # ---------- 情境 1：直接遷移（合成資料的 calib 與閾值） ----------
    calib_syn = ckpt["calib"] if ckpt["use_calib"] else None
    s_eval = combine_peaks(pk_eval, calib_syn)
    s_fault = combine_peaks(pk_fault, calib_syn)
    thr_syn = ckpt["threshold"]
    results["direct_transfer"] = {
        "threshold": float(thr_syn),
        "fpr": float((s_eval > thr_syn).mean()),
        "recall": float((s_fault > thr_syn).mean()),
    }
    print(f"\n[直接遷移] 閾值={thr_syn:.3f}  "
          f"真實正常誤報率={results['direct_transfer']['fpr']:.3f}  "
          f"真實異常偵測率={results['direct_transfer']['recall']:.3f}")

    # ---------- 情境 2：重新校準（一半真實正常 wafer） ----------
    calib_real = pk_calib.mean(axis=0)
    sc_calib = combine_peaks(pk_calib, calib_real)
    sc_eval = combine_peaks(pk_eval, calib_real)
    sc_fault = combine_peaks(pk_fault, calib_real)
    thr_real = float(sc_calib.mean() + 3 * sc_calib.std())

    detected = sc_fault > thr_real
    # AUC（閾值無關的分離度指標）
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.zeros(len(sc_eval)), np.ones(len(sc_fault))])
    auc = float(roc_auc_score(y, np.concatenate([sc_eval, sc_fault])))

    results["recalibrated"] = {
        "threshold": thr_real,
        "fpr": float((sc_eval > thr_real).mean()),
        "recall": float(detected.mean()),
        "auc": auc,
        "per_fault": {n: bool(d) for n, d in zip(fnames, detected)},
    }
    print(f"\n[重新校準] 閾值={thr_real:.3f}  "
          f"誤報率={results['recalibrated']['fpr']:.3f}  "
          f"偵測率={detected.mean():.3f}  AUC={auc:.3f}")
    for n, s, dflag in sorted(zip(fnames, sc_fault, detected), key=lambda z: -z[1]):
        print(f"  {'[O]' if dflag else '[X]'} {n:10s} score={s:.2f}")

    # ---------- SPC 對照 ----------
    spc_eval, spc_fault = spc_on_real(normal_calib, normal_eval, faulty)
    results["spc"] = {"fpr": float(spc_eval.mean()), "recall": float(spc_fault.mean())}
    print(f"\n[SPC X-bar] 誤報率={spc_eval.mean():.3f}  偵測率={spc_fault.mean():.3f}")

    (OUTPUT_DIR / "real_validation.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n結果已存檔：{OUTPUT_DIR / 'real_validation.json'}")

    # ---------- 圖：真實資料驗證 ----------
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.5),
                             gridspec_kw={"height_ratios": [1, 1.4]})

    # (a) 分數分布：正常（評估組）vs faulty
    ax = axes[0]
    rj = np.random.default_rng(1)
    ax.scatter(rj.uniform(-0.15, 0.15, len(sc_eval)), sc_eval, s=16,
               color=COLORS["normal"], alpha=0.6, edgecolors="none",
               label=f"真實正常（n={len(sc_eval)}）")
    ax.scatter(1 + rj.uniform(-0.15, 0.15, len(sc_fault)), sc_fault, s=20,
               color=COLORS["faulty"], alpha=0.75, edgecolors="none",
               label=f"真實 faulty（n={len(sc_fault)}）")
    ax.axhline(thr_real, color=COLORS["ink"], linestyle="--", linewidth=1.4)
    ax.text(1.35, thr_real * 1.04, "閾值（重新校準）", fontsize=9, ha="right",
            color=COLORS["ink2"])
    ax.set_xticks([0, 1], ["Normal", "Faulty"])
    ax.set_yscale("log")
    ax.set_ylabel("Anomaly score")
    ax.set_title(f"(a) 真實 wafer 異常分數（AUC = {auc:.3f}）")
    ax.legend(frameon=False, fontsize=9)

    # (b) 每個 fault 的分數（依大小排序）
    ax = axes[1]
    idx_sorted = np.argsort(sc_fault)[::-1]
    labels = [fnames[i] for i in idx_sorted]
    vals = sc_fault[idx_sorted]
    colors = [COLORS["faulty"] if v > thr_real else COLORS["muted"] for v in vals]
    bars = ax.bar(range(len(vals)), vals, color=colors, width=0.7)
    ax.axhline(thr_real, color=COLORS["ink"], linestyle="--", linewidth=1.4)
    ax.set_xticks(range(len(vals)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Anomaly score")
    ax.set_title("(b) 每片真實 faulty wafer 的分數（紅=偵測到，灰=漏掉）")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_real_validation.png", bbox_inches="tight")
    print(f"圖已存檔：{FIGURE_DIR / '05_real_validation.png'}")


if __name__ == "__main__":
    main()
