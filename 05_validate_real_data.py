# -*- coding: utf-8 -*-
"""
Step 5：真實資料最終驗證（最重要的 sanity check）。

方法論：模型只用合成資料訓練；真實資料分兩組——
- 統計組（60%，Step 1 提取統計特性用；此處兼作閾值校準）
- 保留組（40%，從未參與統計提取；當最終驗證的負樣本）
- 20 片真實 faulty wafer（設定點偏移型異常）當正樣本

評估兩種情境：
1. 直接遷移：閾值沿用合成驗證集的設定（考驗 synthetic→real gap）
2. 重新校準：用統計組真實 wafer 重算 calib 與閾值（實務部署做法）

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
                    PROCESS_STEPS, MIN_WAFER_LEN, RESAMPLE_LEN, COLORS,
                    set_plot_style)
from models import (LSTMAutoEncoder, DenseAutoEncoder, pointwise_errors,
                    sensor_peak_scores, combine_peaks, make_threshold)


def load_real_wafers():
    mat = scipy.io.loadmat(DATA_MAT)
    lam = mat["LAMDATA"][0, 0]
    fault_names = [str(n).strip() for n in lam["fault_names"]]

    def prep(w):
        w = w[np.isin(w[:, STEP_COL], PROCESS_STEPS)]
        return w[:, SENSOR_IDX].astype(float)

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


def main():
    set_plot_style()
    ckpt = torch.load(OUTPUT_DIR / "lstm_ae.pt", weights_only=False)
    stats = json.loads((OUTPUT_DIR / "sensor_stats.json").read_text(encoding="utf-8"))
    model = LSTMAutoEncoder(len(SENSOR_IDX), ckpt["hidden_size"], ckpt["latent_size"])
    model.load_state_dict(ckpt["state_dict"])
    mu, sd, window = ckpt["mu"], ckpt["sd"], int(ckpt["window"])

    normal, faulty, fnames = load_real_wafers()
    # 沿用 Step 1 的切分：統計組 = 校準；保留組 = 最終驗證負樣本
    normal_calib = [normal[i] for i in stats["stats_idx"]]
    normal_eval = [normal[i] for i in stats["holdout_idx"]]
    print(f"真實正常 wafer：統計組 {len(normal_calib)} 片（校準）+ "
          f"保留組 {len(normal_eval)} 片（驗證）；真實 faulty：{len(faulty)} 片")

    def peaks_of(wafers):
        z = [(w - mu) / sd for w in wafers]
        return sensor_peak_scores(pointwise_errors(model, z), window)

    pk_calib = peaks_of(normal_calib)
    pk_eval = peaks_of(normal_eval)
    pk_fault = peaks_of(faulty)

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

    # ---------- 情境 2：重新校準（統計組真實 wafer） ----------
    calib_real = pk_calib.mean(axis=0) if ckpt["use_calib"] else None
    sc_calib = combine_peaks(pk_calib, calib_real)
    sc_eval = combine_peaks(pk_eval, calib_real)
    sc_fault = combine_peaks(pk_fault, calib_real)
    thr_real = make_threshold(sc_calib, ckpt["thr_rule"])

    detected = sc_fault > thr_real
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.zeros(len(sc_eval)), np.ones(len(sc_fault))])
    auc = float(roc_auc_score(y, np.concatenate([sc_eval, sc_fault])))

    # 有監測 / 未監測 sensor 上的故障分組（sensor 覆蓋取捨的量化）
    monitored_kw = ("Pr ", "Cl2", "He")
    mon = np.array([any(k in n for k in monitored_kw) for n in fnames])
    results["recalibrated"] = {
        "threshold": float(thr_real),
        "fpr": float((sc_eval > thr_real).mean()),
        "recall": float(detected.mean()),
        "auc": auc,
        "recall_monitored": float(detected[mon].mean()),
        "recall_unmonitored": float(detected[~mon].mean()),
        "per_fault": {f"{n}#{i}": bool(dd)
                      for i, (n, dd) in enumerate(zip(fnames, detected))},
    }
    print(f"\n[重新校準] 閾值={thr_real:.3f}  "
          f"誤報率={results['recalibrated']['fpr']:.3f}  "
          f"偵測率={detected.mean():.3f}  AUC={auc:.3f}")
    print(f"  有監測 sensor 的故障（{mon.sum()} 片）偵測率 = "
          f"{results['recalibrated']['recall_monitored']:.2f}")
    print(f"  未監測 sensor 的故障（{(~mon).sum()} 片）偵測率 = "
          f"{results['recalibrated']['recall_unmonitored']:.2f}")
    for n, s, dflag in sorted(zip(fnames, sc_fault, detected), key=lambda z: -z[1]):
        print(f"  {'[O]' if dflag else '[X]'} {n:10s} score={s:.2f}")

    # ---------- SPC 對照（真實 faults 是設定點偏移型，SPC 的主場） ----------
    finals_c = np.stack([w[-1] for w in normal_calib])
    m_, s_ = finals_c.mean(axis=0), finals_c.std(axis=0)
    ucl, lcl = m_ + 3 * s_, m_ - 3 * s_

    def spc_flag(wafers):
        f = np.stack([w[-1] for w in wafers])
        return ((f > ucl) | (f < lcl)).any(axis=1).astype(int)

    results["spc"] = {"fpr": float(spc_flag(normal_eval).mean()),
                      "recall": float(spc_flag(faulty).mean())}
    print(f"\n[SPC X-bar] 誤報率={results['spc']['fpr']:.3f}  "
          f"偵測率={results['spc']['recall']:.3f}")

    # ---------- Dense AE 對照（重採樣固定長度 + 重新校準，與 LSTM 同待遇） ----------
    dense_pt = OUTPUT_DIR / "dense_ae.pt"
    if dense_pt.exists():
        dck = torch.load(dense_pt, weights_only=False)
        dense = DenseAutoEncoder(dck["seq_len"], len(SENSOR_IDX))
        dense.load_state_dict(dck["state_dict"])

        def resample(w, n=RESAMPLE_LEN):
            t_src = np.linspace(0, 1, len(w))
            t_dst = np.linspace(0, 1, n)
            return np.stack([np.interp(t_dst, t_src, w[:, f])
                             for f in range(w.shape[1])], axis=1)

        def d_peaks(wafers):
            z = [(resample(w) - mu) / sd for w in wafers]
            return sensor_peak_scores(pointwise_errors(dense, z), dck["window"])

        dpk_calib, dpk_eval, dpk_fault = (d_peaks(normal_calib),
                                          d_peaks(normal_eval), d_peaks(faulty))
        d_calib = dpk_calib.mean(axis=0) if dck["use_calib"] else None
        ds_calib = combine_peaks(dpk_calib, d_calib)
        ds_eval = combine_peaks(dpk_eval, d_calib)
        ds_fault = combine_peaks(dpk_fault, d_calib)
        d_thr = make_threshold(ds_calib, dck["thr_rule"])
        d_auc = float(roc_auc_score(
            np.concatenate([np.zeros(len(ds_eval)), np.ones(len(ds_fault))]),
            np.concatenate([ds_eval, ds_fault])))
        results["dense_ae"] = {"threshold": float(d_thr),
                               "fpr": float((ds_eval > d_thr).mean()),
                               "recall": float((ds_fault > d_thr).mean()),
                               "auc": d_auc}
        print(f"\n[Dense AE]  誤報率={results['dense_ae']['fpr']:.3f}  "
              f"偵測率={results['dense_ae']['recall']:.3f}  AUC={d_auc:.3f}")

    (OUTPUT_DIR / "real_validation.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n結果已存檔：{OUTPUT_DIR / 'real_validation.json'}")

    # ---------- 圖 ----------
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.5),
                             gridspec_kw={"height_ratios": [1, 1.4]})

    ax = axes[0]
    rj = np.random.default_rng(1)
    ax.scatter(rj.uniform(-0.15, 0.15, len(sc_eval)), sc_eval, s=16,
               color=COLORS["normal"], alpha=0.6, edgecolors="none",
               label=f"真實正常・保留組（n={len(sc_eval)}）")
    ax.scatter(1 + rj.uniform(-0.15, 0.15, len(sc_fault)), sc_fault, s=20,
               color=COLORS["faulty"], alpha=0.75, edgecolors="none",
               label=f"真實 faulty（n={len(sc_fault)}）")
    ax.axhline(thr_real, color=COLORS["ink"], linestyle="--", linewidth=1.4)
    ax.text(1.35, thr_real * 1.04, "閾值（重新校準）", fontsize=9, ha="right",
            color=COLORS["ink2"])
    ax.set_xticks([0, 1], ["Normal", "Faulty"])
    ax.set_yscale("log")
    ax.set_ylabel("Anomaly score")
    ax.set_title(f"(a) 真實 wafer 最終驗證（AUC = {auc:.3f}）")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    idx_sorted = np.argsort(sc_fault)[::-1]
    labels = [fnames[i] for i in idx_sorted]
    vals = sc_fault[idx_sorted]
    colors = [COLORS["faulty"] if v > thr_real else COLORS["muted"] for v in vals]
    ax.bar(range(len(vals)), vals, color=colors, width=0.7)
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
