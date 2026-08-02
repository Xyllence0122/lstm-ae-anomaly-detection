# -*- coding: utf-8 -*-
"""
Step 1：從 LAM 9600 真實資料提取「正常製程的統計特性」。

方法論約束：
- 真實資料只做兩件事：提取統計特性（本步）、最終驗證（Step 5）
- 模型只用合成資料訓練；合成資料完全由本步提取的統計特性生成
- 真實正常 wafer 先切 60/40：統計組（提取特性）/ 保留組（最終驗證的負樣本），
  避免「提取統計的資料」同時當「驗證負樣本」造成洩漏

提取的統計特性（皆存入 sensor_stats.json）：
- mean profile：每個 sensor 的平均波形（正規化時間軸 101 點），
  自然涵蓋 ramp 方向、step 4→5 轉換暫態
- 殘差統計：扣掉 profile 與片間 offset 後的 within-wafer std、lag-1 自相關
- between_wafer_std：片間 offset 的變異
- quant_step：量化階距（真實訊號是整數量化的）
- transient_amp：製程前段暫態幅度（判斷「暫態到位過快」異常適用於哪些 sensors）
- 長度分布：step 4+5 製程段的長度範圍

輸出：
- outputs/sensor_stats.json
- figures/01_real_waveforms.png
"""
import json

import numpy as np
import scipy.io

from config import (DATA_MAT, OUTPUT_DIR, FIGURE_DIR, VAR_NAMES,
                    SELECTED_SENSORS, SENSOR_IDX, STEP_COL, PROCESS_STEPS,
                    MIN_WAFER_LEN, COLORS, set_plot_style)
import matplotlib.pyplot as plt

PROFILE_GRID = 101   # mean profile 的正規化時間軸點數
SPLIT_SEED = 123     # 統計組/保留組切分（固定，確保 Step 5 用同一份切分）
STATS_FRACTION = 0.6


def load_lam_data():
    mat = scipy.io.loadmat(DATA_MAT)
    lam = mat["LAMDATA"][0, 0]
    file_vars = [str(v).strip() for v in lam["variables"]]
    normal = [lam["calibration"][i, 0] for i in range(lam["calibration"].shape[0])]
    faulty = [lam["test"][i, 0] for i in range(lam["test"].shape[0])]
    fault_names = [str(n).strip() for n in lam["fault_names"]]
    return normal, faulty, fault_names, file_vars


def keep_process_steps(wafer):
    return wafer[np.isin(wafer[:, STEP_COL], PROCESS_STEPS)]


def resample_to_grid(x, n_grid=PROFILE_GRID):
    """把任意長度的訊號線性重採樣到正規化時間軸"""
    t_src = np.linspace(0, 1, len(x))
    t_dst = np.linspace(0, 1, n_grid)
    return np.interp(t_dst, t_src, x)


def quant_step(values):
    """量化階距 = 出現值之間的最小正間距"""
    u = np.unique(values)
    d = np.diff(u)
    d = d[d > 1e-9]
    return float(d.min()) if len(d) else 0.0


def main():
    set_plot_style()
    normal, faulty, fault_names, file_vars = load_lam_data()
    print(f".mat 內變數名稱：{file_vars}")

    # 注意：長度過濾用「原始長度」（trim 前），Step 5 用同一規則，索引才對得上
    normal_ok = [keep_process_steps(w) for w in normal if w.shape[0] >= MIN_WAFER_LEN]
    n_dropped = len(normal) - len(normal_ok)
    print(f"真實正常 wafer：{len(normal_ok)}（剔除 {n_dropped} 片過短紀錄）")

    # ---------- 統計組 / 保留組切分 ----------
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(normal_ok)).tolist()
    n_stats = int(len(normal_ok) * STATS_FRACTION)
    stats_idx, holdout_idx = order[:n_stats], order[n_stats:]
    stats_wafers = [normal_ok[i] for i in stats_idx]
    print(f"統計組 {len(stats_idx)} 片（提取特性）/ 保留組 {len(holdout_idx)} 片（最終驗證）")

    lens = [w.shape[0] for w in stats_wafers]
    print(f"製程段長度：min={min(lens)}, max={max(lens)}, mean={np.mean(lens):.1f}")

    # Derive the sampling interval from the Time column instead of assuming
    # exactly 1 Hz. The dataset variable name does not encode a unit; the Hz
    # conversion below is conditional on the documented/assumed unit being s.
    time_deltas = np.concatenate([
        np.diff(w[:, 0].astype(float)) for w in stats_wafers
    ])
    time_deltas = time_deltas[np.isfinite(time_deltas) & (time_deltas > 0)]
    if not len(time_deltas):
        raise RuntimeError("No positive sampling intervals found in the Time column")
    median_dt = float(np.median(time_deltas))
    sampling = {
        "time_column": VAR_NAMES[0],
        "time_unit_assumption": "seconds",
        "n_positive_intervals": int(len(time_deltas)),
        "median_interval": median_dt,
        "p05_interval": float(np.percentile(time_deltas, 5)),
        "p95_interval": float(np.percentile(time_deltas, 95)),
        "min_interval": float(time_deltas.min()),
        "max_interval": float(time_deltas.max()),
        "median_rate_hz_if_seconds": float(1.0 / median_dt),
    }
    print(f"取樣間隔中位數={median_dt:.4f}，P05~P95="
          f"{sampling['p05_interval']:.4f}~{sampling['p95_interval']:.4f} "
          f"（若 Time 單位為秒，中位取樣率約 "
          f"{sampling['median_rate_hz_if_seconds']:.3f} Hz）")

    stats = {
        "n_normal_total": len(normal_ok),
        "split_seed": SPLIT_SEED,
        "stats_idx": stats_idx,
        "holdout_idx": holdout_idx,
        "len_min": int(min(lens)),
        "len_max": int(max(lens)),
        "profile_grid": PROFILE_GRID,
        "sampling": sampling,
        "sensors": {},
    }

    for idx in SENSOR_IDX:
        name, meaning = SELECTED_SENSORS[idx]
        traces = [w[:, idx] for w in stats_wafers]

        # mean profile（正規化時間軸）
        resampled = np.stack([resample_to_grid(x) for x in traces])
        profile = resampled.mean(axis=0)

        # 殘差 = 原始訊號 - 依長度縮放回去的 profile - 片間 offset
        offsets, resid_stds, lag1s = [], [], []
        for x in traces:
            prof_t = np.interp(np.linspace(0, 1, len(x)),
                               np.linspace(0, 1, PROFILE_GRID), profile)
            off = float((x - prof_t).mean())
            r = x - prof_t - off
            offsets.append(off)
            resid_stds.append(r.std())
            if r.std() > 1e-9:
                lag1s.append(float(np.corrcoef(r[:-1], r[1:])[0, 1]))

        all_vals = np.concatenate(traces)
        # 暫態幅度：profile 穩態段（前 20% 之後）的平均值 − 起始值，
        # 表示製程開頭有多大的 ramp/暫態（判斷 A 型異常適用性）
        n20 = PROFILE_GRID // 5
        transient_amp = float(profile[n20:].mean() - profile[0])

        stats["sensors"][name] = {
            "index": idx,
            "physical_meaning": meaning,
            "mean": float(all_vals.mean()),
            "std": float(all_vals.std()),
            "range": [float(all_vals.min()), float(all_vals.max())],
            "profile": profile.tolist(),
            "within_wafer_std": float(np.mean(resid_stds)),
            "between_wafer_std": float(np.std(offsets)),
            # 殘差幾乎為常數時 lag1 無法定義，記 0.0（等同白噪聲假設）
            "lag1_autocorr": float(np.mean(lag1s)) if lag1s else 0.0,
            "quant_step": quant_step(all_vals),
            "transient_amp": transient_amp,
        }
        s = stats["sensors"][name]
        print(f"\n[{name}] {meaning}")
        print(f"  mean={s['mean']:.2f}  範圍=[{s['range'][0]:.0f}, {s['range'][1]:.0f}]  "
              f"量化階距={s['quant_step']:.2f}")
        print(f"  殘差 std={s['within_wafer_std']:.3f}  片間 std={s['between_wafer_std']:.3f}  "
              f"lag1={s['lag1_autocorr']:.2f}  暫態幅度={s['transient_amp']:+.2f}"
              f"（{abs(s['transient_amp']) / max(s['within_wafer_std'], 1e-9):.1f}σ）")

    out = OUTPUT_DIR / "sensor_stats.json"
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n統計已存檔：{out}")

    # ---------- 波形圖：正常 3 片 vs 異常 1 片 + mean profile ----------
    faulty_ok = [(keep_process_steps(w), n) for w, n in zip(faulty, fault_names)
                 if w.shape[0] >= MIN_WAFER_LEN]
    fault_pick = next((i for i, (_, n) in enumerate(faulty_ok) if "Pr" in n), 0)
    fw, fname = faulty_ok[fault_pick]

    fig, axes = plt.subplots(len(SENSOR_IDX), 1, figsize=(9, 9.5), sharex=True)
    rng2 = np.random.default_rng(0)
    show = rng2.choice(len(stats_wafers), 3, replace=False)
    for ax, idx in zip(axes, SENSOR_IDX):
        name, _ = SELECTED_SENSORS[idx]
        for k, wi in enumerate(show):
            ax.plot(stats_wafers[wi][:, idx], color=COLORS["normal"], alpha=0.45,
                    linewidth=1.4, label="Normal" if k == 0 else None)
        prof = np.array(stats["sensors"][name]["profile"])
        t_prof = np.linspace(0, np.mean(lens) - 1, PROFILE_GRID)
        ax.plot(t_prof, prof, color=COLORS["ink"], linewidth=2.0, label="Mean profile")
        ax.plot(fw[:, idx], color=COLORS["faulty"], linewidth=1.8,
                linestyle="--", label=f"Faulty ({fname})")
        ax.set_ylabel(name, fontsize=9)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    axes[-1].set_xlabel("Time step（step 4+5 製程段）")
    fig.suptitle("LAM 9600 真實波形與 mean profile（統計組）", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_real_waveforms.png", bbox_inches="tight")
    print(f"波形圖已存檔：{FIGURE_DIR / '01_real_waveforms.png'}")


if __name__ == "__main__":
    main()
