# -*- coding: utf-8 -*-
"""
Step 1：從 LAM 9600 真實資料選出物理意義明確的 sensors，
統計每個 sensor 的均值、標準差、波形範圍，作為合成資料的設計依據。

輸出：
- outputs/sensor_stats.json  每個 sensor 的統計特性
- figures/01_real_waveforms.png  正常 vs 異常 wafer 波形比較
"""
import json

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from config import (DATA_MAT, OUTPUT_DIR, FIGURE_DIR, VAR_NAMES,
                    SELECTED_SENSORS, SENSOR_IDX, STEP_COL, PROCESS_STEPS,
                    MIN_WAFER_LEN, COLORS, set_plot_style)


def load_lam_data():
    """載入 .mat，回傳 (正常 wafer list, 異常 wafer list, 異常名稱 list)"""
    mat = scipy.io.loadmat(DATA_MAT)
    lam = mat["LAMDATA"][0, 0]

    # 驗證變數名稱與 config 一致（.mat 內是字串陣列）
    file_vars = [str(v).strip() for v in lam["variables"]]
    for i, (a, b) in enumerate(zip(file_vars, VAR_NAMES)):
        assert a.lower().startswith(b.split()[0].lower()[:1]), f"變數 {i} 不符: {a} vs {b}"

    normal = [lam["calibration"][i, 0] for i in range(lam["calibration"].shape[0])]
    faulty = [lam["test"][i, 0] for i in range(lam["test"].shape[0])]
    fault_names = [str(n).strip() for n in lam["fault_names"]]
    return normal, faulty, fault_names, file_vars


def keep_process_steps(wafer):
    """只保留主蝕刻步驟（step 4、5）的資料"""
    steps = wafer[:, STEP_COL]
    mask = np.isin(steps, PROCESS_STEPS)
    return wafer[mask]


def main():
    set_plot_style()
    normal, faulty, fault_names, file_vars = load_lam_data()
    print(f".mat 內變數名稱：{file_vars}")

    # 剔除紀錄不完整的 wafer
    normal_ok = [keep_process_steps(w) for w in normal if w.shape[0] >= MIN_WAFER_LEN]
    faulty_ok = [(keep_process_steps(w), n) for w, n in zip(faulty, fault_names)
                 if w.shape[0] >= MIN_WAFER_LEN]
    n_dropped = len(normal) - len(normal_ok)
    print(f"正常 wafer：{len(normal_ok)}（剔除 {n_dropped} 片過短紀錄）")
    print(f"異常 wafer：{len(faulty_ok)}，異常型態：{[n for _, n in faulty_ok]}")

    lens = [w.shape[0] for w in normal_ok]
    print(f"製程段（step 4+5）長度：min={min(lens)}, max={max(lens)}, mean={np.mean(lens):.1f}")

    # ---------- 每個 sensor 的統計特性 ----------
    stats = {"n_normal_wafers": len(normal_ok),
             "n_faulty_wafers": len(faulty_ok),
             "process_len_mean": float(np.mean(lens)),
             "sensors": {}}

    for idx in SENSOR_IDX:
        name, meaning = SELECTED_SENSORS[idx]
        all_vals = np.concatenate([w[:, idx] for w in normal_ok])
        wafer_means = np.array([w[:, idx].mean() for w in normal_ok])
        within_stds = np.array([w[:, idx].std() for w in normal_ok])
        finals = np.array([w[-1, idx] for w in normal_ok])

        # lag-1 自相關（合成雜訊要模仿真實訊號的時間相關性）
        acs = []
        for w in normal_ok:
            x = w[:, idx] - w[:, idx].mean()
            if x.std() > 1e-9:
                acs.append(float(np.corrcoef(x[:-1], x[1:])[0, 1]))
        lag1_ac = float(np.mean(acs))

        stats["sensors"][name] = {
            "index": idx,
            "physical_meaning": meaning,
            "mean": float(all_vals.mean()),          # 整體均值
            "std": float(all_vals.std()),            # 整體標準差
            "range": [float(all_vals.min()), float(all_vals.max())],  # 波形範圍
            "between_wafer_std": float(wafer_means.std()),  # 片間變異
            "within_wafer_std": float(within_stds.mean()),  # 片內雜訊
            "final_value_mean": float(finals.mean()),       # 最終值（SPC 用）
            "final_value_std": float(finals.std()),
            "lag1_autocorr": lag1_ac,
        }
        s = stats["sensors"][name]
        print(f"\n[{name}] {meaning}")
        print(f"  mean={s['mean']:.2f}  std={s['std']:.2f}  "
              f"range=[{s['range'][0]:.1f}, {s['range'][1]:.1f}]")
        print(f"  片間 std={s['between_wafer_std']:.2f}  片內 std={s['within_wafer_std']:.2f}  "
              f"lag-1 autocorr={s['lag1_autocorr']:.3f}")

    out = OUTPUT_DIR / "sensor_stats.json"
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n統計已存檔：{out}")

    # ---------- 波形圖：每個 sensor 一格，正常 3 片 vs 異常 1 片 ----------
    fig, axes = plt.subplots(len(SENSOR_IDX), 1, figsize=(9, 11), sharex=True)
    rng = np.random.default_rng(0)
    show_normal = rng.choice(len(normal_ok), 3, replace=False)

    # 挑一片與所選 sensor 相關的異常（Pr +3 → Pressure 偏移）
    fault_pick = next((i for i, (_, n) in enumerate(faulty_ok) if "Pr" in n), 0)
    fw, fname = faulty_ok[fault_pick]

    for ax, idx in zip(axes, SENSOR_IDX):
        name, _ = SELECTED_SENSORS[idx]
        for k, wi in enumerate(show_normal):
            ax.plot(normal_ok[wi][:, idx], color=COLORS["normal"], alpha=0.55,
                    linewidth=1.6, label="Normal" if k == 0 else None)
        ax.plot(fw[:, idx], color=COLORS["faulty"], linewidth=2.0,
                linestyle="--", label=f"Faulty ({fname})")
        ax.set_ylabel(name, fontsize=9)
        ax.legend(loc="upper right", fontsize=8, frameon=False)

    axes[-1].set_xlabel("Time step（step 4+5 製程段）")
    fig.suptitle("LAM 9600 真實波形：正常 vs 異常 wafer", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_real_waveforms.png", bbox_inches="tight")
    print(f"波形圖已存檔：{FIGURE_DIR / '01_real_waveforms.png'}")


if __name__ == "__main__":
    main()
