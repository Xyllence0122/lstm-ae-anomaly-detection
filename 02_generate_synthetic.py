# -*- coding: utf-8 -*-
"""
Step 2：依 LAM 9600 真實統計特性生成合成時間序列。

波形設計（每片 wafer = SEQ_LEN 步 × 5 sensors）：
- 前段 ramp：一階系統響應，從偏離值收斂到設定點（模仿真實 Pressure 的進入穩態行為）
- 穩態段：設定點 + AR(1) 雜訊（φ = 真實 lag-1 自相關，σ = 真實片內 std）
- 片間變異：每片設定點加上 N(0, 真實片間 std)

異常設計（每片隨機影響 1~2 個 sensors，最終值都在正常範圍 → SPC 難以偵測）：
- Type A 升溫過快：時間常數縮短 3 倍 + 欠阻尼過衝震盪，最終達標
- Type B 過程震盪：穩態中段出現 2.5~4σ 的阻尼震盪，結束前恢復正常
- Type C 緩慢漂移：從中段開始線性漂移，最終偏移量控制在 SPC ±3σ 管制界限內

輸出：
- outputs/synthetic_data.npz（train/val/test 資料與標籤）
- figures/02_synthetic_examples.png
"""
import json

import numpy as np
import matplotlib.pyplot as plt

from config import (OUTPUT_DIR, FIGURE_DIR, SENSOR_IDX, SELECTED_SENSORS,
                    SEQ_LEN, RANDOM_SEED, COLORS, set_plot_style)

N_TRAIN_NORMAL = 500
N_VAL_NORMAL = 100
N_TEST_NORMAL = 200
N_TEST_PER_ANOMALY = 100
RAMP_LEN = 18  # ramp 段長度（步）

ANOMALY_TYPES = {0: "Normal", 1: "A: 升溫過快", 2: "B: 過程震盪", 3: "C: 緩慢漂移"}


def load_stats():
    stats = json.loads((OUTPUT_DIR / "sensor_stats.json").read_text(encoding="utf-8"))
    return [stats["sensors"][SELECTED_SENSORS[i][0]] for i in SENSOR_IDX]


def ar1_noise(rng, n, sigma, phi):
    """AR(1) 雜訊，邊際標準差固定為 sigma"""
    phi = float(np.clip(phi, 0.0, 0.95))
    eps = rng.normal(0, sigma * np.sqrt(1 - phi**2), n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def make_wafer(rng, sensor_stats, anomaly=0, n_affected=None):
    """生成一片 wafer 的多變量時間序列 (SEQ_LEN, n_sensors)"""
    n_sensors = len(sensor_stats)
    X = np.zeros((SEQ_LEN, n_sensors))
    t = np.arange(SEQ_LEN, dtype=float)

    if anomaly == 0:
        affected = set()
    else:
        k = n_affected or rng.integers(1, 3)  # 影響 1~2 個 sensors
        affected = set(rng.choice(n_sensors, size=k, replace=False))

    for j, s in enumerate(sensor_stats):
        sp = s["mean"] + rng.normal(0, s["between_wafer_std"])   # 本片設定點
        sig = s["within_wafer_std"]                              # 片內雜訊 σ
        ramp_amp = 6 * sig * rng.uniform(0.8, 1.2)               # ramp 起始偏離量
        start = sp - ramp_amp

        is_bad = j in affected

        # ---------- ramp 段 ----------
        if anomaly == 1 and is_bad:
            # Type A：時間常數縮短 3 倍 + 欠阻尼過衝，最終仍收斂到設定點
            tau = rng.uniform(1.2, 2.0)
            w = 2 * np.pi / rng.uniform(8, 14)
            base = sp + (start - sp) * np.exp(-t / tau) * np.cos(w * t)
        else:
            tau = rng.uniform(4.0, 6.0)  # 正常一階響應
            base = sp + (start - sp) * np.exp(-t / tau)

        # ---------- 穩態雜訊 ----------
        base += ar1_noise(rng, SEQ_LEN, sig, s["lag1_autocorr"])

        # ---------- Type B：中段阻尼震盪，結尾前恢復 ----------
        if anomaly == 2 and is_bad:
            amp = rng.uniform(2.5, 4.0) * sig
            period = rng.uniform(6, 15)
            center = rng.uniform(35, 55)
            half = rng.uniform(10, 18)
            lo, hi = int(center - half), min(int(center + half), SEQ_LEN - 10)
            env = np.zeros(SEQ_LEN)
            seg = np.hanning(hi - lo)          # Hann 包絡：漸入漸出
            env[lo:hi] = seg
            base += amp * env * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))

        # ---------- Type C：緩慢線性漂移（最終偏移 < SPC 管制界限） ----------
        if anomaly == 3 and is_bad:
            t0 = int(rng.uniform(15, 30))
            drift_end = rng.choice([-1, 1]) * rng.uniform(2.0, 2.8) * sig
            drift = np.zeros(SEQ_LEN)
            drift[t0:] = np.linspace(0, drift_end, SEQ_LEN - t0)
            base += drift

        X[:, j] = base
    return X


def gen_set(rng, sensor_stats, n, anomaly=0):
    return np.stack([make_wafer(rng, sensor_stats, anomaly) for _ in range(n)])


def main():
    set_plot_style()
    sensor_stats = load_stats()
    names = [SELECTED_SENSORS[i][0] for i in SENSOR_IDX]
    rng = np.random.default_rng(RANDOM_SEED)

    print("生成合成資料 …")
    X_train = gen_set(rng, sensor_stats, N_TRAIN_NORMAL, 0)
    X_val = gen_set(rng, sensor_stats, N_VAL_NORMAL, 0)

    # 驗證異常集：只用於模型選擇（挑 checkpoint / 超參數），與測試集完全分開
    N_VAL_PER_ANOMALY = 50
    X_val_anom = np.concatenate([gen_set(rng, sensor_stats, N_VAL_PER_ANOMALY, k)
                                 for k in (1, 2, 3)])
    y_val_anom = np.concatenate([np.full(N_VAL_PER_ANOMALY, k) for k in (1, 2, 3)])

    X_test_n = gen_set(rng, sensor_stats, N_TEST_NORMAL, 0)
    X_test_a = gen_set(rng, sensor_stats, N_TEST_PER_ANOMALY, 1)
    X_test_b = gen_set(rng, sensor_stats, N_TEST_PER_ANOMALY, 2)
    X_test_c = gen_set(rng, sensor_stats, N_TEST_PER_ANOMALY, 3)

    X_test = np.concatenate([X_test_n, X_test_a, X_test_b, X_test_c])
    y_test = np.concatenate([np.zeros(N_TEST_NORMAL, dtype=int),
                             np.full(N_TEST_PER_ANOMALY, 1),
                             np.full(N_TEST_PER_ANOMALY, 2),
                             np.full(N_TEST_PER_ANOMALY, 3)])

    np.savez_compressed(
        OUTPUT_DIR / "synthetic_data.npz",
        X_train=X_train, X_val=X_val,
        X_val_anom=X_val_anom, y_val_anom=y_val_anom,
        X_test=X_test, y_test=y_test,
        sensor_names=np.array(names),
    )
    print(f"train(normal)={X_train.shape}  val(normal)={X_val.shape}  "
          f"val_anom={X_val_anom.shape}  "
          f"test={X_test.shape}（normal {N_TEST_NORMAL} + A/B/C 各 {N_TEST_PER_ANOMALY}）")
    print(f"已存檔：{OUTPUT_DIR / 'synthetic_data.npz'}")

    # ---------- 範例圖：以 Pressure 為例，正常 vs 三種異常 ----------
    # 為了讓「異常形狀」清楚可見：對照組共用設定點（片間 std 設 0）與相同雜訊種子
    j = names.index("Pressure")
    stats_demo = [{**s, "between_wafer_std": 0.0} for s in sensor_stats]
    n_all = len(stats_demo)  # 範例圖讓所有 sensors 都帶異常，確保示範的 sensor 有效果
    examples = {
        "Normal": make_wafer(np.random.default_rng(11), stats_demo, 0),
        "A: 升溫過快（最終達標）": make_wafer(np.random.default_rng(11), stats_demo, 1, n_affected=n_all),
        "B: 過程震盪（最終達標）": make_wafer(np.random.default_rng(11), stats_demo, 2, n_affected=n_all),
        "C: 緩慢漂移": make_wafer(np.random.default_rng(11), stats_demo, 3, n_affected=n_all),
    }
    ref = examples["Normal"][:, j]

    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True, sharey=True)
    keys = list(examples)
    for ax, key in zip(axes, keys):
        if key != "Normal":
            ax.plot(ref, color=COLORS["normal"], alpha=0.45, linewidth=1.6,
                    label="Normal 參考")
        color = COLORS["normal"] if key == "Normal" else COLORS["faulty"]
        ax.plot(examples[key][:, j], color=color, linewidth=2.0, label=key)
        ax.legend(loc="lower right", fontsize=9, frameon=False)
        ax.set_ylabel("Pressure (mTorr)", fontsize=9)

    axes[-1].set_xlabel("Time step")
    fig.suptitle("合成資料範例（Pressure）：正常 vs 三種異常型態", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_synthetic_examples.png", bbox_inches="tight")
    print(f"範例圖已存檔：{FIGURE_DIR / '02_synthetic_examples.png'}")


if __name__ == "__main__":
    main()
