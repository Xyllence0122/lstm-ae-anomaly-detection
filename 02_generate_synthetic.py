# -*- coding: utf-8 -*-
"""
Step 2：依 LAM 9600 真實統計特性生成合成時間序列（v2.1）。

v2 相對 v1 的修正（縮小 synthetic→real domain gap）：
- 波形骨架改用真實 mean profile（自然涵蓋 ramp 方向與 step 4→5 暫態），
  不再用人工假設的「從下方 6σ ramp」
- 加入整數量化（真實訊號量化階距 = 1）
- 序列長度改為變長（取樣自真實長度範圍 95~112）
- 「升溫過快」只發生在真實有暫態的 sensors（Pressure、Vat Valve）
- 驗證異常集（模型選擇用）與測試集完全分開

v2.1 修正：
- AR(1) 噪聲改為「平穩初始化」：x[0] ~ N(0, σ)。v2 版 x[0]=0，
  序列開頭的噪聲變異數偏小（非平穩），會讓模型把「開頭很安靜」學成正常特徵
- AR(1) 遞迴改用 scipy.signal.lfilter 向量化（等價、較快）
- Type A 若無任何 sensor 具備足夠暫態，直接報錯而不是靜默生成
  「標了異常標籤但實際正常」的錯誤樣本

異常設計（每片隨機影響 1~2 個 sensors，最終值都在正常範圍）：
- Type A 暫態到位過快：暫態段時間軸壓縮 2.5~4 倍 + 過衝 ringing，最終達標
- Type B 過程震盪：中段 2.5~4σ 阻尼震盪，結束前恢復
- Type C 緩慢漂移：線性漂移，最終偏移 2.0~2.8σ（在 SPC 管制界限內）

輸出：
- outputs/synthetic_data.npz：變長版（LSTM 用，object array）+
  重採樣固定長度版（Dense AE / Isolation Forest 用）
- figures/02_synthetic_examples.png
"""
import json

import numpy as np
from scipy.signal import lfilter

from config import (OUTPUT_DIR, FIGURE_DIR, SENSOR_IDX, SELECTED_SENSORS,
                    RESAMPLE_LEN, RANDOM_SEED, COLORS, set_plot_style)
import matplotlib.pyplot as plt

N_TRAIN_NORMAL = 500
N_VAL_NORMAL = 200
N_VAL_PER_ANOMALY = 50
N_TEST_NORMAL = 200
N_TEST_PER_ANOMALY = 100

TRANSIENT_FRAC = 0.2        # profile 前 20% 視為暫態段
A_ELIGIBLE_SIGMA = 2.0      # 暫態幅度 ≥ 2σ 的 sensor 才可能發生「升溫過快」

ANOMALY_TYPES = {
    0: "Normal",
    1: "A: 暫態到位過快",
    2: "B: 過程震盪",
    3: "C: 緩慢漂移",
}


def load_stats():
    stats = json.loads((OUTPUT_DIR / "sensor_stats.json").read_text(encoding="utf-8"))
    sensors = [stats["sensors"][SELECTED_SENSORS[i][0]] for i in SENSOR_IDX]
    return stats, sensors


def ar1_noise(rng, n, sigma, phi):
    """
    平穩 AR(1) 噪聲：x[t] = φ·x[t-1] + ε[t]，ε ~ N(0, σ√(1-φ²))。
    x[0] 直接取自平穩分布 N(0, σ)，整段序列每一點的邊際變異數都是 σ²。
    以 lfilter 實作遞迴（與逐步迴圈等價，向量化較快）。
    """
    phi = float(np.clip(phi, 0.0, 0.95))
    eps = rng.normal(0, sigma * np.sqrt(1 - phi**2), n)
    eps[0] = rng.normal(0, sigma)               # 平穩初始化
    return lfilter([1.0], [1.0, -phi], eps)


def make_wafer(rng, sensors, len_range, anomaly=0, n_affected=None,
               fixed_len=None, no_offset=False, return_metadata=False):
    """生成一片變長多變量時間序列。

    ``return_metadata=True`` 時一併回傳異常注入起點與受影響 sensor 位置，
    供串流模型計算首次告警延遲。這些欄位描述合成注入規則，不是假裝已知的
    真實設備故障起點。
    """
    n_sensors = len(sensors)
    T = fixed_len or int(rng.integers(len_range[0], len_range[1] + 1))
    u = np.linspace(0, 1, T)                       # 正規化時間軸
    X = np.zeros((T, n_sensors))

    # 選擇受影響的 sensors（A 型只能發生在有真實暫態的 sensor）
    if anomaly == 0:
        affected = set()
    else:
        pool = list(range(n_sensors))
        if anomaly == 1:
            pool = [j for j in pool
                    if abs(sensors[j]["transient_amp"])
                    >= A_ELIGIBLE_SIGMA * sensors[j]["within_wafer_std"]]
            if not pool:
                raise ValueError(
                    "沒有任何 sensor 的暫態幅度達到 A_ELIGIBLE_SIGMA，"
                    "無法生成 Type A 異常（檢查 sensor_stats.json 的 transient_amp，"
                    "或調低 A_ELIGIBLE_SIGMA）")
        k = min(n_affected or int(rng.integers(1, 3)), len(pool))
        affected = set(rng.choice(pool, size=k, replace=False))

    onset_indices = []
    end_indices = []
    for j, s in enumerate(sensors):
        profile = np.asarray(s["profile"])
        grid = np.linspace(0, 1, len(profile))
        sig = s["within_wafer_std"]
        is_bad = j in affected

        # ---------- 波形骨架：mean profile（A 型壓縮暫態段時間軸） ----------
        if anomaly == 1 and is_bad:
            speed = rng.uniform(2.5, 4.0)
            uu = u.copy()
            m = uu < TRANSIENT_FRAC
            uu[m] = TRANSIENT_FRAC * (uu[m] / TRANSIENT_FRAC) ** (1.0 / speed)
            base = np.interp(uu, grid, profile)
            # 過衝 ringing：幅度與暫態成比例，快速衰減，最終達標
            amp = abs(s["transient_amp"]) * rng.uniform(0.10, 0.20)
            tau = rng.uniform(2.0, 4.0)
            period = rng.uniform(6.0, 12.0)
            t_steps = np.arange(T, dtype=float)
            base = base + amp * np.exp(-t_steps / tau) * np.sin(
                2 * np.pi * t_steps / period)
            onset_indices.append(0)
            end_indices.append(min(int(np.ceil(TRANSIENT_FRAC * T)), T - 1))
        else:
            base = np.interp(u, grid, profile)

        # ---------- 片間 offset + AR(1) 殘差 ----------
        if not no_offset:
            base = base + rng.normal(0, s["between_wafer_std"])
        base = base + ar1_noise(rng, T, sig, s["lag1_autocorr"])

        # ---------- Type B：中段阻尼震盪，結尾前恢復 ----------
        if anomaly == 2 and is_bad:
            amp = rng.uniform(2.5, 4.0) * sig
            period = rng.uniform(6, 15)
            center = rng.uniform(0.35, 0.70) * T
            half = rng.uniform(0.12, 0.20) * T
            lo = max(int(center - half), 0)
            hi = min(int(center + half), T - 8)
            env = np.zeros(T)
            env[lo:hi] = np.hanning(hi - lo)
            t_steps = np.arange(T, dtype=float)
            base = base + amp * env * np.sin(
                2 * np.pi * t_steps / period + rng.uniform(0, 2 * np.pi))
            onset_indices.append(lo)
            end_indices.append(max(hi - 1, lo))

        # ---------- Type C：緩慢線性漂移（最終偏移 < SPC 管制界限） ----------
        if anomaly == 3 and is_bad:
            t0 = int(rng.uniform(0.20, 0.35) * T)
            drift_end = rng.choice([-1, 1]) * rng.uniform(2.0, 2.8) * sig
            drift = np.zeros(T)
            drift[t0:] = np.linspace(0, drift_end, T - t0)
            base = base + drift
            onset_indices.append(t0)
            end_indices.append(T - 1)

        # ---------- 整數量化（真實訊號的量化階距） ----------
        q = s["quant_step"]
        if q > 0:
            base = np.round(base / q) * q

        X[:, j] = base
    metadata = {
        "anomaly_type": int(anomaly),
        "anomaly_name": ANOMALY_TYPES[int(anomaly)],
        "affected_sensor_positions": sorted(int(j) for j in affected),
        "onset_index": min(onset_indices) if onset_indices else None,
        "end_index": max(end_indices) if end_indices else None,
        "sequence_length": int(T),
    }
    if metadata["onset_index"] is not None:
        metadata["onset_fraction"] = (
            metadata["onset_index"] / max(metadata["sequence_length"] - 1, 1))
    else:
        metadata["onset_fraction"] = None
    return (X, metadata) if return_metadata else X


def resample_fixed(X, n=RESAMPLE_LEN):
    """重採樣到固定長度（Dense AE / Isolation Forest 用）"""
    T, F = X.shape
    t_src = np.linspace(0, 1, T)
    t_dst = np.linspace(0, 1, n)
    return np.stack([np.interp(t_dst, t_src, X[:, f]) for f in range(F)], axis=1)


def gen_set(rng, sensors, len_range, n, anomaly=0, with_metadata=False):
    generated = [make_wafer(rng, sensors, len_range, anomaly,
                            return_metadata=with_metadata) for _ in range(n)]
    if not with_metadata:
        return generated
    wafers, metadata = zip(*generated)
    return list(wafers), list(metadata)


def metadata_array(items):
    """Store metadata as portable JSON strings instead of a pickle object array."""
    return np.asarray([json.dumps(item, ensure_ascii=False) for item in items])


def to_obj(lst):
    a = np.empty(len(lst), dtype=object)
    for i, x in enumerate(lst):
        a[i] = x
    return a


def main():
    set_plot_style()
    stats, sensors = load_stats()
    names = [SELECTED_SENSORS[i][0] for i in SENSOR_IDX]
    len_range = (stats["len_min"], stats["len_max"])
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"生成合成資料（變長 {len_range[0]}~{len_range[1]} 步、含量化）…")
    X_train = gen_set(rng, sensors, len_range, N_TRAIN_NORMAL, 0)
    X_val = gen_set(rng, sensors, len_range, N_VAL_NORMAL, 0)
    X_val_anom, y_val_anom, val_metadata = [], [], []
    for k in (1, 2, 3):
        wafers, metadata = gen_set(
            rng, sensors, len_range, N_VAL_PER_ANOMALY, k, with_metadata=True)
        X_val_anom += wafers
        val_metadata += metadata
        y_val_anom += [k] * N_VAL_PER_ANOMALY
    X_test, test_metadata = gen_set(
        rng, sensors, len_range, N_TEST_NORMAL, 0, with_metadata=True)
    y_test = [0] * N_TEST_NORMAL
    for k in (1, 2, 3):
        wafers, metadata = gen_set(
            rng, sensors, len_range, N_TEST_PER_ANOMALY, k, with_metadata=True)
        X_test += wafers
        test_metadata += metadata
        y_test += [k] * N_TEST_PER_ANOMALY

    np.savez_compressed(
        OUTPUT_DIR / "synthetic_data.npz",
        X_train=to_obj(X_train), X_val=to_obj(X_val),
        X_val_anom=to_obj(X_val_anom), y_val_anom=np.array(y_val_anom),
        X_test=to_obj(X_test), y_test=np.array(y_test),
        X_train_fixed=np.stack([resample_fixed(x) for x in X_train]),
        X_val_fixed=np.stack([resample_fixed(x) for x in X_val]),
        X_val_anom_fixed=np.stack([resample_fixed(x) for x in X_val_anom]),
        X_test_fixed=np.stack([resample_fixed(x) for x in X_test]),
        val_metadata=metadata_array(val_metadata),
        test_metadata=metadata_array(test_metadata),
        sensor_names=np.array(names),
    )
    print(f"train={len(X_train)}  val={len(X_val)}  val_anom={len(X_val_anom)}  "
          f"test={len(X_test)}（normal {N_TEST_NORMAL} + A/B/C 各 {N_TEST_PER_ANOMALY}）")
    print(f"已存檔：{OUTPUT_DIR / 'synthetic_data.npz'}")

    # ---------- 範例圖：Pressure，正常 vs 三種異常（共用設定點與種子） ----------
    j = names.index("Pressure")
    demo = {}
    for key, anom in [("Normal", 0), ("A: 暫態到位過快（最終達標）", 1),
                      ("B: 過程震盪（最終達標）", 2), ("C: 緩慢漂移", 3)]:
        demo[key] = make_wafer(np.random.default_rng(11), sensors, len_range,
                               anomaly=anom, n_affected=len(sensors),
                               fixed_len=100, no_offset=True)
    ref = demo["Normal"][:, j]

    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True, sharey=True)
    for ax, key in zip(axes, demo):
        if key != "Normal":
            ax.plot(ref, color=COLORS["normal"], alpha=0.45, linewidth=1.6,
                    label="Normal 參考")
        color = COLORS["normal"] if key == "Normal" else COLORS["faulty"]
        ax.plot(demo[key][:, j], color=color, linewidth=2.0, label=key)
        ax.legend(loc="lower right", fontsize=9, frameon=False)
        ax.set_ylabel("Pressure (mTorr)", fontsize=9)
    axes[-1].set_xlabel("Time step")
    fig.suptitle("合成資料 v2.1 範例（Pressure）：mean profile 骨架 + 量化", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_synthetic_examples.png", bbox_inches="tight")
    print(f"範例圖已存檔：{FIGURE_DIR / '02_synthetic_examples.png'}")


if __name__ == "__main__":
    main()
