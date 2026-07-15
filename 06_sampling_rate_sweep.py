# -*- coding: utf-8 -*-
"""
Step 6：取樣率掃描實驗——「取樣多快，才能看到多快的異常？」

回答的研究問題：
    同一個「到位過快」異常（例如設定點本來 2 秒到位，異常時 0.5 秒或
    0.05 秒就衝到），在不同取樣率下模型的偵測率是多少？
    → 證明「取樣率是偵測瓶頸」，因此需要能就地高頻取樣、就地推論的
      邊緣運算架構（kHz 級資料不可能全部上傳雲端）。

為什麼不直接宣稱「毫秒級異常偵測」：
    真實 LAM 9600 資料是 1 Hz 取樣，秒以下的物理特性沒有被記錄，
    任何毫秒級合成資料的微觀行為都無法用真實資料驗證。本實驗改為
    「固定異常、掃描取樣率」——每一步都有明確的物理模型與可陳述的
    假設，不需要宣稱合成資料在毫秒尺度上與真實世界一致。

物理模型（連續時間，取樣率只影響「拍照的密度」，不影響物理本身）：
    x(t) = steady + (start − steady)·exp(−t/τ) + OU 噪聲 + 片間 offset，再量化
    - 錨定自真實資料（sensor_stats.json 的 Pressure）：
        steady/start 位準、殘差強度 σ（within_wafer_std）、
        噪聲相關時間 θ（由 1 Hz 的 lag-1 自相關反推：θ = −1/ln(ρ₁)，
        假設殘差為 OU 過程——AR(1) 正是 OU 的離散取樣）、
        片間 offset 變異、量化階距
    - 情境假設（顯式陳述，錨不到就不假裝錨得到）：
        正常事件 = 2 秒內到位的設定點轉換（呼應「溫度 2 秒到 200」情境；
        真實資料集本身的 20 秒暫態已由主管線在 1 Hz 下處理）
    - 異常 = 到位時間壓縮（0.05~1 秒）+ 與事件時間尺度綁定的過衝 ringing

實驗設計：
    - 對每個取樣率各訓練一個 LSTM-AE（只用該取樣率的正常資料），
      沿用主管線的選擇協議（checkpoint × window × 校正 × 閾值網格）
    - 對每個（取樣率 × 異常到位時間）組合評估偵測率；同時報告誤報率
    - 每個取樣率的結果單獨存檔（outputs/sweep/），中斷可續跑

輸出：
    - outputs/sampling_sweep.json      全部數據
    - figures/06_waveform_demo.png     同一異常在不同取樣率下的樣子（快門比喻）
    - figures/06_detection_vs_rate.png 偵測率 vs 取樣率曲線（論文主圖）

執行時間：CPU 約 30~60 分鐘（取樣率越高序列越長越慢）；CUDA GPU 快很多。
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

from config import OUTPUT_DIR, FIGURE_DIR, COLORS, set_plot_style

# ---------- 情境參數 ----------
SENSOR = "Pressure"          # 錨定用 sensor（真實暫態最明顯、物理意義清楚）
WINDOW_SEC = 20.0            # 監測窗：事件開始後 20 秒
RAMP_NOMINAL_SEC = 2.0       # 情境假設：正常 2 秒到位（呼應溫度 2 秒到 200）
RATES_HZ = [0.5, 1, 2, 5, 10, 20]          # 掃描的取樣率（真實資料 ≈ 1 Hz）
ANOM_RAMP_SECS = [1.0, 0.5, 0.2, 0.05]     # 異常到位時間（0.05 ≈「瞬間衝到」）
VAL_ANOM_SECS = [1.0, 0.5, 0.2]            # 模型選擇用的驗證異常（與測試分開生成）

# ---------- 資料量與訓練 ----------
N_TRAIN, N_VAL = 200, 100
N_VAL_PER = 20               # 驗證異常：每種到位時間 20 片
N_TEST_NORMAL = 100
N_TEST_PER = 60              # 測試異常：每種到位時間 60 片
EPOCHS, CKPT_EVERY = 200, 20
HIDDEN_SIZE, LATENT_SIZE = 64, 16
SEED = 42


# ====================================================================
# 連續時間物理模型（純 NumPy，與 torch 無關，可獨立測試）
# ====================================================================

def load_anchor():
    """從 sensor_stats.json 錨定物理參數；錨不到的（事件時長）是情境假設"""
    stats = json.loads((OUTPUT_DIR / "sensor_stats.json").read_text(encoding="utf-8"))
    s = stats["sensors"][SENSOR]
    prof = np.asarray(s["profile"])
    rho1 = s["lag1_autocorr"]
    return {
        "start": float(prof[0]),                       # 事件起點位準
        "steady": float(prof[len(prof) // 3: -5].mean()),  # 穩態位準
        "sigma": s["within_wafer_std"],                # 殘差強度（1 Hz 下量到的）
        # OU 相關時間：ρ(Δt)=exp(−Δt/θ)，用 Δt=1s 的 lag-1 反推
        "theta": (-1.0 / np.log(rho1)) if 0.0 < rho1 < 1.0 else None,
        "between": s["between_wafer_std"],
        "quant": s["quant_step"],
    }


def ou_noise(rng, n, dt, sigma, theta):
    """OU 過程的精確離散取樣：任何 dt 下邊際變異數都是 σ²、相關時間都是 θ"""
    if theta is None:                       # 1 Hz 下已無相關 → 視為白噪聲
        return rng.normal(0, sigma, n)
    phi = np.exp(-dt / theta)
    eps = rng.normal(0, sigma * np.sqrt(1 - phi**2), n)
    eps[0] = rng.normal(0, sigma)           # 平穩初始化
    return lfilter([1.0], [1.0, -phi], eps)


def make_window(rng, anchor, rate_hz, ramp_sec, anomalous):
    """
    生成一段監測窗訊號，shape (T, 1)。
    連續時間軌跡固定，rate_hz 只決定「快門」拍多密。
    """
    dt = 1.0 / rate_hz
    n = max(int(round(WINDOW_SEC * rate_hz)), 4)
    t = np.arange(n) * dt

    tau = ramp_sec / 3.0                    # τ 取 ramp/3 → t=ramp 時已到位 ~95%
    delta = anchor["steady"] - anchor["start"]
    x = anchor["start"] + delta * (1.0 - np.exp(-t / tau))

    if anomalous:
        # 到位過快伴隨過衝 ringing：幅度固定比例，週期與衰減都綁在事件時間尺度上
        # （事件越快，餘波頻率越高、消失越快——這正是低取樣率拍不到的部分）
        amp = 0.12 * abs(delta)
        x = x + amp * np.exp(-t / ramp_sec) * np.sin(2 * np.pi * t / ramp_sec)

    x = x + rng.normal(0, anchor["between"])                    # 片間 offset
    x = x + ou_noise(rng, n, dt, anchor["sigma"], anchor["theta"])
    if anchor["quant"] > 0:                                     # 感測器量化
        x = np.round(x / anchor["quant"]) * anchor["quant"]
    return x[:, None]


def gen_set(rng, anchor, rate_hz, n, ramp_sec, anomalous):
    return [make_window(rng, anchor, rate_hz, ramp_sec, anomalous)
            for _ in range(n)]


# ====================================================================
# 每個取樣率：訓練 → 選擇 → 評估（沿用主管線協議）
# ====================================================================

def run_one_rate(rate_hz, anchor):
    from models import (LSTMAutoEncoder, train_collect_checkpoints, grid_select,
                        pointwise_errors, sensor_peak_scores, combine_peaks,
                        make_threshold)

    rng = np.random.default_rng(SEED)
    Xtr = gen_set(rng, anchor, rate_hz, N_TRAIN, RAMP_NOMINAL_SEC, False)
    Xva = gen_set(rng, anchor, rate_hz, N_VAL, RAMP_NOMINAL_SEC, False)
    Xva_an, y_va = [], []
    for k, rs in enumerate(VAL_ANOM_SECS, start=1):
        Xva_an += gen_set(rng, anchor, rate_hz, N_VAL_PER, rs, True)
        y_va += [k] * N_VAL_PER
    Xte_n = gen_set(rng, anchor, rate_hz, N_TEST_NORMAL, RAMP_NOMINAL_SEC, False)
    Xte_anoms = {rs: gen_set(rng, anchor, rate_hz, N_TEST_PER, rs, True)
                 for rs in ANOM_RAMP_SECS}

    # z-score（只用訓練集統計）
    all_tr = np.concatenate(Xtr)
    mu, sd = all_tr.mean(axis=0), all_tr.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    z = lambda L: [(x - mu) / sd for x in L]
    Xtr, Xva, Xva_an, Xte_n = z(Xtr), z(Xva), z(Xva_an), z(Xte_n)
    Xte_anoms = {k: z(v) for k, v in Xte_anoms.items()}

    # 短序列時平滑窗口不能超過序列長度
    T_min = min(len(x) for x in Xtr)
    windows = tuple(w for w in (5, 9) if w <= T_min) or (max(T_min - 1, 1),)

    model = LSTMAutoEncoder(1, HIDDEN_SIZE, LATENT_SIZE)
    _, ckpts = train_collect_checkpoints(model, Xtr, Xva, epochs=EPOCHS,
                                         ckpt_every=CKPT_EVERY, seed=SEED,
                                         verbose=False)
    best = grid_select(model, ckpts, Xva, Xva_an, y_va,
                       windows=windows, verbose=False)
    model.load_state_dict(best["state"])

    va_peaks = sensor_peak_scores(pointwise_errors(model, Xva), best["window"])
    calib = va_peaks.mean(axis=0) if best["use_calib"] else None
    thr = make_threshold(combine_peaks(va_peaks, calib), best["thr_rule"])

    def detect_rate(X_list):
        s = combine_peaks(sensor_peak_scores(
            pointwise_errors(model, X_list), best["window"]), calib)
        return float((s > thr).mean())

    result = {
        "rate_hz": rate_hz,
        "n_steps": len(Xtr[0]),
        "val_f1": best["f1"], "epoch": best["epoch"],
        "window": best["window"], "thr_rule": best["thr_rule"],
        "fpr": detect_rate(Xte_n),
        "recall": {str(rs): detect_rate(Xte_anoms[rs]) for rs in ANOM_RAMP_SECS},
    }
    print(f"[{rate_hz:>4g} Hz] steps={result['n_steps']:>3d}  "
          f"FPR={result['fpr']:.2f}  " +
          "  ".join(f"到位{rs}s:{result['recall'][str(rs)]:.2f}"
                    for rs in ANOM_RAMP_SECS))
    return result


# ====================================================================
# 圖
# ====================================================================

def fig_waveform_demo(anchor):
    """同一個「0.5 秒到位」異常，用三種快門速度看——低取樣率下異常憑空消失"""
    set_plot_style()
    demo_rates = [1, 5, 20]
    fig, axes = plt.subplots(len(demo_rates), 1, figsize=(9, 8.5), sharex=True)
    for ax, r in zip(axes, demo_rates):
        # 連續時間「真相」（用 200 Hz 近似，無噪聲版）
        t_c = np.arange(0, 8, 1 / 200)
        delta = anchor["steady"] - anchor["start"]
        x_n = anchor["start"] + delta * (1 - np.exp(-t_c / (RAMP_NOMINAL_SEC / 3)))
        x_a = anchor["start"] + delta * (1 - np.exp(-t_c / (0.5 / 3)))
        x_a = x_a + 0.12 * abs(delta) * np.exp(-t_c / 0.5) * np.sin(2 * np.pi * t_c / 0.5)
        ax.plot(t_c, x_n, color=COLORS["grid"], linewidth=1.2,
                label="連續時間真相（正常 2s 到位）")
        ax.plot(t_c, x_a, color="#f2c4c4", linewidth=1.2,
                label="連續時間真相（異常 0.5s 到位）")
        # 取樣點（無噪聲，凸顯取樣效果本身）
        rng = np.random.default_rng(0)
        for ramp, color, lab in [(RAMP_NOMINAL_SEC, COLORS["normal"], "取樣：正常"),
                                 (0.5, COLORS["faulty"], "取樣：異常")]:
            tt = np.arange(0, 8, 1 / r)
            xx = anchor["start"] + delta * (1 - np.exp(-tt / (ramp / 3)))
            if ramp != RAMP_NOMINAL_SEC:
                xx = xx + 0.12 * abs(delta) * np.exp(-tt / ramp) * np.sin(
                    2 * np.pi * tt / ramp)
            ax.plot(tt, xx, "o-", color=color, markersize=5, linewidth=1.6,
                    alpha=0.9, label=lab)
        ax.set_ylabel(f"{SENSOR}\n@ {r} Hz", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="lower right", ncols=2)
    axes[0].set_title("同一個異常，三種「快門速度」：取樣太慢時，異常在資料裡不存在")
    axes[-1].set_xlabel("時間（秒）")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "06_waveform_demo.png", bbox_inches="tight")
    print(f"示範圖已存檔：{FIGURE_DIR / '06_waveform_demo.png'}")


def fig_detection_curves(results):
    set_plot_style()
    rates = [r["rate_hz"] for r in results]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    palette = [COLORS["series3"], COLORS["series4"], COLORS["series5"],
               COLORS["faulty"]]
    for color, rs in zip(palette, ANOM_RAMP_SECS):
        ys = [r["recall"][str(rs)] for r in results]
        ax.plot(rates, ys, "o-", color=color, linewidth=2,
                label=f"異常到位 {rs} s")
    ax.plot(rates, [r["fpr"] for r in results], "s--", color=COLORS["muted"],
            linewidth=1.5, label="誤報率（正常被誤判）")
    ax.set_xscale("log")
    ax.set_xticks(rates, [f"{r:g}" for r in rates])
    ax.set_xlabel("取樣率（Hz，log 尺度）")
    ax.set_ylabel("偵測率")
    ax.set_ylim(-0.05, 1.1)
    ax.axvline(1.0, color=COLORS["grid"], linewidth=1.2)
    ax.text(1.02, 1.05, "← 真實資料集的取樣率", fontsize=8, color=COLORS["muted"])
    ax.set_title(f"偵測率 vs 取樣率（正常事件 {RAMP_NOMINAL_SEC:g}s 到位；"
                 f"事件越快，需要的取樣率越高）")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "06_detection_vs_rate.png", bbox_inches="tight")
    print(f"曲線圖已存檔：{FIGURE_DIR / '06_detection_vs_rate.png'}")


# ====================================================================

def main():
    anchor = load_anchor()
    theta_txt = f"{anchor['theta']:.3f} s" if anchor["theta"] else "白噪聲"
    print(f"錨定參數（{SENSOR}）：start={anchor['start']:.1f}  "
          f"steady={anchor['steady']:.1f}  σ={anchor['sigma']:.2f}  "
          f"θ={theta_txt}  量化={anchor['quant']:g}")
    print(f"情境假設：正常 {RAMP_NOMINAL_SEC:g}s 到位；監測窗 {WINDOW_SEC:g}s\n")

    fig_waveform_demo(anchor)

    sweep_dir = OUTPUT_DIR / "sweep"
    sweep_dir.mkdir(exist_ok=True)
    results = []
    for rate in RATES_HZ:
        f = sweep_dir / f"rate_{rate:g}.json"
        if f.exists():
            results.append(json.loads(f.read_text(encoding="utf-8")))
            print(f"[{rate:>4g} Hz] 已完成，跳過")
            continue
        res = run_one_rate(rate, anchor)
        f.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                     encoding="utf-8")
        results.append(res)

    (OUTPUT_DIR / "sampling_sweep.json").write_text(json.dumps(
        {"scenario": {"sensor": SENSOR, "window_sec": WINDOW_SEC,
                      "ramp_nominal_sec": RAMP_NOMINAL_SEC,
                      "anchor": {k: v for k, v in load_anchor().items()}},
         "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n結果已存檔：{OUTPUT_DIR / 'sampling_sweep.json'}")

    fig_detection_curves(results)

    # 經驗法則：偵測率 ≥ 0.9 所需的「事件內取樣點數」
    print("\n===== 經驗法則（論文論點）=====")
    for rs in ANOM_RAMP_SECS:
        ok = [r["rate_hz"] for r in results if r["recall"][str(rs)] >= 0.9]
        if ok:
            r_min = min(ok)
            print(f"到位 {rs}s 的異常：至少 {r_min:g} Hz（事件內約 "
                  f"{r_min * rs:.1f} 個取樣點）才能穩定偵測")
        else:
            print(f"到位 {rs}s 的異常：本掃描範圍（≤{max(RATES_HZ):g} Hz）內"
                  f"無法穩定偵測 → 需要更高取樣率")
    print("外插：0.01 秒級事件依同樣法則約需數百 Hz——"
          "這正是必須在邊緣端就地取樣、就地推論的原因。")


if __name__ == "__main__":
    main()
