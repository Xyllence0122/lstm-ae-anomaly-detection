# -*- coding: utf-8 -*-
"""
Step 3：訓練 LSTM AutoEncoder（只用合成正常資料，5 個隨機種子）。

模型選擇協議：
- 每 20 epochs 存 checkpoint，在驗證集（正常 + 驗證異常）上掃
  「checkpoint × 平滑窗口 × 峰值校正 × 閾值規則」網格，以 F1 選最佳組合
- 驗證異常集與測試集完全分開；測試集每個 seed 只評估一次
- 5 個 seeds 報告 mean ± std（單一 seed 的數字不可信）；
  發佈模型 = 驗證 F1 最高的 seed

輸出：
- outputs/lstm_ae.pt           發佈模型 + 標準化參數 + 分數設定 + 閾值
- outputs/lstm_ae_metrics.json 各 seed 測試指標 + 平均 ± 標準差
- figures/03_training_curve.png / 03_error_distribution.png（發佈 seed）
"""
import hashlib
import json

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

from config import OUTPUT_DIR, FIGURE_DIR, COLORS, set_plot_style
import matplotlib.pyplot as plt
from models import (DEVICE, LSTMAutoEncoder, train_collect_checkpoints,
                    grid_select, pointwise_errors, sensor_peak_scores,
                    combine_peaks, make_threshold)

EPOCHS = 400
CKPT_EVERY = 20
HIDDEN_SIZE = 64
LATENT_SIZE = 16
SEEDS = [42, 43, 44, 45, 46]
PIPELINE_VERSION = 3

ANOMALY_LABELS = {1: "A: 暫態到位過快", 2: "B: 過程震盪", 3: "C: 緩慢漂移"}


def load_data():
    d = np.load(OUTPUT_DIR / "synthetic_data.npz", allow_pickle=True)
    return (list(d["X_train"]), list(d["X_val"]),
            list(d["X_val_anom"]), d["y_val_anom"],
            list(d["X_test"]), d["y_test"])


def zscore_list(X_list, mu, sd):
    return [(x - mu) / sd for x in X_list]


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def run_signature(data_path, seed):
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "data_sha256": digest,
        "seed": seed,
        "epochs": EPOCHS,
        "checkpoint_every": CKPT_EVERY,
        "hidden_size": HIDDEN_SIZE,
        "latent_size": LATENT_SIZE,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def evaluate_test(model, Xte, y_test, window, use_calib, thr_rule, Xva):
    """依選定設定計算閾值（驗證集）與測試指標"""
    va_peaks = sensor_peak_scores(pointwise_errors(model, Xva), window)
    calib = va_peaks.mean(axis=0) if use_calib else None
    val_scores = combine_peaks(va_peaks, calib)
    threshold = make_threshold(val_scores, thr_rule)

    test_scores = combine_peaks(
        sensor_peak_scores(pointwise_errors(model, Xte), window), calib)
    y_true = (y_test > 0).astype(int)
    y_pred = (test_scores > threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    per_type = {label: float(y_pred[y_test == k].mean())
                for k, label in ANOMALY_LABELS.items()}
    fpr = float(y_pred[y_test == 0].mean())
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1),
            "fpr": fpr, "per_type_recall": per_type,
            "threshold": threshold}, test_scores, calib


def main():
    set_plot_style()
    print(f"訓練裝置：{DEVICE}")
    X_train, X_val, X_val_anom, y_val_anom, X_test, y_test = load_data()

    # z-score 標準化（只用訓練集統計；防止零變異 sensor 造成除以零）
    all_train = np.concatenate(X_train)
    mu, sd = all_train.mean(axis=0), all_train.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Xtr = zscore_list(X_train, mu, sd)
    Xva = zscore_list(X_val, mu, sd)
    Xva_an = zscore_list(X_val_anom, mu, sd)
    Xte = zscore_list(X_test, mu, sd)
    n_feat = Xtr[0].shape[1]

    # 每個 seed 的結果單獨存檔 → 中斷後重跑會自動跳過已完成的 seed
    seed_dir = OUTPUT_DIR / "seeds"
    seed_dir.mkdir(exist_ok=True)
    seed_results = []
    data_path = OUTPUT_DIR / "synthetic_data.npz"

    for seed in SEEDS:
        jf = seed_dir / f"seed_{seed}.json"
        pf = seed_dir / f"seed_{seed}.pt"
        signature, signature_payload = run_signature(data_path, seed)
        if jf.exists() and pf.exists():
            m = json.loads(jf.read_text(encoding="utf-8"))
            if m.get("run_signature") == signature:
                seed_results.append(m)
                print(f"seed {seed} 快取一致，跳過（測試 F1={m['f1']:.3f}）")
                continue
            print(f"seed {seed} 快取已過期（資料或設定不同），重新訓練")

        print(f"\n===== seed {seed} =====")
        # Seed must be set before module construction to control initial weights.
        torch.manual_seed(seed)
        model = LSTMAutoEncoder(n_feat, HIDDEN_SIZE, LATENT_SIZE)
        hist, ckpts = train_collect_checkpoints(
            model, Xtr, Xva, epochs=EPOCHS, ckpt_every=CKPT_EVERY, seed=seed)
        best = grid_select(model, ckpts, Xva, Xva_an, y_val_anom)
        model.load_state_dict(best["state"])
        m, test_scores, calib = evaluate_test(
            model, Xte, y_test, best["window"], best["use_calib"],
            best["thr_rule"], Xva)
        m.update({"seed": seed, "best_epoch": best["epoch"],
                  "window": best["window"], "use_calib": best["use_calib"],
                  "thr_rule": best["thr_rule"], "val_f1": best["f1"],
                  "run_signature": signature,
                  "signature_payload": signature_payload})
        seed_results.append(m)
        print(f"seed {seed}: 測試 F1={m['f1']:.3f}  P={m['precision']:.3f}  "
              f"R={m['recall']:.3f}  FPR={m['fpr']:.3f}  "
              f"per-type={ {k: round(v, 2) for k, v in m['per_type_recall'].items()} }")

        jf.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        torch.save({"state_dict": cpu_state(model), "hist": hist,
                    "test_scores": test_scores, "calib": calib,
                    "threshold": m["threshold"]}, pf)

    # 發佈模型 = 驗證 F1 最高的 seed（從存檔載回）
    rel_m = max(seed_results, key=lambda m: m["val_f1"])
    rel_pt = torch.load(seed_dir / f"seed_{rel_m['seed']}.pt", weights_only=False)
    release = {"model_state": rel_pt["state_dict"], "seed": rel_m["seed"],
               "val_f1": rel_m["val_f1"], "window": rel_m["window"],
               "use_calib": rel_m["use_calib"], "thr_rule": rel_m["thr_rule"],
               "threshold": rel_pt["threshold"], "calib": rel_pt["calib"],
               "hist": rel_pt["hist"], "best_epoch": rel_m["best_epoch"],
               "test_scores": rel_pt["test_scores"]}

    # ---------- 彙總 ----------
    agg = {}
    for key in ("precision", "recall", "f1", "fpr"):
        vals = [m[key] for m in seed_results]
        agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    for label in ANOMALY_LABELS.values():
        vals = [m["per_type_recall"][label] for m in seed_results]
        agg[label] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    print("\n===== 5 seeds 彙總（mean ± std）=====")
    print(f"F1 = {agg['f1']['mean']:.3f} ± {agg['f1']['std']:.3f}   "
          f"Precision = {agg['precision']['mean']:.3f} ± {agg['precision']['std']:.3f}   "
          f"Recall = {agg['recall']['mean']:.3f} ± {agg['recall']['std']:.3f}")
    for label in ANOMALY_LABELS.values():
        print(f"  {label}: {agg[label]['mean']:.2f} ± {agg[label]['std']:.2f}")
    print(f"發佈模型：seed {release['seed']}（驗證 F1={release['val_f1']:.3f}，"
          f"epoch {release['best_epoch']}, window {release['window']}, "
          f"閾值規則 {release['thr_rule']}）")

    # ---------- 存檔 ----------
    torch.save({"state_dict": release["model_state"], "mu": mu, "sd": sd,
                "threshold": release["threshold"], "window": release["window"],
                "calib": release["calib"], "use_calib": release["use_calib"],
                "thr_rule": release["thr_rule"],
                "hidden_size": HIDDEN_SIZE, "latent_size": LATENT_SIZE,
                "seed": release["seed"]},
               OUTPUT_DIR / "lstm_ae.pt")
    (OUTPUT_DIR / "lstm_ae_metrics.json").write_text(json.dumps(
        {"per_seed": seed_results, "aggregate": agg,
         "release_seed": release["seed"]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"模型與指標已存檔：{OUTPUT_DIR}")

    # ---------- 圖 1：訓練曲線（發佈 seed） ----------
    hist = release["hist"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(hist["train"], color=COLORS["normal"], label="Train loss")
    ax.plot(hist["val"], color=COLORS["series3"], label="Validation loss")
    ax.axvline(release["best_epoch"], color=COLORS["muted"], linestyle="--",
               linewidth=1.2)
    ax.text(release["best_epoch"] + 4, ax.get_ylim()[1] * 0.97,
            f"選中 epoch {release['best_epoch']}\n（驗證異常 F1 最高）",
            fontsize=9, color=COLORS["ink2"], va="top")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title(f"LSTM-AE 訓練曲線（seed {release['seed']}，"
                 f"checkpoint 由驗證異常 F1 挑選）")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_training_curve.png", bbox_inches="tight")

    # ---------- 圖 2：異常分數分布（發佈 seed） ----------
    test_err = release["test_scores"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    groups = [("Normal", y_test == 0, COLORS["normal"]),
              ("A: 暫態到位過快", y_test == 1, COLORS["series3"]),
              ("B: 過程震盪", y_test == 2, COLORS["series4"]),
              ("C: 緩慢漂移", y_test == 3, COLORS["series5"])]
    rng = np.random.default_rng(0)
    for gi, (label, mask, color) in enumerate(groups):
        x = gi + rng.uniform(-0.18, 0.18, mask.sum())
        ax.scatter(x, test_err[mask], s=14, color=color, alpha=0.6,
                   edgecolors="none", label=label)
    ax.axhline(release["threshold"], color=COLORS["faulty"], linewidth=1.5,
               linestyle="--")
    ax.text(3.45, release["threshold"] * 1.05, f"閾值（{release['thr_rule']}）",
            color=COLORS["faulty"], fontsize=9, ha="right", va="bottom")
    ax.set_xticks(range(4), [g[0] for g in groups])
    ax.set_ylabel("Anomaly score（平滑誤差峰值）")
    ax.set_yscale("log")
    ax.set_title("LSTM-AE 異常分數：正常 vs 三種異常")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_error_distribution.png", bbox_inches="tight")
    print(f"圖已存檔：{FIGURE_DIR}")


if __name__ == "__main__":
    main()
