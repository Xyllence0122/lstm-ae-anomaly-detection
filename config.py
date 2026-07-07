# -*- coding: utf-8 -*-
"""
專案共用設定：路徑、sensor 選擇、繪圖樣式
LSTM-AE 半導體異常偵測（LAM 9600 Metal Etcher）
"""
from pathlib import Path

# ---------- 路徑 ----------
PROJECT_DIR = Path(__file__).parent
DATA_MAT = Path(r"C:\Max\Paper\AIoT\LAM 9600 Metal Etcher Dataset\MACHINE_Data.mat")
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = PROJECT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

# ---------- LAM 9600 的 21 個工程變數 ----------
VAR_NAMES = [
    "Time",            # 0
    "Step Number",     # 1
    "BCl3 Flow",       # 2
    "Cl2 Flow",        # 3
    "RF Btm Pwr",      # 4
    "RF Btm Rfl Pwr",  # 5
    "Endpt A",         # 6
    "He Press",        # 7
    "Pressure",        # 8
    "RF Tuner",        # 9
    "RF Load",         # 10
    "RF Phase Err",    # 11
    "RF Pwr",          # 12
    "RF Impedance",    # 13
    "TCP Tuner",       # 14
    "TCP Phase Err",   # 15
    "TCP Impedance",   # 16
    "TCP Top Pwr",     # 17
    "TCP Rfl Pwr",     # 18
    "TCP Load",        # 19
    "Vat Valve",       # 20
]

# ---------- 選用的 sensors（從設備控制角度選擇，物理意義明確） ----------
# index -> (名稱, 物理意義)
# 敘事：被控變數（壓力）、執行器（閥門開度）、冷卻系統（He 壓力）、流量迴路（MFC）
SELECTED_SENSORS = {
    3:  ("Cl2 Flow",  "製程氣體流量 (sccm)，MFC 流量控制迴路"),
    7:  ("He Press",  "晶圓背面 He 冷卻壓力，散熱系統壓力控制"),
    8:  ("Pressure",  "反應腔壓力 (mTorr)，壓力控制迴路的被控變數"),
    20: ("Vat Valve", "節流閥開度，調節腔體壓力的執行器"),
}
SENSOR_IDX = sorted(SELECTED_SENSORS.keys())
SENSOR_NAMES = [SELECTED_SENSORS[i][0] for i in SENSOR_IDX]

STEP_COL = 1          # Step Number 欄位
PROCESS_STEPS = (4, 5)  # 主蝕刻製程步驟（慣例：只分析 step 4、5）
MIN_WAFER_LEN = 60    # 過短的 wafer 視為紀錄不完整，剔除

SEQ_LEN = 90          # 合成資料的時間序列長度
RANDOM_SEED = 42

# ---------- 繪圖樣式（validated palette，light mode） ----------
COLORS = {
    "normal":  "#2a78d6",  # blue   - 正常
    "faulty":  "#e34948",  # red    - 異常
    "series3": "#eda100",  # yellow
    "series4": "#008300",  # green
    "series5": "#4a3aa7",  # violet
    "surface": "#fcfcfb",
    "grid":    "#e1e0d9",
    "muted":   "#898781",
    "ink":     "#0b0b0b",
    "ink2":    "#52514e",
}


def set_plot_style():
    """套用一致的 matplotlib 樣式（繁中字型、細格線、隱藏多餘外框）"""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft JhengHei", "SimHei", "Arial"],
        "axes.unicode_minus": False,
        "figure.facecolor": COLORS["surface"],
        "axes.facecolor": COLORS["surface"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["ink2"],
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.6,
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.facecolor": COLORS["surface"],
    })
