import pandas as pd
import os

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# 中文字型設定
# =========================
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 1. 讀取資料
# =========================
df = pd.read_csv("master_dataset_final_2011_2025.csv", encoding="utf-8-sig")
df.columns = df.columns.str.strip()

df["月份"] = pd.to_datetime(df["月份"], format="%Y-%m")
df = df[
    (df["月份"] >= pd.to_datetime("2011-01")) &
    (df["月份"] <= pd.to_datetime("2025-12"))
].copy()

# =========================
# 2. 金額轉億元
# =========================
df["電腦型彩券總銷售額_億元"] = df["電腦型彩券總銷售額"] / 100000000
df["大樂透銷售總額_億元"] = df["大樂透銷售總額"] / 100000000
df["威力彩銷售總額_億元"] = df["威力彩銷售總額"] / 100000000
df["今彩539銷售總額_億元"] = df["今彩539銷售總額"] / 100000000
df["電腦型彩券最高連槓獎金_億元"] = df["電腦型彩券最高連槓獎金"] / 100000000


def add_event_markers(ax, include_labels=True):
    covid_start = pd.to_datetime("2020-01")
    covid_end = pd.to_datetime("2022-12")
    inflation_start = pd.to_datetime("2022-01")

    ax.axvspan(
        covid_start,
        covid_end,
        alpha=0.12,
        color="#7A7A7A",
        label="COVID-19期間" if include_labels else None,
    )
    ax.axvline(
        covid_start,
        linestyle="--",
        linewidth=1.2,
        color="#555555",
        label="COVID-19開始" if include_labels else None,
    )
    ax.axvline(
        inflation_start,
        linestyle=":",
        linewidth=1.8,
        color="#C2410C",
        label="通膨衝擊開始" if include_labels else None,
    )


# =========================
# 3. 圖1：總銷售額趨勢
# =========================
plt.figure(figsize=(14, 6))
plt.plot(df["月份"], df["電腦型彩券總銷售額_億元"], label="電腦型彩券總銷售額(億)", linewidth=2)
add_event_markers(plt.gca())

plt.title("2011-2025 電腦型彩券總銷售額(億)月趨勢")
plt.xlabel("月份")
plt.ylabel("電腦型彩券總銷售額(億)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("04_電腦型彩券總銷售額月趨勢.png", dpi=300)
plt.close()

# =========================
# 4. 圖2：三種彩券趨勢比較
# =========================
plt.figure(figsize=(14, 6))
plt.plot(df["月份"], df["大樂透銷售總額_億元"], label="大樂透銷售總額(億)", linewidth=2)
plt.plot(df["月份"], df["威力彩銷售總額_億元"], label="威力彩銷售總額(億)", linewidth=2)
plt.plot(df["月份"], df["今彩539銷售總額_億元"], label="今彩539銷售總額(億)", linewidth=2)
add_event_markers(plt.gca())

plt.title("三種電腦型彩券月銷售額趨勢比較")
plt.xlabel("月份")
plt.ylabel("銷售額(億)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("04_三種彩券月銷售額趨勢比較.png", dpi=300)
plt.close()

# =========================
# 5. 雙軸圖函數
# =========================
def dual_axis_plot(y2_col, y2_label, title, filename):
    fig, ax1 = plt.subplots(figsize=(14, 6))

    line1 = ax1.plot(
        df["月份"],
        df["電腦型彩券總銷售額_億元"],
        label="電腦型彩券總銷售額(億)",
        linewidth=2
    )

    ax1.set_xlabel("月份")
    ax1.set_ylabel("電腦型彩券總銷售額(億)")

    ax2 = ax1.twinx()

    line2 = ax2.plot(
        df["月份"],
        df[y2_col],
        label=y2_label,
        linewidth=2,
        linestyle="--"
    )

    ax2.set_ylabel(y2_label)

    add_event_markers(ax1)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    plt.title(title)
    fig.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)

# =========================
# 6. 輸出雙軸圖
# =========================
dual_axis_plot(
    "通膨率",
    "CPI年增率",
    "電腦型彩券總銷售額(億)與CPI年增率趨勢",
    "04_彩券總銷售額_vs_通膨率.png"
)

dual_axis_plot(
    "失業率年變動_pp",
    "失業率年變動(pp)",
    "電腦型彩券總銷售額(億)與失業率年變動趨勢",
    "04_彩券總銷售額_vs_失業率年變動.png"
)

dual_axis_plot(
    "CCI年變動",
    "CCI年變動",
    "電腦型彩券總銷售額(億)與CCI年變動趨勢",
    "04_彩券總銷售額_vs_CCI年變動.png"
)

dual_axis_plot(
    "TAIEX年增率",
    "TAIEX年增率(%)",
    "電腦型彩券總銷售額(億)與TAIEX年增率趨勢",
    "04_彩券總銷售額_vs_TAIEX年增率.png"
)

dual_axis_plot(
    "電腦型彩券最高連槓獎金_億元",
    "連槓獎金(億)",
    "電腦型彩券總銷售額(億)與最高連槓累積獎金趨勢",
    "04_彩券總銷售額_vs_最高連槓累積獎金.png"
)

print("已輸出雙軸折線圖。")
