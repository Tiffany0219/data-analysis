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

DISPLAY_NAMES = {
    "電腦型彩券總銷售額_億元": "電腦型彩券總銷售額(億)",
    "大樂透銷售總額_億元": "大樂透銷售總額(億)",
    "威力彩銷售總額_億元": "威力彩銷售總額(億)",
    "今彩539銷售總額_億元": "今彩539銷售總額(億)",
    "通膨率": "CPI年增率",
    "失業率年變動_pp": "失業率年變動(pp)",
    "CCI年變動": "CCI年變動",
    "TAIEX年增率": "TAIEX年增率(%)",
    "電腦型彩券最高連槓獎金_億元": "連槓獎金(億)",
}

# =========================
# 2. 金額轉億元
# =========================
df["電腦型彩券總銷售額_億元"] = df["電腦型彩券總銷售額"] / 100000000
df["大樂透銷售總額_億元"] = df["大樂透銷售總額"] / 100000000
df["威力彩銷售總額_億元"] = df["威力彩銷售總額"] / 100000000
df["今彩539銷售總額_億元"] = df["今彩539銷售總額"] / 100000000
df["電腦型彩券最高連槓獎金_億元"] = df["電腦型彩券最高連槓獎金"] / 100000000

# =========================
# 3. 選擇變數
# =========================
cols = [
    "電腦型彩券總銷售額_億元",
    "大樂透銷售總額_億元",
    "威力彩銷售總額_億元",
    "今彩539銷售總額_億元",
    "通膨率",
    "失業率年變動_pp",
    "CCI年變動",
    "TAIEX年增率",
    "電腦型彩券最高連槓獎金_億元",
    "春節",
    "端午",
    "中秋"
]

df_analysis = df.dropna(subset=cols).copy()

corr = df_analysis[cols].corr().round(2)
corr = corr.rename(index=DISPLAY_NAMES, columns=DISPLAY_NAMES)

# =========================
# 4. 畫熱力圖
# =========================
plt.figure(figsize=(13, 10))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Pearson相關係數")

plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.index)), corr.index)

for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

plt.title("Pearson相關矩陣熱力圖")
plt.tight_layout()
plt.savefig("03_相關矩陣熱力圖.png", dpi=300)
plt.close()

print("已輸出：03_相關矩陣熱力圖.png")
