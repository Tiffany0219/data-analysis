import pandas as pd

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
# 3. Pearson 相關分析
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

corr = df_analysis[cols].corr().round(3)
corr = corr.rename(index=DISPLAY_NAMES, columns=DISPLAY_NAMES)

# 與總銷售額的相關係數
target_corr = corr[["電腦型彩券總銷售額(億)"]].sort_values(
    by="電腦型彩券總銷售額(億)",
    ascending=False
)

print("--- Pearson 相關矩陣 ---")
print(corr)

print("\n--- 各變數與電腦型彩券總銷售額的相關係數 ---")
print(target_corr)

corr.to_csv("02_Pearson相關矩陣.csv", encoding="utf-8-sig")
target_corr.to_csv("02_與總銷售額相關係數.csv", encoding="utf-8-sig")

print("\n已輸出：")
print("02_Pearson相關矩陣.csv")
print("02_與總銷售額相關係數.csv")
