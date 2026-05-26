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
# 2. 金額轉成億元
# =========================
money_cols = [
    "電腦型彩券總銷售額",
    "大樂透銷售總額",
    "威力彩銷售總額",
    "今彩539銷售總額",
    "電腦型彩券最高連槓獎金",
]

for col in money_cols:
    if col in df.columns:
        df[col + "_億元"] = df[col] / 100000000

# =========================
# 3. 描述性統計欄位
# =========================
cols = [
    "電腦型彩券總銷售額_億元",
    "大樂透銷售總額_億元",
    "威力彩銷售總額_億元",
    "今彩539銷售總額_億元",
    "電腦型彩券最高連槓獎金_億元",
    "通膨率",
    "失業率年變動_pp",
    "CCI年變動",
    "TAIEX年增率"
]

df_analysis = df.dropna(subset=cols).copy()

desc = df_analysis[cols].describe().T

desc = desc.rename(columns={
    "count": "樣本數",
    "mean": "平均值",
    "std": "標準差",
    "min": "最小值",
    "25%": "第一四分位數",
    "50%": "中位數",
    "75%": "第三四分位數",
    "max": "最大值"
}).round(2)
desc = desc.rename(index=DISPLAY_NAMES)

# =========================
# 4. 節慶 dummy 次數表
# =========================
festival = pd.DataFrame({
    "節慶": ["春節", "端午", "中秋"],
    "標記月份數": [
        int(df_analysis["春節"].sum()),
        int(df_analysis["端午"].sum()),
        int(df_analysis["中秋"].sum())
    ],
    "總月份數": [len(df_analysis), len(df_analysis), len(df_analysis)]
})

festival["比例"] = (festival["標記月份數"] / festival["總月份數"]).round(3)

# =========================
# 5. 輸出
# =========================
print("--- 描述性統計 ---")
print(desc)

print("\n--- 節慶 dummy 次數表 ---")
print(festival)

desc.to_csv("01_描述性統計.csv", encoding="utf-8-sig")
festival.to_csv("01_節慶dummy次數表.csv", index=False, encoding="utf-8-sig")

print("\n已輸出：")
print("01_描述性統計.csv")
print("01_節慶dummy次數表.csv")
