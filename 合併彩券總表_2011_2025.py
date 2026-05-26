import pandas as pd

# =========================
# 1. 讀取三種彩券月資料
# =========================
big_lotto = pd.read_csv("大樂透_monthly_2011_2025.csv", encoding="utf-8-sig")
power_lotto = pd.read_csv("威力彩_monthly_2011_2025.csv", encoding="utf-8-sig")
daily_cash = pd.read_csv("今彩539_monthly_2011_2025.csv", encoding="utf-8-sig")

# =========================
# 2. 清理欄位名稱與月份格式
# =========================
for df in [big_lotto, power_lotto, daily_cash]:
    df.columns = df.columns.str.strip()
    df["月份"] = df["月份"].astype(str).str.strip()

# =========================
# 3. 依照月份合併三種彩券
# =========================
lottery = big_lotto.merge(power_lotto, on="月份", how="outer")
lottery = lottery.merge(daily_cash, on="月份", how="outer")

# =========================
# 4. 依月份排序
# =========================
lottery["月份_dt"] = pd.to_datetime(lottery["月份"], format="%Y-%m", errors="coerce")
lottery = lottery.sort_values("月份_dt").drop(columns=["月份_dt"])

# =========================
# 5. 數值欄位轉換
# =========================
numeric_cols = [
    "大樂透開獎期數",
    "大樂透銷售總額",
    "大樂透銷售注數",
    "大樂透單月最高總獎金",
    "大樂透平均每期銷售額",

    "威力彩開獎期數",
    "威力彩銷售總額",
    "威力彩銷售注數",
    "威力彩單月最高總獎金",
    "威力彩平均每期銷售額",

    "今彩539開獎期數",
    "今彩539銷售總額",
    "今彩539銷售注數",
    "今彩539單月最高總獎金",
    "今彩539平均每期銷售額"
]

for col in numeric_cols:
    if col not in lottery.columns:
        raise ValueError(f"缺少欄位：{col}")
    lottery[col] = pd.to_numeric(lottery[col], errors="coerce")

# 如果某月份某券種沒有資料，先補 0
lottery[numeric_cols] = lottery[numeric_cols].fillna(0)

# =========================
# 6. 建立電腦型彩券總指標
# =========================

# 三種彩券月銷售額加總
lottery["電腦型彩券總銷售額"] = (
    lottery["大樂透銷售總額"]
    + lottery["威力彩銷售總額"]
    + lottery["今彩539銷售總額"]
)

# 三種彩券月銷售注數加總
lottery["電腦型彩券總銷售注數"] = (
    lottery["大樂透銷售注數"]
    + lottery["威力彩銷售注數"]
    + lottery["今彩539銷售注數"]
)

# 三種彩券月開獎期數加總
lottery["電腦型彩券總開獎期數"] = (
    lottery["大樂透開獎期數"]
    + lottery["威力彩開獎期數"]
    + lottery["今彩539開獎期數"]
)

# 當月最高總獎金：三種彩券各自單月最高總獎金中取最大
lottery["當月最高總獎金"] = lottery[
    [
        "大樂透單月最高總獎金",
        "威力彩單月最高總獎金",
        "今彩539單月最高總獎金"
    ]
].max(axis=1)

# 電腦型彩券平均每期銷售額
lottery["電腦型彩券平均每期銷售額"] = (
    lottery["電腦型彩券總銷售額"] / lottery["電腦型彩券總開獎期數"]
)

# =========================
# 7. 篩選研究期間
# =========================
lottery = lottery[
    (lottery["月份"] >= "2010-01") &
    (lottery["月份"] <= "2025-12")
].copy()

# =========================
# 8. 整理欄位順序
# =========================
final_cols = [
    "月份",

    # 三種彩券銷售額
    "大樂透銷售總額",
    "威力彩銷售總額",
    "今彩539銷售總額",
    "電腦型彩券總銷售額",

    # 三種彩券銷售注數
    "大樂透銷售注數",
    "威力彩銷售注數",
    "今彩539銷售注數",
    "電腦型彩券總銷售注數",

    # 開獎期數
    "大樂透開獎期數",
    "威力彩開獎期數",
    "今彩539開獎期數",
    "電腦型彩券總開獎期數",

    # 獎金誘因
    "大樂透單月最高總獎金",
    "威力彩單月最高總獎金",
    "今彩539單月最高總獎金",
    "當月最高總獎金",

    # 平均每期銷售額
    "大樂透平均每期銷售額",
    "威力彩平均每期銷售額",
    "今彩539平均每期銷售額",
    "電腦型彩券平均每期銷售額"
]

lottery = lottery[final_cols]

# =========================
# 9. 檢查結果
# =========================
pd.options.display.float_format = "{:,.2f}".format

print("--- 合併後彩券月資料 ---")
print(lottery.head())
print(lottery.tail())

print("\n月資料筆數：", len(lottery))
print("月份範圍：", lottery["月份"].min(), "到", lottery["月份"].max())

print("\n缺失值檢查：")
print(lottery.isnull().sum())

print("\n銷售額檢查：")
print(lottery[[
    "月份",
    "大樂透銷售總額",
    "威力彩銷售總額",
    "今彩539銷售總額",
    "電腦型彩券總銷售額"
]].head())

print("\n當月最高總獎金檢查：")
print(lottery[[
    "月份",
    "大樂透單月最高總獎金",
    "威力彩單月最高總獎金",
    "今彩539單月最高總獎金",
    "當月最高總獎金"
]].head())

# =========================
# 10. 輸出 CSV
# =========================
lottery.to_csv(
    "lottery_monthly_2011_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n已輸出：lottery_monthly_2011_2025.csv")
