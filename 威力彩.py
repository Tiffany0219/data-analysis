import pandas as pd
import glob
import os

# =========================
# 威力彩：2010-2025 月度彙總
# =========================

folder_path = "威力彩"
files = sorted(glob.glob(os.path.join(folder_path, "威力彩_*.csv")))
required_columns = [
    "遊戲名稱",
    "期別",
    "開獎日期",
    "銷售總額",
    "銷售注數",
    "總獎金",
]

print("找到威力彩檔案數量：", len(files))

all_data = []

for file in files:
    print("正在讀取：", file)

    # 只讀需要的欄位，避免不同年份尾端多餘逗號造成錯位或越界
    df = pd.read_csv(
        file,
        encoding="utf-8-sig",
        engine="python",
        index_col=False,
        usecols=lambda col: col.strip() in required_columns,
    )

    df.columns = df.columns.astype(str).str.strip()

    df = df[required_columns].copy()

    df["開獎日期"] = df["開獎日期"].astype(str).str.strip()

    df["開獎日期"] = pd.to_datetime(
        df["開獎日期"],
        format="%Y/%m/%d",
        errors="coerce"
    )

    for col in ["銷售總額", "銷售注數", "總獎金"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["開獎日期", "銷售總額", "銷售注數", "總獎金"])

    all_data.append(df)

if len(all_data) == 0:
    raise ValueError("沒有讀到任何威力彩資料，請確認資料夾與檔名")

df_all = pd.concat(all_data, ignore_index=True)

df_all["月份"] = df_all["開獎日期"].dt.to_period("M").astype(str)

df_all = df_all[
    (df_all["月份"] >= "2010-01") &
    (df_all["月份"] <= "2025-12")
].copy()

monthly = df_all.groupby("月份", as_index=False).agg(
    威力彩開獎期數=("期別", "count"),
    威力彩銷售總額=("銷售總額", "sum"),
    威力彩銷售注數=("銷售注數", "sum"),
    威力彩單月最高總獎金=("總獎金", "max")
)

monthly["威力彩平均每期銷售額"] = (
    monthly["威力彩銷售總額"] / monthly["威力彩開獎期數"]
)

pd.options.display.float_format = "{:,.2f}".format

print("\n--- 威力彩月度彙總 ---")
print(monthly.head())
print(monthly.tail())

print("\n原始逐期資料筆數：", len(df_all))
print("月資料筆數：", len(monthly))
print("月份範圍：", monthly["月份"].min(), "到", monthly["月份"].max())

print("\n缺失值檢查：")
print(monthly.isnull().sum())

monthly.to_csv(
    "威力彩_monthly_2011_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n已輸出：威力彩_monthly_2011_2025.csv")
