import pandas as pd
import glob
import os

# =========================
# 今彩539：2010-2025 月度彙總
# =========================

folder_path = "今彩539"
required_cols = [
    "遊戲名稱",
    "期別",
    "開獎日期",
    "銷售總額",
    "銷售注數",
    "總獎金",
]

# 讀取 今彩539/ 資料夾底下所有 今彩539_*.csv
files = sorted(glob.glob(os.path.join(folder_path, "今彩539_*.csv")))

print("找到今彩539檔案數量：", len(files))

if len(files) == 0:
    raise ValueError("找不到今彩539 CSV 檔案，請確認資料夾名稱是「今彩539」，檔名格式是「今彩539_2011.csv」")

all_data = []

for file in files:
    print("正在讀取：", file)

    # 不使用 usecols=range，避免欄位數不足造成錯誤
    df = pd.read_csv(
        file,
        encoding="utf-8-sig",
        engine="python",
        index_col=False,
        usecols=lambda col: col.strip() in required_cols,
    )

    # 清理欄位名稱
    df.columns = df.columns.astype(str).str.strip()

    # 檢查欄位是否存在
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"{file} 缺少欄位：{missing_cols}\n"
            f"目前欄位：{list(df.columns)}"
        )

    # 只保留需要欄位
    df = df[required_cols].copy()

    # =========================
    # 日期轉換
    # =========================
    df["開獎日期"] = df["開獎日期"].astype(str).str.strip()

    df["開獎日期"] = pd.to_datetime(
        df["開獎日期"],
        format="%Y/%m/%d",
        errors="coerce"
    )

    # =========================
    # 數值欄位轉換
    # =========================
    for col in ["銷售總額", "銷售注數", "總獎金"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 刪除無效資料
    df = df.dropna(subset=["開獎日期", "銷售總額", "銷售注數", "總獎金"])

    all_data.append(df)

# =========================
# 合併所有年份資料
# =========================
df_all = pd.concat(all_data, ignore_index=True)

# =========================
# 建立月份欄位
# =========================
df_all["月份"] = df_all["開獎日期"].dt.to_period("M").astype(str)

# =========================
# 篩選研究期間：2010-01 到 2025-12
# =========================
df_all = df_all[
    (df_all["月份"] >= "2010-01") &
    (df_all["月份"] <= "2025-12")
].copy()

# =========================
# 月度彙總
# =========================
monthly = df_all.groupby("月份", as_index=False).agg(
    今彩539開獎期數=("期別", "count"),
    今彩539銷售總額=("銷售總額", "sum"),
    今彩539銷售注數=("銷售注數", "sum"),
    今彩539單月最高總獎金=("總獎金", "max")
)

# 平均每期銷售額
monthly["今彩539平均每期銷售額"] = (
    monthly["今彩539銷售總額"] / monthly["今彩539開獎期數"]
)

# =========================
# 排序
# =========================
monthly["月份_dt"] = pd.to_datetime(monthly["月份"], format="%Y-%m", errors="coerce")
monthly = (
    monthly.sort_values("月份_dt")
    .drop(columns=["月份_dt"])
    .reset_index(drop=True)
)

pd.options.display.float_format = "{:,.2f}".format

print("\n--- 今彩539月度彙總 ---")
print(monthly.head())
print(monthly.tail())

print("\n原始逐期資料筆數：", len(df_all))
print("月資料筆數：", len(monthly))
print("月份範圍：", monthly["月份"].min(), "到", monthly["月份"].max())

print("\n缺失值檢查：")
print(monthly.isnull().sum())

monthly.to_csv(
    "今彩539_monthly_2011_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n已輸出：今彩539_monthly_2011_2025.csv")
