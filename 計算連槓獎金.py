import pandas as pd
import glob
import os

# =========================
# 1. 讀取單一券種逐期資料
# =========================
def read_lottery_raw(folder_path, prefix):
    files = sorted(glob.glob(os.path.join(folder_path, f"{prefix}_*.csv")))

    print(f"找到 {prefix} 檔案數量：", len(files))

    if len(files) == 0:
        raise ValueError(f"找不到 {prefix} CSV 檔案，請確認資料夾與檔名")

    all_data = []

    for file in files:
        print(f"正在讀取 {prefix}：", file)

        df = pd.read_csv(
            file,
            encoding="utf-8-sig",
            engine="python",
            index_col=False
        )

        df.columns = df.columns.astype(str).str.strip()

        required_cols = [
            "遊戲名稱",
            "期別",
            "開獎日期",
            "銷售總額",
            "銷售注數",
            "總獎金"
        ]

        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(
                f"{file} 缺少欄位：{missing_cols}\n"
                f"目前欄位：{list(df.columns)}"
            )

        df = df[required_cols].copy()

        df["開獎日期"] = pd.to_datetime(
            df["開獎日期"].astype(str).str.strip(),
            format="%Y/%m/%d",
            errors="coerce"
        )

        df["期別"] = pd.to_numeric(df["期別"], errors="coerce")

        for col in ["銷售總額", "銷售注數", "總獎金"]:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["開獎日期", "期別", "總獎金"])

        all_data.append(df)

    df_all = pd.concat(all_data, ignore_index=True)

    df_all = df_all.sort_values(["開獎日期", "期別"]).reset_index(drop=True)

    df_all["月份"] = df_all["開獎日期"].dt.to_period("M").astype(str)

    df_all = df_all[
        (df_all["月份"] >= "2010-01") &
        (df_all["月份"] <= "2025-12")
    ].copy()

    return df_all


# =========================
# 2. 計算連槓獎金代理變數
# =========================
def calculate_rollover(df, prefix):
    df = df.copy()

    # 上一期總獎金
    df["上一期總獎金"] = df["總獎金"].shift(1)

    # 是否獎金上升
    df["獎金上升"] = (df["總獎金"] > df["上一期總獎金"]).astype(int)

    # 是否重置：本期總獎金低於上一期
    df["獎金重置"] = (df["總獎金"] < df["上一期總獎金"]).astype(int)

    # 第一筆資料視為新一輪
    df.loc[df.index[0], "獎金重置"] = 1

    # 連槓輪次：每次重置就開新一輪
    df["連槓輪次"] = df["獎金重置"].cumsum()

    # 每一輪內目前最高獎金
    df[f"{prefix}連槓獎金"] = df.groupby("連槓輪次")["總獎金"].cummax()

    # 如果你只想把「真的上升」才視為連槓，可以打開這段
    # 但目前先保留每輪最高值，對月資料比較穩
    # df.loc[df["獎金上升"] == 0, f"{prefix}連槓獎金"] = df["總獎金"]

    monthly = df.groupby("月份", as_index=False).agg(
        **{
            f"{prefix}連槓獎金": (f"{prefix}連槓獎金", "max"),
            f"{prefix}連槓期數": ("獎金上升", "sum"),
            f"{prefix}重置次數": ("獎金重置", "sum")
        }
    )

    return monthly


# =========================
# 3. 分別處理三種彩券
# =========================
big_raw = read_lottery_raw("大樂透", "大樂透")
power_raw = read_lottery_raw("威力彩", "威力彩")
cash_raw = read_lottery_raw("今彩539", "今彩539")

big_rollover = calculate_rollover(big_raw, "大樂透")
power_rollover = calculate_rollover(power_raw, "威力彩")
cash_rollover = calculate_rollover(cash_raw, "今彩539")

# =========================
# 4. 合併三種彩券連槓獎金
# =========================
rollover = big_rollover.merge(power_rollover, on="月份", how="outer")
rollover = rollover.merge(cash_rollover, on="月份", how="outer")

rollover = rollover.sort_values("月份").reset_index(drop=True)

# 缺值補 0
numeric_cols = [col for col in rollover.columns if col != "月份"]
rollover[numeric_cols] = rollover[numeric_cols].fillna(0)

# 電腦型彩券最高連槓獎金
rollover["電腦型彩券最高連槓獎金"] = rollover[
    [
        "大樂透連槓獎金",
        "威力彩連槓獎金",
        "今彩539連槓獎金"
    ]
].max(axis=1)

# 三種彩券連槓期數加總
rollover["電腦型彩券連槓期數"] = rollover[
    [
        "大樂透連槓期數",
        "威力彩連槓期數",
        "今彩539連槓期數"
    ]
].sum(axis=1)

# =========================
# 5. 檢查結果
# =========================
pd.options.display.float_format = "{:,.2f}".format

print("\n--- 連槓獎金月資料 ---")
print(rollover.head())
print(rollover.tail())

print("\n資料筆數：", len(rollover))
print("月份範圍：", rollover["月份"].min(), "到", rollover["月份"].max())

print("\n缺失值檢查：")
print(rollover.isnull().sum())

# =========================
# 6. 輸出
# =========================
rollover.to_csv(
    "lottery_rollover_monthly_2011_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n已輸出：lottery_rollover_monthly_2011_2025.csv")
