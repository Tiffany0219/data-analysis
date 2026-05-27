import pandas as pd


def roc_month_to_yyyy_mm(value):
    value = str(value).strip().replace('"', "")
    if "年" not in value or "月" not in value:
        return pd.NA

    year_text, month_text = value.split("年", 1)
    month_text = month_text.replace("月", "").strip()

    year = pd.to_numeric(year_text, errors="coerce")
    month = pd.to_numeric(month_text, errors="coerce")
    if pd.isna(year) or pd.isna(month):
        return pd.NA

    return f"{int(year) + 1911:04d}-{int(month):02d}"


def read_regular_monthly_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["月份"] = df["月份"].astype(str).str.strip()
    return df


def read_cpi_monthly(path):
    df = pd.read_csv(path, encoding="utf-8-sig", skiprows=2, engine="python")
    df.columns = df.columns.str.strip()
    df["月份"] = df["統計期"].apply(roc_month_to_yyyy_mm)
    df = df.dropna(subset=["月份"]).copy()
    df = df.rename(
        columns={
            "總指數": "CPI總指數",
            "一.食物類": "CPI食物類",
            "二.衣著類": "CPI衣著類",
            "三.居住類": "CPI居住類",
            "四.交通及通訊類": "CPI交通及通訊類",
            "五.醫藥保健類": "CPI醫藥保健類",
            "六.教養娛樂類": "CPI教養娛樂類",
            "七.雜項類": "CPI雜項類",
        }
    )
    keep_cols = [
        "月份",
        "CPI總指數",
        "CPI食物類",
        "CPI衣著類",
        "CPI居住類",
        "CPI交通及通訊類",
        "CPI醫藥保健類",
        "CPI教養娛樂類",
        "CPI雜項類",
    ]
    return df[keep_cols]


def read_unemployment_monthly(path):
    df = pd.read_csv(path, encoding="utf-8-sig", skiprows=2, engine="python")
    df.columns = df.columns.str.strip()
    df["月份"] = df["統計期"].apply(roc_month_to_yyyy_mm)
    df = df.dropna(subset=["月份"]).copy()
    df = df.rename(
        columns={
            "總人口數(千人)": "總人口數千人",
            "15歲以上民間人口(千人)": "15歲以上民間人口千人",
            "勞動力(千人)": "勞動力千人",
            "就業人數(千人)": "就業人數千人",
            "失業人數(千人)": "失業人數千人",
            "勞動力參與率(%)": "勞動力參與率",
            "失業率(%)": "失業率",
        }
    )
    keep_cols = [
        "月份",
        "總人口數千人",
        "15歲以上民間人口千人",
        "勞動力千人",
        "就業人數千人",
        "失業人數千人",
        "勞動力參與率",
        "失業率",
    ]
    return df[keep_cols]


# =========================
# 1. 讀取五份月資料
# =========================
lottery = read_regular_monthly_csv("lottery_monthly_2011_2025.csv")
cpi = read_cpi_monthly("cpi_monthly_2011_2025.csv")
unemployment = read_unemployment_monthly("unemployment_monthly_2011_2025.csv")
cci = read_regular_monthly_csv("CCI_monthly_2011_2025.csv")
taiex = read_regular_monthly_csv("TAIEX_monthly_2011_2025_with_estimates.csv")
rollover = pd.read_csv("lottery_rollover_monthly_2011_2025.csv", encoding="utf-8-sig")

# =========================
# 2. 清理欄位名稱與月份格式
# =========================
dataframes = [lottery, cpi, unemployment, cci, taiex, rollover]

for df in dataframes:
    df.columns = df.columns.str.strip()
    df["月份"] = df["月份"].astype(str).str.strip()

# =========================
# 3. 合併資料
# =========================
master = lottery.merge(cpi, on="月份", how="left")
master = master.merge(unemployment, on="月份", how="left")
master = master.merge(cci, on="月份", how="left")
master = master.merge(taiex, on="月份", how="left")
master = master.merge(rollover, on="月份", how="left")

# =========================
# 4. 篩選期間 2010-01 到 2025-12
# =========================
master = master[
    (master["月份"] >= "2010-01") &
    (master["月份"] <= "2025-12")
].copy()

# =========================
# 5. 建立節慶虛擬變數
# =========================
master["春節"] = 0
master["端午"] = 0
master["中秋"] = 0

spring_months = [
    "2010-02",
    "2011-02", "2012-01", "2013-02", "2014-01", "2015-02",
    "2016-02", "2017-01", "2018-02", "2019-02", "2020-01",
    "2021-02", "2022-02", "2023-01", "2024-02", "2025-01",
    "2025-02"
]

dragon_boat_months = [
    "2010-06",
    "2011-06", "2012-06", "2013-06", "2014-06", "2015-06",
    "2016-06", "2017-05", "2018-06", "2019-06", "2020-06",
    "2021-06", "2022-06", "2023-06", "2024-06", "2025-05",
    "2025-06"
]

mid_autumn_months = [
    "2010-09",
    "2011-09", "2012-09", "2013-09", "2014-09", "2015-09",
    "2016-09", "2017-10", "2018-09", "2019-09", "2020-10",
    "2021-09", "2022-09", "2023-09", "2024-09", "2025-10"
]

master.loc[master["月份"].isin(spring_months), "春節"] = 1
master.loc[master["月份"].isin(dragon_boat_months), "端午"] = 1
master.loc[master["月份"].isin(mid_autumn_months), "中秋"] = 1

# =========================
# 6. 數值欄位轉型
# =========================
for col in master.columns:
    if col != "月份":
        master[col] = pd.to_numeric(master[col], errors="coerce")

# =========================
# 7. 排序
# =========================
master["月份_dt"] = pd.to_datetime(master["月份"], format="%Y-%m", errors="coerce")
master = master.sort_values("月份_dt").reset_index(drop=True)

# =========================
# 8. 計算年變動變數
# =========================
if "CPI總指數" not in master.columns:
    raise ValueError("找不到 CPI總指數 欄位，請確認 master 檔案欄位名稱")
if "失業率" not in master.columns:
    raise ValueError("找不到 失業率 欄位，請確認 master 檔案欄位名稱")
if "CCI" not in master.columns:
    raise ValueError("找不到 CCI 欄位，請確認 master 檔案欄位名稱")
if "TAIEX月平均收盤指數" not in master.columns:
    raise ValueError("找不到 TAIEX月平均收盤指數 欄位，請確認 master 檔案欄位名稱")

master["通膨率"] = master["CPI總指數"].pct_change(periods=12) * 100
master["通膨率"] = master["通膨率"].round(2)
master["失業率年變動_pp"] = master["失業率"] - master["失業率"].shift(12)
master["失業率年變動_pp"] = master["失業率年變動_pp"].round(2)
master["CCI年變動"] = master["CCI"] - master["CCI"].shift(12)
master["CCI年變動"] = master["CCI年變動"].round(2)
master["TAIEX年增率"] = master["TAIEX月平均收盤指數"].pct_change(periods=12) * 100
master["TAIEX年增率"] = master["TAIEX年增率"].round(2)
master = master.drop(columns=["月份_dt"])

# =========================
# 9. 檢查結果
# =========================
pd.options.display.float_format = "{:,.2f}".format

print("--- Master Dataset Final ---")
print(master.head())
print(master.tail())

print("\n資料筆數：", len(master))
print("月份範圍：", master["月份"].min(), "到", master["月份"].max())

print("\n缺失值檢查：")
print(master.isnull().sum())

print("\nCPI 與通膨率檢查：")
print(master[["月份", "CPI總指數", "通膨率"]].head(15))
print(master[["月份", "CPI總指數", "通膨率"]].tail())

print("\n失業率與失業率年變動檢查：")
print(master[["月份", "失業率", "失業率年變動_pp"]].head(15))
print(master[["月份", "失業率", "失業率年變動_pp"]].tail())

print("\nCCI 與 CCI 年變動檢查：")
print(master[["月份", "CCI", "CCI年變動"]].head(15))
print(master[["月份", "CCI", "CCI年變動"]].tail())

print("\nTAIEX 與 TAIEX 年增率檢查：")
print(master[["月份", "TAIEX月平均收盤指數", "TAIEX年增率"]].head(15))
print(master[["月份", "TAIEX月平均收盤指數", "TAIEX年增率"]].tail())

print("\n節慶 dummy 檢查：")
print("春節月份數：", int(master["春節"].sum()))
print("端午月份數：", int(master["端午"].sum()))
print("中秋月份數：", int(master["中秋"].sum()))

if "TAIEX是否估計" in master.columns:
    print("\nTAIEX 估計資料筆數：", int(master["TAIEX是否估計"].sum()))
    print(master.loc[
        master["TAIEX是否估計"] == 1,
        ["月份", "TAIEX月平均收盤指數", "TAIEX是否估計"]
    ])

# =========================
# 10. 輸出
# =========================
master.to_csv(
    "master_dataset_final_2011_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n已輸出：master_dataset_final_2011_2025.csv")
