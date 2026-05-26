import pandas as pd

# =========================
# 1. 讀取 CCI Excel
# =========================
file_path = "未來半年國內經濟景氣.xlsx"

df = pd.read_excel(
    file_path,
    sheet_name=0,
    engine="openpyxl"
)

# =========================
# 2. 清理欄位名稱
# =========================
df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.replace("\n", "", regex=False)
    .str.replace(" ", "", regex=False)
)

print("原始欄位：")
print(df.columns)

# =========================
# 3. 重新命名欄位
# =========================
# 你的檔案第一欄是月份，第二欄是未來半年國內經濟景氣
df = df.iloc[:, [0, 1]].copy()
df.columns = ["月份", "CCI"]

# =========================
# 4. 月份轉成 YYYY-MM
# =========================
df["月份"] = pd.to_datetime(df["月份"], errors="coerce")
df["月份"] = df["月份"].dt.to_period("M").astype(str)

# =========================
# 5. CCI 轉數值
# =========================
df["CCI"] = pd.to_numeric(df["CCI"], errors="coerce")

# =========================
# 6. 清理無效資料
# =========================
df = df.dropna(subset=["月份", "CCI"])

# =========================
# 7. 篩選 2011-01 到 2025-12
# =========================
df = df[
    (df["月份"] >= "2011-01") &
    (df["月份"] <= "2025-12")
].copy()

# =========================
# 8. 排序與去重
# =========================
df = df.drop_duplicates(subset=["月份"], keep="last")
df = df.sort_values("月份").reset_index(drop=True)

# =========================
# 9. 檢查結果
# =========================
print("\n--- CCI 月資料 ---")
print(df.head())
print(df.tail())

print("\n資料筆數：", len(df))
print("月份範圍：", df["月份"].min(), "到", df["月份"].max())

print("\n缺失值檢查：")
print(df.isnull().sum())

# =========================
# 10. 輸出 CSV
# =========================
df.to_csv(
    "CCI_monthly_2011_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n已輸出：CCI_monthly_2011_2025.csv")