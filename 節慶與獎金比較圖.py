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
df = df.dropna(subset=["通膨率", "失業率年變動_pp", "CCI年變動", "TAIEX年增率"]).copy()

# =========================
# 2. 金額轉億元
# =========================
df["電腦型彩券總銷售額_億元"] = df["電腦型彩券總銷售額"] / 100000000
df["大樂透銷售總額_億元"] = df["大樂透銷售總額"] / 100000000
df["威力彩銷售總額_億元"] = df["威力彩銷售總額"] / 100000000
df["今彩539銷售總額_億元"] = df["今彩539銷售總額"] / 100000000
df["電腦型彩券最高連槓獎金_億元"] = df["電腦型彩券最高連槓獎金"] / 100000000

# =========================
# 3. 建立節慶月份
# =========================
df["是否節慶月份"] = (
    (df["春節"] == 1) |
    (df["端午"] == 1) |
    (df["中秋"] == 1)
).astype(int)

df["月份類別"] = df["是否節慶月份"].map({
    0: "一般月份",
    1: "節慶月份"
})

# =========================
# 4. 節慶月份 vs 一般月份平均銷售額
# =========================
festival_compare = df.groupby("月份類別", as_index=False).agg(
    平均總銷售額_億元=("電腦型彩券總銷售額_億元", "mean"),
    平均大樂透銷售額_億元=("大樂透銷售總額_億元", "mean"),
    平均威力彩銷售額_億元=("威力彩銷售總額_億元", "mean"),
    平均今彩539銷售額_億元=("今彩539銷售總額_億元", "mean")
).round(2)
festival_compare = festival_compare.rename(
    columns={
        "平均總銷售額_億元": "平均電腦型彩券總銷售額(億)",
        "平均大樂透銷售額_億元": "平均大樂透銷售總額(億)",
        "平均威力彩銷售額_億元": "平均威力彩銷售總額(億)",
        "平均今彩539銷售額_億元": "平均今彩539銷售總額(億)",
    }
)

print("--- 節慶月份 vs 一般月份 ---")
print(festival_compare)

festival_compare.to_csv("05_節慶月份_vs_一般月份平均銷售額.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(8, 6))
plt.bar(festival_compare["月份類別"], festival_compare["平均電腦型彩券總銷售額(億)"])
plt.title("一般月份與節慶月份平均電腦型彩券總銷售額(億)比較")
plt.xlabel("月份類別")
plt.ylabel("平均電腦型彩券總銷售額(億)")
plt.tight_layout()
plt.savefig("05_一般月份_vs_節慶月份平均銷售額.png", dpi=300)
plt.close()

# =========================
# 5. 春節、端午、中秋比較
# =========================
festival_types = ["春節", "端午", "中秋"]

festival_rows = []

for festival in festival_types:
    festival_rows.append({
        "節慶": festival,
        "平均總銷售額_億元": df.loc[df[festival] == 1, "電腦型彩券總銷售額_億元"].mean(),
        "平均大樂透銷售額_億元": df.loc[df[festival] == 1, "大樂透銷售總額_億元"].mean(),
        "平均威力彩銷售額_億元": df.loc[df[festival] == 1, "威力彩銷售總額_億元"].mean(),
        "平均今彩539銷售額_億元": df.loc[df[festival] == 1, "今彩539銷售總額_億元"].mean()
    })

festival_table = pd.DataFrame(festival_rows).round(2)
festival_table = festival_table.rename(
    columns={
        "平均總銷售額_億元": "平均電腦型彩券總銷售額(億)",
        "平均大樂透銷售額_億元": "平均大樂透銷售總額(億)",
        "平均威力彩銷售額_億元": "平均威力彩銷售總額(億)",
        "平均今彩539銷售額_億元": "平均今彩539銷售總額(億)",
    }
)

print("\n--- 三大節慶平均銷售額 ---")
print(festival_table)

festival_table.to_csv("05_三大節慶平均銷售額.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(8, 6))
plt.bar(festival_table["節慶"], festival_table["平均電腦型彩券總銷售額(億)"])
plt.title("三大節慶月份平均電腦型彩券總銷售額(億)比較")
plt.xlabel("節慶")
plt.ylabel("平均電腦型彩券總銷售額(億)")
plt.tight_layout()
plt.savefig("05_三大節慶平均銷售額比較.png", dpi=300)
plt.close()

# =========================
# 6. 節慶箱型圖
# =========================
box_data = [
    df.loc[df["春節"] == 1, "電腦型彩券總銷售額_億元"],
    df.loc[df["端午"] == 1, "電腦型彩券總銷售額_億元"],
    df.loc[df["中秋"] == 1, "電腦型彩券總銷售額_億元"],
    df.loc[df["是否節慶月份"] == 0, "電腦型彩券總銷售額_億元"]
]

plt.figure(figsize=(10, 6))
plt.boxplot(box_data, tick_labels=["春節", "端午", "中秋", "一般月份"])
plt.title("三大節慶月份與一般月份電腦型彩券總銷售額(億)分布比較")
plt.xlabel("月份類別")
plt.ylabel("電腦型彩券總銷售額(億)")
plt.tight_layout()
plt.savefig("05_節慶月份與一般月份箱型圖.png", dpi=300)
plt.close()

# =========================
# 7. 高連槓獎金月份 vs 一般月份
# 使用前25%作為高連槓獎金月份
# =========================
threshold = df["電腦型彩券最高連槓獎金_億元"].quantile(0.75)

df["是否高連槓獎金月份"] = (df["電腦型彩券最高連槓獎金_億元"] >= threshold).astype(int)
df["獎金月份類別"] = df["是否高連槓獎金月份"].map({
    0: "一般連槓獎金月份",
    1: "高連槓獎金月份"
})

jackpot_compare = df.groupby("獎金月份類別", as_index=False).agg(
    平均總銷售額_億元=("電腦型彩券總銷售額_億元", "mean"),
    平均最高連槓獎金_億元=("電腦型彩券最高連槓獎金_億元", "mean")
).round(2)
jackpot_compare = jackpot_compare.rename(
    columns={
        "平均總銷售額_億元": "平均總銷售額(億)",
        "平均最高連槓獎金_億元": "平均連槓獎金(億)",
    }
)

print("\n--- 高連槓獎金月份 vs 一般連槓獎金月份 ---")
print(jackpot_compare)

jackpot_compare.to_csv("05_高連槓獎金月份_vs_一般連槓獎金月份.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(8, 6))
plt.bar(jackpot_compare["獎金月份類別"], jackpot_compare["平均總銷售額(億)"])
plt.title("高連槓獎金月份與一般連槓獎金月份平均電腦型彩券總銷售額(億)比較")
plt.xlabel("月份類別")
plt.ylabel("平均電腦型彩券總銷售額(億)")
plt.tight_layout()
plt.savefig("05_高連槓獎金月份_vs_一般連槓獎金月份.png", dpi=300)
plt.close()

print("\n已輸出節慶與獎金比較圖。")
