import os

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


df = pd.read_csv("master_dataset_final_2011_2025.csv", encoding="utf-8-sig")
df.columns = df.columns.str.strip()
df = df.dropna(subset=["通膨率", "失業率年變動_pp", "CCI年變動", "TAIEX年增率"]).copy()

df["電腦型彩券總銷售額_億元"] = df["電腦型彩券總銷售額"] / 100000000
df["電腦型彩券最高連槓獎金_億元"] = df["電腦型彩券最高連槓獎金"] / 100000000

threshold = df["電腦型彩券最高連槓獎金_億元"].quantile(0.75)
df["連槓獎金組別"] = df["電腦型彩券最高連槓獎金_億元"].apply(
    lambda value: "高連槓獎金月份" if value >= threshold else "一般連槓獎金月份"
)

box_data = [
    df.loc[df["連槓獎金組別"] == "一般連槓獎金月份", "電腦型彩券總銷售額_億元"],
    df.loc[df["連槓獎金組別"] == "高連槓獎金月份", "電腦型彩券總銷售額_億元"],
]

summary = (
    df.groupby("連槓獎金組別", as_index=False)
    .agg(
        樣本數=("電腦型彩券總銷售額_億元", "count"),
        平均總銷售額_億元=("電腦型彩券總銷售額_億元", "mean"),
        中位數總銷售額_億元=("電腦型彩券總銷售額_億元", "median"),
        平均最高連槓獎金_億元=("電腦型彩券最高連槓獎金_億元", "mean"),
    )
    .round(2)
)
summary = summary.rename(
    columns={
        "平均總銷售額_億元": "平均總銷售額(億)",
        "中位數總銷售額_億元": "中位數總銷售額(億)",
        "平均最高連槓獎金_億元": "平均連槓獎金(億)",
    }
)

plt.figure(figsize=(8, 6))
plt.boxplot(
    box_data,
    tick_labels=["一般連槓獎金月份", "高連槓獎金月份"],
    patch_artist=True,
    boxprops={"facecolor": "#D8E8F6"},
    medianprops={"color": "#B42318", "linewidth": 2},
)
plt.title("高連槓獎金月份與一般月份電腦型彩券總銷售額(億)分布")
plt.xlabel("月份組別")
plt.ylabel("電腦型彩券總銷售額(億)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("09_高連槓獎金月份銷售額箱型圖.png", dpi=300)
plt.close()

summary.to_csv("09_高連槓獎金月份銷售額箱型圖摘要.csv", index=False, encoding="utf-8-sig")

print("--- 高連槓獎金月份箱型圖摘要 ---")
print(summary)
print("已輸出：09_高連槓獎金月份銷售額箱型圖.png")
print("已輸出：09_高連槓獎金月份銷售額箱型圖摘要.csv")
