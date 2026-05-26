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

df["月份_dt"] = pd.to_datetime(df["月份"], format="%Y-%m", errors="coerce")
df = df[
    (df["月份_dt"] >= pd.to_datetime("2011-01")) &
    (df["月份_dt"] <= pd.to_datetime("2025-12"))
].copy()
df["年度"] = df["月份_dt"].dt.year
df["月份序號"] = df["月份_dt"].dt.month
df["電腦型彩券總銷售額_億元"] = df["電腦型彩券總銷售額"] / 100000000

heatmap_data = df.pivot(
    index="年度",
    columns="月份序號",
    values="電腦型彩券總銷售額_億元",
).sort_index()

month_labels = [f"{month}月" for month in range(1, 13)]

fig, ax = plt.subplots(figsize=(12, 7))
image = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd")

ax.set_title("2011-2025 電腦型彩券總銷售額(億)季節性熱力圖")
ax.set_xlabel("月份")
ax.set_ylabel("年度")
ax.set_xticks(range(12))
ax.set_xticklabels(month_labels)
ax.set_yticks(range(len(heatmap_data.index)))
ax.set_yticklabels(heatmap_data.index)

for i, year in enumerate(heatmap_data.index):
    for j, month in enumerate(heatmap_data.columns):
        value = heatmap_data.loc[year, month]
        if pd.notna(value):
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=7)

colorbar = fig.colorbar(image, ax=ax)
colorbar.set_label("電腦型彩券總銷售額(億)")

fig.tight_layout()
plt.savefig("10_電腦型彩券月銷售額季節性熱力圖.png", dpi=300)
plt.close(fig)

heatmap_data.to_csv("10_電腦型彩券月銷售額季節性熱力圖資料.csv", encoding="utf-8-sig")

print("已輸出：10_電腦型彩券月銷售額季節性熱力圖.png")
print("已輸出：10_電腦型彩券月銷售額季節性熱力圖資料.csv")
