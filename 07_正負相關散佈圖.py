import pandas as pd
import numpy as np
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

DISPLAY_NAMES = {
    "電腦型彩券總銷售額_億元": "電腦型彩券總銷售額(億)",
    "通膨率": "CPI年增率",
    "失業率年變動_pp": "失業率年變動(pp)",
    "CCI年變動": "CCI年變動",
    "TAIEX年增率": "TAIEX年增率(%)",
    "電腦型彩券最高連槓獎金_億元": "連槓獎金(億)",
}

print("目前欄位：")
print(df.columns.tolist())

# =========================
# 2. 建立分析欄位
# =========================
# Y：電腦型彩券總銷售額(億)
if "電腦型彩券總銷售額" not in df.columns:
    raise ValueError("找不到『電腦型彩券總銷售額』欄位")

df["電腦型彩券總銷售額_億元"] = df["電腦型彩券總銷售額"] / 100000000

# 通膨率：如果沒有，就用 CPI總指數算
if "通膨率" not in df.columns:
    if "CPI總指數" in df.columns:
        df["月份_dt"] = pd.to_datetime(df["月份"], format="%Y-%m", errors="coerce")
        df = df.sort_values("月份_dt").reset_index(drop=True)
        df["通膨率"] = df["CPI總指數"].pct_change(periods=12) * 100
        df["通膨率"] = df["通膨率"].round(2)
        if "月份_dt" in df.columns:
            df = df.drop(columns=["月份_dt"])
    else:
        print("⚠️ 找不到『通膨率』也找不到『CPI總指數』，將略過通膨率圖")

# 連槓獎金欄位
rollover_col = "電腦型彩券最高連槓獎金"
rollover_100m_col = "電腦型彩券最高連槓獎金_億元"

if rollover_col in df.columns:
    df[rollover_100m_col] = df[rollover_col] / 100000000
    print(f"連槓獎金使用欄位：{rollover_col}")
else:
    print("⚠️ 找不到『電腦型彩券最高連槓獎金』欄位，將略過連槓獎金圖")

# =========================
# 3. 畫散佈圖 + 趨勢線函數
# =========================
def scatter_with_trend(data, x_col, y_col, x_label, y_label, title, filename):
    plot_df = data[[x_col, y_col]].dropna().copy()

    if len(plot_df) < 2:
        print(f"資料不足，無法繪製：{title}")
        return None

    x = plot_df[x_col]
    y = plot_df[y_col]

    # Pearson 相關係數
    corr = x.corr(y)

    # 線性趨勢線
    slope, intercept = np.polyfit(x, y, 1)
    trend_y = slope * x + intercept

    # 判斷關係方向
    if corr > 0:
        relation = "正相關"
    elif corr < 0:
        relation = "負相關"
    else:
        relation = "無明顯線性相關"

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.7)
    order = np.argsort(x.to_numpy())
    plt.plot(x.to_numpy()[order], trend_y.to_numpy()[order], linewidth=2)

    plt.title(f"{title}\nPearson r = {corr:.3f}，{relation}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    return {
        "變數": x_col,
        "相關係數": round(corr, 3),
        "關係方向": relation,
        "趨勢線斜率": round(slope, 4),
        "樣本數": len(plot_df)
    }

# =========================
# 4. 設定要畫的變數
# =========================
plot_tasks = []

if "通膨率" in df.columns:
    plot_tasks.append({
        "x_col": "通膨率",
        "x_label": "CPI年增率",
        "title": "CPI年增率與電腦型彩券總銷售額(億)關係",
        "filename": "07_通膨率_vs_彩券總銷售額_散佈圖.png"
    })

if "失業率年變動_pp" in df.columns:
    plot_tasks.append({
        "x_col": "失業率年變動_pp",
        "x_label": "失業率年變動(pp)",
        "title": "失業率年變動與電腦型彩券總銷售額(億)關係",
        "filename": "07_失業率年變動_vs_彩券總銷售額_散佈圖.png"
    })

if "CCI年變動" in df.columns:
    plot_tasks.append({
        "x_col": "CCI年變動",
        "x_label": "CCI年變動",
        "title": "CCI年變動與電腦型彩券總銷售額(億)關係",
        "filename": "07_CCI年變動_vs_彩券總銷售額_散佈圖.png"
    })

if "TAIEX年增率" in df.columns:
    plot_tasks.append({
        "x_col": "TAIEX年增率",
        "x_label": "TAIEX年增率(%)",
        "title": "TAIEX年增率與電腦型彩券總銷售額(億)關係",
        "filename": "07_TAIEX年增率_vs_彩券總銷售額_散佈圖.png"
    })

if rollover_col in df.columns:
    plot_tasks.append({
        "x_col": rollover_100m_col,
        "x_label": "連槓獎金(億)",
        "title": "連槓獎金(億)與電腦型彩券總銷售額(億)關係",
        "filename": "07_電腦型彩券最高連槓獎金_vs_彩券總銷售額_散佈圖.png"
    })

# =========================
# 5. 執行畫圖
# =========================
results = []

for task in plot_tasks:
    result = scatter_with_trend(
        data=df,
        x_col=task["x_col"],
        y_col="電腦型彩券總銷售額_億元",
        x_label=task["x_label"],
        y_label="電腦型彩券總銷售額(億)",
        title=task["title"],
        filename=task["filename"]
    )

    if result is not None:
        results.append(result)

# =========================
# 6. 輸出結果表
# =========================
if len(results) > 0:
    result_df = pd.DataFrame(results)
    result_df["變數"] = result_df["變數"].replace(DISPLAY_NAMES)
    print("\n--- 正負相關判斷表 ---")
    print(result_df)

    result_df.to_csv("07_正負相關判斷表.csv", index=False, encoding="utf-8-sig")
    print("\n已輸出：07_正負相關判斷表.csv")
else:
    print("沒有可繪製的圖。")
