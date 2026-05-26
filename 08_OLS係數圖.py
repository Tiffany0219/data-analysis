import os

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


coef = pd.read_csv("06_OLS四模型係數表.csv", encoding="utf-8-sig")
coef.columns = coef.columns.str.strip()

main_model = coef[
    (coef["模型"] == "整體電腦型彩券") &
    (coef["變數"] != "const")
].copy()

main_model["95%信賴區間"] = 1.96 * main_model["標準誤"]
main_model["顯著性"] = main_model["p值"].apply(lambda p: "顯著" if p < 0.05 else "不顯著")
main_model["顏色"] = main_model["p值"].apply(lambda p: "#2878B5" if p < 0.05 else "#A7A7A7")

main_model = main_model.sort_values("係數")

fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(
    main_model["變數"],
    main_model["係數"],
    xerr=main_model["95%信賴區間"],
    color=main_model["顏色"],
    alpha=0.9,
    capsize=4,
)
ax.axvline(0, color="black", linewidth=1)

x_min = (main_model["係數"] - main_model["95%信賴區間"]).min()
x_max = (main_model["係數"] + main_model["95%信賴區間"]).max()
x_range = x_max - x_min
ax.set_xlim(x_min - x_range * 0.12, x_max + x_range * 0.16)

for y_pos, (_, row) in enumerate(main_model.iterrows()):
    beta = row["係數"]
    ci = row["95%信賴區間"]
    label = f"β={beta:+.3f}"
    if beta >= 0:
        x_pos = beta + ci + x_range * 0.015
        ha = "left"
    else:
        x_pos = beta - ci - x_range * 0.015
        ha = "right"
    ax.text(x_pos, y_pos, label, va="center", ha=ha, fontsize=9, color="#333333")

ax.set_title("整體電腦型彩券 OLS 係數圖")
ax.set_xlabel("迴歸係數（依變數：電腦型彩券總銷售額(億)）")
ax.set_ylabel("自變數")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig("08_OLS整體模型係數圖.png", dpi=300)
plt.close(fig)

main_model.drop(columns=["顏色"]).to_csv(
    "08_OLS整體模型係數圖資料.csv",
    index=False,
    encoding="utf-8-sig",
)

print("已輸出：08_OLS整體模型係數圖.png")
print("已輸出：08_OLS整體模型係數圖資料.csv")
