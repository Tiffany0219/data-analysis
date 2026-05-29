import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =========================
# 1. 讀取資料
# =========================
df = pd.read_csv("master_dataset_final_2011_2025.csv", encoding="utf-8-sig")
df.columns = df.columns.str.strip()

DISPLAY_NAMES = {
    "電腦型彩券總銷售額_億元": "電腦型彩券總銷售額(億)",
    "大樂透銷售總額_億元": "大樂透銷售總額(億)",
    "威力彩銷售總額_億元": "威力彩銷售總額(億)",
    "今彩539銷售總額_億元": "今彩539銷售總額(億)",
    "通膨率": "CPI年增率",
    "失業率年變動_pp": "失業率年變動(pp)",
    "CCI年變動": "CCI年變動",
    "TAIEX年增率": "TAIEX年增率(%)",
    "電腦型彩券最高連槓獎金_億元": "連槓獎金(億)",
    "大樂透連槓獎金_億元": "大樂透連槓獎金(億)",
    "威力彩連槓獎金_億元": "威力彩連槓獎金(億)",
    "今彩539連槓獎金_億元": "今彩539連槓獎金(億)",
}

# =========================
# 2. 金額轉億元
# =========================
df["電腦型彩券總銷售額_億元"] = df["電腦型彩券總銷售額"] / 100000000
df["大樂透銷售總額_億元"] = df["大樂透銷售總額"] / 100000000
df["威力彩銷售總額_億元"] = df["威力彩銷售總額"] / 100000000
df["今彩539銷售總額_億元"] = df["今彩539銷售總額"] / 100000000

df["電腦型彩券最高連槓獎金_億元"] = df["電腦型彩券最高連槓獎金"] / 100000000
df["大樂透連槓獎金_億元"] = df["大樂透連槓獎金"] / 100000000
df["威力彩連槓獎金_億元"] = df["威力彩連槓獎金"] / 100000000
df["今彩539連槓獎金_億元"] = df["今彩539連槓獎金"] / 100000000

# =========================
# 3. 確認分析欄位
# =========================
base_x_cols = [
    "通膨率",
    "失業率年變動_pp",
    "CCI年變動",
    "TAIEX年增率",
    "電腦型彩券最高連槓獎金_億元",
    "春節",
    "端午",
    "中秋"
]

check_cols = ["電腦型彩券總銷售額_億元"] + base_x_cols

print("--- 缺失值檢查 ---")
print(df[check_cols].isnull().sum())

df_model = df.dropna(subset=check_cols).copy()

# =========================
# 4. VIF 共線性檢查
# =========================
X_vif = sm.add_constant(df_model[base_x_cols])

vif = pd.DataFrame()
vif["變數"] = X_vif.columns
vif["VIF"] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]

vif = vif.round(3)
vif["變數"] = vif["變數"].replace(DISPLAY_NAMES)

print("\n--- VIF 共線性檢查 ---")
print(vif)

vif.to_csv("06_VIF共線性檢查.csv", index=False, encoding="utf-8-sig")

# =========================
# 5. OLS 函數
# =========================
def run_ols(model_name, y_col, jackpot_col):
    x_cols = [
        "通膨率",
        "失業率年變動_pp",
        "CCI年變動",
        "TAIEX年增率",
        jackpot_col,
        "春節",
        "端午",
        "中秋"
    ]

    data = df.dropna(subset=[y_col] + x_cols).copy()

    X = data[x_cols]
    y = data[y_col].copy()
    y.name = DISPLAY_NAMES.get(y_col, y_col)

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    print(f"\n--- OLS 回歸結果：{model_name} ---")
    print(model.summary())

    coef_table = pd.DataFrame({
        "模型": model_name,
        "變數": model.params.index,
        "係數": model.params.values,
        "標準誤": model.bse.values,
        "t值": model.tvalues.values,
        "p值": model.pvalues.values
    }).round(4)
    coef_table["變數"] = coef_table["變數"].replace(DISPLAY_NAMES)

    summary_table = pd.DataFrame({
        "模型": [model_name],
        "R平方": [model.rsquared],
        "調整後R平方": [model.rsquared_adj],
        "F統計量": [model.fvalue],
        "F檢定p值": [model.f_pvalue],
        "樣本數": [int(model.nobs)]
    }).round(4)

    with open(f"06_OLS_{model_name}.txt", "w", encoding="utf-8") as f:
        f.write(str(model.summary()))

    return coef_table, summary_table


def run_time_split_backtest(model_name, y_col, jackpot_col, train_ratio=0.8):
    x_cols = [
        "通膨率",
        "失業率年變動_pp",
        "CCI年變動",
        "TAIEX年增率",
        jackpot_col,
        "春節",
        "端午",
        "中秋"
    ]

    data = df.dropna(subset=["月份", y_col] + x_cols).copy()
    data["月份_dt"] = pd.to_datetime(data["月份"], format="%Y-%m", errors="coerce")
    data = data.dropna(subset=["月份_dt"]).sort_values("月份_dt").reset_index(drop=True)

    split_index = int(len(data) * train_ratio)
    train = data.iloc[:split_index].copy()
    test = data.iloc[split_index:].copy()

    X_train = sm.add_constant(train[x_cols])
    y_train = train[y_col]
    X_test = sm.add_constant(test[x_cols], has_constant="add")

    model = sm.OLS(y_train, X_train).fit()
    y_pred = model.predict(X_test)
    y_actual = test[y_col].reset_index(drop=True)
    error = y_pred.reset_index(drop=True) - y_actual

    mae = error.abs().mean()
    rmse = (error.pow(2).mean()) ** 0.5
    bias = error.mean()
    mape = (error.abs() / y_actual.replace(0, pd.NA)).mean() * 100
    test_sse = error.pow(2).sum()
    test_sst = ((y_actual - y_actual.mean()) ** 2).sum()
    test_r2 = 1 - test_sse / test_sst if test_sst != 0 else pd.NA

    summary = pd.DataFrame({
        "模型": [model_name],
        "訓練期間": [f"{train['月份'].iloc[0]} 至 {train['月份'].iloc[-1]}"],
        "測試期間": [f"{test['月份'].iloc[0]} 至 {test['月份'].iloc[-1]}"],
        "訓練筆數": [len(train)],
        "測試筆數": [len(test)],
        "訓練R平方": [model.rsquared],
        "測試R平方": [test_r2],
        "MAE_億元": [mae],
        "RMSE_億元": [rmse],
        "平均偏誤_億元": [bias],
        "MAPE_%": [mape],
    }).round(4)

    detail = pd.DataFrame({
        "模型": model_name,
        "月份": test["月份"].values,
        "實際銷售額_億元": y_actual.values,
        "預測銷售額_億元": y_pred.values,
        "誤差_億元": error.values,
        "絕對誤差_億元": error.abs().values,
    }).round(4)

    return summary, detail

# =========================
# 6. 四個模型
# =========================
models = []
backtests = []

models.append(run_ols(
    model_name="整體電腦型彩券",
    y_col="電腦型彩券總銷售額_億元",
    jackpot_col="電腦型彩券最高連槓獎金_億元"
))
backtests.append(run_time_split_backtest(
    model_name="整體電腦型彩券",
    y_col="電腦型彩券總銷售額_億元",
    jackpot_col="電腦型彩券最高連槓獎金_億元"
))

models.append(run_ols(
    model_name="大樂透",
    y_col="大樂透銷售總額_億元",
    jackpot_col="大樂透連槓獎金_億元"
))
backtests.append(run_time_split_backtest(
    model_name="大樂透",
    y_col="大樂透銷售總額_億元",
    jackpot_col="大樂透連槓獎金_億元"
))

models.append(run_ols(
    model_name="威力彩",
    y_col="威力彩銷售總額_億元",
    jackpot_col="威力彩連槓獎金_億元"
))
backtests.append(run_time_split_backtest(
    model_name="威力彩",
    y_col="威力彩銷售總額_億元",
    jackpot_col="威力彩連槓獎金_億元"
))

models.append(run_ols(
    model_name="今彩539",
    y_col="今彩539銷售總額_億元",
    jackpot_col="今彩539連槓獎金_億元"
))
backtests.append(run_time_split_backtest(
    model_name="今彩539",
    y_col="今彩539銷售總額_億元",
    jackpot_col="今彩539連槓獎金_億元"
))

coef_all = pd.concat([m[0] for m in models], ignore_index=True)
summary_all = pd.concat([m[1] for m in models], ignore_index=True)
backtest_summary_all = pd.concat([b[0] for b in backtests], ignore_index=True)
backtest_detail_all = pd.concat([b[1] for b in backtests], ignore_index=True)

coef_all.to_csv("06_OLS四模型係數表.csv", index=False, encoding="utf-8-sig")
summary_all.to_csv("06_OLS四模型摘要表.csv", index=False, encoding="utf-8-sig")
backtest_summary_all.to_csv("06_OLS_80_20時間切分回測摘要表.csv", index=False, encoding="utf-8-sig")
backtest_detail_all.to_csv("06_OLS_80_20時間切分回測明細.csv", index=False, encoding="utf-8-sig")

print("\n已輸出：")
print("06_VIF共線性檢查.csv")
print("06_OLS四模型係數表.csv")
print("06_OLS四模型摘要表.csv")
print("06_OLS_80_20時間切分回測摘要表.csv")
print("06_OLS_80_20時間切分回測明細.csv")
print("06_OLS_整體電腦型彩券.txt")
print("06_OLS_大樂透.txt")
print("06_OLS_威力彩.txt")
print("06_OLS_今彩539.txt")
