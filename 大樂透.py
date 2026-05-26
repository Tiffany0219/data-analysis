import glob
import os
from pathlib import Path

import pandas as pd


GAME_NAME = "大樂透"
SOURCE_DIR = Path(GAME_NAME)
OUTPUT_DIR = Path("分析結果")
NUMBER_COLUMNS = ["獎號1", "獎號2", "獎號3", "獎號4", "獎號5", "獎號6"]
MONEY_COLUMNS = ["銷售總額", "銷售注數", "總獎金"]


def load_lottery_data(folder_path: Path) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not files:
        raise FileNotFoundError(f"找不到 CSV 檔案：{folder_path}")

    required_columns = [
        "遊戲名稱",
        "期別",
        "開獎日期",
        *MONEY_COLUMNS,
        *NUMBER_COLUMNS,
        "特別號",
    ]
    all_data = []
    for file in files:
        print(f"正在讀取：{file}")
        df = pd.read_csv(
            file,
            sep=",",
            engine="python",
            index_col=False,
            encoding="utf-8-sig",
            usecols=lambda col: col.strip() in required_columns,
        )

        df.columns = df.columns.str.strip()
        df = df[required_columns].copy()

        df["開獎日期"] = pd.to_datetime(
            df["開獎日期"].astype(str).str.strip(),
            format="%Y/%m/%d",
            errors="coerce",
        )

        for col in MONEY_COLUMNS + NUMBER_COLUMNS + ["特別號"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["開獎日期"])
        all_data.append(df)

    df_all = pd.concat(all_data, ignore_index=True).sort_values("開獎日期")
    df_all = df_all[
        (df_all["開獎日期"] >= "2010-01-01") &
        (df_all["開獎日期"] <= "2025-12-31")
    ].copy()
    return df_all


def summarize_by_month(df: pd.DataFrame) -> pd.DataFrame:
    df_all = df.assign(月份=df["開獎日期"].dt.to_period("M").astype(str))
    monthly = (
        df_all.groupby("月份", as_index=False)
        .agg(
            大樂透開獎期數=("期別", "count"),
            大樂透銷售總額=("銷售總額", "sum"),
            大樂透銷售注數=("銷售注數", "sum"),
            大樂透單月最高總獎金=("總獎金", "max"),
        )
    )
    monthly["大樂透平均每期銷售額"] = (
        monthly["大樂透銷售總額"] / monthly["大樂透開獎期數"]
    )
    return monthly


def summarize_by_year(df: pd.DataFrame) -> pd.DataFrame:
    yearly = (
        df.assign(年度=df["開獎日期"].dt.year)
        .groupby("年度", as_index=False)
        .agg(
            開獎次數=("期別", "count"),
            銷售總額=("銷售總額", "sum"),
            銷售注數=("銷售注數", "sum"),
            總獎金=("總獎金", "sum"),
            平均每期銷售額=("銷售總額", "mean"),
            平均每期總獎金=("總獎金", "mean"),
        )
    )
    yearly["獎金銷售比"] = yearly["總獎金"] / yearly["銷售總額"]
    return yearly


def count_numbers(df: pd.DataFrame, columns: list[str], output_name: str) -> pd.DataFrame:
    numbers = (
        df[columns]
        .melt(value_name="號碼")
        .dropna(subset=["號碼"])
        .assign(號碼=lambda data: data["號碼"].astype(int))
    )
    frequency = (
        numbers.groupby("號碼", as_index=False)
        .size()
        .rename(columns={"size": "出現次數"})
        .sort_values(["出現次數", "號碼"], ascending=[False, True])
    )
    frequency["排名"] = frequency["出現次數"].rank(method="min", ascending=False).astype(int)
    frequency["類型"] = output_name
    return frequency[["類型", "排名", "號碼", "出現次數"]]


def find_extreme_draws(df: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "銷售總額最高": "銷售總額",
        "銷售注數最高": "銷售注數",
        "總獎金最高": "總獎金",
    }
    rows = []
    for label, column in metrics.items():
        row = df.loc[df[column].idxmax()]
        rows.append(
            {
                "項目": label,
                "期別": row["期別"],
                "開獎日期": row["開獎日期"].date().isoformat(),
                column: int(row[column]),
                "獎號": " ".join(f"{int(row[col]):02d}" for col in NUMBER_COLUMNS),
                "特別號": f"{int(row['特別號']):02d}",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_lottery_data(SOURCE_DIR)
    monthly = summarize_by_month(df)
    yearly = summarize_by_year(df)
    main_number_frequency = count_numbers(df, NUMBER_COLUMNS, "一般獎號")
    special_number_frequency = count_numbers(df, ["特別號"], "特別號")
    extremes = find_extreme_draws(df)

    all_draws_path = OUTPUT_DIR / "大樂透_全部開獎資料_2011_2025.csv"
    monthly_path = Path("大樂透_monthly_2011_2025.csv")
    yearly_path = OUTPUT_DIR / "大樂透_年度彙總_2011_2025.csv"
    frequency_path = OUTPUT_DIR / "大樂透_號碼出現頻率_2011_2025.csv"
    extremes_path = OUTPUT_DIR / "大樂透_最高紀錄_2011_2025.csv"

    df.to_csv(all_draws_path, index=False, encoding="utf-8-sig")
    monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    yearly.to_csv(yearly_path, index=False, encoding="utf-8-sig")
    pd.concat([main_number_frequency, special_number_frequency], ignore_index=True).to_csv(
        frequency_path,
        index=False,
        encoding="utf-8-sig",
    )
    extremes.to_csv(extremes_path, index=False, encoding="utf-8-sig")

    print("\n分析完成")
    print(f"資料期間：{df['開獎日期'].min().date()} 至 {df['開獎日期'].max().date()}")
    print(f"總期數：{len(df):,}")
    print(f"銷售總額：{df['銷售總額'].sum():,.0f}")
    print(f"總獎金：{df['總獎金'].sum():,.0f}")

    print("\n一般獎號出現次數 Top 10")
    print(main_number_frequency.head(10).to_string(index=False))

    print("\n特別號出現次數 Top 10")
    print(special_number_frequency.head(10).to_string(index=False))

    print("\n已輸出：")
    for path in [all_draws_path, monthly_path, yearly_path, frequency_path, extremes_path]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
