import shutil
import warnings
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup
from pypdf import PdfReader


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

YEAR = 2026
TARGET_GAMES = ["大樂透", "威力彩", "今彩539"]

TAIWAN_LOTTERY_API = "https://api.taiwanlottery.com/TLCAPIWeB/Lottery/ResultDownload"
DGBAS_API_BASE = "https://nstatdb.dgbas.gov.tw/dgbasAll/webMain.aspx?sdmx"
NCU_CCI_URL = "https://rcted.ncu.edu.tw/cci.asp"
TWSE_TAIEX_API = "https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST"

DOWNLOAD_DIR = Path("台彩年度下載")
MASTER_PATH = Path("master_dataset_final_2011_2025.csv")

HEADERS = {"User-Agent": "Mozilla/5.0"}


def month_range_2026():
    current = date.today()
    end_month = current.month if current.year == YEAR else 12
    return [f"{YEAR}-{month:02d}" for month in range(1, end_month + 1)]


def to_dgbas_month(month_text):
    month = pd.to_datetime(month_text, format="%Y-%m")
    return f"{month.year}-M{month.month}"


def parse_single_sdmx_value(payload):
    datasets = payload.get("data", {}).get("dataSets", [])
    if not datasets:
        return None

    series = datasets[0].get("series", {})
    for item in series.values():
        observations = item.get("observations", {})
        if observations:
            first_key = sorted(observations.keys(), key=lambda value: int(value))[0]
            return float(observations[first_key][0])
    return None


def fetch_dgbas_series(function_code, dimension, month_text):
    period = to_dgbas_month(month_text)
    url = f"{DGBAS_API_BASE}/{function_code}/{dimension}&startTime={period}&endTime={period}"
    response = requests.get(url, timeout=20, verify=False)
    response.raise_for_status()
    return parse_single_sdmx_value(response.json())


def fetch_cpi_unemployment(months):
    rows = []
    for month_text in months:
        cpi = fetch_dgbas_series("A030101015", "1...M", month_text)
        unemployment = fetch_dgbas_series("A040107010", "12.1..M", month_text)
        if cpi is not None and cpi <= 0:
            cpi = None
        if unemployment is not None and unemployment <= 0:
            unemployment = None
        if cpi is not None or unemployment is not None:
            rows.append(
                {
                    "月份": month_text,
                    "CPI總指數": cpi,
                    "失業率": unemployment,
                }
            )
    return pd.DataFrame(rows)


def roc_report_month_to_ad(text):
    match = pd.Series([text]).str.extract(r"(\d{3})年\s*(\d{1,2})月份").iloc[0]
    if match.isna().any():
        return None
    return f"{int(match[0]) + 1911}-{int(match[1]):02d}"


def extract_domestic_economy_cci(text):
    compact_text = "".join(str(text).split())
    match = pd.Series([compact_text]).str.extract(
        r"未來半年國內經濟景氣[，,」]*本月調查為(\d+(?:\.\d+)?)點"
    ).iloc[0, 0]
    if pd.isna(match):
        return None
    return float(match)


def fetch_cci_2026():
    response = requests.get(NCU_CCI_URL, headers=HEADERS, timeout=20, verify=False)
    response.raise_for_status()
    response.encoding = "utf-8"

    soup = BeautifulSoup(response.text, "html.parser")
    rows = []
    for link in soup.find_all("a"):
        title = link.get_text(strip=True)
        href = link.get("href")
        month_text = roc_report_month_to_ad(title)
        if not href or month_text is None or not month_text.startswith(f"{YEAR}-"):
            continue

        pdf_url = requests.compat.urljoin(NCU_CCI_URL, href)
        pdf_response = requests.get(pdf_url, headers=HEADERS, timeout=30, verify=False)
        pdf_response.raise_for_status()

        reader = PdfReader(BytesIO(pdf_response.content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages[:8])
        cci = extract_domestic_economy_cci(text)
        if cci is not None:
            rows.append({"月份": month_text, "CCI": cci, "CCI來源": pdf_url})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates("月份").sort_values("月份").reset_index(drop=True)


def fetch_taiex_2026(months):
    rows = []
    for month_text in months:
        date_text = month_text.replace("-", "") + "01"
        response = requests.get(
            TWSE_TAIEX_API,
            params={"date": date_text, "response": "json"},
            headers=HEADERS,
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("stat") != "OK" or not payload.get("data"):
            continue

        closes = []
        for row in payload["data"]:
            closes.append(float(str(row[4]).replace(",", "")))

        rows.append(
            {
                "月份": month_text,
                "TAIEX月平均收盤指數": round(sum(closes) / len(closes), 2),
                "TAIEX交易日數": len(closes),
                "TAIEX是否估計": 0,
            }
        )
    return pd.DataFrame(rows)


def download_taiwan_lottery_2026():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        TAIWAN_LOTTERY_API,
        params={"year": YEAR},
        headers=HEADERS,
        timeout=20,
        verify=False,
    )
    response.raise_for_status()
    data = response.json()
    content = data.get("content") or {}
    download_url = content.get("path")
    file_name = content.get("fileName") or str(YEAR)
    if not download_url:
        raise ValueError(f"台彩 {YEAR} 年 API 找不到下載網址：{data}")

    zip_path = DOWNLOAD_DIR / f"{file_name}.zip"
    with requests.get(
        download_url,
        headers=HEADERS,
        timeout=60,
        stream=True,
        verify=False,
    ) as download:
        download.raise_for_status()
        with zip_path.open("wb") as output:
            for chunk in download.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)

    extract_dir = DOWNLOAD_DIR / file_name
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(extract_dir)

    for game in TARGET_GAMES:
        game_dir = Path(game)
        game_dir.mkdir(parents=True, exist_ok=True)
        for csv_path in extract_dir.glob(f"{game}_*.csv"):
            shutil.copy2(csv_path, game_dir / csv_path.name)


def read_lottery_raw_2026():
    frames = []
    for game in TARGET_GAMES:
        path = Path(game) / f"{game}_{YEAR}.csv"
        if not path.exists():
            path = DOWNLOAD_DIR / str(YEAR) / f"{game}_{YEAR}.csv"
        if not path.exists():
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.ParserWarning)
            df = pd.read_csv(path, encoding="utf-8-sig", engine="python", index_col=False)
        df.columns = df.columns.astype(str).str.strip()
        required = ["開獎日期", "銷售總額", "銷售注數", "總獎金"]
        if any(col not in df.columns for col in required):
            continue

        df = df[required].copy()
        df["遊戲名稱"] = game
        df["開獎日期"] = pd.to_datetime(df["開獎日期"], format="%Y/%m/%d", errors="coerce")
        for col in ["銷售總額", "銷售注數", "總獎金"]:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce",
            )
        df = df.dropna(subset=["開獎日期", "銷售總額", "銷售注數", "總獎金"])
        df["月份"] = df["開獎日期"].dt.to_period("M").astype(str)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["遊戲名稱", "開獎日期"])


def build_lottery_monthly_2026(raw):
    if raw.empty:
        return pd.DataFrame()

    parts = []
    for game, game_df in raw.groupby("遊戲名稱"):
        game_df = game_df.sort_values("開獎日期").copy()
        game_df["上一期總獎金"] = game_df["總獎金"].shift(1)
        game_df["獎金重置"] = (game_df["總獎金"] < game_df["上一期總獎金"]).astype(int)
        game_df.loc[game_df.index[0], "獎金重置"] = 1
        game_df["連槓輪次"] = game_df["獎金重置"].cumsum()
        game_df["連槓獎金"] = game_df.groupby("連槓輪次")["總獎金"].cummax()
        parts.append(game_df)

    data = pd.concat(parts, ignore_index=True)
    by_game = data.groupby(["月份", "遊戲名稱"], as_index=False).agg(
        銷售總額=("銷售總額", "sum"),
        銷售注數=("銷售注數", "sum"),
        開獎期數=("銷售總額", "count"),
        單月最高總獎金=("總獎金", "max"),
        最高連槓獎金=("連槓獎金", "max"),
    )

    rows = []
    for month_text, month_df in by_game.groupby("月份"):
        row = {"月份": month_text}
        for game in TARGET_GAMES:
            game_row = month_df[month_df["遊戲名稱"] == game]
            if game_row.empty:
                row[f"{game}銷售總額"] = 0
                row[f"{game}銷售注數"] = 0
                row[f"{game}開獎期數"] = 0
                row[f"{game}單月最高總獎金"] = 0
            else:
                item = game_row.iloc[0]
                row[f"{game}銷售總額"] = item["銷售總額"]
                row[f"{game}銷售注數"] = item["銷售注數"]
                row[f"{game}開獎期數"] = item["開獎期數"]
                row[f"{game}單月最高總獎金"] = item["單月最高總獎金"]
        row["電腦型彩券總銷售額"] = sum(row[f"{game}銷售總額"] for game in TARGET_GAMES)
        row["電腦型彩券總銷售額_億"] = round(row["電腦型彩券總銷售額"] / 100000000, 2)
        row["電腦型彩券最高連槓獎金"] = month_df["最高連槓獎金"].max()
        row["連槓獎金(億)"] = round(row["電腦型彩券最高連槓獎金"] / 100000000, 2)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("月份").reset_index(drop=True)


def add_derived_columns(df):
    if not MASTER_PATH.exists() or df.empty:
        return df

    master = pd.read_csv(MASTER_PATH, encoding="utf-8-sig")
    master.columns = master.columns.str.strip()
    master["月份"] = master["月份"].astype(str).str.strip()

    output = df.copy()
    for idx, row in output.iterrows():
        base_month = (pd.to_datetime(row["月份"]) - pd.DateOffset(years=1)).strftime("%Y-%m")
        base = master[master["月份"] == base_month]
        if base.empty:
            continue
        base = base.iloc[0]
        output.loc[idx, "基準月份"] = base_month
        if pd.notna(row.get("CPI總指數")):
            output.loc[idx, "CPI年增率"] = round(
                (row["CPI總指數"] - base["CPI總指數"]) / base["CPI總指數"] * 100,
                2,
            )
        if pd.notna(row.get("失業率")):
            output.loc[idx, "失業率年變動(pp)"] = round(row["失業率"] - base["失業率"], 2)
        if pd.notna(row.get("CCI")):
            output.loc[idx, "CCI年變動"] = round(row["CCI"] - base["CCI"], 2)
        if pd.notna(row.get("TAIEX月平均收盤指數")):
            output.loc[idx, "TAIEX年增率(%)"] = round(
                (row["TAIEX月平均收盤指數"] - base["TAIEX月平均收盤指數"])
                / base["TAIEX月平均收盤指數"]
                * 100,
                2,
            )
    return output


def main():
    months = month_range_2026()

    print("1. 更新台彩 2026 年度資料...")
    download_taiwan_lottery_2026()
    lottery = build_lottery_monthly_2026(read_lottery_raw_2026())
    lottery.to_csv("lottery_monthly_2026_latest.csv", index=False, encoding="utf-8-sig")

    print("2. 抓取主計總處 CPI / 失業率...")
    cpi_unemp = fetch_cpi_unemployment(months)
    cpi_unemp.to_csv("cpi_unemployment_monthly_2026_latest.csv", index=False, encoding="utf-8-sig")

    print("3. 抓取中央大學 CCI...")
    cci = fetch_cci_2026()
    cci.to_csv("CCI_monthly_2026_latest.csv", index=False, encoding="utf-8-sig")

    print("4. 抓取證交所 TAIEX...")
    taiex = fetch_taiex_2026(months)
    taiex.to_csv("TAIEX_monthly_2026_latest.csv", index=False, encoding="utf-8-sig")

    print("5. 合併 2026 可用資料...")
    merged = pd.DataFrame({"月份": months})
    for item in [lottery, cpi_unemp, cci, taiex]:
        if not item.empty:
            merged = merged.merge(item, on="月份", how="left")

    merged = add_derived_columns(merged)
    merged.to_csv("forecast_inputs_2026_latest.csv", index=False, encoding="utf-8-sig")

    print("\n已輸出：")
    print("- lottery_monthly_2026_latest.csv")
    print("- cpi_unemployment_monthly_2026_latest.csv")
    print("- CCI_monthly_2026_latest.csv")
    print("- TAIEX_monthly_2026_latest.csv")
    print("- forecast_inputs_2026_latest.csv")
    print("\n2026 彙整資料：")
    print(merged)


if __name__ == "__main__":
    main()
