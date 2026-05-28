import pandas as pd
import re
import requests
import streamlit as st
import urllib3
import altair as alt
from bs4 import BeautifulSoup
from io import BytesIO
from pathlib import Path
from pypdf import PdfReader
import warnings


MASTER_PATH = "master_dataset_final_2011_2025.csv"
COEF_PATH = "06_OLS四模型係數表.csv"
FORECAST_2026_PATH = Path("forecast_inputs_2026_latest.csv")

MODEL_NAME = "整體電腦型彩券"
MODEL_NAMES = ["整體電腦型彩券", "大樂透", "威力彩", "今彩539"]
Y_LABEL = "電腦型彩券總銷售額"

COEF_ALIASES = {
    "const": ["const", "Intercept", "截距"],
    "cpi_yoy": ["CPI年增率", "通膨率"],
    "unemp_change": ["失業率年變動(pp)", "失業率年變動", "失業率"],
    "cci_change": ["CCI年變動", "CCI"],
    "taiex_yoy": ["TAIEX年增率(%)", "TAIEX年增率", "TAIEX月平均收盤指數"],
    "jackpot": [
        "連槓獎金(億)",
        "電腦型彩券最高連槓獎金_億元",
        "大樂透連槓獎金(億)",
        "威力彩連槓獎金(億)",
        "今彩539連槓獎金(億)",
    ],
    "spring": ["春節"],
    "dragon": ["端午"],
    "mid_autumn": ["中秋"],
}

LOTTERY_GAMES = ["大樂透", "威力彩", "今彩539"]
DGBAS_API_BASE = "https://nstatdb.dgbas.gov.tw/dgbasAll/webMain.aspx?sdmx"
NCU_CCI_URL = "https://rcted.ncu.edu.tw/cci.asp"

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@st.cache_data
def load_master():
    df = pd.read_csv(MASTER_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["月份"] = normalize_month_series(df["月份"])
    return df


def load_coefficients_for_model(model_name):
    coef = pd.read_csv(COEF_PATH, encoding="utf-8-sig")
    coef.columns = coef.columns.str.strip()
    coef = coef[coef["模型"] == model_name].copy()

    values = {}
    for key, aliases in COEF_ALIASES.items():
        match = coef.loc[coef["變數"].isin(aliases), "係數"]
        if match.empty:
            raise ValueError(f"{model_name} 係數表找不到變數：{' / '.join(aliases)}")
        values[key] = float(match.iloc[0])
    return values


def load_coefficients():
    return load_coefficients_for_model(MODEL_NAME)


@st.cache_data
def load_forecast_2026():
    if not FORECAST_2026_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(FORECAST_2026_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["月份"] = normalize_month_series(df["月份"])
    return df


def normalize_month_series(series):
    parsed = pd.to_datetime(series.astype(str).str.strip(), errors="coerce")
    return parsed.dt.to_period("M").astype(str)


def month_to_datetime(month_text):
    return pd.to_datetime(str(month_text).strip(), format="%Y-%m", errors="raise")


def previous_year_month(month_text):
    month = month_to_datetime(month_text)
    return (month - pd.DateOffset(years=1)).strftime("%Y-%m")


def find_base_row(df, month_text):
    base_month = previous_year_month(month_text)
    data = df.copy()
    data["月份_dt"] = pd.to_datetime(data["月份"], format="%Y-%m", errors="coerce")
    data = data.dropna(subset=["月份_dt"]).sort_values("月份_dt").reset_index(drop=True)

    matched = data[data["月份"] == base_month]
    if not matched.empty:
        row = matched.iloc[0].drop(labels=["月份_dt"])
        return base_month, row

    target_dt = month_to_datetime(base_month)
    earlier = data[data["月份_dt"] <= target_dt]
    if not earlier.empty:
        row = earlier.iloc[-1].drop(labels=["月份_dt"])
        return row["月份"], row

    if len(data) > 12:
        row = data.iloc[-13].drop(labels=["月份_dt"])
        return row["月份"], row

    if not data.empty:
        row = data.iloc[0].drop(labels=["月份_dt"])
        return row["月份"], row

    return base_month, None


def find_current_row(df, month_text):
    matched = df[df["月份"] == month_text]
    if matched.empty:
        return None
    return matched.iloc[0]


def find_forecast_row(df, month_text):
    if df.empty:
        return None
    matched = df[df["月份"] == month_text]
    if matched.empty:
        return None
    return matched.iloc[0]


def get_festival_flags(selected_month_str):
    spring_months = {
        "2010-02",
        "2011-02", "2012-01", "2013-02", "2014-01", "2015-02",
        "2016-02", "2017-01", "2018-02", "2019-02", "2020-01",
        "2021-02", "2022-02", "2023-01", "2024-02", "2025-01",
        "2026-02",
    }
    dragon_boat_months = {
        "2010-06",
        "2011-06", "2012-06", "2013-06", "2014-06", "2015-06",
        "2016-06", "2017-05", "2018-06", "2019-06", "2020-06",
        "2021-06", "2022-06", "2023-06", "2024-06", "2025-05",
        "2026-06",
    }
    mid_autumn_months = {
        "2010-09",
        "2011-09", "2012-09", "2013-09", "2014-09", "2015-09",
        "2016-09", "2017-10", "2018-09", "2019-09", "2020-10",
        "2021-09", "2022-09", "2023-09", "2024-09", "2025-10",
        "2026-09",
    }

    return {
        "spring": selected_month_str in spring_months,
        "dragon": selected_month_str in dragon_boat_months,
        "mid_autumn": selected_month_str in mid_autumn_months,
    }


def build_month_options(df, future_months=12):
    historical = df["月份"].sort_values().tolist()
    latest = pd.to_datetime(historical[-1], format="%Y-%m")
    future = [
        (latest + pd.DateOffset(months=i)).strftime("%Y-%m")
        for i in range(1, future_months + 1)
    ]
    return historical + future


@st.cache_data
def load_lottery_raw_data():
    frames = []

    for game in LOTTERY_GAMES:
        candidates = sorted(Path(game).glob(f"{game}_*.csv"))
        candidates += sorted(Path("台彩年度下載").glob(f"*/{game}_*.csv"))

        seen = set()
        for csv_path in candidates:
            if csv_path.name in seen:
                continue
            seen.add(csv_path.name)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", pd.errors.ParserWarning)
                    raw = pd.read_csv(
                        csv_path,
                        encoding="utf-8-sig",
                        engine="python",
                        index_col=False,
                    )
            except Exception:
                continue

            raw.columns = raw.columns.astype(str).str.strip()
            required = ["開獎日期", "銷售總額", "銷售注數", "總獎金"]
            if any(col not in raw.columns for col in required):
                continue

            raw = raw[required].copy()
            raw["遊戲名稱"] = game
            raw["開獎日期"] = pd.to_datetime(
                raw["開獎日期"].astype(str).str.strip(),
                format="%Y/%m/%d",
                errors="coerce",
            )

            for col in ["銷售總額", "銷售注數", "總獎金"]:
                raw[col] = (
                    raw[col]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .str.strip()
                )
                raw[col] = pd.to_numeric(raw[col], errors="coerce")

            raw = raw.dropna(subset=["開獎日期", "銷售總額", "總獎金"])
            frames.append(raw)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["遊戲名稱", "開獎日期"]).reset_index(drop=True)
    data["月份"] = data["開獎日期"].dt.to_period("M").astype(str)
    return data


def lottery_month_snapshot(raw_data, month_text):
    if raw_data.empty:
        return None

    data = raw_data[raw_data["月份"] <= month_text].copy()
    if data.empty or month_text not in set(data["月份"]):
        return None

    rollover_parts = []
    for game, game_df in data.groupby("遊戲名稱"):
        game_df = game_df.sort_values("開獎日期").copy()
        game_df["上一期總獎金"] = game_df["總獎金"].shift(1)
        game_df["獎金重置"] = (game_df["總獎金"] < game_df["上一期總獎金"]).astype(int)
        if len(game_df) > 0:
            game_df.loc[game_df.index[0], "獎金重置"] = 1
        game_df["連槓輪次"] = game_df["獎金重置"].cumsum()
        game_df["連槓獎金"] = game_df.groupby("連槓輪次")["總獎金"].cummax()
        rollover_parts.append(game_df)

    data = pd.concat(rollover_parts, ignore_index=True)
    selected = data[data["月份"] == month_text].copy()
    if selected.empty:
        return None

    by_game = selected.groupby("遊戲名稱", as_index=False).agg(
        銷售總額=("銷售總額", "sum"),
        開獎期數=("銷售總額", "count"),
        最高連槓獎金=("連槓獎金", "max"),
    )

    total_sales = by_game["銷售總額"].sum()
    max_jackpot = by_game["最高連槓獎金"].max()

    return {
        "month": month_text,
        "total_sales": float(total_sales),
        "total_sales_100m": round(float(total_sales) / 100000000, 2),
        "jackpot_100m": round(float(max_jackpot) / 100000000, 2),
        "by_game": by_game,
    }


def to_dgbas_month(month_text):
    month = pd.to_datetime(month_text, format="%Y-%m", errors="raise")
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
            value = observations[first_key][0]
            return float(value)
    return None


def fetch_dgbas_series(function_code, dimension, month_text):
    period = to_dgbas_month(month_text)
    url = f"{DGBAS_API_BASE}/{function_code}/{dimension}&startTime={period}&endTime={period}"
    response = requests.get(url, timeout=20, verify=False)
    response.raise_for_status()
    payload = response.json()
    return parse_single_sdmx_value(payload), url


def fetch_dgbas_cpi_unemployment(month_text):
    cpi, cpi_url = fetch_dgbas_series("A030101015", "1...M", month_text)
    unemp, unemp_url = fetch_dgbas_series("A040107010", "12.1..M", month_text)

    if cpi is None and unemp is None:
        raise ValueError(f"主計總處尚未提供 {month_text} 的 CPI 與失業率資料")

    return {
        "cpi": cpi,
        "unemp": unemp,
        "cpi_url": cpi_url,
        "unemp_url": unemp_url,
    }


def roc_report_month_to_ad(month_text):
    match = re.search(r"(\d{3})年\s*(\d{1,2})月份", month_text)
    if not match:
        return None
    year = int(match.group(1)) + 1911
    month = int(match.group(2))
    return f"{year}-{month:02d}"


def extract_domestic_economy_cci(text):
    compact_text = "".join(str(text).split())
    pattern = r"未來半年國內經濟景氣[，,」]*本月調查為(\d+(?:\.\d+)?)點"
    match = re.search(pattern, compact_text)
    if not match:
        return None
    return float(match.group(1))


def fetch_ncu_cci_reports():
    response = requests.get(NCU_CCI_URL, timeout=20, verify=False)
    response.raise_for_status()
    response.encoding = "utf-8"

    soup = BeautifulSoup(response.text, "html.parser")
    reports = []
    for link in soup.find_all("a"):
        title = link.get_text(strip=True)
        href = link.get("href")
        if href and "消費者信心指數調查報告" in title:
            reports.append((title, requests.compat.urljoin(NCU_CCI_URL, href)))

    if not reports:
        raise ValueError("中央大學 CCI 頁面找不到調查報告連結")

    values = []
    for title, pdf_url in reports[:15]:
        pdf_response = requests.get(pdf_url, timeout=30, verify=False)
        pdf_response.raise_for_status()
        reader = PdfReader(BytesIO(pdf_response.content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages[:8])
        cci_value = extract_domestic_economy_cci(text)
        report_month = roc_report_month_to_ad(title)
        if report_month is not None and cci_value is not None:
            values.append(
                {
                    "month": report_month,
                    "cci": cci_value,
                    "title": title,
                    "url": pdf_url,
                }
            )

    if not values:
        raise ValueError("已找到 CCI 報告，但無法解析「未來半年國內經濟景氣」數值")

    return values


def add_ncu_cci_change(value, all_values=None):
    all_values = all_values or []
    base_month = previous_year_month(value["month"])
    base_match = [item for item in all_values if item["month"] == base_month]
    value["base_month"] = base_month
    value["base_cci"] = base_match[0]["cci"] if base_match else None
    value["cci_change"] = (
        round(value["cci"] - value["base_cci"], 2)
        if value["base_cci"] is not None
        else None
    )
    return value


def fetch_ncu_cci_latest():
    values = fetch_ncu_cci_reports()
    return add_ncu_cci_change(values[0], values)


def fetch_ncu_cci_for_month(month_text):
    values = fetch_ncu_cci_reports()
    matched = [item for item in values if item["month"] == month_text]
    if not matched:
        latest_month = values[0]["month"]
        raise ValueError(f"中央大學目前找不到 {month_text} 的 CCI 報告，最新為 {latest_month}")
    return add_ncu_cci_change(matched[0], values)


def set_ncu_cci_values(values):
    st.session_state["cci"] = values["cci"]
    st.session_state["ncu_cci_month"] = values["month"]
    st.session_state["ncu_cci_url"] = values["url"]
    st.session_state["ncu_cci_change"] = values.get("cci_change")
    st.session_state["ncu_cci_base_month"] = values.get("base_month")
    st.session_state["ncu_cci_base_value"] = values.get("base_cci")


def set_ncu_cci_from_forecast_row(row):
    st.session_state["cci"] = float(row["CCI"])
    st.session_state["ncu_cci_month"] = row["月份"]
    st.session_state["ncu_cci_url"] = row.get("CCI來源", "")
    st.session_state["ncu_cci_change"] = (
        float(row["CCI年變動"])
        if "CCI年變動" in row and pd.notna(row["CCI年變動"])
        else None
    )
    st.session_state["ncu_cci_base_month"] = row.get("基準月份")
    st.session_state["ncu_cci_base_value"] = None


def calculate_features(current, base):
    cpi_yoy = (current["cpi"] - base["CPI總指數"]) / base["CPI總指數"] * 100
    unemp_change = current["unemp"] - base["失業率"]
    cci_change = current["cci"] - base["CCI"]
    taiex_yoy = (
        (current["taiex"] - base["TAIEX月平均收盤指數"])
        / base["TAIEX月平均收盤指數"]
        * 100
    )

    return {
        "cpi_yoy": round(cpi_yoy, 2),
        "unemp_change": round(unemp_change, 2),
        "cci_change": round(cci_change, 2),
        "taiex_yoy": round(taiex_yoy, 2),
        "jackpot": current["jackpot"],
        "spring": int(current["spring"]),
        "dragon": int(current["dragon"]),
        "mid_autumn": int(current["mid_autumn"]),
    }


def predict_sales(features, beta):
    return (
        beta["const"]
        + beta["cpi_yoy"] * features["cpi_yoy"]
        + beta["unemp_change"] * features["unemp_change"]
        + beta["cci_change"] * features["cci_change"]
        + beta["taiex_yoy"] * features["taiex_yoy"]
        + beta["jackpot"] * features["jackpot"]
        + beta["spring"] * features["spring"]
        + beta["dragon"] * features["dragon"]
        + beta["mid_autumn"] * features["mid_autumn"]
    )


def classify_heat_level(master_df, prediction):
    if Y_LABEL not in master_df.columns:
        return "無法判斷"

    historical_sales = pd.to_numeric(master_df[Y_LABEL], errors="coerce").dropna() / 100000000
    if historical_sales.empty:
        return "無法判斷"

    q25, q50, q75 = historical_sales.quantile([0.25, 0.50, 0.75])
    if prediction < q25:
        return "一般買氣"
    if prediction < q50:
        return "中高買氣"
    if prediction < q75:
        return "高買氣"
    return "極高買氣"


def diff_from_historical_average(master_df, prediction):
    if Y_LABEL not in master_df.columns:
        return None

    historical_sales = pd.to_numeric(master_df[Y_LABEL], errors="coerce").dropna() / 100000000
    if historical_sales.empty:
        return None

    return prediction - float(historical_sales.mean())


def build_contribution_table(features, beta):
    contribution = pd.DataFrame(
        [
            ["CPI年增率", beta["cpi_yoy"] * features["cpi_yoy"]],
            ["失業率年變動", beta["unemp_change"] * features["unemp_change"]],
            ["CCI年變動", beta["cci_change"] * features["cci_change"]],
            ["TAIEX年增率", beta["taiex_yoy"] * features["taiex_yoy"]],
            ["連槓獎金", beta["jackpot"] * features["jackpot"]],
            ["春節", beta["spring"] * features["spring"]],
            ["端午", beta["dragon"] * features["dragon"]],
            ["中秋", beta["mid_autumn"] * features["mid_autumn"]],
        ],
        columns=["因素", "貢獻值_億元"],
    )
    contribution["影響方向"] = contribution["貢獻值_億元"].apply(contribution_direction)
    contribution["絕對影響"] = contribution["貢獻值_億元"].abs()
    return contribution.sort_values("絕對影響", ascending=False).reset_index(drop=True)


def metric_card(label, value, note=None, tone="neutral"):
    note_html = f'<div class="metric-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="metric-card metric-card-{tone}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def heat_level_tone(heat_level):
    if heat_level == "極高買氣":
        return "hot"
    if heat_level == "高買氣":
        return "warm"
    if heat_level == "中高買氣":
        return "mild"
    return "neutral"


def contribution_direction(value):
    if abs(value) < 0.005:
        return "無影響"
    if value > 0:
        return "推升"
    return "拉低"


def build_interpretation(prediction, features, jackpot_source=None):
    festival_names = []
    if features["spring"]:
        festival_names.append("春節")
    if features["dragon"]:
        festival_names.append("端午")
    if features["mid_autumn"]:
        festival_names.append("中秋")

    festival_text = "、".join(festival_names) if festival_names else "非主要節慶月份"
    jackpot_text = (
        "連槓獎金偏高，對買氣有正向支撐"
        if features["jackpot"] >= 10
        else "連槓獎金未達高額門檻，買氣主要受一般月份與總體變數影響"
    )
    if jackpot_source == "歷史實際資料":
        jackpot_source_text = "連槓獎金使用該月份歷史實際資料，"
    elif jackpot_source:
        jackpot_source_text = f"連槓獎金採用「{jackpot_source}」作為情境輸入，"
    else:
        jackpot_source_text = ""
    return (
        f"本月預測電腦型彩券總銷售額約為 {prediction:.2f} 億元。"
        f"節慶判斷為「{festival_text}」，{jackpot_text}。"
        f"{jackpot_source_text}"
        "此結果為 OLS 模型估計值，適合用來比較不同月份或不同連槓獎金情境下的買氣變化，"
        "不代表中獎機率或購買建議。"
    )


def build_month_sales_chart_data(master_df, selected_month_str, prediction):
    selected_dt = month_to_datetime(selected_month_str)
    target_month = selected_dt.month

    data = master_df.copy()
    data["月份_dt"] = pd.to_datetime(data["月份"], format="%Y-%m", errors="coerce")
    data = data.dropna(subset=["月份_dt"])
    data = data[data["月份_dt"].dt.month == target_month].copy()

    if data.empty or "電腦型彩券總銷售額" not in data.columns:
        return pd.DataFrame()

    chart = pd.DataFrame(
        {
            "月份": data["月份"],
            "歷史實際銷售額": data["電腦型彩券總銷售額"] / 100000000,
            "本次預測銷售額": pd.NA,
        }
    )

    prediction_row = pd.DataFrame(
        [
            {
                "月份": selected_month_str,
                "歷史實際銷售額": pd.NA,
                "本次預測銷售額": prediction,
            }
        ]
    )
    chart = pd.concat([chart, prediction_row], ignore_index=True)
    chart["月份_dt"] = pd.to_datetime(chart["月份"], format="%Y-%m", errors="coerce")
    chart = chart.sort_values("月份_dt").drop(columns=["月份_dt"])
    return chart


def render_prediction_analysis(master_df, selected_month_str, prediction, features, beta, jackpot_source=None):
    st.markdown('<div class="section-heading">預測結果分析</div>', unsafe_allow_html=True)
    analysis_left, analysis_right = st.columns([1.25, 1], gap="large")

    with analysis_left:
        st.write("模型解讀")
        st.write(build_interpretation(prediction, features, jackpot_source))

        st.write("同月份歷史銷售額與本次預測")
        chart_data = build_month_sales_chart_data(master_df, selected_month_str, prediction)
        if chart_data.empty:
            st.info("目前沒有足夠的同月份歷史資料可以繪圖。")
        else:
            history = chart_data.dropna(subset=["歷史實際銷售額"]).copy()
            prediction_point = chart_data.dropna(subset=["本次預測銷售額"]).copy()

            history_chart = (
                alt.Chart(history)
                .mark_line(point=True, color="#4C78A8")
                .encode(
                    x=alt.X("月份:N", title="月份"),
                    y=alt.Y("歷史實際銷售額:Q", title="銷售額（億元）"),
                    tooltip=[
                        alt.Tooltip("月份:N", title="月份"),
                        alt.Tooltip("歷史實際銷售額:Q", title="歷史實際銷售額", format=".2f"),
                    ],
                )
            )

            prediction_chart = (
                alt.Chart(prediction_point)
                .mark_point(size=180, color="#E45756", filled=True)
                .encode(
                    x=alt.X("月份:N", title="月份"),
                    y=alt.Y("本次預測銷售額:Q", title="銷售額（億元）"),
                    tooltip=[
                        alt.Tooltip("月份:N", title="預測月份"),
                        alt.Tooltip("本次預測銷售額:Q", title="本次預測銷售額", format=".2f"),
                    ],
                )
            )

            prediction_label = (
                alt.Chart(prediction_point)
                .mark_text(align="left", dx=8, dy=-8, color="#E45756")
                .encode(
                    x="月份:N",
                    y="本次預測銷售額:Q",
                    text=alt.Text("本次預測銷售額:Q", format=".2f"),
                )
            )

            st.altair_chart(
                (history_chart + prediction_chart + prediction_label).properties(height=340),
                width="stretch",
            )

    with analysis_right:
        st.write("主要影響因素排行")
        st.caption(
            "貢獻值 = OLS 係數 × 本次帶入的變數值；正值代表推升預測銷售額，負值代表拉低。"
            "此表不含模型截距，主要用來比較各因素相對影響。"
        )
        contribution = build_contribution_table(features, beta)
        visible_contribution = contribution[contribution["絕對影響"] >= 0.005].copy()
        display_contribution = visible_contribution[["因素", "影響方向", "貢獻值_億元"]].copy()
        display_contribution["貢獻值_億元"] = display_contribution["貢獻值_億元"].map(lambda value: f"{value:+.2f}")
        st.table(display_contribution.style.hide(axis="index"))

        contribution_chart = (
            alt.Chart(visible_contribution)
            .mark_bar()
            .encode(
                x=alt.X("貢獻值_億元:Q", title="貢獻值（億元）"),
                y=alt.Y("因素:N", sort=visible_contribution["因素"].tolist(), title="因素"),
                color=alt.Color(
                    "影響方向:N",
                    scale=alt.Scale(
                        domain=["推升", "拉低", "無影響"],
                        range=["#E45756", "#4C78A8", "#BAB0AC"],
                    ),
                    title="影響方向",
                ),
                tooltip=[
                    alt.Tooltip("因素:N", title="因素"),
                    alt.Tooltip("影響方向:N", title="影響方向"),
                    alt.Tooltip("貢獻值_億元:Q", title="貢獻值", format=".2f"),
                ],
            )
            .properties(height=340)
        )
        st.altair_chart(contribution_chart, width="stretch")


def default_latest_values(df):
    latest = df.sort_values("月份").iloc[-1]
    return values_from_row(latest)


def values_from_row(row):
    return {
        "month": row["月份"],
        "cpi": float(row["CPI總指數"]),
        "unemp": float(row["失業率"]),
        "cci": float(row["CCI"]),
        "taiex": float(row["TAIEX月平均收盤指數"]),
        "jackpot": round(float(row["電腦型彩券最高連槓獎金"]) / 100000000, 2),
        "spring": bool(row["春節"]),
        "dragon": bool(row["端午"]),
        "mid_autumn": bool(row["中秋"]),
    }


def set_defaults(values, include_month=True):
    for key, value in values.items():
        if key == "month" and not include_month:
            continue
        st.session_state[key] = value


def set_lottery_snapshot(snapshot, include_actual=True):
    st.session_state["jackpot"] = snapshot["jackpot_100m"]
    if include_actual:
        st.session_state["actual_sales_100m"] = snapshot["total_sales_100m"]
        st.session_state["actual_sales_month"] = snapshot["month"]


def actual_sales_for_month(forecast_row, lottery_snapshot):
    if (
        forecast_row is not None
        and "電腦型彩券總銷售額_億" in forecast_row
        and pd.notna(forecast_row["電腦型彩券總銷售額_億"])
        and float(forecast_row["電腦型彩券總銷售額_億"]) > 0
    ):
        return float(forecast_row["電腦型彩券總銷售額_億"])

    if lottery_snapshot is not None:
        return float(lottery_snapshot["total_sales_100m"])

    return None


def actual_sales_by_game_for_month(forecast_row, lottery_snapshot):
    values = {}

    if lottery_snapshot is not None and "by_game" in lottery_snapshot:
        by_game = lottery_snapshot["by_game"]
        for game in LOTTERY_GAMES:
            matched = by_game[by_game["遊戲名稱"] == game]
            if not matched.empty:
                values[game] = round(float(matched.iloc[0]["銷售總額"]) / 100000000, 2)

    if forecast_row is not None:
        for game in LOTTERY_GAMES:
            column = f"{game}銷售總額"
            if column in forecast_row and pd.notna(forecast_row[column]):
                value = float(forecast_row[column])
                if value > 0:
                    values[game] = round(value / 100000000, 2)

    return values


def game_jackpots_for_month(master_df, selected_month_str, lottery_snapshot):
    values = {}

    if lottery_snapshot is not None and "by_game" in lottery_snapshot:
        by_game = lottery_snapshot["by_game"]
        for game in LOTTERY_GAMES:
            matched = by_game[by_game["遊戲名稱"] == game]
            if not matched.empty:
                values[game] = round(float(matched.iloc[0]["最高連槓獎金"]) / 100000000, 2)

    current_row = find_current_row(master_df, selected_month_str)
    if current_row is not None:
        for game in LOTTERY_GAMES:
            column = f"{game}連槓獎金"
            if column in current_row and pd.notna(current_row[column]) and float(current_row[column]) > 0:
                values[game] = round(float(current_row[column]) / 100000000, 2)

    _, base = find_base_row(master_df, selected_month_str)
    for game in LOTTERY_GAMES:
        if game in values:
            continue
        column = f"{game}連槓獎金"
        if base is not None and column in base and pd.notna(base[column]) and float(base[column]) > 0:
            values[game] = round(float(base[column]) / 100000000, 2)
        else:
            values[game] = 0.0

    return values


def set_dgbas_macro_values(values, month_text):
    if values.get("cpi") is not None:
        st.session_state["cpi"] = values["cpi"]
    if values.get("unemp") is not None:
        st.session_state["unemp"] = values["unemp"]
    st.session_state["dgbas_macro_month"] = month_text
    st.session_state["dgbas_cpi_url"] = values.get("cpi_url", "")
    st.session_state["dgbas_unemp_url"] = values.get("unemp_url", "")


def set_if_valid(key, value):
    if pd.notna(value) and float(value) > 0:
        st.session_state[key] = float(value)


def set_forecast_2026_values(row, include_actual=True):
    set_if_valid("cpi", row.get("CPI總指數"))
    set_if_valid("unemp", row.get("失業率"))
    set_if_valid("cci", row.get("CCI"))
    set_if_valid("taiex", row.get("TAIEX月平均收盤指數"))
    set_if_valid("jackpot", row.get("連槓獎金(億)"))

    month_text = row["月份"]
    if (
        include_actual
        and pd.notna(row.get("電腦型彩券總銷售額_億"))
        and row["電腦型彩券總銷售額_億"] > 0
    ):
        st.session_state["actual_sales_100m"] = float(row["電腦型彩券總銷售額_億"])
        st.session_state["actual_sales_month"] = month_text

    st.session_state["spring"] = month_text == "2026-02"
    st.session_state["dragon"] = month_text == "2026-06"
    st.session_state["mid_autumn"] = month_text == "2026-09"
    st.session_state["forecast_2026_month"] = month_text


def update_2026_latest_data():
    from 抓取2026即時資料 import main as update_data

    update_data()
    load_lottery_raw_data.clear()
    load_forecast_2026.clear()


def latest_forecast_value(df, column, default):
    if df.empty or column not in df.columns:
        return default
    values = pd.to_numeric(df[column], errors="coerce")
    values = values[values.notna()]
    if values.empty:
        return default
    return float(values.iloc[-1])


def latest_positive_value(*sources):
    for source, column in sources:
        if source is None or source.empty or column not in source.columns:
            continue
        values = pd.to_numeric(source[column], errors="coerce")
        values = values[(values.notna()) & (values > 0)]
        if not values.empty:
            return float(values.iloc[-1])
    return 0.0


def latest_available_jackpot(master_df, forecast_df=None, before_month=None):
    rows = []

    def append_candidates(df, month_col, value_col, source_label, divisor=1):
        if df is None or df.empty or month_col not in df.columns or value_col not in df.columns:
            return

        data = df[[month_col, value_col]].copy()
        data["月份"] = normalize_month_series(data[month_col])
        data["月份_dt"] = pd.to_datetime(data["月份"], format="%Y-%m", errors="coerce")
        data["jackpot"] = pd.to_numeric(data[value_col], errors="coerce") / divisor
        data = data[(data["月份_dt"].notna()) & (data["jackpot"].notna()) & (data["jackpot"] > 0)]
        if before_month is not None:
            before_dt = month_to_datetime(before_month)
            data = data[data["月份_dt"] < before_dt]

        for _, item in data.iterrows():
            rows.append(
                {
                    "月份": item["月份"],
                    "月份_dt": item["月份_dt"],
                    "jackpot": float(item["jackpot"]),
                    "source": source_label,
                }
            )

    append_candidates(
        master_df,
        "月份",
        "電腦型彩券最高連槓獎金",
        "master_dataset 最新可得資料",
        divisor=100000000,
    )
    append_candidates(
        forecast_df,
        "月份",
        "連槓獎金(億)",
        "2026 最新彙整資料",
        divisor=1,
    )

    if not rows:
        latest = default_latest_values(master_df)
        return latest["jackpot"], "資料庫最新一期", latest["month"]

    latest_row = pd.DataFrame(rows).sort_values("月份_dt").iloc[-1]
    return (
        round(float(latest_row["jackpot"]), 2),
        str(latest_row["source"]),
        str(latest_row["月份"]),
    )


def get_positive_value(row, column):
    if row is None or column not in row or pd.isna(row[column]):
        return None
    value = float(row[column])
    if value <= 0:
        return None
    return value


def recent_average(master_df, column, months=36, default=0.0):
    data = master_df[master_df["月份"] >= "2011-01"].tail(months)
    if column not in data.columns:
        return default
    values = pd.to_numeric(data[column], errors="coerce").dropna()
    if values.empty:
        return default
    return float(values.mean())


def recent_average_before(master_df, column, selected_month_str, months=36, default=0.0):
    if column not in master_df.columns:
        return default

    data = master_df.copy()
    data["月份_dt"] = pd.to_datetime(data["月份"], format="%Y-%m", errors="coerce")
    selected_dt = month_to_datetime(selected_month_str)
    data = data[(data["月份_dt"].notna()) & (data["月份_dt"] < selected_dt)]
    data = data.sort_values("月份_dt").tail(months)

    values = pd.to_numeric(data[column], errors="coerce").dropna()
    if values.empty:
        return default
    return float(values.mean())


def fallback_future_value(master_df, selected_month_str, target_column):
    _, base = find_base_row(master_df, selected_month_str)
    if base is None:
        latest = default_latest_values(master_df)
        return latest[target_column]

    if target_column == "cpi":
        growth = recent_average(master_df, "通膨率", default=1.5)
        return round(float(base["CPI總指數"]) * (1 + growth / 100), 2)

    if target_column == "unemp":
        change = recent_average(master_df, "失業率年變動_pp", default=0.0)
        return round(max(float(base["失業率"]) + change, 0), 2)

    if target_column == "cci":
        change = recent_average(master_df, "CCI年變動", default=0.0)
        return round(max(float(base["CCI"]) + change, 0), 2)

    if target_column == "taiex":
        growth = recent_average(master_df, "TAIEX年增率", default=0.0)
        return round(float(base["TAIEX月平均收盤指數"]) * (1 + growth / 100), 2)

    return default_latest_values(master_df)[target_column]


def estimate_future_current_values(master_df, selected_month_str):
    _, base = find_base_row(master_df, selected_month_str)
    if base is None:
        return None

    cpi_growth = recent_average_before(master_df, "通膨率", selected_month_str, default=1.5)
    unemp_change = recent_average_before(master_df, "失業率年變動_pp", selected_month_str, default=0.0)
    cci_change = recent_average_before(master_df, "CCI年變動", selected_month_str, default=0.0)
    taiex_growth = recent_average_before(master_df, "TAIEX年增率", selected_month_str, default=0.0)
    jackpot, _, _ = latest_available_jackpot(master_df, before_month=selected_month_str)
    festival = get_festival_flags(selected_month_str)

    return {
        "cpi": float(base["CPI總指數"]) * (1 + cpi_growth / 100),
        "unemp": max(float(base["失業率"]) + unemp_change, 0),
        "cci": max(float(base["CCI"]) + cci_change, 0),
        "taiex": float(base["TAIEX月平均收盤指數"]) * (1 + taiex_growth / 100),
        "jackpot": jackpot,
        "spring": festival["spring"],
        "dragon": festival["dragon"],
        "mid_autumn": festival["mid_autumn"],
    }


def build_future_method_backtest(master_df, beta):
    candidate_months = [
        "2024-02",
        "2024-06",
        "2024-09",
        "2024-12",
        "2025-01",
        "2025-02",
        "2025-06",
        "2025-10",
        "2025-12",
    ]

    rows = []
    for month_text in candidate_months:
        current_row = find_current_row(master_df, month_text)
        base_month, base = find_base_row(master_df, month_text)
        estimated_current = estimate_future_current_values(master_df, month_text)

        if current_row is None or base is None or estimated_current is None:
            continue

        features = calculate_features(estimated_current, base)
        prediction = predict_sales(features, beta)
        actual = float(current_row["電腦型彩券總銷售額"]) / 100000000
        error = prediction - actual
        festival_names = []
        if features["spring"]:
            festival_names.append("春節")
        if features["dragon"]:
            festival_names.append("端午")
        if features["mid_autumn"]:
            festival_names.append("中秋")

        rows.append(
            {
                "月份": month_text,
                "去年同月": base_month,
                "月份類型": "、".join(festival_names) if festival_names else "一般月",
                "預測銷售額(億)": round(prediction, 2),
                "實際銷售額(億)": round(actual, 2),
                "誤差(億)": round(error, 2),
                "絕對誤差(億)": round(abs(error), 2),
                "誤差率": round(error / actual * 100, 2) if actual else 0,
            }
        )

    return pd.DataFrame(rows)


def render_future_method_backtest(master_df, beta):
    backtest = build_future_method_backtest(master_df, beta)
    if backtest.empty:
        st.info("目前沒有足夠資料可以做回測。")
        return

    avg_abs_error = backtest["絕對誤差(億)"].mean()
    avg_abs_pct_error = backtest["誤差率"].abs().mean()

    st.caption(
        "回測做法：假裝下列月份還沒發生，用「去年同月 + 當時可得的近三年平均變化」先估總體變數，"
        "再代入 OLS 模型，最後和該月實際銷售額比較。"
    )
    m1, m2 = st.columns(2)
    with m1:
        metric_card("平均絕對誤差", f"{avg_abs_error:.2f} 億元")
    with m2:
        metric_card("平均絕對誤差率", f"{avg_abs_pct_error:.1f}%")

    display = backtest.copy()
    display["誤差率"] = display["誤差率"].map(lambda value: f"{value:+.2f}%")
    display["誤差(億)"] = display["誤差(億)"].map(lambda value: f"{value:+.2f}")
    st.table(display.drop(columns=["絕對誤差(億)"]).style.hide(axis="index"))

    chart_data = backtest.melt(
        id_vars=["月份"],
        value_vars=["預測銷售額(億)", "實際銷售額(億)"],
        var_name="類型",
        value_name="銷售額(億)",
    )
    chart = (
        alt.Chart(chart_data)
        .mark_line(point=True)
        .encode(
            x=alt.X("月份:N", title="月份"),
            y=alt.Y("銷售額(億):Q", title="銷售額（億元）"),
            color=alt.Color("類型:N", title="類型"),
            tooltip=[
                alt.Tooltip("月份:N", title="月份"),
                alt.Tooltip("類型:N", title="類型"),
                alt.Tooltip("銷售額(億):Q", title="銷售額", format=".2f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")


def selected_month_defaults(master_df, forecast_df, selected_month_str):
    current_row = find_current_row(master_df, selected_month_str)
    forecast_row = find_forecast_row(forecast_df, selected_month_str)

    if current_row is not None:
        return values_from_row(current_row)

    cpi = get_positive_value(forecast_row, "CPI總指數")
    unemp = get_positive_value(forecast_row, "失業率")
    cci = get_positive_value(forecast_row, "CCI")
    taiex = get_positive_value(forecast_row, "TAIEX月平均收盤指數")
    jackpot, jackpot_source, jackpot_month = latest_available_jackpot(master_df, forecast_df)

    return {
        "cpi": cpi or fallback_future_value(master_df, selected_month_str, "cpi"),
        "unemp": unemp or fallback_future_value(master_df, selected_month_str, "unemp"),
        "cci": cci or fallback_future_value(master_df, selected_month_str, "cci"),
        "taiex": taiex or fallback_future_value(master_df, selected_month_str, "taiex"),
        "jackpot": jackpot,
        "jackpot_source": jackpot_source,
        "jackpot_month": jackpot_month,
    }


def macro_inputs_are_valid(current):
    invalid = []
    if current["cpi"] <= 0:
        invalid.append("CPI 總指數")
    if current["cci"] <= 0:
        invalid.append("CCI")
    if current["taiex"] <= 0:
        invalid.append("TAIEX 月平均收盤指數")
    if current["unemp"] <= 0:
        invalid.append("失業率")
    return invalid


def render_ticket_type_comparison(master_df, forecast_df, lottery_raw_df):
    st.subheader("分券種買氣比較")
    st.caption("使用同一組總體情境與連槓獎金，分別代入大樂透、威力彩、今彩539 的 OLS 模型。")

    month_options = build_month_options(master_df)
    latest_values = default_latest_values(master_df)
    default_index = month_options.index("2026-05") if "2026-05" in month_options else len(month_options) - 1
    selected_month_str = st.selectbox(
        "選擇預測月份",
        options=month_options,
        index=default_index,
        key="ticket_type_month",
    )

    base_month, base = find_base_row(master_df, selected_month_str)
    if base is None:
        st.error(f"找不到去年同月 `{base_month}` 的基準資料，無法計算年增率。")
        return

    level_defaults = selected_month_defaults(master_df, forecast_df, selected_month_str)
    festival_defaults = get_festival_flags(selected_month_str)
    forecast_row = find_forecast_row(forecast_df, selected_month_str)
    lottery_snapshot = lottery_month_snapshot(lottery_raw_df, selected_month_str)
    actual_by_game = actual_sales_by_game_for_month(forecast_row, lottery_snapshot)
    jackpot_by_game = game_jackpots_for_month(master_df, selected_month_str, lottery_snapshot)
    has_actual_game_sales = bool(actual_by_game)
    input_label_prefix = "當月" if has_actual_game_sales else "預估"
    source_status = "使用實際資料" if has_actual_game_sales else "使用基準情境估算"

    c1, c2 = st.columns(2)
    with c1:
        cpi = st.number_input(
            f"{input_label_prefix} CPI 總指數",
            min_value=0.0,
            step=0.01,
            format="%.2f",
            value=level_defaults["cpi"],
            key=f"ticket_cpi_{selected_month_str}",
        )
        unemp = st.number_input(
            f"{input_label_prefix}失業率（%）",
            min_value=0.0,
            step=0.01,
            format="%.2f",
            value=level_defaults["unemp"],
            key=f"ticket_unemp_{selected_month_str}",
        )
        cci = st.number_input(
            f"{input_label_prefix} CCI",
            min_value=0.0,
            step=0.01,
            format="%.2f",
            value=level_defaults["cci"],
            key=f"ticket_cci_{selected_month_str}",
        )
    with c2:
        taiex = st.number_input(
            f"{input_label_prefix} TAIEX 月平均收盤指數",
            min_value=0.0,
            step=0.01,
            format="%.2f",
            value=level_defaults["taiex"],
            key=f"ticket_taiex_{selected_month_str}",
        )

    st.write(f"{input_label_prefix}各券種連槓獎金")
    j1, j2, j3 = st.columns(3)
    with j1:
        lotto_jackpot = st.number_input(
            "大樂透連槓獎金（億）",
            min_value=0.0,
            step=0.5,
            format="%.2f",
            value=max(jackpot_by_game["大樂透"], 0.0),
            key=f"ticket_lotto_jackpot_{selected_month_str}",
        )
    with j2:
        power_jackpot = st.number_input(
            "威力彩連槓獎金（億）",
            min_value=0.0,
            step=0.5,
            format="%.2f",
            value=max(jackpot_by_game["威力彩"], 0.0),
            key=f"ticket_power_jackpot_{selected_month_str}",
        )
    with j3:
        daily539_jackpot = st.number_input(
            "今彩539連槓獎金（億）",
            min_value=0.0,
            step=0.1,
            format="%.2f",
            value=max(jackpot_by_game["今彩539"], 0.0),
            key=f"ticket_daily539_jackpot_{selected_month_str}",
        )

    manual_jackpots = {
        "大樂透": lotto_jackpot,
        "威力彩": power_jackpot,
        "今彩539": daily539_jackpot,
    }

    f1, f2, f3 = st.columns(3)
    with f1:
        spring = st.checkbox("春節", value=festival_defaults["spring"], key=f"ticket_spring_{selected_month_str}")
    with f2:
        dragon = st.checkbox("端午", value=festival_defaults["dragon"], key=f"ticket_dragon_{selected_month_str}")
    with f3:
        mid_autumn = st.checkbox("中秋", value=festival_defaults["mid_autumn"], key=f"ticket_mid_autumn_{selected_month_str}")

    current = {
        "cpi": cpi,
        "unemp": unemp,
        "cci": cci,
        "taiex": taiex,
        "jackpot": max(manual_jackpots.values()),
        "spring": spring,
        "dragon": dragon,
        "mid_autumn": mid_autumn,
    }
    invalid_macro_inputs = macro_inputs_are_valid(current)
    if invalid_macro_inputs:
        st.warning("請先補上：" + "、".join(invalid_macro_inputs))
        return

    features = calculate_features(current, base)
    rows = []
    for model_name in MODEL_NAMES[1:]:
        model_beta = load_coefficients_for_model(model_name)
        model_features = features.copy()
        model_features["jackpot"] = manual_jackpots[model_name]
        rows.append(
            {
                "券種": model_name,
                "連槓獎金(億)": round(manual_jackpots[model_name], 2),
                "預測銷售額(億)": round(predict_sales(model_features, model_beta), 2),
            }
        )

    result = pd.DataFrame(rows).sort_values("預測銷售額(億)", ascending=False)
    if has_actual_game_sales:
        result["實際銷售額(億)"] = result["券種"].map(actual_by_game)
        result["預測誤差(億)"] = result["預測銷售額(億)"] - result["實際銷售額(億)"]
        result["預測誤差(億)"] = result["預測誤差(億)"].round(2)
        result = result.sort_values("實際銷售額(億)", ascending=False)
    st.write(f"預測月份：{selected_month_str}；去年同月：{base_month}")
    st.write(f"資料來源狀態：{source_status}")
    st.warning("提醒：此頁預測的是各券種市場買氣與月銷售額，不預測中獎機率、開獎號碼，也不構成購買建議。")
    st.table(result.style.hide(axis="index"))

    chart_value_col = "實際銷售額(億)" if has_actual_game_sales else "預測銷售額(億)"
    chart_title = "實際銷售額（億元）" if has_actual_game_sales else "預測銷售額（億元）"
    chart_caption = "各券種實際銷售額比較" if has_actual_game_sales else "各券種預測銷售額比較"
    st.write(chart_caption)
    chart = (
        alt.Chart(result)
        .mark_bar(color="#4C78A8")
        .encode(
            x=alt.X(f"{chart_value_col}:Q", title=chart_title),
            y=alt.Y("券種:N", sort="-x", title="券種"),
            tooltip=[
                alt.Tooltip("券種:N", title="券種"),
                alt.Tooltip(f"{chart_value_col}:Q", title=chart_value_col, format=".2f"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, width="stretch")

    feature_table = pd.DataFrame(
        [
            ["CPI年增率", f"{features['cpi_yoy']:.2f}%"],
            ["失業率年變動", f"{features['unemp_change']:.2f}"],
            ["CCI年變動", f"{features['cci_change']:.2f}"],
            ["TAIEX年增率(%)", f"{features['taiex_yoy']:.2f}%"],
            ["大樂透連槓獎金(億)", f"{manual_jackpots['大樂透']:.2f}"],
            ["威力彩連槓獎金(億)", f"{manual_jackpots['威力彩']:.2f}"],
            ["今彩539連槓獎金(億)", f"{manual_jackpots['今彩539']:.2f}"],
            ["春節", "是" if features["spring"] else "否"],
            ["端午", "是" if features["dragon"] else "否"],
            ["中秋", "是" if features["mid_autumn"] else "否"],
        ],
        columns=["變數", "數值"],
    )
    with st.expander("帶入模型的變數"):
        st.dataframe(feature_table, hide_index=True, width="stretch")

    st.caption("分券種模型已分別使用各券種自己的連槓獎金，避免把整體最高獎金套到所有券種。")


st.set_page_config(
    page_title="電腦型彩券銷售額預測",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1480px;
    }
    [data-testid="stSidebar"] {
        background: #f4f6f9;
    }
    .app-header {
        border-bottom: 1px solid #edf0f5;
        margin: 0 0 1.2rem 0;
        padding: 0.15rem 0 1rem 0;
    }
    .app-eyebrow {
        color: #e45756;
        font-size: 0.9rem;
        font-weight: 760;
        letter-spacing: 0.04em;
        margin-bottom: 0.25rem;
    }
    .app-title {
        color: #2f3340;
        font-size: clamp(2rem, 4vw, 3.6rem);
        line-height: 1.05;
        font-weight: 820;
        letter-spacing: 0;
        margin: 0;
    }
    .app-subtitle {
        color: #7d828c;
        font-size: 1rem;
        line-height: 1.55;
        margin: 0.7rem 0 0 0;
        max-width: 760px;
    }
    .section-heading {
        color: #2f3340;
        font-size: 1.55rem;
        line-height: 1.2;
        font-weight: 780;
        margin: 0.25rem 0 1rem 0;
    }
    .compact-note {
        color: #858b96;
        font-size: 0.95rem;
        line-height: 1.5;
        margin: -0.35rem 0 1rem 0;
    }
    .metric-card {
        border: 1px solid #e7e9ee;
        border-radius: 10px;
        padding: 18px 20px;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(20, 24, 35, 0.04);
        margin-bottom: 14px;
    }
    .metric-card-primary {
        border-color: #f3b7b4;
        background: linear-gradient(180deg, #fffafa 0%, #ffffff 100%);
    }
    .metric-card-hot {
        border-color: #e45756;
        background: #fff4f2;
    }
    .metric-card-warm {
        border-color: #f2a45d;
        background: #fff8ef;
    }
    .metric-card-mild {
        border-color: #94b7e6;
        background: #f3f7ff;
    }
    .metric-card-neutral {
        background: #ffffff;
    }
    .metric-label {
        color: #6f7480;
        font-size: 0.95rem;
        font-weight: 650;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #2f3340;
        font-size: clamp(1.65rem, 2.8vw, 2.7rem);
        line-height: 1.15;
        font-weight: 760;
        letter-spacing: 0;
        white-space: normal;
        overflow-wrap: anywhere;
    }
    .metric-card-primary .metric-value {
        font-size: clamp(2.1rem, 4vw, 3.5rem);
    }
    .metric-note {
        color: #8a8f99;
        font-size: 0.9rem;
        margin-top: 8px;
    }
    div[data-testid="stMetric"] {
        background: #fff;
        border: 1px solid #e8ebf0;
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
    }
    div[data-testid="stVerticalBlock"] > div:has(> .stAlert) {
        margin-top: 0.25rem;
    }
    button[kind="primary"] {
        font-weight: 760;
    }
    .model-method {
        border: 1px solid #e7e9ee;
        border-radius: 10px;
        padding: 0.95rem 1.1rem;
        background: #fbfcfe;
        margin: -0.3rem 0 1.2rem 0;
    }
    .model-method-title {
        color: #2f3340;
        font-weight: 780;
        margin-bottom: 0.45rem;
    }
    .model-method ol {
        margin: 0;
        padding-left: 1.25rem;
        color: #5f6570;
        line-height: 1.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <div class="app-eyebrow">LOTTERY DEMAND FORECAST</div>
        <h1 class="app-title">電腦型彩券月銷售額預測系統</h1>
        <p class="app-subtitle">
            選擇月份並帶入當月資料，系統會自動與去年同月比較，換算成 OLS 模型需要的年增率與年變動。
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

master = load_master()
beta = load_coefficients()
latest_defaults = default_latest_values(master)
month_options = build_month_options(master)
lottery_raw = load_lottery_raw_data()
forecast_2026 = load_forecast_2026()

with st.sidebar:
    page = st.radio(
        "頁面",
        ["整體買氣預測", "分券種買氣比較"],
        key="page",
    )

if page == "分券種買氣比較":
    render_ticket_type_comparison(master, forecast_2026, lottery_raw)
    st.stop()

if "month" not in st.session_state:
    set_defaults(latest_defaults)

pending_month = st.session_state.pop("pending_month", None)
if pending_month in month_options:
    st.session_state["month"] = pending_month

with st.sidebar:
    st.header("資料帶入")
    st.write(f"目前資料庫最新月份：`{latest_defaults['month']}`")

    if st.button("帶入資料庫最新一期", width="stretch"):
        set_defaults(latest_defaults)
        st.rerun()

    st.divider()
    if st.button("更新 2026 最新資料", width="stretch"):
        try:
            with st.spinner("正在抓取台彩、主計總處、中央大學 CCI、TAIEX..."):
                update_2026_latest_data()
            st.success("2026 最新資料已更新。")
            st.rerun()
        except Exception as error:
            st.error(f"更新失敗：{error}")

analysis_payload = None
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="section-heading">預測資料設定</div>', unsafe_allow_html=True)

    month = st.selectbox("選擇預測月份", options=month_options, key="month")
    selected_month_str = str(month)
    selected_month_dt = month_to_datetime(selected_month_str)
    festival_defaults = get_festival_flags(selected_month_str)
    current_row = find_current_row(master, selected_month_str)
    forecast_row = find_forecast_row(forecast_2026, selected_month_str)
    lottery_snapshot = lottery_month_snapshot(lottery_raw, selected_month_str)
    actual_sales = actual_sales_for_month(forecast_row, lottery_snapshot)
    future_mode = st.checkbox(
        "只做預測（不顯示實際銷售額與預測誤差）",
        value=selected_month_str > latest_defaults["month"] and actual_sales is None,
        key=f"future_mode_{selected_month_str}",
    )

    selected_month_source = None
    if current_row is not None:
        selected_month_source = "master"

    if forecast_row is not None:
        if current_row is None:
            selected_month_source = "forecast_2026"

    if not future_mode and st.session_state.get("forecast_2026_month") == selected_month_str:
        st.success("已帶入 2026 最新彙整資料。")

    if lottery_snapshot is not None:
        if selected_month_source is None:
            selected_month_source = "lottery_only"

    if not future_mode and selected_month_source is not None:
        if st.button("帶入所選月份資料", type="primary", width="stretch"):
            if selected_month_source == "master":
                set_defaults(values_from_row(current_row), include_month=False)
            elif selected_month_source == "forecast_2026":
                set_forecast_2026_values(forecast_row, include_actual=not future_mode)
            else:
                st.session_state["forecast_2026_month"] = None
                set_lottery_snapshot(lottery_snapshot, include_actual=not future_mode)
            st.rerun()

    with st.expander("選填：單項資料補抓", expanded=False):
        if st.button("抓取主計總處 CPI / 失業率", width="stretch"):
            try:
                macro_values = fetch_dgbas_cpi_unemployment(month)
                input_prefix = "future_" if future_mode else ""
                if macro_values.get("cpi") is not None:
                    st.session_state[f"{input_prefix}cpi_{selected_month_str}"] = macro_values["cpi"]
                if macro_values.get("unemp") is not None:
                    st.session_state[f"{input_prefix}unemp_{selected_month_str}"] = macro_values["unemp"]
                set_dgbas_macro_values(macro_values, month)
                st.rerun()
            except Exception as error:
                st.error(f"抓取失敗：{error}")

        if st.session_state.get("dgbas_macro_month") == month:
            st.success("已帶入主計總處 CPI / 失業率。")

        if st.button("抓取中央大學 CCI", width="stretch"):
            try:
                if (
                    forecast_row is not None
                    and "CCI" in forecast_row
                    and pd.notna(forecast_row["CCI"])
                    and float(forecast_row["CCI"]) > 0
                ):
                    input_prefix = "future_" if future_mode else ""
                    st.session_state[f"{input_prefix}cci_{selected_month_str}"] = float(forecast_row["CCI"])
                    set_ncu_cci_from_forecast_row(forecast_row)
                elif forecast_row is not None and month.startswith("2026-"):
                    available_cci = forecast_2026[
                        forecast_2026.get("CCI", pd.Series(dtype=float)).notna()
                        & (forecast_2026.get("CCI", pd.Series(dtype=float)) > 0)
                    ]
                    latest_cci_month = (
                        available_cci["月份"].max()
                        if not available_cci.empty
                        else "尚未取得"
                    )
                    raise ValueError(f"{month} 尚未公布 CCI，目前本機最新為 {latest_cci_month}")
                else:
                    cci_values = fetch_ncu_cci_for_month(month)
                    input_prefix = "future_" if future_mode else ""
                    st.session_state[f"{input_prefix}cci_{selected_month_str}"] = cci_values["cci"]
                    set_ncu_cci_values(cci_values)
                st.rerun()
            except Exception as error:
                st.error(f"抓取失敗：{error}")

        if st.session_state.get("ncu_cci_month") == month:
            cci_change = st.session_state.get("ncu_cci_change")
            if cci_change is None:
                st.success("已帶入中央大學 CCI。")
            else:
                st.success(
                    "已帶入中央大學 CCI。"
                    f"年變動：{cci_change:+.2f}"
                )

    input_label_prefix = "預估" if future_mode else "當月"
    jackpot_label = "預估最高連槓獎金（億）" if future_mode else "當月最高連槓獎金（億）"
    jackpot_source = "歷史實際資料"

    if future_mode:
        st.write("預測情境設定")
        level_defaults = selected_month_defaults(master, forecast_2026, selected_month_str)
        latest_jackpot = max(float(level_defaults["jackpot"]), 0.0)
        latest_jackpot_month = level_defaults.get("jackpot_month", "最新月份")
        latest_jackpot_source = level_defaults.get("jackpot_source", "最新可得資料")

        c1, c2 = st.columns(2)
        with c1:
            cpi = st.number_input(
                f"{input_label_prefix} CPI 總指數",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                value=level_defaults["cpi"],
                key=f"future_cpi_{selected_month_str}",
            )
            unemp = st.number_input(
                f"{input_label_prefix}失業率（%）",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                value=level_defaults["unemp"],
                key=f"future_unemp_{selected_month_str}",
            )
            cci = st.number_input(
                f"{input_label_prefix} CCI",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                value=level_defaults["cci"],
                key=f"future_cci_{selected_month_str}",
            )
        with c2:
            taiex = st.number_input(
                f"{input_label_prefix} TAIEX 月平均收盤指數",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                value=level_defaults["taiex"],
                key=f"future_taiex_{selected_month_str}",
            )
            jackpot_scenario = st.selectbox(
                "連槓獎金情境",
                options=[
                    "最新可得連槓獎金",
                    "低連槓情境：5 億元",
                    "中連槓情境：10 億元",
                    "高連槓情境：20 億元",
                    "極高連槓情境：30 億元",
                    "自訂",
                ],
                key=f"future_jackpot_scenario_{selected_month_str}",
            )
            scenario_values = {
                "最新可得連槓獎金": (latest_jackpot, f"最新可得資料（{latest_jackpot_source}，{latest_jackpot_month}）"),
                "低連槓情境：5 億元": (5.0, "低連槓情境"),
                "中連槓情境：10 億元": (10.0, "中連槓情境"),
                "高連槓情境：20 億元": (20.0, "高連槓情境"),
                "極高連槓情境：30 億元": (30.0, "極高連槓情境"),
            }
            if jackpot_scenario == "自訂":
                jackpot = st.number_input(
                    "自訂連槓獎金（億）",
                    min_value=0.0,
                    step=0.5,
                    format="%.2f",
                    value=latest_jackpot,
                    key=f"future_custom_jackpot_{selected_month_str}",
                )
                jackpot_source = "使用者自訂"
            else:
                jackpot, jackpot_source = scenario_values[jackpot_scenario]
                st.info(f"目前連槓獎金來源：{jackpot_source}，代入 {jackpot:.2f} 億元。")

        ref_base_month, ref_base = find_base_row(master, selected_month_str)
        if ref_base is not None and "電腦型彩券最高連槓獎金" in ref_base:
            ref_jackpot = float(ref_base["電腦型彩券最高連槓獎金"]) / 100000000
            with st.expander("去年同月連槓獎金參考", expanded=False):
                st.write(f"去年同月：{ref_base_month}")
                st.write(f"去年同月連槓獎金：{ref_jackpot:.2f} 億元，僅供參考。")
    else:
        level_defaults = selected_month_defaults(master, forecast_2026, selected_month_str)
        c1, c2 = st.columns(2)
        with c1:
            cpi = st.number_input(
                f"{input_label_prefix} CPI 總指數",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                value=level_defaults["cpi"],
                key=f"cpi_{selected_month_str}",
            )
            unemp = st.number_input(
                f"{input_label_prefix}失業率（%）",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                value=level_defaults["unemp"],
                key=f"unemp_{selected_month_str}",
            )
            cci = st.number_input(
                f"{input_label_prefix} CCI",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                value=level_defaults["cci"],
                key=f"cci_{selected_month_str}",
            )
        with c2:
            taiex = st.number_input(
                f"{input_label_prefix} TAIEX 月平均收盤指數",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                value=level_defaults["taiex"],
                key=f"taiex_{selected_month_str}",
            )
            jackpot = st.number_input(
                jackpot_label,
                min_value=0.0,
                step=0.1,
                format="%.2f",
                value=max(level_defaults["jackpot"], 0.0),
                key=f"jackpot_{selected_month_str}",
            )
            st.info("目前連槓獎金來源：歷史實際資料。")

    st.write("節慶設定")
    f1, f2, f3 = st.columns(3)
    with f1:
        spring = st.checkbox(
            "春節",
            value=festival_defaults["spring"],
            key=f"spring_{selected_month_str}",
        )
    with f2:
        dragon = st.checkbox(
            "端午",
            value=festival_defaults["dragon"],
            key=f"dragon_{selected_month_str}",
        )
    with f3:
        mid_autumn = st.checkbox(
            "中秋",
            value=festival_defaults["mid_autumn"],
            key=f"mid_autumn_{selected_month_str}",
        )

with right:
    st.markdown('<div class="section-heading">模型預測</div>', unsafe_allow_html=True)

    try:
        base_month, base = find_base_row(master, selected_month_str)
        if base is None:
            st.error(f"找不到去年同月 `{base_month}` 的基準資料，無法計算年增率。")
            st.stop()

        if future_mode:
            current = {
                "cpi": cpi,
                "unemp": unemp,
                "cci": cci,
                "taiex": taiex,
                "jackpot": jackpot,
                "spring": spring,
                "dragon": dragon,
                "mid_autumn": mid_autumn,
            }
            invalid_macro_inputs = macro_inputs_are_valid(current)
            if invalid_macro_inputs:
                st.warning("請先補上：" + "、".join(invalid_macro_inputs))
                st.stop()
            features = calculate_features(current, base)
        else:
            current = {
                "cpi": cpi,
                "unemp": unemp,
                "cci": cci,
                "taiex": taiex,
                "jackpot": jackpot,
                "spring": spring,
                "dragon": dragon,
                "mid_autumn": mid_autumn,
            }
            invalid_macro_inputs = macro_inputs_are_valid(current)
            if invalid_macro_inputs:
                st.warning(
                    "目前只匯入了台彩資料，總體資料還沒有完整帶入。"
                    "請先補上："
                    + "、".join(invalid_macro_inputs)
                    + "。如果這些欄位是 0，年增率會被算成 -100%。"
                )
                st.stop()

            features = calculate_features(current, base)
        prediction = predict_sales(features, beta)
        heat_level = classify_heat_level(master, prediction)
        avg_diff = diff_from_historical_average(master, prediction)

        metric_card(
            "預測電腦型彩券總銷售額",
            f"{prediction:.2f} 億元",
            note=f"預測月份：{selected_month_str}",
            tone="primary",
        )
        m1, m2 = st.columns(2)
        with m1:
            metric_card("買氣等級", heat_level, tone=heat_level_tone(heat_level))
        with m2:
            avg_text = f"{avg_diff:+.2f} 億元" if avg_diff is not None else "無法判斷"
            metric_card("相較歷史平均", avg_text)
        st.warning(
            "提醒：本系統預測的是市場買氣與月銷售額，不預測中獎機率、開獎號碼，也不構成購買建議。"
        )
        actual_sales = actual_sales_for_month(forecast_row, lottery_snapshot)
        if current_row is not None:
            source_status = "使用實際資料"
        elif actual_sales is not None and not future_mode:
            source_status = "使用已匯入實際資料"
        else:
            source_status = "使用基準情境估算；連槓獎金為情境輸入"

        st.write(f"預測月份：{selected_month_str}；去年同月：{base_month}")
        st.write(f"資料來源狀態：{source_status}")
        st.write(f"連槓獎金來源：{jackpot_source}")
        if future_mode:
            st.caption(
                f"系統會自動和 {base_month} 比較，換算成模型需要的年增率與年變動；"
                "連槓獎金因隨機性高，採情境輸入而非精準預測。"
            )
        else:
            st.caption(f"基準月份：{base_month}（用來計算年增率與年變動）")

        if not future_mode:
            if actual_sales is None and st.session_state.get("actual_sales_month") == selected_month_str:
                actual_sales = st.session_state.get("actual_sales_100m")

        if not future_mode and actual_sales is not None:
            metric_card(
                "所選月份實際彩券銷售額",
                f"{actual_sales:.2f} 億元",
                note=f"預測誤差：{prediction - actual_sales:+.2f} 億元",
            )

        with st.expander("帶入模型的變數", expanded=False):
            feature_table = pd.DataFrame(
                [
                    ["CPI年增率", f"{features['cpi_yoy']:.2f}%"],
                    ["失業率年變動", f"{features['unemp_change']:.2f}"],
                    ["CCI年變動", f"{features['cci_change']:.2f}"],
                    ["TAIEX年增率(%)", f"{features['taiex_yoy']:.2f}%"],
                    ["連槓獎金(億)", f"{features['jackpot']:.2f}"],
                    ["連槓獎金來源", jackpot_source],
                    ["春節", "是" if features["spring"] else "否"],
                    ["端午", "是" if features["dragon"] else "否"],
                    ["中秋", "是" if features["mid_autumn"] else "否"],
                ],
                columns=["變數", "數值"],
            )
            st.table(feature_table.style.hide(axis="index"))

        analysis_payload = {
            "master_df": master,
            "selected_month_str": selected_month_str,
            "prediction": prediction,
            "features": features,
            "beta": beta,
            "jackpot_source": jackpot_source,
        }

        with st.expander("目前使用的 OLS 模型係數", expanded=False):
            st.caption("這張表是系統本次預測實際讀取的係數，來源：06_OLS四模型係數表.csv。")
            coef_table = pd.DataFrame(
                [
                    ["截距", beta["const"]],
                    ["CPI年增率", beta["cpi_yoy"]],
                    ["失業率年變動", beta["unemp_change"]],
                    ["CCI年變動", beta["cci_change"]],
                    ["TAIEX年增率(%)", beta["taiex_yoy"]],
                    ["連槓獎金(億)", beta["jackpot"]],
                    ["春節", beta["spring"]],
                    ["端午", beta["dragon"]],
                    ["中秋", beta["mid_autumn"]],
                ],
                columns=["變數", "係數"],
            )
            coef_table["係數"] = coef_table["係數"].round(4)
            st.table(coef_table.style.hide(axis="index"))

    except ValueError:
        st.error("月份格式錯誤，請從月份選單重新選擇。")

if analysis_payload is not None:
    render_prediction_analysis(**analysis_payload)

with st.expander("連槓獎金情境預測", expanded=False):
    try:
        _, base = find_base_row(master, selected_month_str)
        if base is not None:
            current = {
                "cpi": cpi,
                "unemp": unemp,
                "cci": cci,
                "taiex": taiex,
                "jackpot": jackpot,
                "spring": spring,
                "dragon": dragon,
                "mid_autumn": mid_autumn,
            }
            if macro_inputs_are_valid(current):
                st.info("補齊 CPI、失業率、CCI、TAIEX 後，才會顯示連槓獎金情境預測。")
                st.stop()

            rows = []
            for scenario_jackpot in [5, 10, 20, 30, 36.3]:
                current["jackpot"] = scenario_jackpot
                scenario_features = calculate_features(current, base)
                rows.append(
                    {
                        "連槓獎金(億)": scenario_jackpot,
                        "預測銷售額(億)": round(predict_sales(scenario_features, beta), 2),
                    }
                )
            st.table(pd.DataFrame(rows).style.hide(axis="index"))
    except Exception:
        pass

with st.expander("未來估算法回測", expanded=False):
    render_future_method_backtest(master, beta)

with st.expander("模型方法與未來月份估算說明", expanded=False):
    st.markdown(
        """
        <div class="model-method">
            <div class="model-method-title">模型建立流程</div>
            <ol>
                <li>整理 2011-2025 年每月三種電腦型彩券資料：大樂透、威力彩、今彩539。</li>
                <li>合併總體指標：CPI、失業率、CCI、TAIEX，並加入連槓獎金與春節、端午、中秋節慶變數。</li>
                <li>把總體資料轉成和去年同月比較的變數，例如 CPI 年增率、失業率年變動、CCI 年變動、TAIEX 年增率。</li>
                <li>使用 OLS 迴歸模型估計「哪些因素會影響每月彩券銷售額」。</li>
                <li>使用者選擇月份後，系統把當月資料代入模型，輸出預測銷售額、買氣等級與主要影響因素。</li>
            </ol>
            <div class="model-method-title" style="margin-top: 0.9rem;">未來月份資料的預估方式</div>
            <ol>
                <li>預估當月 CPI = 去年同月 CPI × (1 + 近三年平均通膨率 / 100)</li>
                <li>預估當月失業率 = 去年同月失業率 + 近三年平均失業率年變動</li>
                <li>預估當月 CCI = 去年同月 CCI + 近三年平均 CCI 年變動</li>
                <li>預估當月 TAIEX = 去年同月 TAIEX × (1 + 近三年平均 TAIEX 年增率 / 100)</li>
                <li>連槓獎金不使用趨勢外推；未來月份改用最新可得、低 / 中 / 高 / 極高或自訂情境輸入</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("提醒：這是市場買氣與銷售額預測模型，不是中獎機率或開獎號碼預測。")
