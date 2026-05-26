import argparse
import shutil
import zipfile
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_URL = "https://api.taiwanlottery.com/TLCAPIWeB/Lottery/ResultDownload"
DOWNLOAD_DIR = Path("台彩年度下載")
TARGET_GAMES = ["大樂透", "威力彩", "今彩539"]

headers = {
    "User-Agent": "Mozilla/5.0"
}


def get_download_info(year):
    response = requests.get(
        API_URL,
        params={"year": year},
        headers=headers,
        timeout=20,
        verify=False,
    )
    response.raise_for_status()

    data = response.json()
    if data.get("rtCode") != 0:
        raise ValueError(f"{year} 年 API 回傳錯誤：{data}")

    content = data.get("content") or {}
    download_url = content.get("path")
    file_name = content.get("fileName") or str(year)

    if not download_url:
        raise ValueError(f"{year} 年找不到下載網址：{data}")

    return file_name, download_url


def download_file(url, output_path):
    with requests.get(
        url,
        headers=headers,
        timeout=60,
        stream=True,
        verify=False,
    ) as response:
        response.raise_for_status()

        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def extract_zip(zip_path, extract_dir):
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(extract_dir)


def copy_target_csvs(extract_dir):
    copied = []
    for game in TARGET_GAMES:
        game_dir = Path(game)
        game_dir.mkdir(parents=True, exist_ok=True)

        for csv_path in extract_dir.glob(f"{game}_*.csv"):
            output_path = game_dir / csv_path.name
            shutil.copy2(csv_path, output_path)
            copied.append(output_path)

    return copied


def main():
    parser = argparse.ArgumentParser(description="下載台灣彩券各年度開獎結果 zip 檔")
    parser.add_argument("--start-year", type=int, default=2011, help="開始西元年，預設 2011")
    parser.add_argument("--end-year", type=int, default=2026, help="結束西元年，預設 2026")
    parser.add_argument("--extract", action="store_true", help="下載後自動解壓縮")
    parser.add_argument(
        "--copy-targets",
        action="store_true",
        help="解壓縮後把大樂透、威力彩、今彩539 CSV 複製到對應資料夾",
    )
    args = parser.parse_args()

    if args.copy_targets:
        args.extract = True

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    for year in range(args.start_year, args.end_year + 1):
        print(f"\n處理 {year} 年...")

        file_name, download_url = get_download_info(year)
        zip_path = DOWNLOAD_DIR / f"{file_name}.zip"

        print("下載網址：", download_url)

        if zip_path.exists():
            print("已存在，略過下載：", zip_path)
        else:
            download_file(download_url, zip_path)
            print("已下載：", zip_path)

        if args.extract:
            extract_dir = DOWNLOAD_DIR / file_name
            extract_zip(zip_path, extract_dir)
            print("已解壓縮到：", extract_dir)

            if args.copy_targets:
                copied_files = copy_target_csvs(extract_dir)
                if copied_files:
                    print("已複製目標彩券 CSV：")
                    for copied_file in copied_files:
                        print(" -", copied_file)
                else:
                    print("沒有找到大樂透、威力彩、今彩539 CSV")

    print("\n全部完成。")


if __name__ == "__main__":
    main()
