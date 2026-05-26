import requests
import urllib3
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =========================
# 1. 台灣彩券資料下載頁
# =========================
PAGE_URL = "https://www.taiwanlottery.com/lotto/history/result_download/"

TARGET_GAMES = ["大樂透", "威力彩", "今彩539"]

headers = {
    "User-Agent": "Mozilla/5.0"
}

# =========================
# 2. 讀取網頁
# =========================
response = requests.get(
    PAGE_URL,
    headers=headers,
    timeout=20,
    verify=False
)
response.raise_for_status()
response.encoding = "utf-8"

soup = BeautifulSoup(response.text, "html.parser")

# =========================
# 3. 找出所有連結
# =========================
all_links = []
target_links = []

for a in soup.find_all("a"):
    text = a.get_text(strip=True)
    href = a.get("href")

    if not href:
        continue

    full_url = urljoin(PAGE_URL, href)
    decoded_url = unquote(full_url)

    all_links.append({
        "文字": text,
        "網址": decoded_url
    })

    combined = text + " " + decoded_url

    if any(game in combined for game in TARGET_GAMES):
        target_links.append({
            "文字": text,
            "網址": decoded_url
        })

# =========================
# 4. 印出檢查結果
# =========================
print("全部連結數量：", len(all_links))
print("符合大樂透 / 威力彩 / 今彩539 的連結數量：", len(target_links))

print("\n--- 符合目標的連結 ---")
for item in target_links:
    print(item["文字"], "=>", item["網址"])

print("\n--- 前 30 個全部連結，方便檢查 ---")
for item in all_links[:30]:
    print(item["文字"], "=>", item["網址"])
