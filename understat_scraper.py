import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import time

# Daftar liga dan slug nama liga di Understat
LEAGUES = {
    "EPL": "EPL",
    "La_Liga": "La_liga",
    "Bundesliga": "Bundesliga",
    "Serie_A": "Serie_A",
    "Ligue_1": "Ligue_1",
    "RFPL": "RFPL"
}

YEAR = 2023  # atau ubah ke tahun berjalan
BASE_URL = "https://understat.com/league/{league}/{year}"

def extract_team_data(js_code):
    json_start = js_code.find("('") + 2
    json_end = js_code.find("')")  
    json_str = js_code[json_start:json_end].encode('utf8').decode('unicode_escape')
    data = json.loads(json_str)
    return data

all_rows = []

for league_code, league_slug in LEAGUES.items():
    print(f"Scraping {league_code}...")
    url = BASE_URL.format(league=league_slug, year=YEAR)
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "lxml")
    scripts = soup.find_all("script")
    json_data_script = [s for s in scripts if "teamsData" in s.text][0].string

    data = extract_team_data(json_data_script)
    for team_name, team_info in data.items():
        matches = team_info["history"][-5:]  # 5 laga terakhir
        xg_total = sum(float(m["xG"]) for m in matches)
        xga_total = sum(float(m["xGA"]) for m in matches)
        form_points = 0
        for m in matches:
            gf = float(m["goals"])
            ga = float(m["goals_against"])
            if gf > ga:
                form_points += 3
            elif gf == ga:
                form_points += 1
        avg_xg = xg_total / 5
        avg_xga = xga_total / 5

        all_rows.append({
            "team": team_name,
            "league": league_code,
            "xg": round(avg_xg, 2),
            "xa": round(avg_xga, 2),
            "form_points": form_points
        })

    time.sleep(1.5)  # Delay agar tidak diblok

df = pd.DataFrame(all_rows)
df.to_csv("tim_stats_understat.csv", index=False)
print("✅ Data berhasil disimpan ke tim_stats_understat.csv")
