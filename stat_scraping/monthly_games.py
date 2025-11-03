import pandas as pd
import requests
import time
import datetime
from io import StringIO

# function to scrape games for the current month


def get_daily_games():
    # Define seasons
    month_name = datetime.datetime.now().strftime("%B")
    year = datetime.datetime.now().year
    # date = datetime.datetime.now().strftime("%a, %b %-d, %Y")

    cols_needed = ['Date', 'Home/Neutral', 'Visitor/Neutral', 'Start (ET)']

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    # print(f"\n=== Processing {date} ===")
    schedule_url = f"https://www.basketball-reference.com/leagues/NBA_{year+1}_games-{month_name.lower()}.html"
    response = requests.get(schedule_url, headers=headers)
    time.sleep(4)

    games = pd.read_html(StringIO(response.text),
                         match=f"{month_name} Schedule Table")[0]
    games = games[cols_needed].reset_index(drop=True)
    games = games[games['Date'] != 'Date'].reset_index(drop=True)
    return games


if __name__ == "__main__":
    games_this_month = get_daily_games()
    print(games_this_month)
    # games_this_month.to_csv('games_this_month.csv', index = False)
