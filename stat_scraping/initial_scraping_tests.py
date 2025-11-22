import requests
from bs4 import BeautifulSoup, Comment
import time
import pandas as pd
import re 
#----------------------------------------------------------------------------------------------
# NBA Standings from Basketball Reference
standings_url = "https://www.basketball-reference.com/leagues/NBA_2025_standings.html"

# Add headers - Basketball Reference blocks requests without them
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# Get the HTML
response = requests.get(standings_url, headers=headers)
time.sleep(2)  # Be respectful with delays

# Parse the HTML
soup = BeautifulSoup(response.text, "html.parser")

# Basketball Reference hides tables in HTML comments
# Find the comment containing expanded_standings
comments = soup.find_all(string=lambda text: isinstance(text, Comment))

#loop through and find the standings table from comments
expanded_standings_table = None
for comment in comments:
    if 'expanded_standings' in comment:
        # Parse the commented HTML
        comment_soup = BeautifulSoup(comment, "html.parser")
        expanded_standings_table = comment_soup.find('table', {'id': 'expanded_standings'})
        break

# get the individual team links
team_links = expanded_standings_table.find_all('a')
team_links = [l.get("href") for l in team_links]
team_links = [l for l in team_links if '/teams/'in l]
# print(team_links)

# ----------------------------------------------------------------------------------------------

#set up link to go to games section
team_game_urls = []
for l in team_links:
    score_index = l.find('.')
    l = l[:score_index]
    l = 'https://www.basketball-reference.com' + l + '_games.html'
    team_game_urls.append(l)
# print(team_game_urls)

# use pandas to create a dataframe of games inlcuding scores, W-L, and date
team_url = team_game_urls[0]
data = requests.get(team_url, headers=headers)
games = pd.read_html(data.text, match="Regular Season Table")[0]

# account for duplicate header rows within the table
games = games[games['G'] != 'G'].reset_index(drop=True)

#----------------------------------------------------------------------------------------------
# set up link to go to game log and get stats for each game
game_log_urls = []
for l in team_links:
    l = l.replace(".html","/gamelog/")
    l = 'https://www.basketball-reference.com' + l
    game_log_urls.append(l)

game_log = game_log_urls[0]
data = requests.get(game_log, headers=headers)
game_stats = pd.read_html(data.text, match="2024-25 Regular Season Table")[0]

# Drop the multi-level column headers
game_stats.columns = game_stats.columns.droplevel()

# Convert Rk to numeric (this will make non-numeric values NaN)
game_stats['Rk'] = pd.to_numeric(game_stats['Rk'], errors='coerce')

# Keep only rows where Rk is a valid number (removes headers and totals)
game_stats = game_stats[game_stats['Rk'].notna()].reset_index(drop=True)

# Separate team stats from opponent stats
team_stats_only = game_stats.iloc[:, :30]
print(team_stats_only)


#----------------------------------------------------------------------------------------------

# Add game number to both dataframes
games['Game_Num'] = range(1, len(games) + 1)
team_stats_only['Game_Num'] = range(1, len(team_stats_only) + 1)

# Merge on game number instead
team_data = games.merge(
    team_stats_only[["Game_Num", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", 
                     "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF"]], 
    on="Game_Num",
    how="left"
)

print(team_data)

    