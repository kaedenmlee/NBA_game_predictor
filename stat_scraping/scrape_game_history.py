import requests
import time
import pandas as pd

from io import StringIO
from bs4 import BeautifulSoup, Comment


def scrape_nba_data_with_team_names():
    """
    Scrape NBA data with proper team names from standings table
    """

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    # Define seasons
    seasons = [
        ("2025-26", "2026"),
        ("2024-25", "2025"),
        ("2023-24", "2024"),
        ("2022-23", "2023"),
        ("2021-22", "2022")
    ]

    all_data = []

    for season_name, year in seasons:
        print(f"\n=== Processing {season_name} season ===")

        # Get standings page
        standings_url = f"https://www.basketball-reference.com/leagues/NBA_{year}_standings.html"
        response = requests.get(standings_url, headers=headers)
        time.sleep(4)

        soup = BeautifulSoup(response.text, "html.parser")
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))

        # Find expanded standings table
        expanded_standings_table = None
        for comment in comments:
            if 'expanded_standings' in comment:
                comment_soup = BeautifulSoup(comment, "html.parser")
                expanded_standings_table = comment_soup.find(
                    'table', {'id': 'expanded_standings'})
                break

        if expanded_standings_table is None:
            print(f"No standings table found for {season_name}")
            continue

        # Extract team links and names from standings
        team_rows = expanded_standings_table.find_all(
            'tr')[1:]  # Skip header row

        team_info = []
        for row in team_rows:
            team_link = row.find('a', href=True)
            if team_link and '/teams/' in team_link.get('href', ''):
                team_url = team_link.get('href')
                team_name = team_link.get_text().strip()
                team_abbr = team_url.split('/')[-2].replace('.html', '')

                team_info.append({
                    'team_name': team_name,
                    'team_abbr': team_abbr,
                    'team_url': team_url,
                    'season': season_name
                })

        print(f"Found {len(team_info)} teams in standings")

        # Now scrape each team's data
        for i, team in enumerate(team_info):
            try:
                print(
                    f"Processing {team['team_name']} ({i+1}/{len(team_info)})")
                time.sleep(2)
                # Get games data
                games_url = f"https://www.basketball-reference.com{team['team_url'].replace('.html', '_games.html')}"
                games_response = requests.get(games_url, headers=headers)

                games = pd.read_html(
                    StringIO(games_response.text), match="Regular Season Table")[0]
                games = games[games['G'] != 'G'].reset_index(drop=True)

                # filter out future games (games with out a LOG time are not played yet)
                games = games[games['LOG'].notna()].reset_index(drop=True)

                # the game_location is unnamed on the website, so I have to manually check for this column
                location_col = None
                for col in games.columns:
                    if games[col].dtype == object:
                        if (games[col] == '@').any():
                            location_col = col
                            break

                # create Home column, 1 = Home, 0 = Away
                if location_col is not None:
                    games['Home'] = (games[location_col] != '@').astype(int)
                    # if empty, it is a home game
                    games['Home'] = games['Home'].fillna(1).astype(int)
                else:
                    games['Home'] = None

                if len(games) == 0:
                    print(f"No games found for {team['team_name']}")
                    continue

                print(f"Found {len(games)} games for {team['team_name']}")

                # Get game stats data
                game_log_url = f"https://www.basketball-reference.com{team['team_url'].replace('.html', '/gamelog/')}"
                stats_response = requests.get(game_log_url, headers=headers)

                game_stats = pd.read_html(
                    StringIO(stats_response.text), match=f"{season_name} Regular Season Table")[0]
                game_stats.columns = game_stats.columns.droplevel()
                game_stats['Rk'] = pd.to_numeric(
                    game_stats['Rk'], errors='coerce')
                game_stats = game_stats[game_stats['Rk'].notna()].reset_index(
                    drop=True)
                team_stats_only = game_stats.iloc[:, :30]

                # Merge games and stats data
                games['Game_Num'] = range(1, len(games) + 1)
                team_stats_only['Game_Num'] = range(
                    1, len(team_stats_only) + 1)

                # Select available stats columns
                stats_columns = ["Game_Num", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%",
                                 "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF"]
                available_stats_columns = [
                    col for col in stats_columns if col in team_stats_only.columns]

                team_data = games.merge(
                    team_stats_only[available_stats_columns],
                    on="Game_Num",
                    how="left"
                )

                # Add team and season identifiers
                team_data['Team_Name'] = team['team_name']
                team_data['Team_Abbr'] = team['team_abbr']
                team_data['Season'] = season_name

                all_data.append(team_data)
                print(
                    f"  Scraped {len(team_data)} games for {team['team_name']}")

            except Exception as e:
                print(f"  Error processing {team['team_name']}: {e}")
                continue

    # Combine all data
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        # Select columns to keep
        columns_to_keep = [
            'Team_Name', 'Team_Abbr', 'Season', 'G', 'Date', 'Start (ET)', 'Opponent',
            'Home', 'Tm', 'Opp', 'W', 'L', 'Streak', 'Attend.', 'LOG', 'Notes',
            'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%',
            'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF'
        ]

        df_final = combined_data[columns_to_keep].copy()

        # Convert Date column to datetime
        df_final['Date'] = pd.to_datetime(
            df_final['Date'], format='%a, %b %d, %Y', errors='coerce')
        # Sort by Team_Name, Season, and Date
        df_final = df_final.sort_values(
            ['Team_Name', 'Season', 'Date']).reset_index(drop=True)
        # Reassign Game_Number to be sequential within each team/season
        df_final['G'] = df_final.groupby(
            ['Team_Name', 'Season']).cumcount() + 1
        # Convert Date back to string format for consistency
        df_final['Date'] = df_final['Date'].dt.strftime('%a, %b %d, %Y')
        # Save the data
        df_final.to_csv('nba_scrape_data.csv', index=False)

        print(f"\n=== Final Results ===")
        print(f"Total games: {len(df_final)}")
        print(f"Teams: {df_final['Team_Name'].nunique()}")
        print(f"Seasons: {df_final['Season'].nunique()}")
        print(f"Teams: {sorted(df_final['Team_Name'].unique())}")
        print(f"Seasons: {sorted(df_final['Season'].unique())}")

        # Show sample
        print("\nSample data:")
        print(df_final.head())

        return df_final
    else:
        print("No data was scraped")
        return None


if __name__ == "__main__":
    df = scrape_nba_data_with_team_names()
