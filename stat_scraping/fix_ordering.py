import pandas as pd
from datetime import datetime

def fix_game_ordering():
    """
    Fix the ordering of games to be chronological for each team and season
    """
    
    # Read the data
    df = pd.read_csv('nba_data_with_team_names.csv')
    
    # print("Original data shape:", df.shape)
    # print("Sample of current ordering:")
    # print(df[['Team_Name', 'Season', 'Game_Number', 'Date']].head(10))
    
    # Convert Date column to datetime for proper sorting
    df['Date'] = pd.to_datetime(df['Date'], format='%a, %b %d, %Y', errors='coerce')
    
    # Sort by Team_Name, Season, and Date
    df_sorted = df.sort_values(['Team_Name', 'Season', 'Date']).reset_index(drop=True)
    
    # Reassign Game_Number to be sequential within each team/season
    df_sorted['Game_Number'] = df_sorted.groupby(['Team_Name', 'Season']).cumcount() + 1
    
    # Convert Date back to string format for consistency
    df_sorted['Date'] = df_sorted['Date'].dt.strftime('%a, %b %d, %Y')
    
    # Save the properly ordered data
    df_sorted.to_csv('nba_data_ordered.csv', index=False)
    
    # print("\n=== Fixed Ordering ===")
    # print("Sample of corrected ordering:")
    # print(df_sorted[['Team_Name', 'Season', 'Game_Number', 'Date']].head(10))
    
    # # Show ordering for a specific team
    # print("\nSample for Atlanta Hawks 2020-21 season:")
    # atlanta_2021 = df_sorted[(df_sorted['Team_Name'] == 'Atlanta Hawks') & (df_sorted['Season'] == '2020-21')]
    # print(atlanta_2021[['Team_Name', 'Season', 'Game_Number', 'Date', 'Opponent']].head(10))
    
    # print(f"\nFinal data shape: {df_sorted.shape}")
    # print(f"Teams: {df_sorted['Team_Name'].nunique()}")
    # print(f"Seasons: {df_sorted['Season'].nunique()}")
    
    return df_sorted

if __name__ == "__main__":
    df = fix_game_ordering()