import pandas as pd

#------IN PROGRESS------------------------------
# Import the function from fix_ordering.py
from fix_ordering import fix_game_ordering

from sklearn.ensemble import RandomForestClassifier

# Get the sorted dataframe by calling the function
df_sorted = fix_game_ordering()

# Convert Date column to datetime
df_sorted["Date"] = pd.to_datetime(df_sorted["Date"])

# Create new columns
df_sorted["opp_code"] = df_sorted["Opponent"].astype("category").cat.codes
# Extract just the hour from times like "8:00p", "10:30a", etc.
df_sorted["hour"] = df_sorted["Start_Time"].str.extract(r'(\d+)')[0].astype(int)
# Create a column for day of the week
df_sorted["day_code"] = df_sorted["Date"].dt.dayofweek
# Create a column for the result
df_sorted["target"] = (df_sorted["Team_Score"] > df_sorted["Opponent_Score"]).astype(int)

# rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

train = df_sorted[df_sorted["Date"] < "2025-04-13"]
print(df_sorted)