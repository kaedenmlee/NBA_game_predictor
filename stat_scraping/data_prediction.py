import pandas as pd

#------IN PROGRESS------------------------------
# Import the function from fix_ordering.py
from fix_ordering import fix_game_ordering

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

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

# print(df_sorted.head())

# Create the model; estimators = number of trees; min_samples_split = minimum number of samples required to split a node
# rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

# train = df_sorted[df_sorted["Date"] < "2025-04-13"]
# test = df_sorted[df_sorted["Date"] > "2025-04-13"]
# predictors = ["opp_code", "hour", "day_code"]
# rf.fit(train[predictors], train["target"])

# preds = rf.predict(test[predictors])

# acc = accuracy_score(test["target"], preds)

# combined = pd.DataFrame(dict(actual=test["target"],prediction=preds))

# grouped_matches = df_sorted.groupby("Team Name")
