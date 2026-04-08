import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#from collections import Counter
#import warnings
#warnings.filterwarnings("ignore")

# SECTION 1 - CLEAN AND PREPARE THE DATASET
#Load the dataset
df = pd.read_csv("data.csv")

# Convert timestamps into usable time formats
df["visitStartTime"] = pd.to_datetime(df["visitStartTime"], unit="s")
df["hit_time_seconds"] = df["hit_time_ms"] / 1000

print("\n=== Time Range ===")
print("Min visit time:", df["visitStartTime"].min())
print("Max visit time:", df["visitStartTime"].max())

# Create unique Session ID (combining visitor ID and visit ID)
df["session_id"] = df["fullVisitorId"].astype(str) + "_" + df["visitId"].astype(str)

print("\n=== Unique Counts ===")
print("Sessions:", df["session_id"].nunique())
print("Visitors:", df["fullVisitorId"].nunique())

# Sort page hits in the order they occurred within each session
df = df.sort_values(["session_id", "hitNumber"]).reset_index(drop=True)

# Calculate null rates for each column to identify data quality issues
null_report = df.isnull().mean().sort_values(ascending=False)
print("=== Null Rate by Column ===")
print(null_report[null_report > 0].to_string())

# Normalise key behavioural columns
# Boolean columns arrive as True/False strings or 0/1 depending on export
df["isExit"] = df["isExit"].fillna(False).astype(bool)

print("\n=== isExit Distribution ===")
print(df["isExit"].value_counts())

df["is_new_visitor"] = df["is_new_visitor"].fillna(False).astype(bool)

print("\n=== New Visitor Distribution ===")
print(df["is_new_visitor"].value_counts())

df["transactions"] = pd.to_numeric(df["transactions"], errors="coerce").fillna(0)

print("\n=== Transactions Summary ===")
print(df["transactions"].describe())
print("Non-zero transactions:", (df["transactions"] > 0).sum())

# Clean and standardize page URLs
df["pagePath_clean"] = df["pagePath"].str.split("?").str[0].str.rstrip("/").str.lower()
df["pagePath_clean"] = df["pagePath_clean"].replace("", "/") 

print("\n=== Sample Cleaned URLs ===")
print(df[["pagePath", "pagePath_clean"]].head(10))

print("\n=== Sample Session Sequence ===")
sample_session = df["session_id"].iloc[0]
print(df[df["session_id"] == sample_session][["hitNumber", "pagePath_clean"]])

# SECTION 2 - EXTRACT PAGE-LEVEL INFORMATION FROM HIT-LEVEL RECORDS
# Right now, the data set is 1 row = 1 page hit. We need to aggregate this up to 1 row = 1 session to do path analysis and compare converting vs. non-converting sessions
session_metrics = (
    df.groupby("session_id") #group all page hits that belong to the same session together
    .agg(
        pages_viewed      = ("pagePath_clean", "count"),     # total page hits in session
        unique_pages      = ("pagePath_clean", "nunique"),   # loop indicator
        exit_count        = ("isExit", "sum"),               # number of exits in session (should be 1)
        transactions      = ("transactions", "max"),         # whether session converted (max of transactions in session)
        session_duration  = ("hit_time_seconds", "max"),     # approximate total session time (assumes last hit time is end of session)
        device            = ("device_category", "first"),    # device type for session (assumes consistent device category within session)
        channel           = ("channel_group", "first"),      # traffic source for session (assumes consistent channel group within session)
        new_user          = ("is_new_visitor", "first"),     # whether session is from new visitor (assumes consistent visitor type within session)
    )
    .reset_index()
)

print("\n=== Session-Level Metrics")
print(session_metrics.head())

#now we have a session-level dataset with one row per session, we can create some new columns to help with our analysis
session_metrics["is_bounce"]   = session_metrics["pages_viewed"] == 1 # user only visited one page and left
session_metrics["converted"]  = session_metrics["transactions"] > 0 # user completed a transaction
session_metrics["has_loop"]   = session_metrics["pages_viewed"] > session_metrics["unique_pages"] # user visited at least one page more than once, indicating looping behaviour
session_metrics["depth_group"] = pd.cut(
    session_metrics["pages_viewed"],
    bins=[0, 1, 3, 7, np.inf],
    labels=["Bounce (1)", "Shallow (2–3)", "Medium (4–7)", "Deep (8+)"]
) #bucket sessions into groups based on how many pages they viewed to indicate engagement level

print("\n=== Session-Level Summary ===")
print(session_metrics[["pages_viewed", "session_duration", "is_bounce", "converted"]].describe())
print(f"\nOverall bounce rate:     {session_metrics['is_bounce'].mean():.1%}")
print(f"Overall conversion rate: {session_metrics['converted'].mean():.1%}")
print(f"Sessions with looping:   {session_metrics['has_loop'].mean():.1%}")

# Section 3 - Map user journey's using page-path patterns



# Section 4- Segment sessions by device type, traffic source, and new vs. returning visitors



# Section 5 - Conduct Path Analysis to identify common navigation sequences, looping behaviour and high-exit pages



# Section 6 - Compare converting vs. non-converting sessions to identify key differences in user behaviour



# Section 7 - Visualizations and summary statistics








