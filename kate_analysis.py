import pandas as pd
#import numpy as np
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


# Section 2 - Extracting Page-Level Information from hit-level records



# Section 3 - Map user journey's using page-path patterns



# Section 4- Segment sessions by device type, traffic source, and new vs. returning visitors



# Section 5 - Conduct Path Analysis to identify common navigation sequences, looping behaviour and high-exit pages



# Section 6 - Compare converting vs. non-converting sessions to identify key differences in user behaviour



# Section 7 - Visualizations and summary statistics








