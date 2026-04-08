import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from collections import Counter
#import warnings
#warnings.filterwarnings("ignore")

# SECTION 1 - CLEAN AND PREPARE THE DATASET
# 1.1 Load the dataset
df = pd.read_csv("data.csv")

# 1.2 Convert timestamps into usable time formats
df["visitStartTime"] = pd.to_datetime(df["visitStartTime"], unit="s")
df["hit_time_seconds"] = df["hit_time_ms"] / 1000

print("\n=== Time Range ===")
print("Min visit time:", df["visitStartTime"].min())
print("Max visit time:", df["visitStartTime"].max())

# 1.3 Create unique Session ID (combining visitor ID and visit ID)
df["session_id"] = df["fullVisitorId"].astype(str) + "_" + df["visitId"].astype(str)

print("\n=== Unique Counts ===")
print("Sessions:", df["session_id"].nunique())
print("Visitors:", df["fullVisitorId"].nunique())

# 1.4 Sort page hits in the order they occurred within each session
df = df.sort_values(["session_id", "hitNumber"]).reset_index(drop=True)

# 1.5 Calculate null rates for each column to identify data quality issues
null_report = df.isnull().mean().sort_values(ascending=False)
print("=== Null Rate by Column ===")
print(null_report[null_report > 0].to_string())

# 1.5 Normalise key behavioural columns

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

# 1.6 Clean and standardize page URLs
df["pagePath_clean"] = df["pagePath"].str.split("?").str[0].str.rstrip("/").str.lower()
df["pagePath_clean"] = df["pagePath_clean"].replace("", "/") 
print("\n=== Sample Cleaned URLs ===")
print(df[["pagePath", "pagePath_clean"]].head(10))
print("\n=== Sample Session Sequence ===")
sample_session = df["session_id"].iloc[0]
print(df[df["session_id"] == sample_session][["hitNumber", "pagePath_clean"]])

# SECTION 2 - EXTRACT PAGE-LEVEL INFORMATION FROM HIT-LEVEL RECORDS
# Right now, the data set is 1 row = 1 page hit. We need to aggregate this up to 1 row = 1 session to do path analysis and compare converting vs. non-converting sessions
# 2.1 Aggregate page hits into session-level metrics
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

# 2.2 Create new columns to help with analysis
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

# SECTION 3 - MAP USER JOURNEYS USING PAGE-PATH PATTERNS
#now we move from session metrics to user journeys

# 3.1 build user journeys
# aggregate page paths into ordered lists for each session, e.g. /home → /product → /cart becomes ["/home", "/product", "/cart"]
paths = (
    df.groupby("session_id")["pagePath_clean"]
    .apply(list)
    .reset_index(name="page_paths")
)

# convert lists to tuples, count unique paths, and identify most common full paths
path_counts   = Counter(tuple(p) for p in paths["page_paths"])
common_paths  = path_counts.most_common(10)

print("\n=== Top 10 Most Common Full Paths ===")
for path, count in common_paths:
    print(f"  {count:>6,}  {' → '.join(path)}")

# 3.2 page-to-page transitions
# convert /home → /product → /cart to (/home → /product), (/product → /cart)
def get_bigrams(path):
    return list(zip(path[:-1], path[1:]))

# apply get_bigrams to all session paths and count the most common transitions
all_bigrams   = [bg for path in paths["page_paths"] for bg in get_bigrams(path)]
bigram_counts = Counter(all_bigrams)
top_bigrams   = pd.DataFrame(bigram_counts.most_common(20), columns=["transition", "count"])
top_bigrams[["from_page", "to_page"]] = pd.DataFrame(top_bigrams["transition"].tolist())
top_bigrams = top_bigrams.drop(columns="transition")

print("\n=== Top 20 Page Transitions ===")
print(top_bigrams.to_string(index=False))

# 3.3 Exit Points
exit_pages = (
    df[df["isExit"]]                #find exits
    .groupby("pagePath_clean")      #count how many times each page was the exit point
    .size()
    .reset_index(name="exit_count")
    .sort_values("exit_count", ascending=False)
    .head(20)
)

# add total hits per page to compute exit rate (not just raw count)
page_hits = df.groupby("pagePath_clean").size().reset_index(name="total_hits")
exit_pages = exit_pages.merge(page_hits, on="pagePath_clean")
exit_pages["exit_rate"] = exit_pages["exit_count"] / exit_pages["total_hits"]

print("\n=== Top 20 High-Exit Pages ===")
print(exit_pages.to_string(index=False))

# Section 4- Segment sessions by device type, traffic source, and new vs. returning visitors



# Section 5 - Conduct Path Analysis to identify common navigation sequences, looping behaviour and high-exit pages



# Section 6 - Compare converting vs. non-converting sessions to identify key differences in user behaviour



# Section 7 - Visualizations and summary statistics








