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

# SECTION 3 - MAP USER JOURNEYS USING PAGE-PATH PATTERNS AND CONDUCT PATH ANALYSIS
#now we move from session metrics to user journeys

# 3.1 build user journeys
# aggregate page paths into ordered lists for each session, e.g. /home → /product → /cart becomes ["/home", "/product", "/cart"]
paths = (
    df.groupby("session_id")["pagePath_clean"]
    .apply(list)
    .reset_index(name="page_paths")
)

paths["path_length"] = paths["page_paths"].apply(len)

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

# SECTION 4 - COMPARE CONVERTING VS. NON-CONVERTING SESSIONS TO IDENTIFY KEY DIFFERENCES IN USER BEHAVIOUR
# 4.1 joining journey data with session metrics to compare converters vs. non-converters on various dimensions
session_analysis = paths.merge(
    session_metrics[["session_id", "converted", "device", "channel",
                     "is_bounce", "has_loop", "new_user", "session_duration",
                     "depth_group"]],
    on="session_id"
)

# 4.2 comparing converting vs. non-converting sessions on numeric metrics like path length and session duration
numeric_compare = (
    session_analysis
    .groupby("converted")[["path_length", "session_duration"]]
    .agg(["mean", "median"])
)
print("\n=== Path Length & Duration: Converters vs. Non-Converters ===")
print(numeric_compare.to_string())

# 4.3 comparing converting vs. non-converting sessions on categorical metrics like device type, channel, bounce rate, looping behaviour, new vs. returning
device_conv = (
    session_metrics
    .groupby(["device", "converted"])
    .size()
    .unstack(fill_value=0)
    .assign(conversion_rate=lambda x: x[True] / (x[True] + x[False]))
    .rename(columns={False: "non_converted", True: "converted_count"})
)
print("\n=== Conversion Rate by Device ===")
print(device_conv.to_string())

channel_conv = (
    session_metrics
    .groupby(["channel", "converted"])
    .size()
    .unstack(fill_value=0)
    .assign(conversion_rate=lambda x: x[True] / (x[True] + x[False]))
    .rename(columns={False: "non_converted", True: "converted_count"})
    .sort_values("conversion_rate", ascending=False)
)
print("\n=== Conversion Rate by Channel ===")
print(channel_conv.to_string())

new_ret_conv = (
    session_metrics
    .groupby(["new_user", "converted"])
    .size()
    .unstack(fill_value=0)
    .assign(conversion_rate=lambda x: x[True] / (x[True] + x[False]))
    .rename(columns={False: "non_converted", True: "converted_count",
                     "new_user": "is_new_visitor"})
)
print("\n=== Conversion Rate: New vs. Returning Visitors ===")
print(new_ret_conv.to_string())

loop_conv = (
    session_metrics
    .groupby(["has_loop", "converted"])
    .size()
    .unstack(fill_value=0)
    .assign(conversion_rate=lambda x: x[True] / (x[True] + x[False]))
)
print("\n=== Looping Behaviour vs. Conversion ===")
print(loop_conv.to_string())

# Section 7 - Visualizations and summary statistics








