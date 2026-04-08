import pandas as pd
import numpy as np
from collections import Counter

#=============================================================================
# SECTION 1 - CLEAN AND PREPARE THE DATASET
#=============================================================================

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

# 1.6 Normalise key behavioural columns
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

# 1.7 Clean and standardize page URLs
df["pagePath_clean"] = df["pagePath"].str.split("?").str[0].str.rstrip("/").str.lower()
df["pagePath_clean"] = df["pagePath_clean"].replace("", "/") 

print("\n=== Sample Cleaned URLs ===")
print(df[["pagePath", "pagePath_clean"]].head(10))

print("\n=== Sample Session Sequence ===")
sample_session = df["session_id"].iloc[0]
print(df[df["session_id"] == sample_session][["hitNumber", "pagePath_clean"]])

# =============================================================================
# SECTION 2 - EXTRACT SESSION-LEVEL METRICS FROM HIT-LEVEL RECORDS
# =============================================================================

session_metrics = (
    df.groupby("session_id")
    .agg(
        pages_viewed      = ("pagePath_clean", "count"),
        unique_pages      = ("pagePath_clean", "nunique"),
        exit_count        = ("isExit", "sum"),
        transactions      = ("transactions", "max"),
        session_duration  = ("hit_time_seconds", "max"),
        device            = ("device_category", "first"),
        channel           = ("channel_group", "first"),
        new_user          = ("is_new_visitor", "first"),
    )
    .reset_index()
)

print("\n=== Session-Level Metrics ===")
print(session_metrics.head())

# 2.2 Derived behavioural flags
session_metrics["is_bounce"]   = session_metrics["pages_viewed"] == 1
session_metrics["converted"]   = session_metrics["transactions"] > 0
session_metrics["has_loop"]    = session_metrics["pages_viewed"] > session_metrics["unique_pages"]

session_metrics["depth_group"] = pd.cut(
    session_metrics["pages_viewed"],
    bins=[0, 1, 3, 7, np.inf],
    labels=["Bounce (1)", "Shallow (2–3)", "Medium (4–7)", "Deep (8+)"]
)

print("\n=== Session-Level Summary ===")
print(session_metrics[["pages_viewed", "session_duration", "is_bounce", "converted"]].describe())
print(f"\nOverall bounce rate:     {session_metrics['is_bounce'].mean():.1%}")
print(f"Overall conversion rate: {session_metrics['converted'].mean():.1%}")
print(f"Sessions with looping:   {session_metrics['has_loop'].mean():.1%}")

# =============================================================================
# SECTION 3 - MAP USER JOURNEYS AND PATH ANALYSIS
# =============================================================================

# 3.1 Build ordered page paths per session
paths = (
    df.groupby("session_id")["pagePath_clean"]
    .apply(list)
    .reset_index(name="page_paths")
)

paths["path_length"] = paths["page_paths"].apply(len)

# 3.2 Most common full paths
path_counts   = Counter(tuple(p) for p in paths["page_paths"])
common_paths  = path_counts.most_common(10)

print("\n=== Top 10 Most Common Full Paths ===")
for path, count in common_paths:
    print(f"  {count:>6,}  {' → '.join(path)}")

# 3.3 Page-to-page transitions (bigrams)
def get_bigrams(path):
    return list(zip(path[:-1], path[1:]))

all_bigrams   = [bg for path in paths["page_paths"] for bg in get_bigrams(path)]
bigram_counts = Counter(all_bigrams)

top_bigrams   = pd.DataFrame(bigram_counts.most_common(20), columns=["transition", "count"])
top_bigrams[["from_page", "to_page"]] = pd.DataFrame(top_bigrams["transition"].tolist())
top_bigrams = top_bigrams.drop(columns="transition")

print("\n=== Top 20 Page Transitions ===")
print(top_bigrams.to_string(index=False))

# 3.4 Transition probabilities (UPGRADE: behaviour likelihood instead of raw counts)
from_totals = (
    top_bigrams
    .groupby("from_page")["count"]
    .sum()
    .reset_index(name="total_from")
)

top_bigrams = top_bigrams.merge(from_totals, on="from_page")
top_bigrams["transition_prob"] = top_bigrams["count"] / top_bigrams["total_from"]

print("\n=== Top Transitions with Probabilities ===")
print(top_bigrams.sort_values(["from_page", "transition_prob"], ascending=[True, False]).to_string(index=False))

# 3.5 Conversion-based transitions (UPGRADE: compare converters vs non-converters)
paths_with_conv = paths.merge(
    session_metrics[["session_id", "converted"]],
    on="session_id"
)

bigram_data = []
for _, row in paths_with_conv.iterrows():
    path = row["page_paths"]
    conv = row["converted"]
    bigram_data.extend([(a, b, conv) for a, b in zip(path[:-1], path[1:])])

bigram_df = pd.DataFrame(bigram_data, columns=["from_page", "to_page", "converted"])

conv_transitions = (
    bigram_df
    .groupby(["from_page", "to_page", "converted"])
    .size()
    .reset_index(name="count")
)

print("\n=== Transitions by Conversion ===")
print(conv_transitions.head(20))

# 3.6 Entry pages (UPGRADE: where users start)
entry_pages = (
    df.sort_values(["session_id", "hitNumber"])
    .groupby("session_id")
    .first()["pagePath_clean"]
    .value_counts()
    .head(10)
)

print("\n=== Top Entry Pages ===")
print(entry_pages)

# 3.7 Exit pages and exit rate
exit_pages = (
    df[df["isExit"]]
    .groupby("pagePath_clean")
    .size()
    .reset_index(name="exit_count")
    .sort_values("exit_count", ascending=False)
    .head(20)
)

page_hits = df.groupby("pagePath_clean").size().reset_index(name="total_hits")
exit_pages = exit_pages.merge(page_hits, on="pagePath_clean")
exit_pages["exit_rate"] = exit_pages["exit_count"] / exit_pages["total_hits"]

print("\n=== Top 20 High-Exit Pages ===")
print(exit_pages.to_string(index=False))

# 3.8 Exit pages vs conversion (UPGRADE: identify problematic pages)
exit_with_conv = df.merge(
    session_metrics[["session_id", "converted"]],
    on="session_id"
)

exit_conv_rate = (
    exit_with_conv[exit_with_conv["isExit"]]
    .groupby("pagePath_clean")["converted"]
    .mean()
    .reset_index(name="conversion_rate_when_exit")
    .sort_values("conversion_rate_when_exit")
)

print("\n=== Exit Page Conversion Rates ===")
print(exit_conv_rate.head(20))

# =============================================================================
# SECTION 4 - CONVERTING VS NON-CONVERTING SESSION ANALYSIS
# =============================================================================

session_analysis = paths.merge(
    session_metrics[["session_id", "converted", "device", "channel",
                     "is_bounce", "has_loop", "new_user", "session_duration",
                     "depth_group"]],
    on="session_id"
)

# 4.1 Numeric comparison
numeric_compare = (
    session_analysis
    .groupby("converted")[["path_length", "session_duration"]]
    .agg(["mean", "median"])
)

print("\n=== Path Length & Duration: Converters vs. Non-Converters ===")
print(numeric_compare.to_string())

# 4.2 Device-level conversion
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

# 4.3 Channel-level conversion
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

# 4.4 New vs returning users
new_ret_conv = (
    session_metrics
    .groupby(["new_user", "converted"])
    .size()
    .unstack(fill_value=0)
    .assign(conversion_rate=lambda x: x[True] / (x[True] + x[False]))
    .rename(columns={False: "non_converted", True: "converted_count"})
)

print("\n=== Conversion Rate: New vs. Returning Visitors ===")
print(new_ret_conv.to_string())

# 4.5 Looping behaviour vs conversion
loop_conv = (
    session_metrics
    .groupby(["has_loop", "converted"])
    .size()
    .unstack(fill_value=0)
    .assign(conversion_rate=lambda x: x[True] / (x[True] + x[False]))
)

print("\n=== Looping Behaviour vs. Conversion ===")
print(loop_conv.to_string())