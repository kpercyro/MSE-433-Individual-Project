"""
MSE 433 Individual Project — Quantitative UX Analysis
Google Merchandise Store — Google Analytics Sample Dataset
Kate Percy-Robb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ── Plot styling ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 150


# =============================================================================
# 1. DATA PREPARATION & CLEANING
# =============================================================================

df = pd.read_csv("data.csv")

# --- Timestamps --------------------------------------------------------------
df["visitStartTime"] = pd.to_datetime(df["visitStartTime"], unit="s")
df["hit_time_seconds"] = df["hit_time_ms"] / 1000

# --- Session ID --------------------------------------------------------------
df["session_id"] = df["fullVisitorId"].astype(str) + "_" + df["visitId"].astype(str)

# --- Sort hits within each session -------------------------------------------
df = df.sort_values(["session_id", "hitNumber"]).reset_index(drop=True)

# --- Null audit --------------------------------------------------------------
null_report = df.isnull().mean().sort_values(ascending=False)
print("=== Null Rate by Column ===")
print(null_report[null_report > 0].to_string())

# --- Normalise key columns ---------------------------------------------------
# Boolean columns arrive as True/False strings or 0/1 depending on export
df["isExit"] = df["isExit"].fillna(False).astype(bool)
df["is_new_visitor"] = df["is_new_visitor"].fillna(False).astype(bool)
df["transactions"] = pd.to_numeric(df["transactions"], errors="coerce").fillna(0)

# Sanitise page paths: strip query strings so /store?ref=X == /store
df["pagePath_clean"] = df["pagePath"].str.split("?").str[0].str.rstrip("/").str.lower()
df["pagePath_clean"] = df["pagePath_clean"].replace("", "/")   # root page


# =============================================================================
# 2. SESSION-LEVEL ENGAGEMENT & DROP-OFF METRICS
# =============================================================================

session_metrics = (
    df.groupby("session_id")
    .agg(
        pages_viewed      = ("pagePath_clean", "count"),
        unique_pages      = ("pagePath_clean", "nunique"),   # loop indicator
        exit_count        = ("isExit", "sum"),
        transactions      = ("transactions", "max"),
        session_duration  = ("hit_time_seconds", "max"),     # proxy for time-on-site
        device            = ("device_category", "first"),
        channel           = ("channel_group", "first"),
        new_user          = ("is_new_visitor", "first"),
    )
    .reset_index()
)

# --- Derived flags -----------------------------------------------------------
session_metrics["is_bounce"]  = session_metrics["pages_viewed"] == 1
session_metrics["converted"]  = session_metrics["transactions"] > 0
session_metrics["has_loop"]   = session_metrics["pages_viewed"] > session_metrics["unique_pages"]
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
# 3. PATH & JOURNEY ANALYSIS
# =============================================================================

# --- Build ordered page-path lists per session --------------------------------
paths = (
    df.groupby("session_id")["pagePath_clean"]
    .apply(list)
    .reset_index(name="page_paths")
)

# FIX: was "pathe_paths" (typo) in original code
path_counts   = Counter(tuple(p) for p in paths["page_paths"])
common_paths  = path_counts.most_common(10)

print("\n=== Top 10 Most Common Full Paths ===")
for path, count in common_paths:
    print(f"  {count:>6,}  {' → '.join(path)}")

# --- Bigrams: most common page-to-page transitions ---------------------------
def get_bigrams(path):
    return list(zip(path[:-1], path[1:]))

all_bigrams   = [bg for path in paths["page_paths"] for bg in get_bigrams(path)]
bigram_counts = Counter(all_bigrams)
top_bigrams   = pd.DataFrame(bigram_counts.most_common(20), columns=["transition", "count"])
top_bigrams[["from_page", "to_page"]] = pd.DataFrame(top_bigrams["transition"].tolist())
top_bigrams = top_bigrams.drop(columns="transition")

print("\n=== Top 20 Page Transitions ===")
print(top_bigrams.to_string(index=False))

# --- Loop detection (session-level) ------------------------------------------
# (already captured above via unique_pages; kept here for path-level detail)
def has_loop(path):
    return len(path) != len(set(path))

paths["looping"] = paths["page_paths"].apply(has_loop)
paths["path_length"] = paths["page_paths"].apply(len)

# --- High-exit pages ---------------------------------------------------------
exit_pages = (
    df[df["isExit"]]
    .groupby("pagePath_clean")
    .size()
    .reset_index(name="exit_count")
    .sort_values("exit_count", ascending=False)
    .head(20)
)

# Add total hits per page to compute exit rate (not just raw count)
page_hits = df.groupby("pagePath_clean").size().reset_index(name="total_hits")
exit_pages = exit_pages.merge(page_hits, on="pagePath_clean")
exit_pages["exit_rate"] = exit_pages["exit_count"] / exit_pages["total_hits"]

print("\n=== Top 20 High-Exit Pages ===")
print(exit_pages.to_string(index=False))


# =============================================================================
# 4. CONVERTING vs. NON-CONVERTING SESSIONS
# =============================================================================

# FIX: original code used "path" (wrong column name); correct is "page_paths"
session_analysis = paths.merge(
    session_metrics[["session_id", "converted", "device", "channel",
                     "is_bounce", "has_loop", "new_user", "session_duration",
                     "depth_group"]],
    on="session_id"
)

# --- Numeric comparison ------------------------------------------------------
numeric_compare = (
    session_analysis
    .groupby("converted")[["path_length", "session_duration"]]
    .agg(["mean", "median"])
)
print("\n=== Path Length & Duration: Converters vs. Non-Converters ===")
print(numeric_compare.to_string())

# --- Device breakdown --------------------------------------------------------
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

# --- Channel breakdown -------------------------------------------------------
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

# --- New vs. returning -------------------------------------------------------
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

# --- Looping behaviour & conversion ------------------------------------------
loop_conv = (
    session_metrics
    .groupby(["has_loop", "converted"])
    .size()
    .unstack(fill_value=0)
    .assign(conversion_rate=lambda x: x[True] / (x[True] + x[False]))
)
print("\n=== Looping Behaviour vs. Conversion ===")
print(loop_conv.to_string())


# =============================================================================
# 5. SEGMENT-SPECIFIC DROP-OFF ANALYSIS
# =============================================================================

# Bounce rate by device
bounce_device = session_metrics.groupby("device")["is_bounce"].mean().sort_values(ascending=False)
print("\n=== Bounce Rate by Device ===")
print(bounce_device.to_string())

# Bounce rate by channel
bounce_channel = session_metrics.groupby("channel")["is_bounce"].mean().sort_values(ascending=False)
print("\n=== Bounce Rate by Channel ===")
print(bounce_channel.to_string())

# Depth group distribution by device
depth_device = (
    session_metrics
    .groupby(["device", "depth_group"])
    .size()
    .unstack(fill_value=0)
)
print("\n=== Session Depth by Device ===")
print(depth_device.to_string())


# =============================================================================
# 6. VISUALISATIONS
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Google Merchandise Store — UX Analytics Dashboard", fontsize=14, fontweight="bold")

# (A) Bounce rate by device
ax = axes[0, 0]
bounce_device.plot(kind="bar", ax=ax, color=sns.color_palette("muted", len(bounce_device)))
ax.set_title("Bounce Rate by Device")
ax.set_ylabel("Bounce Rate")
ax.set_xlabel("")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.tick_params(axis="x", rotation=30)

# (B) Conversion rate by channel
ax = axes[0, 1]
channel_conv["conversion_rate"].plot(kind="barh", ax=ax, color=sns.color_palette("muted", len(channel_conv)))
ax.set_title("Conversion Rate by Channel")
ax.set_xlabel("Conversion Rate")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

# (C) Session depth distribution
ax = axes[0, 2]
depth_counts = session_metrics["depth_group"].value_counts().sort_index()
depth_counts.plot(kind="bar", ax=ax, color=sns.color_palette("muted", len(depth_counts)))
ax.set_title("Session Depth Distribution")
ax.set_ylabel("Sessions")
ax.tick_params(axis="x", rotation=30)

# (D) Path length: converters vs. non-converters
ax = axes[1, 0]
session_analysis.boxplot(column="path_length", by="converted", ax=ax,
                         showfliers=False, patch_artist=True)
ax.set_title("Path Length by Conversion")
ax.set_xlabel("Converted")
ax.set_ylabel("Pages in Path")
plt.sca(ax); plt.title("Path Length by Conversion")

# (E) Top 15 exit pages (by exit rate, min 50 hits)
ax = axes[1, 1]
top_exit = exit_pages[exit_pages["total_hits"] >= 50].nlargest(15, "exit_rate")
ax.barh(top_exit["pagePath_clean"].str[-40:], top_exit["exit_rate"],
        color=sns.color_palette("muted")[3])
ax.set_title("Top 15 Exit-Rate Pages\n(min 50 hits)")
ax.set_xlabel("Exit Rate")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.invert_yaxis()

# (F) Looping vs. non-looping conversion rate
ax = axes[1, 2]
loop_rates = loop_conv["conversion_rate"].rename(index={True: "Looping", False: "No Loop"})
loop_rates.plot(kind="bar", ax=ax, color=sns.color_palette("muted", 2))
ax.set_title("Conversion Rate:\nLooping vs. Non-Looping Sessions")
ax.set_ylabel("Conversion Rate")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
ax.tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig("ux_analytics_dashboard.png", bbox_inches="tight")
plt.show()
print("\nDashboard saved to ux_analytics_dashboard.png")
