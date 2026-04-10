# MSE 433 Individual Project: Web Analytics Analysis

## Overview

This project analyzes Google Analytics data from an e-commerce website (Google Merchandise Store) to understand user behavior, session patterns, conversion funnels, and page-level performance. The analysis provides insights into customer journeys, bounce rates, conversion rates, and transition probabilities between pages.

## Prerequisites

- Python 3.8 or higher
- Required packages:
  - pandas
  - numpy

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/kpercyro/MSE-433-Individual-Project.git
   cd MSE-433-Individual-Project
   ```

2. Install required dependencies:
   ```bash
   pip install pandas numpy
   ```

## Data Description

The analysis uses Google Analytics hit-level data (`Quantitative Analysis/data.csv`) containing:

- **Visitor Information**: `fullVisitorId`, `visitId`, `visitNumber`
- **Session Data**: `visitStartTime`, `device_category`, `channel_group`
- **Traffic Sources**: `traffic_source`, `traffic_medium`
- **Behavioral Data**: `pageviews`, `bounces`, `transactions`, `revenue_usd`
- **Hit Details**: `hitNumber`, `hit_time_ms`, `hit_type`, `pagePath`, `pageTitle`
- **Page Flow**: `isExit`, `isEntrance`, `is_new_visitor`

## Project Structure

```
MSE-433-Individual-Project/
├── README.md
└── Qualitative Analysis Documents/
    ├── MSE 433 Participant 1 - Transcript.pdf
    ├── MSE 433 Participant 2 - Transcript.pdf
    ├── MSE 433 Participant 3 - Transcript.pdf
    ├── MSE 433 Participant 4 - Transcript.pdf
    ├── MSE 433 Qualitative Analysis.xlsx
    ├── MSE 433 Qualitative Interview Script.pdf
└── Quantitative Analysis/
    ├── data.csv                      (Google Analytics dataset)
    ├── quantitative_analysis.py      (Main analysis script)
    ├── ux_analysis_output.xlsx       (Generated - Analysis results)
    ├── conversion_rate_by_channel.png
    ├── conversion_rate_by_device.png
    ├── path_length_by_conversion.png
    ├── session_depth_distribution.png
    ├── top_exit_pages.png
    └── looping_vs_conversion.png
```

## Usage

Navigate to the Quantitative Analysis folder and run the analysis script:

```bash
cd "Quantitative Analysis"
python quantitative_analysis.py
```

The script will process the data and output various analytics including:

### Key Metrics Generated

1. **Data Quality Assessment**
   - Time range of the dataset
   - Null rates by column
   - Data distributions

2. **Session-Level Analysis**
   - Unique sessions and visitors
   - Pages viewed per session
   - Session duration
   - Bounce rate and conversion rate
   - Device and channel breakdowns

3. **User Journey Mapping**
   - Most common full user paths
   - Page-to-page transition analysis
   - Transition probabilities
   - Entry and exit page analysis

4. **Conversion Analysis**
   - Transaction patterns
   - Conversion funnel analysis
   - Comparative analysis between converters and non-converters

## Analysis Sections

### Section 1: Data Cleaning and Preparation
- Converts timestamps to datetime format
- Creates unique session IDs
- Handles missing values and data normalization
- Standardizes page URLs

### Section 2: Session-Level Metrics
- Aggregates hit-level data to session level
- Calculates behavioral flags (bounce, conversion, looping)
- Creates session depth groupings

### Section 3: Path Analysis and User Journeys
- Maps complete user paths through the website
- Analyzes page transitions and probabilities
- Identifies high-exit pages and entry points
- Compares behavior patterns between converted and non-converted sessions

## Output

The script prints comprehensive analytics to the console, including:

- Dataset summary statistics
- Top user paths and transitions
- Conversion rates and bounce analysis
- Page performance metrics
- Behavioral segmentation insights

## Key Insights

The analysis reveals:
- Most common user journey patterns
- High-performing vs. underperforming pages
- Conversion bottlenecks
- Traffic source effectiveness
- Device and channel preferences
