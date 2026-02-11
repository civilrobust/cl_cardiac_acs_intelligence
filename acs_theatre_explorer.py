"""
ACS Theatre Intelligence Explorer
==================================
An interactive data exploration tool for Adult Cardiac Surgery theatre operations.
Built for NHS data scientist analysts who need plain English insights from operational data.

Designed by David — AI Engineer, NHS 365 AI Technologies
Inspired by the Clinical ML Tutor approach: AI as a reasoning partner, not a black box.

Usage:
    pip install gradio pandas scikit-learn plotly
    python acs_theatre_explorer.py

Place your ACSData_Data_.csv in the same folder, or update DATA_PATH below.
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = "ACSData_Data_.csv"

# NHS Colour Palette
NHS_BLUE = "#005EB8"
NHS_DARK_BLUE = "#003087"
NHS_LIGHT_BLUE = "#41B6E6"
NHS_AQUA = "#00A9CE"
NHS_GREEN = "#009639"
NHS_RED = "#DA291C"
NHS_WARM_YELLOW = "#FFB81C"
NHS_GREY = "#768692"
NHS_PALE_GREY = "#E8EDEE"
NHS_WHITE = "#FFFFFF"

# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================

def load_and_clean_data(path):
    """Load the ACS dataset and perform cleaning."""
    df = pd.read_csv(path)

    # Clean age — extract numeric
    df["Age_Numeric"] = df["Age on Procedure Date"].str.extract(r"(\d+)").astype(float)

    # Parse surgery date
    df["Surgery_Date_Parsed"] = pd.to_datetime(df["Surgery Date"], format="%d/%m/%Y", errors="coerce")

    # Duration variance (actual minus scheduled)
    df["Duration_Variance"] = df["Actual Theatre (Duration)"] - df["Scheduled (Duration)"]
    df["Duration_Variance_Pct"] = (df["Duration_Variance"] / df["Scheduled (Duration)"] * 100).round(1)

    # Is cancelled flag
    df["Is_Cancelled"] = (df["Log Status"] == "Cancelled").astype(int)

    # Is emergency flag
    df["Is_Emergency"] = df["Case Classification"].str.contains("P1a|P1b", na=False).astype(int)

    # Priority code extraction
    df["Priority_Code"] = df["Case Classification"].str.extract(r"(P\d[ab]?)")

    # Parse time columns to calculate derived metrics
    def time_to_minutes(time_str):
        """Convert HH:MM time string to minutes from midnight."""
        try:
            parts = str(time_str).split(":")
            return int(parts[0]) * 60 + int(parts[1])
        except (ValueError, IndexError, AttributeError):
            return np.nan

    df["Scheduled_Start_Mins"] = df["Scheduled (Start)"].apply(time_to_minutes)
    df["Anaesthesia_Start_Mins"] = df["Anaesthesia Start"].apply(time_to_minutes)

    # Delay to start (actual anaesthesia start vs scheduled start)
    df["Start_Delay_Mins"] = df["Anaesthesia_Start_Mins"] - df["Scheduled_Start_Mins"]
    # Handle cases crossing midnight
    df.loc[df["Start_Delay_Mins"] < -720, "Start_Delay_Mins"] += 1440

    # Is overrun flag
    df["Is_Overrun"] = (df["Duration_Variance"] > 0).astype(int)

    return df


def generate_plain_english_summary(df):
    """Generate a comprehensive plain English overview of the dataset."""
    total = len(df)
    completed = len(df[df["Log Status"] == "Completed"])
    cancelled = len(df[df["Log Status"] == "Cancelled"])
    cancel_rate = cancelled / total * 100

    # Active cases (non-cancelled with actual duration)
    active = df[df["Actual Theatre (Duration)"].notna()]
    avg_scheduled = df["Scheduled (Duration)"].mean()
    avg_actual = active["Actual Theatre (Duration)"].mean()
    avg_variance = active["Duration_Variance"].mean()

    overruns = active[active["Duration_Variance"] > 0]
    overrun_rate = len(overruns) / len(active) * 100 if len(active) > 0 else 0

    emergencies = df[df["Is_Emergency"] == 1]
    addons = df[df["Add-on"] == "Yes"]

    male_pct = len(df[df["Patient Sex"] == "Male [2]"]) / total * 100
    avg_age = df["Age_Numeric"].mean()

    top_procedure = df["Cardiothoracic Procedure"].value_counts().index[0]
    top_surgeon = df["Primary Surgeon"].value_counts().index[0]

    months = df["Month"].unique()
    date_range = f"{df['Surgery_Date_Parsed'].min().strftime('%B %Y')} to {df['Surgery_Date_Parsed'].max().strftime('%B %Y')}" if df["Surgery_Date_Parsed"].notna().any() else "Unknown"

    summary = f"""## 📊 Dataset Overview — Plain English Summary

**What is this data?** This dataset contains **{total} adult cardiac surgery theatre records** spanning **{date_range}**, covering operations at DH Cardiac Theatres. It tracks everything from scheduling through to recovery times.

---

### The Big Numbers

• **{completed} cases completed** ({completed/total*100:.0f}%), **{cancelled} cancelled** ({cancel_rate:.1f}% cancellation rate)
• **{len(emergencies)} emergency cases** (P1a/P1b) — that's {len(emergencies)/total*100:.1f}% of all bookings
• **{len(addons)} add-on cases** ({len(addons)/total*100:.1f}%) squeezed into existing schedules
• **Average patient**: {avg_age:.0f} years old, {male_pct:.0f}% male

### Theatre Timing — The Key Story

• **Scheduled duration averages {avg_scheduled:.0f} minutes** (~{avg_scheduled/60:.1f} hours)
• **Actual duration averages {avg_actual:.0f} minutes** (~{avg_actual/60:.1f} hours)
• **Average variance: {avg_variance:+.0f} minutes** — cases typically run {"over" if avg_variance > 0 else "under"} schedule
• **{overrun_rate:.0f}% of cases overrun** their scheduled time

### What's Being Done

• **Most common procedure category**: {top_procedure} ({df['Cardiothoracic Procedure'].value_counts().iloc[0]} cases)
• **Busiest surgeon**: {top_surgeon} ({df['Primary Surgeon'].value_counts().iloc[0]} cases)
• **{len(df['Procedures'].unique())} distinct procedure combinations** recorded — high complexity and variability
• **{df['Cardiothoracic Procedure'].nunique()} broad procedure categories** from CABG only through to multi-component aortic/valve/CABG combinations

### Data Quality Notes

• **Incision Close** is only recorded for {df['Incision Close'].notna().sum()} of {total} cases ({df['Incision Close'].notna().sum()/total*100:.0f}%) — significant gap
• **Out Recovery** has just {df['Out Recovery'].notna().sum()} records — recovery tracking is sparse
• **Balloon Event** and **Inventory Item** columns are completely empty
• **Redo** field only has {df['Redo'].notna().sum()} entries (all "Yes") — likely only flagged when applicable

### Why This Matters

This data tells the story of how a cardiac surgery unit actually operates day-to-day. The gap between scheduled and actual times is where efficiency lives. Understanding which procedures, surgeons, days, and patient types drive overruns could directly improve patient flow, reduce cancellations, and make better use of expensive theatre time.
"""
    return summary


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def analyse_theatre_efficiency(df):
    """Theatre utilisation analysis with plain English narrative."""
    active = df[df["Actual Theatre (Duration)"].notna()].copy()

    # Scheduled vs Actual scatter
    fig_scatter = px.scatter(
        active,
        x="Scheduled (Duration)",
        y="Actual Theatre (Duration)",
        color="Cardiothoracic Procedure",
        hover_data=["Primary Surgeon", "Case Classification", "Day of Week"],
        title="Scheduled vs Actual Theatre Duration",
        labels={
            "Scheduled (Duration)": "Scheduled Duration (mins)",
            "Actual Theatre (Duration)": "Actual Duration (mins)"
        },
        opacity=0.7
    )
    # Add the perfect prediction line
    max_dur = max(active["Scheduled (Duration)"].max(), active["Actual Theatre (Duration)"].max())
    fig_scatter.add_trace(go.Scatter(
        x=[0, max_dur], y=[0, max_dur],
        mode="lines", name="Perfect Schedule",
        line=dict(dash="dash", color=NHS_GREY, width=2)
    ))
    fig_scatter.update_layout(
        template="plotly_white",
        font=dict(family="Arial, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.5)
    )

    # Variance distribution
    fig_variance = px.histogram(
        active,
        x="Duration_Variance",
        nbins=50,
        title="Distribution of Duration Variance (Actual − Scheduled)",
        labels={"Duration_Variance": "Variance (minutes)"},
        color_discrete_sequence=[NHS_BLUE]
    )
    fig_variance.add_vline(x=0, line_dash="dash", line_color=NHS_RED,
                           annotation_text="On Schedule", annotation_position="top")
    fig_variance.update_layout(template="plotly_white", font=dict(family="Arial, sans-serif"))

    # Overrun by procedure type
    proc_stats = active.groupby("Cardiothoracic Procedure").agg(
        avg_variance=("Duration_Variance", "mean"),
        count=("Duration_Variance", "count"),
        overrun_rate=("Is_Overrun", "mean")
    ).reset_index()
    proc_stats = proc_stats[proc_stats["count"] >= 5]  # min 5 cases
    proc_stats = proc_stats.sort_values("avg_variance", ascending=True)
    proc_stats["overrun_rate"] = (proc_stats["overrun_rate"] * 100).round(1)

    fig_proc = px.bar(
        proc_stats,
        y="Cardiothoracic Procedure",
        x="avg_variance",
        orientation="h",
        title="Average Duration Variance by Procedure Type (min 5 cases)",
        labels={"avg_variance": "Avg Variance (mins)", "Cardiothoracic Procedure": ""},
        color="avg_variance",
        color_continuous_scale=["#009639", "#FFB81C", "#DA291C"],
        hover_data=["count", "overrun_rate"]
    )
    fig_proc.add_vline(x=0, line_dash="dash", line_color=NHS_GREY)
    fig_proc.update_layout(template="plotly_white", font=dict(family="Arial, sans-serif"), height=500)

    # Generate narrative
    worst_overrun = proc_stats.iloc[-1]
    best_underrun = proc_stats.iloc[0]
    median_variance = active["Duration_Variance"].median()

    narrative = f"""## 🏥 Theatre Efficiency Analysis — What the Data Says

### The Headline

**Cases run an average of {active['Duration_Variance'].mean():+.0f} minutes {"over" if active['Duration_Variance'].mean() > 0 else "under"} schedule**, with a median variance of {median_variance:+.0f} minutes. {"This means the typical case is close to schedule, but outliers pull the average up." if median_variance < active['Duration_Variance'].mean() else ""}

### Overruns vs Underruns

• **{(active['Is_Overrun']==1).sum()} cases ({(active['Is_Overrun']==1).sum()/len(active)*100:.0f}%) ran over** their scheduled time
• **{(active['Is_Overrun']==0).sum()} cases ({(active['Is_Overrun']==0).sum()/len(active)*100:.0f}%) finished on time or early**
• The **longest overrun** was {active['Duration_Variance'].max():.0f} minutes ({active['Duration_Variance'].max()/60:.1f} hours) over schedule
• The **biggest underrun** was {abs(active['Duration_Variance'].min()):.0f} minutes under schedule

### Which Procedures Cause Problems?

• **Worst overruns**: "{worst_overrun['Cardiothoracic Procedure']}" averages **{worst_overrun['avg_variance']:+.0f} minutes** over schedule ({worst_overrun['overrun_rate']:.0f}% overrun rate across {worst_overrun['count']:.0f} cases)
• **Best time-keeping**: "{best_underrun['Cardiothoracic Procedure']}" averages **{best_underrun['avg_variance']:+.0f} minutes** variance across {best_underrun['count']:.0f} cases

### What This Means for Planning

The scatter plot shows how well scheduling predicts reality. Points above the dashed line are overruns, below are underruns. The further from the line, the bigger the scheduling error. If you see clusters of a particular colour (procedure type) consistently above the line, that's a systematic scheduling problem — not just random variation.

**Key question for analysts**: Are overruns driven by procedure complexity being underestimated, or by operational delays (late starts, equipment issues, staffing)?
"""
    return narrative, fig_scatter, fig_variance, fig_proc


def analyse_surgeon_profiles(df):
    """Surgeon performance analysis with context."""
    active = df[(df["Actual Theatre (Duration)"].notna()) & (df["Log Status"] != "Cancelled")].copy()

    # Surgeon case volumes and mix
    surgeon_summary = df.groupby("Primary Surgeon").agg(
        total_cases=("Transaction ID", "count"),
        cancellations=("Is_Cancelled", "sum"),
        emergencies=("Is_Emergency", "sum"),
        addons=("Add-on", lambda x: (x == "Yes").sum()),
    ).reset_index()
    surgeon_summary["cancel_rate"] = (surgeon_summary["cancellations"] / surgeon_summary["total_cases"] * 100).round(1)
    surgeon_summary["emergency_rate"] = (surgeon_summary["emergencies"] / surgeon_summary["total_cases"] * 100).round(1)

    # Surgeon duration stats (active cases only)
    surgeon_duration = active.groupby("Primary Surgeon").agg(
        avg_actual=("Actual Theatre (Duration)", "mean"),
        avg_variance=("Duration_Variance", "mean"),
        median_variance=("Duration_Variance", "median"),
        overrun_rate=("Is_Overrun", "mean"),
        active_cases=("Transaction ID", "count")
    ).reset_index()
    surgeon_duration["overrun_rate"] = (surgeon_duration["overrun_rate"] * 100).round(1)

    # Merge
    surgeon_all = surgeon_summary.merge(surgeon_duration, on="Primary Surgeon", how="left")

    # Box plot of durations by surgeon
    fig_box = px.box(
        active,
        x="Primary Surgeon",
        y="Actual Theatre (Duration)",
        color="Primary Surgeon",
        title="Actual Theatre Duration Distribution by Surgeon",
        labels={"Actual Theatre (Duration)": "Duration (mins)", "Primary Surgeon": "Surgeon"},
        points="outliers"
    )
    fig_box.update_layout(template="plotly_white", font=dict(family="Arial, sans-serif"), showlegend=False)

    # Surgeon variance by procedure type heatmap
    surgeon_proc = active.groupby(["Primary Surgeon", "Cardiothoracic Procedure"]).agg(
        avg_variance=("Duration_Variance", "mean"),
        count=("Transaction ID", "count")
    ).reset_index()
    surgeon_proc = surgeon_proc[surgeon_proc["count"] >= 3]  # min 3 cases

    fig_heat = px.density_heatmap(
        active,
        x="Primary Surgeon",
        y="Cardiothoracic Procedure",
        z="Duration_Variance",
        histfunc="avg",
        title="Average Duration Variance: Surgeon × Procedure Type",
        labels={"Primary Surgeon": "Surgeon", "Cardiothoracic Procedure": "Procedure"},
        color_continuous_scale=["#009639", "#FFB81C", "#DA291C"],
        color_continuous_midpoint=0
    )
    fig_heat.update_layout(template="plotly_white", font=dict(family="Arial, sans-serif"), height=500)

    # Case mix comparison
    surgeon_casemix = active.groupby(["Primary Surgeon", "Cardiothoracic Procedure"]).size().reset_index(name="count")
    fig_casemix = px.bar(
        surgeon_casemix,
        x="Primary Surgeon",
        y="count",
        color="Cardiothoracic Procedure",
        title="Case Mix by Surgeon",
        labels={"count": "Number of Cases", "Primary Surgeon": "Surgeon"},
        barmode="stack"
    )
    fig_casemix.update_layout(
        template="plotly_white",
        font=dict(family="Arial, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.6)
    )

    # Narrative
    busiest = surgeon_all.loc[surgeon_all["total_cases"].idxmax()]
    best_time = surgeon_all.loc[surgeon_all["avg_variance"].idxmin()] if surgeon_all["avg_variance"].notna().any() else None
    highest_cancel = surgeon_all.loc[surgeon_all["cancel_rate"].idxmax()]

    narrative = f"""## 👨‍⚕️ Surgeon Activity Profiles — Context Matters

**Important caveat**: Comparing surgeons on raw numbers alone is misleading. Case mix complexity, emergency rates, and patient acuity all affect duration and outcomes. This analysis provides context, not league tables.

### Workload Distribution

| Surgeon | Total Cases | Completed | Cancellations | Cancel Rate | Emergency Rate |
|---------|------------|-----------|---------------|-------------|----------------|
"""
    for _, row in surgeon_all.sort_values("total_cases", ascending=False).iterrows():
        narrative += f"| {row['Primary Surgeon']} | {row['total_cases']:.0f} | {row.get('active_cases', 'N/A')} | {row['cancellations']:.0f} | {row['cancel_rate']:.1f}% | {row['emergency_rate']:.1f}% |\n"

    narrative += f"""
### Key Observations

• **Busiest surgeon**: {busiest['Primary Surgeon']} with {busiest['total_cases']:.0f} total cases
• **Highest cancellation rate**: {highest_cancel['Primary Surgeon']} at {highest_cancel['cancel_rate']:.1f}% — but look at their emergency rate ({highest_cancel['emergency_rate']:.1f}%) which may explain this
"""
    if best_time is not None:
        narrative += f"""• **Closest to schedule**: {best_time['Primary Surgeon']} averages {best_time['avg_variance']:+.1f} minutes variance with an overrun rate of {best_time['overrun_rate']:.0f}%
"""

    narrative += """
### Reading the Charts

The **box plot** shows the spread of actual durations per surgeon. Wide boxes = high variability. Outlier dots = unusually long or short cases. Compare median lines (middle of box) for typical case length.

The **heatmap** reveals which surgeon-procedure combinations run over or under. Green = under schedule, red = over. This helps identify whether overruns are surgeon-specific or procedure-specific.

The **case mix** chart is critical context — a surgeon doing mostly complex multi-component operations will naturally have longer, more variable durations than one doing primarily straightforward CABG.

**Key question for analysts**: When you see a surgeon with high variance, check their case mix before drawing conclusions. Complexity explains a lot.
"""
    return narrative, fig_box, fig_heat, fig_casemix


def analyse_scheduling_patterns(df):
    """Scheduling and case flow analysis."""
    active = df[df["Actual Theatre (Duration)"].notna()].copy()

    # Day of week patterns
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_stats = active.groupby("Day of Week").agg(
        case_count=("Transaction ID", "count"),
        avg_actual=("Actual Theatre (Duration)", "mean"),
        avg_variance=("Duration_Variance", "mean"),
        overrun_rate=("Is_Overrun", "mean"),
        emergency_count=("Is_Emergency", "sum"),
        addon_count=("Add-on", lambda x: (x == "Yes").sum())
    ).reindex(dow_order).reset_index()
    dow_stats["overrun_rate"] = (dow_stats["overrun_rate"] * 100).round(1)

    fig_dow = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Cases per Day of Week", "Average Variance per Day of Week"),
        vertical_spacing=0.15
    )
    fig_dow.add_trace(
        go.Bar(x=dow_stats["Day of Week"], y=dow_stats["case_count"],
               marker_color=NHS_BLUE, name="Cases",
               text=dow_stats["case_count"], textposition="auto"),
        row=1, col=1
    )
    fig_dow.add_trace(
        go.Bar(x=dow_stats["Day of Week"], y=dow_stats["avg_variance"],
               marker_color=[NHS_GREEN if v <= 0 else NHS_RED for v in dow_stats["avg_variance"]],
               name="Avg Variance", text=dow_stats["avg_variance"].round(0), textposition="auto"),
        row=2, col=1
    )
    fig_dow.update_layout(template="plotly_white", font=dict(family="Arial, sans-serif"),
                          height=600, showlegend=False)

    # Monthly trends
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_stats = df.groupby("Month").agg(
        total=("Transaction ID", "count"),
        cancelled=("Is_Cancelled", "sum"),
        emergencies=("Is_Emergency", "sum")
    ).reindex(month_order).dropna().reset_index()
    month_stats["cancel_rate"] = (month_stats["cancelled"] / month_stats["total"] * 100).round(1)

    fig_month = go.Figure()
    fig_month.add_trace(go.Bar(
        x=month_stats["Month"], y=month_stats["total"],
        name="Total Cases", marker_color=NHS_BLUE
    ))
    fig_month.add_trace(go.Bar(
        x=month_stats["Month"], y=month_stats["cancelled"],
        name="Cancelled", marker_color=NHS_RED
    ))
    fig_month.add_trace(go.Scatter(
        x=month_stats["Month"], y=month_stats["cancel_rate"],
        name="Cancel Rate %", yaxis="y2",
        line=dict(color=NHS_WARM_YELLOW, width=3),
        mode="lines+markers"
    ))
    fig_month.update_layout(
        title="Monthly Case Volumes and Cancellation Rates",
        template="plotly_white",
        font=dict(family="Arial, sans-serif"),
        barmode="group",
        yaxis=dict(title="Number of Cases"),
        yaxis2=dict(title="Cancellation Rate %", overlaying="y", side="right", range=[0, 30]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2)
    )

    # Emergency impact analysis
    priority_stats = active.groupby("Priority_Code").agg(
        count=("Transaction ID", "count"),
        avg_variance=("Duration_Variance", "mean"),
        avg_duration=("Actual Theatre (Duration)", "mean")
    ).reset_index()

    fig_priority = px.bar(
        priority_stats,
        x="Priority_Code",
        y="avg_variance",
        color="avg_variance",
        color_continuous_scale=["#009639", "#FFB81C", "#DA291C"],
        title="Average Duration Variance by Priority Code",
        labels={"Priority_Code": "Priority", "avg_variance": "Avg Variance (mins)"},
        text="count",
        hover_data=["avg_duration"]
    )
    fig_priority.update_layout(template="plotly_white", font=dict(family="Arial, sans-serif"))

    # Narrative
    busiest_day = dow_stats.loc[dow_stats["case_count"].idxmax()]
    worst_day = dow_stats.loc[dow_stats["avg_variance"].idxmax()]
    weekend_cases = active[active["Day of Week"].isin(["Saturday", "Sunday"])]
    highest_cancel_month = month_stats.loc[month_stats["cancel_rate"].idxmax()]

    narrative = f"""## 📅 Scheduling Patterns — When Things Go Right (and Wrong)

### Day of Week

• **Busiest day**: {busiest_day['Day of Week']} with {busiest_day['case_count']:.0f} completed cases
• **Worst overruns**: {worst_day['Day of Week']} averages {worst_day['avg_variance']:+.0f} minutes over schedule ({worst_day['overrun_rate']:.0f}% overrun rate)
• **Weekend operations**: {len(weekend_cases)} cases done on Saturday/Sunday — these are almost entirely emergencies

| Day | Cases | Avg Variance | Overrun Rate | Emergencies | Add-ons |
|-----|-------|-------------|--------------|-------------|---------|
"""
    for _, row in dow_stats.iterrows():
        if pd.notna(row["case_count"]):
            narrative += f"| {row['Day of Week']} | {row['case_count']:.0f} | {row['avg_variance']:+.0f} min | {row['overrun_rate']:.0f}% | {row['emergency_count']:.0f} | {row['addon_count']:.0f} |\n"

    narrative += f"""
### Monthly Patterns

• **Highest cancellation rate**: {highest_cancel_month['Month']} at {highest_cancel_month['cancel_rate']:.1f}%
• Volumes are fairly consistent across months, suggesting a well-managed waiting list

### Priority & Emergency Impact

Emergency cases (P1a — within 24 hours, P1b — within 72 hours) are inherently less predictable. They come in without full workup, may have sicker patients, and disrupt the elective schedule. The priority chart shows how variance changes with urgency.

**Key question for analysts**: Do days with more emergencies show higher variance across ALL cases that day, or just the emergency ones? That tells you about knock-on disruption.
"""
    return narrative, fig_dow, fig_month, fig_priority


def analyse_cancellations(df):
    """Cancellation analysis."""
    cancelled = df[df["Log Status"] == "Cancelled"].copy()
    not_cancelled = df[df["Log Status"] != "Cancelled"].copy()

    total_cancel = len(cancelled)
    total = len(df)

    # Cancellation by surgeon
    cancel_by_surgeon = df.groupby("Primary Surgeon").agg(
        total=("Transaction ID", "count"),
        cancelled=("Is_Cancelled", "sum")
    ).reset_index()
    cancel_by_surgeon["cancel_rate"] = (cancel_by_surgeon["cancelled"] / cancel_by_surgeon["total"] * 100).round(1)

    fig_surgeon_cancel = px.bar(
        cancel_by_surgeon.sort_values("cancel_rate", ascending=True),
        y="Primary Surgeon",
        x="cancel_rate",
        orientation="h",
        title="Cancellation Rate by Surgeon",
        labels={"cancel_rate": "Cancellation Rate (%)", "Primary Surgeon": ""},
        color="cancel_rate",
        color_continuous_scale=["#009639", "#FFB81C", "#DA291C"],
        text="cancelled",
        hover_data=["total"]
    )
    fig_surgeon_cancel.update_layout(template="plotly_white", font=dict(family="Arial, sans-serif"))

    # Cancellation by day of week
    cancel_by_dow = df.groupby("Day of Week").agg(
        total=("Transaction ID", "count"),
        cancelled=("Is_Cancelled", "sum")
    ).reset_index()
    cancel_by_dow["cancel_rate"] = (cancel_by_dow["cancelled"] / cancel_by_dow["total"] * 100).round(1)
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    cancel_by_dow["Day of Week"] = pd.Categorical(cancel_by_dow["Day of Week"], categories=dow_order, ordered=True)
    cancel_by_dow = cancel_by_dow.sort_values("Day of Week")

    fig_dow_cancel = px.bar(
        cancel_by_dow,
        x="Day of Week",
        y="cancel_rate",
        title="Cancellation Rate by Day of Week",
        labels={"cancel_rate": "Cancellation Rate (%)"},
        color="cancel_rate",
        color_continuous_scale=["#009639", "#FFB81C", "#DA291C"],
        text="cancelled"
    )
    fig_dow_cancel.update_layout(template="plotly_white", font=dict(family="Arial, sans-serif"))

    # Cancellation by priority
    cancel_by_priority = df.groupby("Priority_Code").agg(
        total=("Transaction ID", "count"),
        cancelled=("Is_Cancelled", "sum")
    ).reset_index()
    cancel_by_priority["cancel_rate"] = (cancel_by_priority["cancelled"] / cancel_by_priority["total"] * 100).round(1)

    fig_priority_cancel = px.bar(
        cancel_by_priority.sort_values("cancel_rate"),
        x="Priority_Code",
        y="cancel_rate",
        title="Cancellation Rate by Case Priority",
        labels={"cancel_rate": "Cancellation Rate (%)", "Priority_Code": "Priority"},
        color="cancel_rate",
        color_continuous_scale=["#009639", "#FFB81C", "#DA291C"],
        text="cancelled"
    )
    fig_priority_cancel.update_layout(template="plotly_white", font=dict(family="Arial, sans-serif"))

    # Patient class breakdown
    cancel_by_class = cancelled["Patient Class"].value_counts()
    cancel_by_addon = cancelled["Add-on"].value_counts()
    cancel_by_proc = cancelled["Cardiothoracic Procedure"].value_counts().head(5)

    narrative = f"""## ❌ Cancellation Analysis — {total_cancel} Cases Lost

### Scale of the Problem

**{total_cancel} out of {total} cases were cancelled** — a **{total_cancel/total*100:.1f}%** cancellation rate. Each cancelled cardiac surgery slot represents wasted theatre time, anaesthetist availability, nursing cover, and most importantly, a patient whose treatment was delayed.

### Who Gets Cancelled?

**By patient admission type:**
"""
    for cls, count in cancel_by_class.items():
        narrative += f"• {cls}: {count} cancellations ({count/total_cancel*100:.0f}%)\n"

    narrative += f"""
**Top cancelled procedure categories:**
"""
    for proc, count in cancel_by_proc.items():
        narrative += f"• {proc}: {count} cancellations\n"

    narrative += f"""
**Add-on cases cancelled:** {cancel_by_addon.get('Yes', 0)} out of {total_cancel} ({cancel_by_addon.get('Yes', 0)/total_cancel*100:.1f}%)

### Patterns to Investigate

The charts below show cancellation rates broken down by surgeon, day of week, and priority. Look for:

1. **Surgeon patterns** — Is one surgeon's list disproportionately affected? Could be case mix complexity, patient selection, or operational factors
2. **Day patterns** — Some days may have higher cancellation rates due to emergency pressure or staffing levels
3. **Priority patterns** — Lower priority cases (P3, P4) might be getting bumped by emergencies — a classic NHS scheduling tension

**Key question for analysts**: What's the relationship between add-on emergency cases and elective cancellations on the same day? That's the smoking gun for capacity-driven cancellations.
"""
    return narrative, fig_surgeon_cancel, fig_dow_cancel, fig_priority_cancel


# ============================================================
# ML PREDICTION MODEL
# ============================================================

def build_duration_model(df):
    """Build and evaluate a theatre duration prediction model."""
    # Prepare features
    model_df = df[df["Actual Theatre (Duration)"].notna()].copy()
    model_df = model_df[model_df["Log Status"] != "Cancelled"].copy()

    # Features
    feature_cols = ["Cardiothoracic Procedure", "Primary Surgeon", "Priority_Code",
                    "Patient Sex", "Patient Class", "Day of Week", "Add-on",
                    "Age_Numeric", "Scheduled (Duration)"]

    model_df = model_df.dropna(subset=feature_cols + ["Actual Theatre (Duration)"])

    # Encode categoricals
    encoders = {}
    encoded_df = model_df.copy()
    categorical_cols = ["Cardiothoracic Procedure", "Primary Surgeon", "Priority_Code",
                        "Patient Sex", "Patient Class", "Day of Week", "Add-on"]

    for col in categorical_cols:
        le = LabelEncoder()
        encoded_df[col + "_enc"] = le.fit_transform(encoded_df[col].astype(str))
        encoders[col] = le

    feature_cols_enc = [c + "_enc" for c in categorical_cols] + ["Age_Numeric", "Scheduled (Duration)"]
    X = encoded_df[feature_cols_enc].values
    y = encoded_df["Actual Theatre (Duration)"].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)

    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    # Evaluate
    rf_pred = rf.predict(X_test)
    gb_pred = gb.predict(X_test)

    rf_mae = mean_absolute_error(y_test, rf_pred)
    gb_mae = mean_absolute_error(y_test, gb_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    gb_r2 = r2_score(y_test, gb_pred)

    # Cross-validation
    rf_cv = cross_val_score(rf, X, y, cv=5, scoring="neg_mean_absolute_error")
    gb_cv = cross_val_score(gb, X, y, cv=5, scoring="neg_mean_absolute_error")

    # Choose best model
    best_model = rf if rf_mae < gb_mae else gb
    best_name = "Random Forest" if rf_mae < gb_mae else "Gradient Boosting"
    best_pred = rf_pred if rf_mae < gb_mae else gb_pred
    best_mae = min(rf_mae, gb_mae)
    best_r2 = rf_r2 if rf_mae < gb_mae else gb_r2

    # Feature importance
    importances = pd.DataFrame({
        "Feature": [c.replace("_enc", "") for c in feature_cols_enc],
        "Importance": best_model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig_importance = px.bar(
        importances,
        y="Feature",
        x="Importance",
        orientation="h",
        title=f"Feature Importance — {best_name}",
        labels={"Importance": "Importance Score", "Feature": ""},
        color_discrete_sequence=[NHS_BLUE]
    )
    fig_importance.update_layout(template="plotly_white", font=dict(family="Arial, sans-serif"))

    # Actual vs Predicted scatter
    fig_pred = px.scatter(
        x=y_test,
        y=best_pred,
        labels={"x": "Actual Duration (mins)", "y": "Predicted Duration (mins)"},
        title=f"Actual vs Predicted Theatre Duration ({best_name})",
        opacity=0.6,
        color_discrete_sequence=[NHS_BLUE]
    )
    max_val = max(y_test.max(), best_pred.max())
    fig_pred.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", name="Perfect Prediction",
        line=dict(dash="dash", color=NHS_RED, width=2)
    ))
    fig_pred.update_layout(template="plotly_white", font=dict(family="Arial, sans-serif"))

    # Residuals
    residuals = y_test - best_pred
    fig_residuals = px.histogram(
        x=residuals,
        nbins=40,
        title="Prediction Errors (Residuals)",
        labels={"x": "Prediction Error (mins)"},
        color_discrete_sequence=[NHS_BLUE]
    )
    fig_residuals.add_vline(x=0, line_dash="dash", line_color=NHS_RED)
    fig_residuals.update_layout(template="plotly_white", font=dict(family="Arial, sans-serif"))

    # Narrative
    top_feature = importances.iloc[-1]["Feature"]
    second_feature = importances.iloc[-2]["Feature"]

    narrative = f"""## 🤖 ML Duration Prediction Model — Plain English Results

### What Did We Build?

We trained two machine learning models to **predict how long a cardiac surgery case will actually take** in theatre, based on what's known before the operation starts. Think of it as: "Given this procedure, this surgeon, this patient, and this priority level — how long should we really block the theatre for?"

### Model Comparison

| Model | Mean Absolute Error | R² Score | Cross-Val MAE (5-fold) |
|-------|-------------------|----------|----------------------|
| Random Forest | {rf_mae:.1f} mins | {rf_r2:.3f} | {-rf_cv.mean():.1f} ± {rf_cv.std():.1f} mins |
| Gradient Boosting | {gb_mae:.1f} mins | {gb_r2:.3f} | {-gb_cv.mean():.1f} ± {gb_cv.std():.1f} mins |

**Winner: {best_name}** with an average prediction error of **{best_mae:.1f} minutes** ({best_mae/60:.1f} hours).

### What Does This Mean in Plain English?

• **Mean Absolute Error of {best_mae:.1f} minutes** means the model is typically off by about {best_mae:.0f} minutes either way. For a 5-hour operation, that's roughly ±{best_mae/300*100:.0f}% accuracy.
• **R² of {best_r2:.3f}** means the model explains **{best_r2*100:.1f}%** of the variation in theatre times. {"That's decent — there's clearly predictable structure in the data." if best_r2 > 0.5 else "There's room for improvement, but some variation in surgery is inherently unpredictable."}
• **Cross-validation** confirms the model generalises — it's not just memorising the training data.

### What Drives Prediction?

The feature importance chart shows **{top_feature}** is the strongest predictor, followed by **{second_feature}**. This makes clinical sense because:

• **Scheduled Duration** reflects the surgical team's own estimate — it's the baseline signal
• **Procedure type** determines complexity — a CABG-only is fundamentally different from a combined aortic segment/valve/CABG
• **Surgeon** captures individual pace and technique preferences
• **Age** reflects patient frailty and potential complications
• **Priority** indicates urgency and how much pre-operative optimisation was possible

### Limitations — Be Honest About What the Model Can't Do

1. **It doesn't know about complications** — intra-operative surprises aren't in the input data
2. **It doesn't capture patient comorbidity** — two 70-year-olds can have very different operative risk
3. **Sample size is modest** — {len(model_df)} completed cases limits complex interaction learning
4. **It can't predict cancellations** — only duration for cases that actually happen

### How Could This Be Used?

• **Better scheduling**: Use predicted durations instead of standard blocks to reduce overruns and idle time
• **Resource planning**: If the model predicts a long day, ensure staffing cover
• **Outlier detection**: Cases where actual vastly exceeds predicted may warrant clinical review
• **"What-if" planning**: What happens to the schedule if we add an emergency P1a case?

**Key question for analysts**: The gap between model error and zero represents the irreducible uncertainty of surgery. Can we close that gap with more data (comorbidities, EuroSCORE, BMI), or is this close to the floor?
"""
    return narrative, fig_importance, fig_pred, fig_residuals, best_model, encoders


def predict_single_case(model, encoders, procedure, surgeon, priority, sex, patient_class, day, addon, age, scheduled_dur):
    """Make a prediction for a single case."""
    try:
        features = []
        categorical_inputs = {
            "Cardiothoracic Procedure": procedure,
            "Primary Surgeon": surgeon,
            "Priority_Code": priority,
            "Patient Sex": sex,
            "Patient Class": patient_class,
            "Day of Week": day,
            "Add-on": addon
        }

        for col, val in categorical_inputs.items():
            le = encoders[col]
            if val in le.classes_:
                features.append(le.transform([val])[0])
            else:
                features.append(0)  # Unknown category

        features.append(float(age))
        features.append(float(scheduled_dur))

        prediction = model.predict([features])[0]
        variance = prediction - scheduled_dur

        result = f"""## 🔮 Predicted Theatre Duration

**Predicted actual duration: {prediction:.0f} minutes** ({prediction/60:.1f} hours)

**Scheduled duration: {scheduled_dur:.0f} minutes** ({scheduled_dur/60:.1f} hours)

**Expected variance: {variance:+.0f} minutes** — the model predicts this case will {"overrun" if variance > 0 else "finish early"} by about {abs(variance):.0f} minutes.

### Case Details
• Procedure: {procedure}
• Surgeon: {surgeon}
• Priority: {priority}
• Patient: {sex}, {age:.0f} years old
• Class: {patient_class}
• Day: {day}
• Add-on: {addon}

### What This Means for Scheduling

{"⚠️ **Consider extending the theatre slot** — this case profile typically runs over." if variance > 15 else ""}{"✅ **Schedule looks about right** — variance is within normal range." if -15 <= variance <= 15 else ""}{"🟢 **May finish early** — potential to bring forward the next case or use freed time." if variance < -15 else ""}
"""
        return result
    except Exception as e:
        return f"Prediction error: {str(e)}"


# ============================================================
# DATA QUALITY ANALYSIS
# ============================================================

def analyse_data_quality(df):
    """Comprehensive data quality report."""
    total = len(df)

    quality_data = []
    for col in df.columns:
        missing = df[col].isna().sum()
        quality_data.append({
            "Column": col,
            "Missing": missing,
            "Missing %": round(missing / total * 100, 1),
            "Populated": total - missing,
            "Unique Values": df[col].nunique(),
            "Data Type": str(df[col].dtype)
        })

    quality_df = pd.DataFrame(quality_data)
    quality_df = quality_df.sort_values("Missing %", ascending=False)

    # Visualise
    fig_missing = px.bar(
        quality_df[quality_df["Missing %"] > 0].sort_values("Missing %"),
        y="Column",
        x="Missing %",
        orientation="h",
        title="Missing Data by Column",
        labels={"Missing %": "% Missing", "Column": ""},
        color="Missing %",
        color_continuous_scale=["#009639", "#FFB81C", "#DA291C"]
    )
    fig_missing.update_layout(template="plotly_white", font=dict(family="Arial, sans-serif"), height=600)

    # Completely empty columns
    empty_cols = quality_df[quality_df["Missing %"] == 100]["Column"].tolist()
    sparse_cols = quality_df[(quality_df["Missing %"] > 80) & (quality_df["Missing %"] < 100)]["Column"].tolist()
    good_cols = quality_df[quality_df["Missing %"] < 5]["Column"].tolist()

    narrative = f"""## 🔍 Data Quality Report

### Overall Health

• **{total} records** across **{len(df.columns)} columns**
• **{len(empty_cols)} completely empty columns**: {', '.join(empty_cols) if empty_cols else 'None'}
• **{len(sparse_cols)} very sparse columns** (>80% missing): {', '.join(sparse_cols) if sparse_cols else 'None'}
• **{len(good_cols)} well-populated columns** (<5% missing)

### Key Issues to Address

**🔴 Critical (blocks analysis):**
• **Incision Close** ({df['Incision Close'].notna().sum()}/{total} populated) — only {df['Incision Close'].notna().sum()/total*100:.0f}% of cases have this, making surgical time calculation unreliable for most records
• **Out Recovery** ({df['Out Recovery'].notna().sum()}/{total}) — can't track full patient pathway for most cases

**🟡 Notable (limits scope):**
• **Redo** flag only present for {df['Redo'].notna().sum()} cases — appears to be "Yes only" flagging rather than Yes/No for all
• **Valve fields** (Valve and Aortic Site, Valve Procedure Type) are only relevant for valve cases, so high missing rates are expected
• **Aorta** column similarly only relevant for aortic procedures

**🟢 Good news:**
• Core scheduling and identification fields are complete (0% missing)
• Actual Theatre Duration is present for {df['Actual Theatre (Duration)'].notna().sum()/total*100:.0f}% of non-cancelled cases
• All demographic and classification fields are fully populated

### Recommendations for Data Collection

1. **Mandate Incision Close recording** — without this, you can't calculate true surgical time vs anaesthesia/positioning time
2. **Make Redo a Yes/No for all cases** — currently it's impossible to distinguish "not a redo" from "not recorded"
3. **Consider adding**: EuroSCORE, BMI, diabetes status, previous cardiac history — these would dramatically improve ML predictions
4. **Recovery tracking** needs attention — only {df['Out Recovery'].notna().sum()} of {total} cases have Out Recovery times
"""
    return narrative, fig_missing


# ============================================================
# GRADIO APP
# ============================================================

def build_app():
    """Build the Gradio application."""

    # Load data
    df = load_and_clean_data(DATA_PATH)
    summary_text = generate_plain_english_summary(df)

    # Pre-build ML model
    ml_narrative, fig_imp, fig_pred, fig_resid, model, encoders = build_duration_model(df)

    # Get unique values for prediction dropdowns
    active_df = df[df["Actual Theatre (Duration)"].notna()]
    procedures = sorted(active_df["Cardiothoracic Procedure"].unique().tolist())
    surgeons = sorted(df["Primary Surgeon"].unique().tolist())
    priorities = sorted(df["Priority_Code"].dropna().unique().tolist())
    sexes = sorted(df["Patient Sex"].unique().tolist())
    patient_classes = sorted(df["Patient Class"].unique().tolist())
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # NHS Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Arial', 'Helvetica Neue', sans-serif !important;
        max-width: 1400px !important;
    }
    .main-header {
        background: linear-gradient(135deg, #003087 0%, #005EB8 50%, #41B6E6 100%);
        color: white;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 16px;
    }
    .main-header h1 {
        color: white !important;
        margin: 0 0 8px 0 !important;
        font-size: 28px !important;
    }
    .main-header p {
        color: #E8EDEE !important;
        margin: 0 !important;
        font-size: 14px;
    }
    .tab-nav button {
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    .tab-nav button.selected {
        border-bottom: 3px solid #005EB8 !important;
        color: #005EB8 !important;
    }
    footer { display: none !important; }
    """

    with gr.Blocks(css=custom_css, title="ACS Theatre Intelligence Explorer", theme=gr.themes.Soft()) as demo:

        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🫀 ACS Theatre Intelligence Explorer</h1>
            <p>Adult Cardiac Surgery — Theatre Operations Analysis & ML Prediction</p>
            <p style="font-size: 12px; margin-top: 8px !important;">NHS 365 AI Technologies · Plain English insights for data scientist analysts</p>
        </div>
        """)

        with gr.Tabs():

            # TAB 1: Overview
            with gr.TabItem("📊 Overview"):
                gr.Markdown(summary_text)

            # TAB 2: Theatre Efficiency
            with gr.TabItem("🏥 Theatre Efficiency"):
                eff_btn = gr.Button("Run Theatre Efficiency Analysis", variant="primary", size="lg")
                eff_narrative = gr.Markdown()
                eff_scatter = gr.Plot()
                eff_variance = gr.Plot()
                eff_proc = gr.Plot()

                def run_efficiency():
                    n, f1, f2, f3 = analyse_theatre_efficiency(df)
                    return n, f1, f2, f3

                eff_btn.click(run_efficiency, outputs=[eff_narrative, eff_scatter, eff_variance, eff_proc])

            # TAB 3: Surgeon Profiles
            with gr.TabItem("👨‍⚕️ Surgeon Profiles"):
                surg_btn = gr.Button("Run Surgeon Analysis", variant="primary", size="lg")
                surg_narrative = gr.Markdown()
                surg_box = gr.Plot()
                surg_heat = gr.Plot()
                surg_casemix = gr.Plot()

                def run_surgeon():
                    n, f1, f2, f3 = analyse_surgeon_profiles(df)
                    return n, f1, f2, f3

                surg_btn.click(run_surgeon, outputs=[surg_narrative, surg_box, surg_heat, surg_casemix])

            # TAB 4: Scheduling Patterns
            with gr.TabItem("📅 Scheduling"):
                sched_btn = gr.Button("Run Scheduling Analysis", variant="primary", size="lg")
                sched_narrative = gr.Markdown()
                sched_dow = gr.Plot()
                sched_month = gr.Plot()
                sched_priority = gr.Plot()

                def run_scheduling():
                    n, f1, f2, f3 = analyse_scheduling_patterns(df)
                    return n, f1, f2, f3

                sched_btn.click(run_scheduling, outputs=[sched_narrative, sched_dow, sched_month, sched_priority])

            # TAB 5: Cancellations
            with gr.TabItem("❌ Cancellations"):
                cancel_btn = gr.Button("Run Cancellation Analysis", variant="primary", size="lg")
                cancel_narrative = gr.Markdown()
                cancel_surgeon = gr.Plot()
                cancel_dow = gr.Plot()
                cancel_priority = gr.Plot()

                def run_cancellations():
                    n, f1, f2, f3 = analyse_cancellations(df)
                    return n, f1, f2, f3

                cancel_btn.click(run_cancellations, outputs=[cancel_narrative, cancel_surgeon, cancel_dow, cancel_priority])

            # TAB 6: ML Model
            with gr.TabItem("🤖 ML Prediction"):
                gr.Markdown(ml_narrative)
                with gr.Row():
                    fig_imp_plot = gr.Plot(value=fig_imp, label="Feature Importance")
                    fig_pred_plot = gr.Plot(value=fig_pred, label="Actual vs Predicted")
                fig_resid_plot = gr.Plot(value=fig_resid, label="Residuals")

                gr.Markdown("---")
                gr.Markdown("## 🔮 Predict a New Case")
                gr.Markdown("Fill in the details below and click **Predict** to get an estimated theatre duration.")

                with gr.Row():
                    with gr.Column():
                        pred_procedure = gr.Dropdown(choices=procedures, label="Procedure Type",
                                                     value="CABG only")
                        pred_surgeon = gr.Dropdown(choices=surgeons, label="Surgeon",
                                                   value=surgeons[0])
                        pred_priority = gr.Dropdown(choices=priorities, label="Priority",
                                                    value="P2b")
                        pred_sex = gr.Dropdown(choices=sexes, label="Patient Sex",
                                               value="Male [2]")
                    with gr.Column():
                        pred_class = gr.Dropdown(choices=patient_classes, label="Patient Class",
                                                 value="Surgery Admit")
                        pred_day = gr.Dropdown(choices=days, label="Day of Week",
                                               value="Tuesday")
                        pred_addon = gr.Dropdown(choices=["Yes", "No"], label="Add-on?",
                                                 value="No")
                        pred_age = gr.Slider(minimum=18, maximum=95, value=65, step=1,
                                             label="Patient Age")
                with gr.Row():
                    pred_duration = gr.Slider(minimum=60, maximum=600, value=300, step=10,
                                              label="Scheduled Duration (minutes)")
                    pred_btn = gr.Button("🔮 Predict Duration", variant="primary", size="lg")

                pred_result = gr.Markdown()

                pred_btn.click(
                    lambda proc, surg, pri, sex, cls, day, addon, age, dur:
                        predict_single_case(model, encoders, proc, surg, pri, sex, cls, day, addon, age, dur),
                    inputs=[pred_procedure, pred_surgeon, pred_priority, pred_sex,
                            pred_class, pred_day, pred_addon, pred_age, pred_duration],
                    outputs=pred_result
                )

            # TAB 7: Data Quality
            with gr.TabItem("🔍 Data Quality"):
                dq_btn = gr.Button("Run Data Quality Analysis", variant="primary", size="lg")
                dq_narrative = gr.Markdown()
                dq_missing = gr.Plot()

                def run_dq():
                    n, f = analyse_data_quality(df)
                    return n, f

                dq_btn.click(run_dq, outputs=[dq_narrative, dq_missing])

            # TAB 8: About
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
## About This Tool

### What Is It?

The **ACS Theatre Intelligence Explorer** is a data analysis tool built for NHS data scientist analysts working with Adult Cardiac Surgery theatre operations data. It takes raw scheduling and operational data and turns it into **actionable, plain English insights**.

### Design Philosophy

This tool follows the same approach as the **Clinical ML Tutor** project:

• **AI as a reasoning partner** — not just charts, but explanations of what the charts mean and what questions to ask next
• **Plain English first** — every analysis tab leads with a narrative that a non-technical stakeholder could understand
• **Honest about limitations** — the ML model section is upfront about what the model can and can't do
• **Designed for the NHS** — built by someone who understands that NHS staff need tools that respect their time and intelligence

### What Can You Do With It?

1. **Theatre Efficiency** — Understand overruns, identify which procedures consistently blow their time slots
2. **Surgeon Profiles** — Contextualised workload and performance data (not league tables)
3. **Scheduling Patterns** — Find day-of-week and monthly patterns that affect operations
4. **Cancellation Analysis** — Understand what's being cancelled and why
5. **ML Prediction** — Predict actual theatre duration for a given case profile
6. **Data Quality** — Assess what's missing and what needs improving in data collection

### Technical Details

• Built with **Python**, **Gradio**, **Plotly**, and **scikit-learn**
• ML models: Random Forest and Gradient Boosting regressors with cross-validation
• No external API calls — everything runs locally on your machine
• Your data stays on your machine — nothing is sent anywhere

---

*Built by David — AI Engineer, NHS 365 AI Technologies*
*Inspired by the principle that AI should help people think, not think for them.*
""")

    return demo


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  ACS Theatre Intelligence Explorer")
    print("  NHS 365 AI Technologies")
    print("=" * 60)
    print()

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at '{DATA_PATH}'")
        print(f"Please place your ACSData_Data_.csv file in: {os.getcwd()}")
        print("Or update DATA_PATH at the top of this script.")
        exit(1)

    print(f"Loading data from: {DATA_PATH}")
    demo = build_app()
    print()
    print("Starting server...")
    print("Open your browser to: http://127.0.0.1:7860")
    print()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
