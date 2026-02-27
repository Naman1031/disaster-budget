import requests
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
from dotenv import load_dotenv

# ============================================================
# CONFIG
# ============================================================
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

STATES = ["Assam", "Odisha", "Rajasthan", "Maharashtra", "Tamil Nadu", "Uttar Pradesh"]

state_capitals = {
    "Assam":         "Guwahati",
    "Odisha":        "Bhubaneswar",
    "Rajasthan":     "Jaipur",
    "Maharashtra":   "Mumbai",
    "Tamil Nadu":    "Chennai",
    "Uttar Pradesh": "Lucknow"
}

HAZARDS = ["flood", "storm", "extreme temperature", "earthquake"]

# ============================================================
# STEP 0 — REBUILD BASELINE FROM RAW DATA
# This avoids all CSV column mismatch bugs entirely.
# ============================================================
print("="*70)
print("STEP 0: Rebuilding baseline from historical_disaster_data.csv ...")
print("="*70)

try:
    raw = pd.read_csv("../data/historical_disaster_data.csv")
except FileNotFoundError:
    print("[ERROR] historical_disaster_data.csv not found.")
    sys.exit(1)

# Filter to India natural disasters only
raw = raw[raw["Country"] == "India"]
raw = raw[raw["Disaster Group"] == "Natural"]

# Keep only 4 hazard types
raw["Disaster Type"] = raw["Disaster Type"].str.strip().str.lower()
raw = raw[raw["Disaster Type"].isin(HAZARDS)]

# Extract state from Location field
def extract_state(location):
    if pd.isna(location):
        return None
    for s in STATES:
        if s.lower() in location.lower():
            return s
    return None

raw["state"] = raw["Location"].apply(extract_state)
raw = raw[raw["state"].notna()]

# Keep only single-state events
def count_states_in_loc(location):
    return sum(1 for s in STATES if pd.notna(location) and s.lower() in location.lower())

raw = raw[raw["Location"].apply(count_states_in_loc) == 1]

# Clean numeric columns
raw["Total Damage ('000 US$)"] = pd.to_numeric(raw["Total Damage ('000 US$)"], errors="coerce")
raw["Total Deaths"]             = pd.to_numeric(raw["Total Deaths"], errors="coerce")
raw["Total Affected"]          = pd.to_numeric(raw["Total Affected"], errors="coerce")

# Impute missing damage with group median, then national median fallback
for col in ["Total Damage ('000 US$)", "Total Deaths", "Total Affected"]:
    raw[col] = raw.groupby(["state", "Disaster Type"])[col]\
                  .transform(lambda x: x.fillna(x.median()) if x.notna().sum() > 0 else x)
    raw[col] = raw[col].fillna(raw[col].median())

# ── Build baseline aggregates ──────────────────────────────
baseline = (
    raw.groupby(["state", "Disaster Type"])
       .agg(
           total_events      = ("Disaster Type", "count"),
           avg_damage        = ("Total Damage ('000 US$)", "median"),
           avg_deaths        = ("Total Deaths",            "median"),
           avg_affected      = ("Total Affected",          "mean"),
           first_year        = ("Start Year",              "min"),
           last_year         = ("Start Year",              "max")
       )
       .reset_index()
       .rename(columns={"Disaster Type": "disaster_type"})
)

# ── True avg_events_per_year  ──────────────────────────────
# Count events per (state, disaster_type, year), then average across years
# This is the CORRECT frequency — not total_events / some fixed denominator
yearly = (
    raw.groupby(["state", "Disaster Type", "Start Year"])
       .size()
       .reset_index(name="n")
)
freq = (
    yearly.groupby(["state", "Disaster Type"])
          .agg(avg_events_per_year=("n", "mean"))
          .reset_index()
          .rename(columns={"Disaster Type": "disaster_type"})
)

baseline = baseline.merge(freq, on=["state", "disaster_type"], how="left")

# ── Observation window for Poisson rate ───────────────────
# Lambda for Poisson = total_events / years_observed
# years_observed = last_year - first_year + 1  (handles gaps properly)
baseline["years_observed"] = (baseline["last_year"] - baseline["first_year"] + 1).clip(lower=1)
baseline["poisson_lambda"] = baseline["total_events"] / baseline["years_observed"]

print(f"\nBaseline rebuilt: {len(baseline)} state-hazard pairs")
print(baseline[["state","disaster_type","total_events","avg_damage",
                "avg_events_per_year","poisson_lambda"]].to_string(index=False))

# ============================================================
# STEP 1 — POISSON OCCURRENCE PROBABILITY
# P(at least 1 event in coming year) = 1 - e^(-lambda)
# Uses total_events / years_observed as lambda — more accurate
# than avg_events_per_year which was averaging year-counts
# ============================================================
baseline["poisson_prob"] = 1 - np.exp(-baseline["poisson_lambda"])

# ============================================================
# STEP 2 — FETCH LIVE WEATHER FORECAST
# ============================================================
print("\n" + "="*70)
print("STEP 2: Fetching live 24h weather forecasts ...")
print("="*70)

def fetch_weather_forecast(city):
    url = (f"https://api.openweathermap.org/data/2.5/forecast"
           f"?q={city}&appid={API_KEY}&units=metric")
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [WARNING] {city}: {e}. Using fallback.")
        return {"avg_temp": 28.0, "total_rain_24h": 0.0, "avg_wind": 5.0}

    temps, rains, winds = [], [], []
    for entry in data["list"][:8]:
        temps.append(entry["main"]["temp"])
        rains.append(entry.get("rain", {}).get("3h", 0))
        winds.append(entry["wind"]["speed"])

    return {
        "avg_temp":       round(sum(temps) / len(temps), 2),
        "total_rain_24h": round(sum(rains), 2),
        "avg_wind":       round(sum(winds) / len(winds), 2)
    }

weather_records = []
for state, city in state_capitals.items():
    print(f"  Fetching {city} ({state})...")
    w = fetch_weather_forecast(city)
    weather_records.append({"state": state, **w})

weather_df = pd.DataFrame(weather_records)

# ── Seasonal Climatological Baseline ──────────────────────
# Feb weather is calm everywhere — direct weather severity
# would give 0 for all rows, making live weather useless.
# Instead we use a DEPARTURE FROM SEASONAL NORMAL approach:
#   weather_signal = (actual - seasonal_normal) / seasonal_range
#
# Seasonal normals for Feb (approximate climatological values):
SEASONAL_NORMALS = {
    #            (normal_rain_mm, normal_temp_c, normal_wind_ms)
    "Assam":         (2.0,  18.0, 2.5),
    "Odisha":        (1.0,  22.0, 3.0),
    "Rajasthan":     (0.5,  16.0, 2.5),
    "Maharashtra":   (0.0,  22.0, 3.0),
    "Tamil Nadu":    (0.5,  25.0, 4.5),
    "Uttar Pradesh": (0.5,  16.0, 3.0),
}
# Seasonal range = typical monsoon peak - Feb normal (what a real anomaly looks like)
SEASONAL_RANGES = {
    #            (rain_range, temp_range, wind_range)
    "Assam":         (50.0, 15.0, 8.0),
    "Odisha":        (80.0, 12.0, 10.0),
    "Rajasthan":     (30.0, 20.0, 10.0),
    "Maharashtra":   (80.0, 12.0, 10.0),
    "Tamil Nadu":    (60.0, 10.0, 10.0),
    "Uttar Pradesh": (40.0, 18.0, 8.0),
}

def compute_weather_anomaly(row):
    state = row["state"]
    norm  = SEASONAL_NORMALS[state]
    rng   = SEASONAL_RANGES[state]

    rain_anomaly = np.clip((row["total_rain_24h"] - norm[0]) / rng[0], 0, 1)
    temp_anomaly = np.clip((row["avg_temp"]       - norm[1]) / rng[1], 0, 1)
    wind_anomaly = np.clip((row["avg_wind"]        - norm[2]) / rng[2], 0, 1)

    return pd.Series({
        "rain_anomaly": rain_anomaly,
        "temp_anomaly": temp_anomaly,
        "wind_anomaly": wind_anomaly
    })

anomalies = weather_df.apply(compute_weather_anomaly, axis=1)
weather_df = pd.concat([weather_df, anomalies], axis=1)

print("\n[Live Weather + Anomaly Scores]")
print(weather_df.to_string(index=False))

# ============================================================
# STEP 3 — MERGE & COMPUTE WEATHER SEVERITY PER HAZARD
# ============================================================
df = baseline.merge(weather_df, on="state", how="left")

def get_weather_severity(row):
    d = row["disaster_type"]
    if d == "flood":
        return row["rain_anomaly"]
    elif d == "extreme temperature":
        return row["temp_anomaly"]
    elif d in ["storm", "cyclone"]:
        return row["wind_anomaly"]
    else:
        return 0.0  # earthquake: no weather proxy

df["weather_severity"] = df.apply(get_weather_severity, axis=1)

# ============================================================
# STEP 4 — COMPOSITE OCCURRENCE PROBABILITY
#
# P_composite = P_poisson × (1 + α × weather_severity)
#
# Weather AMPLIFIES historical probability — it cannot create
# risk where none historically existed, but it CAN push a
# high-risk state closer to certainty during extreme weather.
#
# α = 0.40 → extreme weather can boost probability by up to 40%
# ============================================================
ALPHA = 0.40

df["composite_prob"] = (
    df["poisson_prob"] * (1 + ALPHA * df["weather_severity"])
).clip(upper=0.98)

# ============================================================
# STEP 5 — THREE-SCENARIO DAMAGE MODEL
#
# Low    = 0.5 × avg_damage  (below-average event)
# Medium = 1.0 × avg_damage  (typical event)
# High   = 2.5 × avg_damage  (severe tail event)
#
# Weights shift toward High scenario when weather is extreme:
#   w_high   = 0.10 + 0.50 × weather_severity
#   w_low    = 0.40 - 0.20 × weather_severity
#   w_medium = remaining
# ============================================================
df["loss_low"]    = 0.50 * df["avg_damage"]
df["loss_medium"] = 1.00 * df["avg_damage"]
df["loss_high"]   = 2.50 * df["avg_damage"]

def scenario_expected_damage(row):
    ws      = row["weather_severity"]
    w_high  = 0.10 + 0.50 * ws
    w_low   = 0.40 - 0.20 * ws
    w_med   = 1.0 - w_high - w_low
    return (w_low  * row["loss_low"] +
            w_med  * row["loss_medium"] +
            w_high * row["loss_high"])

df["expected_damage"] = df.apply(scenario_expected_damage, axis=1)

# ============================================================
# STEP 6 — AVERAGE ANNUAL LOSS (AAL)
#
# AAL = P(event occurs) × E[Damage | event] × avg_events_per_year
#
# avg_events_per_year accounts for years with 2+ events
# (e.g. Assam floods 1.42 times per year on average)
# ============================================================
df["average_annual_loss"] = (
    df["composite_prob"] *
    df["expected_damage"] *
    df["avg_events_per_year"]
)

# ============================================================
# STEP 7 — 90th PERCENTILE BUDGET (Risk Load)
#
# Governments must budget above the mean loss (AAL).
# We approximate the 90th percentile using:
#
#   Budget = AAL × (1 + Z_90 × CV)
#
# where:
#   Z_90 = 1.28  (90th percentile z-score)
#   CV   = coefficient of variation
#        = (loss_high - loss_low) / (2 × avg_damage)
#        = (2.5 - 0.5) × avg_damage / (2 × avg_damage)
#        = 1.0  (constant — our scenario spread is fixed)
#
# So: Budget = AAL × (1 + 1.28 × 1.0) = AAL × 2.28
# This is the government's 90th-percentile reserve.
# ============================================================
Z_90    = 1.28
CV      = (df["loss_high"] - df["loss_low"]) / (2 * df["avg_damage"].replace(0, np.nan)).fillna(1.0)
df["risk_load_factor"]    = 1 + Z_90 * CV
df["recommended_budget"]  = df["average_annual_loss"] * df["risk_load_factor"]

# ============================================================
# STEP 8 — PRIORITY SCORE & RISK TIER
# ============================================================
raw_score = df["composite_prob"] * df["recommended_budget"]
s_min, s_max = raw_score.min(), raw_score.max()
df["priority_score"] = ((raw_score - s_min) / (s_max - s_min) * 100).round(1)

df["risk_tier"] = pd.cut(
    df["priority_score"],
    bins=[-1, 15, 40, 70, 101],
    labels=["Low", "Medium", "High", "Critical"]
)

df["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")

# ============================================================
# OUTPUT
# ============================================================
df["disaster_type"] = df["disaster_type"].str.title()

output = df[[
    "state", "disaster_type",
    "poisson_prob",
    "weather_severity",
    "composite_prob",
    "average_annual_loss",
    "recommended_budget",
    "priority_score",
    "risk_tier",
    "last_updated"
]].sort_values("recommended_budget", ascending=False).reset_index(drop=True)
output.index += 1

pd.set_option("display.float_format", "{:,.2f}".format)
pd.set_option("display.max_columns", 12)
pd.set_option("display.width", 140)

print("\n" + "="*140)
print(f"  DYNAMIC DISASTER BUDGET RECOMMENDATIONS  |  Amounts in '000 USD  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("="*140)
print(output.to_string())

output.to_csv("../outputs/budget_recommendations.csv", index=True)
print("\n[Saved] budget_recommendations.csv")

# Tier summary
print("\n[Summary by Risk Tier]")
summary = (
    df.groupby("risk_tier", observed=True)
      .agg(
          count                    = ("state", "count"),
          total_recommended_budget = ("recommended_budget", "sum"),
          avg_composite_prob       = ("composite_prob", "mean")
      )
      .sort_values("total_recommended_budget", ascending=False)
)
print(summary.to_string())

from sqlalchemy import create_engine

engine = create_engine(
    "mysql+pymysql://root:naman@localhost/climate_risk"
)

output.to_sql(
    "dynamic_disaster_budget",
    engine,
    if_exists="replace",
    index=False
)

print("\n[Uploaded to SQL: dynamic_disaster_budget]")