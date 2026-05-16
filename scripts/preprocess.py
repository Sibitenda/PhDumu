import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# LOAD RAW DATA
df = pd.read_csv("data/raw/uganda_covid_data.csv")

print("\nOriginal Dataset")
print(df.head())

# BASIC CLEANING
# Convert date column
df["date"] = pd.to_datetime(
    df["date"]
)

# Fill missing values
df["new_confirmed"] = (
    df["new_confirmed"]
    .fillna(0)
)

df["new_deceased"] = (
    df["new_deceased"]
    .fillna(0)
)

# Remove negative values
df["new_confirmed"] = (
    df["new_confirmed"]
    .clip(lower=0)
)

df["new_deceased"] = (
    df["new_deceased"]
    .clip(lower=0)
)

# FEATURE ENGINEERING
# Previous day cases
df["previous_day_cases"] = (
    df["new_confirmed"]
    .shift(1)
)

# 7-day rolling average
df["rolling_avg_cases"] = (
    df["new_confirmed"]
    .rolling(window=7)
    .mean()
)

# Growth rate
df["growth_rate"] = (
    df["new_confirmed"]
    .pct_change()
)

# Replace infinity values
df["growth_rate"] = (
    df["growth_rate"]
    .replace([np.inf, -np.inf], 0)
)

# Fill missing values
df["growth_rate"] = (
    df["growth_rate"]
    .fillna(0)
)

# Death ratio
df["death_ratio"] = (
    df["new_deceased"] /
    (df["new_confirmed"] + 1)
)

# Future cases (Regression Target)
df["future_cases"] = (
    df["new_confirmed"]
    .shift(-1)
)
# Replace remaining infinities
df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

# Remove missing values
df = df.dropna()


df = df.dropna()

# MULTICLASS RISK LEVELS
def classify_risk(cases):
    if cases <= 50:
        return 0  # Low
    elif cases <= 200:
        return 1  # Moderate
    elif cases <= 500:
        return 2  # High
    else:
        return 3  # Critical

df["risk_level"] = df["future_cases"].apply(classify_risk)

print("\nRisk Level Distribution")
print(df["risk_level"].value_counts())

risk_labels = {
    0: "Low",
    1: "Moderate",
    2: "High",
    3: "Critical"
}

print("\nRisk Labels")
print(risk_labels)

# RISK CATEGORY VISUALIZATION
df["year"] = df["date"].dt.year

df["risk_category"] = df["risk_level"].map(risk_labels)

risk_counts = df.groupby(["year", "risk_category"]).size().unstack(fill_value=0)

print("\nRisk Distribution by Year")
print(risk_counts)

# BAR CHART (RISK CATEGORIES BY YEAR)
risk_counts.plot(kind="bar", figsize=(12, 6))
plt.title("COVID-19 Risk Categories by Year")
plt.xlabel("Year")
plt.ylabel("Number of Days")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("data/processed/risk_categories_by_year.png")
plt.show()

# STACKED AREA CHART (RISK EVOLUTION)
risk_counts.plot(kind="area", stacked=True, figsize=(12, 6))
plt.title("Evolution of COVID-19 Risk Categories")
plt.xlabel("Year")
plt.ylabel("Number of Days")
plt.tight_layout()
plt.savefig("data/processed/risk_evolution.png")
plt.show()

# DAILY CASES BAR GRAPH
plt.figure(figsize=(12, 5))
plt.bar(df["date"], df["new_confirmed"])
plt.title("Daily COVID-19 Cases in Uganda (Bar Chart)")
plt.xlabel("Date")
plt.ylabel("New Confirmed Cases")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("data/processed/daily_cases_bar_chart.png")
plt.show()

# FEATURE CORRELATION
feature_columns = [
    "new_confirmed",
    "new_deceased",
    "previous_day_cases",
    "rolling_avg_cases",
    "growth_rate",
    "death_ratio"
]

correlation_matrix = df[feature_columns].corr()

print("\nFeature Correlation Matrix")
print(correlation_matrix)

# FEATURE IMPORTANCE INSIGHTS
target_correlations = correlation_matrix["new_confirmed"].sort_values(ascending=False)

print("\nMost Important Features")
print(target_correlations)

selected_features = [
    "previous_day_cases",
    "rolling_avg_cases",
    "growth_rate"
]

print("\nSelected Features")
print(selected_features)

# SAVE PROCESSED DATA
df.to_csv("data/processed/uganda_covid_processed.csv", index=False)

print("\nProcessed dataset saved successfully")