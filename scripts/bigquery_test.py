from google.cloud import bigquery
import pandas as pd
import os

# Set credentials path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials/credentials.json"

# Create BigQuery client
client = bigquery.Client()

print("Connected to BigQuery Successfully")

# Query COVID dataset
query = """
SELECT
    country_name,
    date,
    new_confirmed,
    new_deceased,
    cumulative_confirmed
FROM
    `bigquery-public-data.covid19_open_data.covid19_open_data`
WHERE
    country_name = 'Uganda'
ORDER BY
    date
LIMIT 1000
"""

# Run query
df = client.query(query).to_dataframe()

# Preview data
print(df.head())

# Save raw dataset
df.to_csv("data/raw/uganda_covid_data.csv", index=False)

print("Dataset saved successfully")