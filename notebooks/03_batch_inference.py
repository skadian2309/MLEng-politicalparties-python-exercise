# Databricks notebook source
import requests
import json
from pyspark.sql import SparkSession
import pandas as pd

# Endpoint URL and headers
url = "https://dbc-ba3cda01-8312.cloud.databricks.com/serving-endpoints/mle_sumit_kadian_endpoint/invocations"
headers = {
    "Authorization": f"Bearer {dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()}",
    "Content-Type": "application/json"
}

# Input texts
batch_texts = [
    {"text": "The party proposed a new healthcare policy"},
    {"text": "Immigration reform should be a priority"},
    {"text": "We need stricter environmental regulations"}
]

# Send request
response = requests.post(url, headers=headers, data=json.dumps({"dataframe_records": batch_texts}))
predictions = response.json()

# Create DataFrame with predictions
df = pd.DataFrame(batch_texts)
df["prediction"] = predictions["predictions"]

# Print predictions
print(df)

# Save to Unity Catalog table
spark_df = SparkSession.builder.getOrCreate().createDataFrame(df)
spark_df.write.format("delta").mode("append").saveAsTable("mle_batch_catalog_2025_q2.sumit_kadian.batch_predictions_from_endpoint")

print(f"Inference complete. Saved {len(df)} rows.")

# COMMAND ----------

import requests
import json

# Set your endpoint name
endpoint_name = "mle_sumit_kadian_endpoint"

# Databricks REST API base URL
workspace_url = "https://dbc-ba3cda01-8312.cloud.databricks.com"

# Get auth token securely from current context
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Build request
url = f"{workspace_url}/api/2.0/serving-endpoints/{endpoint_name}"
headers = {
    "Authorization": f"Bearer {token}"
}

# Call Databricks API
response = requests.get(url, headers=headers)
data = response.json()

# Parse model version info
model_info = data["config"]["served_entities"][0]
model_name = model_info["entity_name"]
model_version = model_info["entity_version"]

print(f"Endpoint '{endpoint_name}' is serving model:")
print(f"   - Name   : {model_name}")
print(f"   - Version: {model_version}")
