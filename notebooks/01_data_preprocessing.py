# Databricks notebook source
# Databricks Notebook: 01_data_preprocessing (Cleaned version without plots)

import pandas as pd
import os
import joblib
from pyspark.sql import SparkSession
import sys

# Step 0: Import DataLoader from source
sys.path.append("/Workspace/Users/sumit.kadian@thoughtworks.com/MLEng-politicalparties-python-exercise/src")
from text_loader.loader import DataLoader

# Step 1: Initialize Spark session
spark = SparkSession.builder.getOrCreate()

# Step 2: Load and preprocess data via DataLoader
loader = DataLoader(filepath="/Workspace/Users/sumit.kadian@thoughtworks.com/MLEng-politicalparties-python-exercise/data/Tweets.csv")
loader.load_data()

# Step 3: Drop NA and assign TweetId
loader.data.dropna(subset=["Tweet", "Party"], inplace=True)
loader.data.reset_index(drop=True, inplace=True)
loader.data["TweetId"] = loader.data.index

# Step 4: Preprocess (for side effects — we only save the vectorizer)
loader.preprocess_tweets()      # Trains and attaches TfidfVectorizer
loader.preprocess_parties()     # Optionally also trains LabelEncoder

# Step 5: Save only the vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(loader.vectorizer, "models/vectorizer.pkl")

# Step 6: Prepare cleaned DataFrame
features_df = loader.data[["TweetId", "Tweet", "Party"]].copy()

# Step 7: Save to Unity Catalog Delta Table
spark_df = spark.createDataFrame(features_df)
spark_df.write.format("delta").mode("overwrite").saveAsTable(
    "mle_batch_catalog_2025_q2.sumit_kadian.party_classifier_features"
)

print(f" Vectorizer saved and {len(features_df)} cleaned rows written to Delta table.")


# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Compute tweet lengths if not already done
if "TweetLength" not in features_df.columns:
    features_df["TweetLength"] = features_df["Tweet"].apply(len)

# Plot 1: Tweet count per party
plt.figure(figsize=(10, 5))
sns.countplot(data=features_df, x="Party", order=features_df["Party"].value_counts().index)
plt.title("Tweet Count per Party")
plt.xlabel("Party")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 2: Tweet length distribution
plt.figure(figsize=(10, 5))
sns.histplot(features_df["TweetLength"], bins=30, kde=True)
plt.title("Tweet Length Distribution")
plt.xlabel("Length of Tweet")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Optional summary
print("EDA Summary:")
print(f"- Unique Parties: {features_df['Party'].nunique()}")
print(f"- Average Tweet Length: {features_df['TweetLength'].mean():.2f}")