# Databricks Notebook: 02_model_training (uses pre-trained vectorizer)

import pandas as pd
import os
import mlflow
import mlflow.pyfunc
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Step 1: Load features from Delta table
spark_df = spark.read.table("mle_batch_catalog_2025_q2.sumit_kadian.party_classifier_features")
features_df = spark_df.toPandas()

# Step 2: Load pre-trained vectorizer and transform
vectorizer = joblib.load("models/vectorizer.pkl")
X = vectorizer.transform(features_df["Tweet"])  # Use transform, NOT fit_transform
y = features_df["Party"]

# Step 3: Split for training/eval
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Save model artifact
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/logreg_model.pkl")

# Step 6: Define PyFunc wrapper
class TextClassifier(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.vectorizer = joblib.load(context.artifacts["vectorizer"])
        self.model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input):
        vectorized = self.vectorizer.transform(model_input["text"])
        return self.model.predict(vectorized)

# Step 7: Register model to Unity Catalog
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Users/sumit.kadian@thoughtworks.com/mle_sumit_kadian_experiment")

from mlflow.models.signature import infer_signature
input_df = pd.DataFrame({"text": features_df["Tweet"].head(5)})
output_df = model.predict(vectorizer.transform(input_df["text"]))
signature = infer_signature(input_df, output_df)

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="text_classifier_model",
        python_model=TextClassifier(),
        artifacts={
            "vectorizer": "models/vectorizer.pkl",
            "model": "models/logreg_model.pkl"
        },
        input_example=input_df,
        signature=signature,
        registered_model_name="mle_batch_catalog_2025_q2.sumit_kadian.mle_sumit_kadian_logreg"
    )

print("Model trained using pre-saved vectorizer and registered successfully.")
