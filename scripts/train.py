import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split

# REGRESSION MODELS
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# CLASSIFICATION MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# REGRESSION METRICS
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# CLASSIFICATION METRICS
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# LOAD DATA
df = pd.read_csv("data/processed/uganda_covid_processed.csv")

print("\nProcessed Dataset")
print(df.head())

# FEATURE SELECTION
features = [
    "previous_day_cases",
    "rolling_avg_cases",
    "growth_rate"
]

X = df[features]

# REGRESSION PROBLEM
print("\n")
print("=" * 50)
print("REGRESSION MODELS")
print("=" * 50)

y_reg = df["future_cases"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X,
    y_reg,
    test_size=0.2,
    random_state=42
)

# LINEAR REGRESSION
linear_model = LinearRegression()
linear_model.fit(X_train_r, y_train_r)
linear_predictions = linear_model.predict(X_test_r)

# RANDOM FOREST REGRESSOR
forest_regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

forest_regressor.fit(X_train_r, y_train_r)
forest_predictions = forest_regressor.predict(X_test_r)

# REGRESSION EVALUATION
def evaluate_regression(name, y_true, predictions):
    mae = mean_absolute_error(y_true, predictions)
    mse = mean_squared_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)

    print(f"\n{name}")
    print("-" * 30)
    print("MAE:", round(mae, 2))
    print("MSE:", round(mse, 2))
    print("R2 Score:", round(r2, 2))

evaluate_regression("Linear Regression", y_test_r, linear_predictions)
evaluate_regression("Random Forest Regressor", y_test_r, forest_predictions)

# FEATURE IMPORTANCE
importance = forest_regressor.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance")
print(importance_df)

plt.figure(figsize=(8, 5))
plt.bar(importance_df["Feature"], importance_df["Importance"])
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("data/processed/feature_importance.png")
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(y_test_r.values[:50], label="Actual Cases")
plt.plot(forest_predictions[:50], label="Predicted Cases")
plt.title("Regression: Actual vs Predicted COVID Cases")
plt.xlabel("Sample")
plt.ylabel("Cases")
plt.legend()
plt.tight_layout()
plt.savefig("data/processed/regression_predictions.png")
plt.show()

# MULTICLASS CLASSIFICATION
print("\n")
print("=" * 50)
print("MULTICLASS CLASSIFICATION")
print("=" * 50)

y_class = df["risk_level"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X,
    y_class,
    test_size=0.2,
    random_state=42
)

# LOGISTIC REGRESSION
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_c, y_train_c)
logistic_predictions = logistic_model.predict(X_test_c)

# RANDOM FOREST CLASSIFIER
forest_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

forest_classifier.fit(X_train_c, y_train_c)
forest_class_predictions = forest_classifier.predict(X_test_c)

# CLASSIFICATION EVALUATION
def evaluate_classification(name, y_true, predictions):
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, average="weighted")
    recall = recall_score(y_true, predictions, average="weighted")
    f1 = f1_score(y_true, predictions, average="weighted")

    print(f"\n{name}")
    print("-" * 30)
    print("Accuracy:", round(accuracy, 2))
    print("Precision:", round(precision, 2))
    print("Recall:", round(recall, 2))
    print("F1 Score:", round(f1, 2))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_true, predictions))

    print("\nClassification Report")
    print(classification_report(y_true, predictions))

evaluate_classification("Logistic Regression", y_test_c, logistic_predictions)
evaluate_classification("Random Forest Classifier", y_test_c, forest_class_predictions)

# CONFUSION MATRIX GRAPH
cm = confusion_matrix(y_test_c, forest_class_predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("data/processed/confusion_matrix.png")
plt.show()

# SAVE MODELS
joblib.dump(forest_regressor, "models/covid_regression_model.pkl")
joblib.dump(forest_classifier, "models/covid_classification_model.pkl")

print("\nModels saved successfully")