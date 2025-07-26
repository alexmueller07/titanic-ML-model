import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# Load and split the dataset
# ===============================

# Load Titanic training dataset
titanic_data = pd.read_csv('data/train.csv')

from sklearn.model_selection import StratifiedShuffleSplit

# Perform stratified sampling based on Survived, Pclass, and Sex to preserve distribution
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

for train_indices, test_indicies in split.split(titanic_data, titanic_data[["Survived", "Pclass", "Sex"]]):
    strat_train_set = titanic_data.loc[train_indices]
    strat_test_set = titanic_data.loc[test_indicies]

# ===============================
# Preprocessing pipeline classes
# ===============================

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

# Custom transformer for imputing missing Age values
class AgeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        imputer = SimpleImputer(strategy="mean")
        X['Age'] = imputer.fit_transform(X[["Age"]])
        return X

from sklearn.preprocessing import OneHotEncoder

# Custom transformer to encode categorical features: Embarked and Sex
class FeatureEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        encoder = OneHotEncoder()

        # Encode Embarked
        matrix = encoder.fit_transform(X[["Embarked"]]).toarray()
        column_names = ["C", "S", "Q", "N"]
        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]

        # Encode Sex
        matrix = encoder.fit_transform(X[["Sex"]]).toarray()
        column_names = ["Female", "Male"]
        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]

        return X

# Custom transformer to drop unused or redundant features
class FeatureDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(["Embarked", "Name", "Ticket", "Cabin", "Sex", "N"], axis=1, errors="ignore")

# ===============================
# Build preprocessing pipeline
# ===============================

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("ageimputer", AgeImputer()),
    ("featureencoder", FeatureEncoder()),
    ("featuredropper", FeatureDropper())
])

# Apply transformations to the training set
strat_train_set = pipeline.fit_transform(strat_train_set)

# ===============================
# Prepare features and labels
# ===============================

from sklearn.preprocessing import StandardScaler

X = strat_train_set.drop(["Survived"], axis=1)
y = strat_train_set["Survived"]

# Standardize features
scaler = StandardScaler()
X_data = scaler.fit_transform(X)
y_data = y.to_numpy()

# ===============================
# Model training (Random Forest + Grid Search)
# ===============================

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

clf = RandomForestClassifier()

# Define hyperparameter grid
param_grid = [
    {
        "n_estimators": [10, 100, 200, 500, 750],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 3, 4]
    }
]

# Grid search with 3-fold cross-validation
grid_search = GridSearchCV(clf, param_grid, cv=3, scoring="accuracy", return_train_score=True)
grid_search.fit(X_data, y_data)

# Save the best model from grid search
final_clf = grid_search.best_estimator_

# ===============================
# Evaluate on stratified test set
# ===============================

# Apply transformations to the test set
strat_test_set = pipeline.fit_transform(strat_test_set)

X_test = strat_test_set.drop(["Survived"], axis=1)
y_test = strat_test_set["Survived"]

scaler = StandardScaler()
X_data_test = scaler.fit_transform(X_test)
y_data_test = y_test.to_numpy()

# Evaluate accuracy
final_clf.score(X_data_test, y_data_test)

# ===============================
# Retrain on full dataset
# ===============================

# Combine train + test sets for final model training
final_data = pipeline.fit_transform(titanic_data)

X_final = final_data.drop(["Survived"], axis=1)
y_final = final_data["Survived"]

scaler = StandardScaler()
X_data_final = scaler.fit_transform(X_final)
y_data_final = y_final.to_numpy()

prod_clf = RandomForestClassifier()

# Reuse the same hyperparameter grid
grid_search = GridSearchCV(prod_clf, param_grid, cv=3, scoring="accuracy", return_train_score=True)
grid_search.fit(X_data_final, y_data_final)

# Final trained model
prod_final_clf = grid_search.best_estimator_

# ===============================
# Predict on official test set
# ===============================

# Load Titanic test data for submission
titanic_test_data = pd.read_csv("data/test.csv")

# Apply pipeline
final_test_data = pipeline.fit_transform(titanic_test_data)

# Fill any remaining missing values
X_final_test = final_test_data.fillna(method="ffill")

# Standardize features
scaler = StandardScaler()
X_data_final_test = scaler.fit_transform(X_final_test)

# Predict survival
predictions = prod_final_clf.predict(X_data_final_test)

# ===============================
# Generate submission CSV
# ===============================

final_df = pd.DataFrame(titanic_test_data["PassengerId"])
final_df["Survived"] = predictions

# Save predictions to CSV
final_df.to_csv("data/predictions.csv", index=False)

# Display resulting DataFrame
print(final_df)

print("===== DONE =====")
