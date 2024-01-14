import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

df = pd.read_csv("./datasets/cleaned.csv")
df = df[80000:95000].copy()

X = df.drop('category', axis=1)
y = pd.Categorical(df['category'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train['clean_text'].values.astype('U'))
X_test_vec = vectorizer.transform(X_test['clean_text'].values.astype('U'))

# Hyper-parameter tuning for Logistic Regression
# Define the hyperparameter space
param_grid = {
   'C': [0.001, 0.01, 0.1, 1, 10, 100],
   'penalty': ['l1', 'l2', 'elasticnet'],
   'solver': ['newton-cg', 'liblinear', 'sag', 'saga'],
   'max_iter': [1000, 10000],
   'random_state': [42]
}

# Create a base model
lr = LogisticRegression()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5)

# Fit the grid search to the data
grid_search.fit(X_train_vec, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Logistic Regression
model_lr = LogisticRegression(**best_params)
model_lr.fit(X_train_vec, y_train)
pred_lr = model_lr.predict(X_test_vec)
result_lr = accuracy_score(y_test, pred_lr)
print("Logistic Regression Accuracy:", result_lr)

print("\n", classification_report(y_test, pred_lr, zero_division=1))
