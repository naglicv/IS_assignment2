import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import models

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("./datasets/cleaned.csv")[:50000]

#print(df[['clean_head','headline', 'clean_desc','short_description']][:10])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['clean_head', 'clean_desc', 'headline', 'short_description']], df['category'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()

# Repeat the headline 2 times
X_train['clean_text'] = X_train['clean_head'] + ' ' + X_train['clean_head'] + ' ' + X_train['clean_desc']
X_test['clean_text'] = X_test['clean_head'] + ' ' + X_test['clean_head'] + ' ' + X_test['clean_desc']

X_train_vec = vectorizer.fit_transform(X_train['clean_text'].values.astype('U'))
X_test_vec = vectorizer.transform(X_test['clean_text'].values.astype('U'))

print(X_train_vec)


"""
# Random Forest Classifier
print("starting random forest")
rf_model = RandomForestClassifier()
rf_model.fit(X_train_vec, y_train)
rf_predictions = rf_model.predict(X_test_vec)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)"""

# Logistic Regression
print("starting Logistic Regression")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)
lr_predictions = lr_model.predict(X_test_vec)
lr_accuracy = accuracy_score(y_test, lr_predictions)
print("Logistic Regression Accuracy:", lr_accuracy)

print(classification_report(y_test, lr_predictions, zero_division=1))

scores = cross_val_score(lr_model, X_train_vec, y_train, cv=5)
print("Cross-Validation Scores:", scores)

'''
# XGBoosting
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['category'])
X = X_train_vec.append(X_test_vec, ignore_index=True)
X_t, X_ts, y_t, y_ts = train_test_split(X, y_encoded, test_size=0.2, random_state=8678686)

# XGBoost
model_xgb = XGBClassifier()
model_xgb.fit(X_t, y_t)
pred_xgb = model_xgb.predict(X_ts)
result_xgb = accuracy_score(y_ts, pred_xgb)
print("XGBoost Accuracy:", result_xgb)'''
