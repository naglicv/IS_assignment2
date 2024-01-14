import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./datasets/cleaned.csv")
df = df[:15000].copy()
#len = df.clean_text.map(len).max()
#print(len)

X = df.drop('category', axis=1)
#X['clean_text'] = X['clean_text'] + " " + X['clean_desc']
y = pd.Categorical(df['category'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train['clean_text'].values.astype('U'))
X_test_vec = vectorizer.transform(X_test['clean_text'].values.astype('U'))

# Decision Tree


#param_grid = {'n_estimators': [50, 60]}  # Adjust the values as needed
param_grid = { 'n_neighbors': range(5, 30)}
#base_estimator = KNeighborsClassifier(n_neighbors=9)  # You can use another base estimator if needed
#
#bagging_model = BaggingClassifier(base_estimator=base_estimator)
knnnn = KNeighborsClassifier()
grid_search = GridSearchCV(knnnn, param_grid, cv=4, scoring='accuracy')  # You can adjust the number of folds (cv) as needed
grid_search.fit(X_train_vec, y_train)  # Assuming X_train and y_train are your training data

best_bagging_model = grid_search.best_estimator_

accuracy = best_bagging_model.score(X_test_vec, y_test)  # Assuming X_test and y_test are your test data
print(f"Best n_estimators: {grid_search.best_params_}, Best Model Accuracy: {accuracy}")

