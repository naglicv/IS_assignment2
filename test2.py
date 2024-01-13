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
df = df[:10000].copy()
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


'''dt = DecisionTreeClassifier(random_state=42)
params = {
    'max_depth': [10, 50, 100, 150],
    'min_samples_split': [20, 30, 50, 100],
    'criterion': ['gini', 'entropy']
}
grid_search = GridSearchCV(estimator=dt, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")
model_DT = DecisionTreeClassifier()
grid_search.fit(X_train_vec, y_train)
print("Decision Tree Accuracy:", grid_search.best_estimator_)'''

model_DT = DecisionTreeClassifier(max_depth=10, min_samples_split=50, random_state=42)
model_DT.fit(X_train_vec, y_train)
pred_DT = model_DT.predict(X_test_vec)
result_ca_DT = accuracy_score(y_test, pred_DT)
print("Decision Tree Accuracy:", result_ca_DT)