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
from sklearn.utils.class_weight import compute_class_weight



df = pd.read_csv("./datasets/cleaned.csv")
df = df[:10000].copy()

class_weights = compute_class_weight(class_weight ='balanced',
classes=np.unique(df['category']),y = df['category'])
class_weights.sort()

print(class_weights)

#print(df[['clean_head','headline', 'clean_desc','short_description']][:10])
# Split the data into features (X) and target variable (y)
X = df.drop('category', axis=1)
X['clean_text'] = X['clean_text'] + " " + X['clean_desc']
y = pd.Categorical(df['category'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train['clean_text'].values.astype('U'))
X_test_vec = vectorizer.transform(X_test['clean_text'].values.astype('U'))



model_rf = RandomForestClassifier(criterion='gini', n_estimators=150, random_state=42)
model_rf.fit(X_train_vec, y_train)
pred_rf = model_rf.predict(X_test_vec)
result_rf = accuracy_score(y_test, pred_rf)
print("Random Forest Accuracy:", result_rf)