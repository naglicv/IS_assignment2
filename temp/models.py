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
df = df[:25000].copy()

#print(df[['clean_head','headline', 'clean_desc','short_description']][:10])
# Split the data into features (X) and target variable (y)
X = df.drop('category', axis=1)
y = pd.Categorical(df['category'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train['clean_text'].values.astype('U'))
X_test_vec = vectorizer.transform(X_test['clean_text'].values.astype('U'))

# Decision Tree
model_DT = DecisionTreeClassifier()
model_DT.fit(X_train_vec, y_train)
pred_DT = model_DT.predict(X_test_vec)
result_ca_DT = accuracy_score(y_test, pred_DT)
print("Decision Tree Accuracy:", result_ca_DT)

# Naive Bayes
model_NB = MultinomialNB()
model_NB.fit(X_train_vec, y_train)
pred_NB = model_NB.predict(X_test_vec)
result_ca_NB = accuracy_score(y_test, pred_NB)
print("Naive Bayes Accuracy:", result_ca_NB)

# K-Nearest Neighbors
model_KNN = KNeighborsClassifier(n_neighbors=5)
model_KNN.fit(X_train_vec, y_train)
pred_KNN = model_KNN.predict(X_test_vec)
result_ca_KNN = accuracy_score(y_test, pred_KNN)
print("K-Nearest Neighbors Accuracy:", result_ca_KNN)

# Hard Voting
model_voting = VotingClassifier(estimators=[('dt', model_DT), ('nb', model_NB), ('knn', model_KNN)], voting='hard')
model_voting.fit(X_train_vec, y_train)
pred_voting = model_voting.predict(X_test_vec)
result_voting = accuracy_score(y_test, pred_voting)
print("Hard Voting Accuracy:", result_voting)

# Soft Voting
model_voting = VotingClassifier(estimators=[('dt', model_DT), ('nb', model_NB), ('knn', model_KNN)], voting='soft')
model_voting.fit(X_train_vec, y_train)
pred_voting = model_voting.predict(X_test_vec)
result_voting = accuracy_score(y_test, pred_voting)
print("Soft Voting Accuracy:", result_voting)

# Weighted Voting 
pred_DT_prob = model_DT.predict_proba(X_test_vec)
pred_NB_prob = model_NB.predict_proba(X_test_vec)
pred_KNN_prob = model_KNN.predict_proba(X_test_vec)

weighted_DT_prob = result_ca_DT * pred_DT_prob
weighted_NB_prob = result_ca_NB * pred_NB_prob
weighted_KNN_prob = result_ca_KNN * pred_KNN_prob

pred_prob = weighted_DT_prob + weighted_NB_prob + weighted_KNN_prob
predicted_labels = np.argmax(pred_prob, axis=1)

# int2class conversion dict
class_conv = {
    0: 'BLACK VOICES',
    1: 'BUSINESS',
    2: 'COMEDY',
    3: 'ENTERTAINMENT',
    4: 'FOOD & DRINK',
    5: 'HEALTHY LIVING',
    6: 'HOME & LIVING',
    7: 'PARENTING',
    8: 'PARENTS',
    9: 'POLITICS',
    10: 'QUEER VOICES',
    11: 'SPORTS',
    12: 'STYLE & BEAUTY',
    13: 'TRAVEL',
    14: 'WELLNESS'
}

# Evaluate
predicted_labels = [class_conv[i] for i in predicted_labels]
result_wvoting = accuracy_score(y_test, predicted_labels)
print("Weighted Voting Accuracy:", result_wvoting)

# Bagging
model_bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=30, random_state=42)
model_bagging.fit(X_train_vec, y_train)
pred_bagging = model_bagging.predict(X_test_vec)
result_bagging = accuracy_score(y_test, pred_bagging)
print("Bagging Accuracy:", result_bagging)

# Random Forest
model_rf = RandomForestClassifier(random_state=8678686)
model_rf.fit(X_train_vec, y_train)
pred_rf = model_rf.predict(X_test_vec)
result_rf = accuracy_score(y_test, pred_rf)
print("Random Forest Accuracy:", result_rf)

model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train_vec, y_train)
pred_lr = model_lr.predict(X_test_vec)
result_lr = accuracy_score(y_test, pred_lr)
print("Logistic Regression Accuracy:", result_lr)

# Boosting
model_boosting = AdaBoostClassifier(estimator=DecisionTreeClassifier(), random_state=4242345)
model_boosting.fit(X_train_vec, y_train)
pred_boosting = model_boosting.predict(X_test_vec)
result_boosting = accuracy_score(y_test, pred_boosting)
print("Boosting Accuracy:", result_boosting)

# Label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=8678686)

X_train_vec = vectorizer.fit_transform(X_train['clean_text'].values.astype('U'))
X_test_vec = vectorizer.transform(X_test['clean_text'].values.astype('U'))

# XGBoost
model_xgb = XGBClassifier()
model_xgb.fit(X_train_vec, y_train)
pred_xgb = model_xgb.predict(X_test_vec)
result_xgb = accuracy_score(y_test, pred_xgb)
print("XGBoost Accuracy:", result_xgb)

# Visualization
algo_names = ["Decision Tree", "Naive Bayes", "K-Nearest Neighbors", "Voting", "Weighted Voting", "Bagging", "Random Forest", "Logistic Regression", "Boosting", "XGBoost"]
performances = [result_ca_DT, result_ca_NB, result_ca_KNN, result_voting, result_wvoting, result_bagging, result_rf, result_lr, result_boosting, result_xgb]
ensemble_model = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

result_df = pd.DataFrame({'Algorithm': algo_names, 'Performance': performances, 'Ensemble Model': ensemble_model})
result_df = result_df.sort_values(by='Performance', ascending=False)

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x='Performance', y='Algorithm', hue='Ensemble Model', data=result_df, dodge=False, palette={0: 'red', 1: 'green'})
plt.xlabel('Accuracy')
plt.ylabel('Algorithm') 
plt.title('Performance Comparison')
plt.legend(title='Ensemble Model', loc='lower right')
plt.show()
