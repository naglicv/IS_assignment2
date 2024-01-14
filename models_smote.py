import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score, GridSearchCV 
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
from imblearn.over_sampling import SMOTE
import time
from sklearn.utils import compute_class_weight
from scipy.stats import randint

df = pd.read_csv("./datasets/cleaned.csv")
df = df[:15000].copy()

X = df.drop('category', axis=1)
y = pd.Categorical(df['category'])

# Label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['category'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train['clean_text'].values.astype('U'))
X_test_vec = vectorizer.transform(X_test['clean_text'].values.astype('U'))

# Apply SMOTE to the training data
smote = SMOTE(random_state=42, sampling_strategy='auto', n_jobs=-1)
X_train_vec, y_train = smote.fit_resample(X_train_vec.toarray(), y_train)

# Decision Tree
model_DT = DecisionTreeClassifier(random_state=42)
# Start the timer
start_time = time.time()
model_DT.fit(X_train_vec, y_train)
pred_DT = model_DT.predict(X_test_vec)
result_ca_DT = accuracy_score(y_test, pred_DT)
print("Decision Tree Accuracy:", result_ca_DT)

print("\n", classification_report(y_test, pred_DT, zero_division=1))
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"\tTime taken: {elapsed_time} seconds")

# Naive Bayes
model_NB = MultinomialNB()
# Start the timer
start_time = time.time()
model_NB.fit(X_train_vec, y_train)
pred_NB = model_NB.predict(X_test_vec)
result_ca_NB = accuracy_score(y_test, pred_NB)
print("Naive Bayes Accuracy:", result_ca_NB)

print("\n", classification_report(y_test, pred_NB, zero_division=1))
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"\tTime taken: {elapsed_time} seconds")

# K-Nearest Neighbors
model_KNN = KNeighborsClassifier(n_neighbors=9)
# Start the timer
start_time = time.time()
model_KNN.fit(X_train_vec, y_train)
pred_KNN = model_KNN.predict(X_test_vec)
result_ca_KNN = accuracy_score(y_test, pred_KNN)
print("K-Nearest Neighbors Accuracy:", result_ca_KNN)

print("\n", classification_report(y_test, pred_KNN, zero_division=1))
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"\tTime taken: {elapsed_time} seconds")

# Bagging
model_bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=14, random_state=8678686)
# Start the timer
start_time = time.time()
model_bagging.fit(X_train_vec, y_train)
pred_bagging = model_bagging.predict(X_test_vec)
result_bagging = accuracy_score(y_test, pred_bagging)
print("Bagging Accuracy:", result_bagging)

print("\n", classification_report(y_test, pred_bagging, zero_division=1))
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"\tTime taken: {elapsed_time} seconds")

# Random Forest
model_rf = RandomForestClassifier(random_state=8678686)
# Start the timer
start_time = time.time()
model_rf.fit(X_train_vec, y_train)
pred_rf = model_rf.predict(X_test_vec)
result_rf = accuracy_score(y_test, pred_rf)
print("Random Forest Accuracy:", result_rf)

print("\n", classification_report(y_test, pred_rf, zero_division=1))
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"\tTime taken: {elapsed_time} seconds")

# Logistic Regression
model_lr = LogisticRegression(max_iter=1000, random_state=42)
# Start the timer
start_time = time.time()
model_lr.fit(X_train_vec, y_train)
pred_lr = model_lr.predict(X_test_vec)
result_lr = accuracy_score(y_test, pred_lr)
print("Logistic Regression Accuracy:", result_lr)

print("\n", classification_report(y_test, pred_lr, zero_division=1))
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"\tTime taken: {elapsed_time} seconds")

# Boosting
model_boosting = AdaBoostClassifier(estimator=DecisionTreeClassifier(), random_state=8678686)
# Start the timer
start_time = time.time()
model_boosting.fit(X_train_vec, y_train)
pred_boosting = model_boosting.predict(X_test_vec)
result_boosting = accuracy_score(y_test, pred_boosting)
print("Boosting Accuracy:", result_boosting)

print("\n", classification_report(y_test, pred_boosting, zero_division=1))
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"\tTime taken: {elapsed_time} seconds")

# Hard Voting
model_voting_hard = VotingClassifier(estimators=[('rf', model_rf), ('lr', model_lr), ('bg', model_bagging)], voting='hard')
# Start the timer
start_time = time.time()
model_voting_hard.fit(X_train_vec, y_train)
pred_voting_hard = model_voting_hard.predict(X_test_vec)
result_voting_hard = accuracy_score(y_test, pred_voting_hard)
print("Hard Voting Accuracy:", result_voting_hard)

print("\n", classification_report(y_test, pred_voting_hard, zero_division=1))
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"\tTime taken: {elapsed_time} seconds")

# Soft Voting
model_voting_soft = VotingClassifier(estimators=[('rf', model_rf), ('lr', model_lr), ('bg', model_bagging)], voting='soft')
# Start the timer
start_time = time.time()
model_voting_soft.fit(X_train_vec, y_train)
pred_voting_soft = model_voting_soft.predict(X_test_vec)
result_voting_soft = accuracy_score(y_test, pred_voting_soft)
print("Soft Voting Accuracy:", result_voting_soft)

print("\n", classification_report(y_test, pred_voting_soft, zero_division=1))
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"\tTime taken: {elapsed_time} seconds")

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

# Start the timer
start_time = time.time()
model_xgb = XGBClassifier()
model_xgb.fit(X_train_vec, y_train)
pred_xgb = model_xgb.predict(X_test_vec)
result_xgb = accuracy_score(y_test, pred_xgb)
print("XGBoost Accuracy:", result_xgb)

print("\n", classification_report(y_test, pred_xgb, zero_division=1))
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"\tTime taken: {elapsed_time} seconds")


# Store models, their predictions, and their performances in a dictionary
models = {
    "Decision Tree": {"model": model_DT, "prediction": pred_DT, "performance": result_ca_DT, "is_ensemble": False},
    "Naive Bayes": {"model": model_NB, "prediction": pred_NB, "performance": result_ca_NB, "is_ensemble": False},
    "K-Nearest Neighbors": {"model": model_KNN, "prediction": pred_KNN, "performance": result_ca_KNN, "is_ensemble": False},
    "Logistic Regression": {"model": model_lr, "prediction": pred_lr, "performance": result_lr, "is_ensemble": False},
    "Hard Voting": {"model": model_voting_hard, "prediction": pred_voting_hard, "performance": result_voting_hard, "is_ensemble": True},
    "Soft Voting": {"model": model_voting_soft, "prediction": pred_voting_soft, "performance": result_voting_soft, "is_ensemble": True},
    "Bagging": {"model": model_bagging, "prediction": pred_bagging, "performance": result_bagging, "is_ensemble": True},
    "Random Forest": {"model": model_rf, "prediction": pred_rf, "performance": result_rf, "is_ensemble": True},
    "Boosting": {"model": model_boosting, "prediction": pred_boosting, "performance": result_boosting, "is_ensemble": True},
    "XGBoost": {"model": model_xgb, "prediction": pred_xgb, "performance": result_xgb, "is_ensemble": True}
}

# Find the index of the maximum performance
max_performance_index = max(models, key=lambda x: models[x]['performance'])

# Get the best model's name, model, and predictions
best_model_name = max_performance_index
best_model = models[best_model_name]['model']
best_predictions = models[best_model_name]['prediction']

# Reverse the class_conv dictionary
rev_class_conv = {v: k for k, v in class_conv.items()}

# Convert best_predictions to integers using the reversed class_conv dictionary
best_predictions = [rev_class_conv[prediction] for prediction in best_predictions]

# Print the model with the highest accuracy
print(f"\nThe model with the highest accuracy is: {best_model_name}\n")

# Print the classification report for the best model
print(classification_report(y_test, best_predictions, zero_division=1))

# Sort models by performance
sorted_models = sorted(models.items(), key=lambda x: x[1]['performance'], reverse=True)

# Calculate cross-validation scores for the best model
if best_model is not None:
    scores = cross_val_score(best_model, X_train_vec, y_train, cv=5)
    print(" -> Cross-Validation Scores:", scores)
       
# Visualization
result_df = pd.DataFrame([(name, model['performance'], model['is_ensemble']) for name, model in models.items()], columns=['Algorithm', 'Performance', 'Ensemble Model'])
result_df = result_df.sort_values(by='Performance', ascending=False)

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x='Performance', y='Algorithm', hue='Ensemble Model', data=result_df, dodge=False, palette={0: 'red', 1: 'green'})
plt.xlabel('Accuracy')
plt.ylabel('Algorithm') 
plt.title('Performance Comparison')
plt.legend(title='Ensemble Model', loc='lower right')
plt.show()