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
df = df[:10000].copy()

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
# Start the timer
start_time = time.time()
model_DT.fit(X_train_vec, y_train)
pred_DT = model_DT.predict(X_test_vec)
result_ca_DT = accuracy_score(y_test, pred_DT)
print("Decision Tree Accuracy:", result_ca_DT)
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
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"\tTime taken: {elapsed_time} seconds")

# K-Nearest Neighbors
model_KNN = KNeighborsClassifier(n_neighbors=2)
# Start the timer
start_time = time.time()
model_KNN.fit(X_train_vec, y_train)
pred_KNN = model_KNN.predict(X_test_vec)
result_ca_KNN = accuracy_score(y_test, pred_KNN)
print("K-Nearest Neighbors Accuracy:", result_ca_KNN)
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
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"\tTime taken: {elapsed_time} seconds")

# Logistic Regression
model_lr = LogisticRegression(max_iter=1000)
# Start the timer
start_time = time.time()
model_lr.fit(X_train_vec, y_train)
pred_lr = model_lr.predict(X_test_vec)
result_lr = accuracy_score(y_test, pred_lr)
print("Logistic Regression Accuracy:", result_lr)
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
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"\tTime taken: {elapsed_time} seconds")

# Weighted Voting 
# Start the timer
start_time = time.time()
pred_rf_prob = model_rf.predict_proba(X_test_vec)
pred_lr_prob = model_lr.predict_proba(X_test_vec)
pred_bagging_prob = model_bagging.predict_proba(X_test_vec)

weighted_rf_prob = result_rf * pred_rf_prob
weighted_lr_prob = result_lr * pred_lr_prob
weighted_bagging_prob = result_bagging * pred_bagging_prob

pred_prob = weighted_rf_prob + weighted_lr_prob + weighted_bagging_prob
predicted_labels = np.argmax(pred_prob, axis=1)

"""# Equal Weights
equal_weight = 1/3

weighted_DT_prob = equal_weight * pred_DT_prob
weighted_NB_prob = equal_weight * pred_NB_prob
weighted_KNN_prob = equal_weight * pred_KNN_prob

pred_prob = weighted_DT_prob + weighted_NB_prob + weighted_KNN_prob
predicted_labels = np.argmax(pred_prob, axis=1)"""

"""# Calculate total accuracy
total_accuracy = result_ca_DT + result_ca_NB + result_ca_KNN
total_accuracy = result_rf + result_lr + result_bagging

# Calculate weights
weight_rf = result_rf / total_accuracy
weight_lr = result_lr / total_accuracy
weight_bagging = result_bagging / total_accuracy

# Weighted Voting
pred_rf_prob = model_rf.predict_proba(X_test_vec)
pred_lr_prob = model_lr.predict_proba(X_test_vec)
pred_bagging_prob = model_bagging.predict_proba(X_test_vec)

weighted_rf_prob = weight_rf * pred_rf_prob
weighted_lr_prob = weight_lr * pred_lr_prob
weighted_bagging_prob = weight_bagging * pred_bagging_prob

pred_prob = weighted_rf_prob + weighted_lr_prob + weighted_bagging_prob
predicted_labels = np.argmax(pred_prob, axis=1)"""


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
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"\tTime taken: {elapsed_time} seconds")

# Label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=8678686)

X_train_vec = vectorizer.fit_transform(X_train['clean_text'].values.astype('U'))
X_test_vec = vectorizer.transform(X_test['clean_text'].values.astype('U'))

# XGBoost

"""
model_xgb = XGBClassifier()

param_grid = {
   'n_estimators': [100],
   'max_depth': [50, 100],
   'learning_rate': [0.01, 0.1, 0.2],
   'subsample': [0.5, 0.7, 1],
   'colsample_bytree': [0.5, 0.7, 1],
   'gamma': [0, 0.1, 0.2],
   'reg_lambda': [1, 10, 100],
}

grid_search = GridSearchCV(estimator=model_xgb, param_grid=param_grid, cv=3)
grid_search.fit(X_train_vec, y_train)
best_params = grid_search.best_params_
print("Best parameters: ", best_params)
print("Best score: ", grid_search.best_score_)
# Start the timer
start_time = time.time()
#model_xgb = XGBClassifier(colsample_bytree=0.7, gamma=0.1, learning_rate=0.1, max_depth=5, min_child_weight=1, subsample=0.7)
model_xgb = XGBClassifier(**grid_search.best_params_)
model_xgb.fit(X_train_vec, y_train)
pred_xgb = model_xgb.predict(X_test_vec)
result_xgb = accuracy_score(y_test, pred_xgb)
print("XGBoost Accuracy (grid search):", result_xgb)
# Stop the timer
end_time = time.time()


param_dist = {
   'n_estimators': randint(100, 500),
   'max_depth': randint(3, 9),
   'learning_rate': [0.01, 0.1, 0.2],
   'subsample': [0.5, 0.7, 1],
   'colsample_bytree': [0.5, 0.7, 1],
   'gamma': [0, 0.1, 0.2],
   'reg_lambda': [1, 10, 100],
}


model_xgb = XGBClassifier()

random_search = RandomizedSearchCV(estimator=model_xgb, param_distributions=param_dist, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(X_train_vec, y_train)
print("Best parameters: ", random_search.best_params_)
print("Best score: ", random_search.best_score_)


# Start the timer
start_time = time.time()
#model_xgb = XGBClassifier(colsample_bytree=0.7, gamma=0.1, learning_rate=0.1, max_depth=5, min_child_weight=1, subsample=0.7)
model_xgb = XGBClassifier(**random_search.best_params_)
model_xgb.fit(X_train_vec, y_train)
pred_xgb = model_xgb.predict(X_test_vec)
result_xgb = accuracy_score(y_test, pred_xgb)
print("XGBoost Accuracy (randomized search):", result_xgb)
# Stop the timer
end_time = time.time()
"""

# Start the timer
start_time = time.time()
model_xgb = XGBClassifier()
model_xgb.fit(X_train_vec, y_train)
pred_xgb = model_xgb.predict(X_test_vec)
result_xgb = accuracy_score(y_test, pred_xgb)
print("XGBoost Accuracy:", result_xgb)
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
    "Weighted Voting": {"model": None, "prediction": predicted_labels, "performance": result_wvoting, "is_ensemble": True},
    "Bagging": {"model": model_bagging, "prediction": pred_bagging, "performance": result_bagging, "is_ensemble": True},
    "Random Forest": {"model": model_rf, "prediction": pred_rf, "performance": result_rf, "is_ensemble": True},
    "Boosting": {"model": model_boosting, "prediction": pred_boosting, "performance": result_boosting, "is_ensemble": True},
    "XGBoost": {"model": model_xgb, "prediction": pred_xgb, "performance": result_xgb, "is_ensemble": False}
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

# Calculate cross-validation scores for the best model
# Note: For weighted voting, cross-validation is not directly applicable as it involves combining predictions
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
