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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import time
from sklearn.utils import compute_class_weight
from scipy.stats import randint
import matplotlib.pyplot as plt


df = pd.read_csv("./datasets/cleaned.csv")
#df = df[80000:95000].copy()
df = df[:15000].copy()

#print(df[['clean_head','headline', 'clean_desc','short_description']][:10])
# Split the data into features (X) and target variable (y)
X = df.drop('category', axis=1)
y = pd.Categorical(df['category'])
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['category'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train['clean_text'].values.astype('U'))
X_test_vec = vectorizer.transform(X_test['clean_text'].values.astype('U'))

print("start smote training")
#Apply SMOTE to the training data

smote = SMOTE(random_state=42, sampling_strategy='auto')
X_train_vec, y_train = smote.fit_resample(X_train_vec.toarray(), y_train)
print("end smote training")
print(len(X_train_vec))

category_counts = y_train.value_counts()

print(category_counts)

category_counts.plot(kind='bar')

plt.xlabel('Category')
plt.ylabel('Stevilo')
plt.title('Novice po kategorijah')

# Show the plot
plt.show()


