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

category_counts = df['category'].value_counts()

print(sum(category_counts[0:3])/len(df))

#print(category_counts)

#category_counts.plot(kind='bar')

#plt.xlabel('Category')
#plt.ylabel('Stevilo')
#plt.title('Novice po kategorijah')
#print(df)



#print(df[['clean_head','headline', 'clean_desc','short_description']][:10])
#Split the data into features (X) and target variable (y)
#cleaned = df.groupby('category').head(1000).reset_index(drop=True)

#cleaned.to_csv('./datasets/1000.csv', index=False)
#print(cleaned)


a = 3955