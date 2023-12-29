import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_text(text):
    # Tokenize the text into words
    #print(text)
    words = word_tokenize(text.lower())  # Convert text to lowercase

    # Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    words = [word.translate(table) for word in words if word.isalpha()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    # Stemming (uncomment if you want to use stemming)
    stemmed_words = [stemmer.stem(word) for word in words]

    # Join the words back into a string
    preprocessed_text = ' '.join(lemmatized_words)
    return preprocessed_text

df = pd.read_json("./datasets/News_Category_Dataset_IS_course.json", lines=True)

new_data = df[['headline', 'short_description', 'category']][:50000].copy()

# Replace None with a default string and apply function
new_data['clean_head'] = new_data['headline'].fillna('').apply(preprocess_text)
new_data['clean_desc'] = new_data['short_description'].fillna('').apply(preprocess_text)
 
print(new_data[['clean_head','headline', 'clean_desc','short_description']][:10])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(new_data[['clean_head', 'clean_desc', 'headline', 'short_description']], new_data['category'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train['clean_desc'])
X_test_vec = vectorizer.transform(X_test['clean_desc'])

# Random Forest Classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train_vec, y_train)
rf_predictions = rf_model.predict(X_test_vec)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)
lr_predictions = lr_model.predict(X_test_vec)
lr_accuracy = accuracy_score(y_test, lr_predictions)
print("Logistic Regression Accuracy:", lr_accuracy)

print(classification_report(y_test, lr_predictions))

scores = cross_val_score(lr_model, X_train_vec, y_train, cv=5)
print("Cross-Validation Scores:", scores)
