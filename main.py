import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset
import json

import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

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

new_data = df[['headline', 'short_description', 'category']].copy()

# Replace None with a default string and apply function
new_data['clean_headline'] = new_data['headline'].fillna('').apply(preprocess_text)
new_data['clean_short_description'] = new_data['short_description'].fillna('').apply(preprocess_text)

#new_data['clean_desc'] = new_data['short_description'].apply(preprocess_text)
 
print(new_data[['clean_headline','headline', 'clean_short_description','short_description']][:10])