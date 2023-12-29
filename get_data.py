import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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

def get_data():
    df = pd.read_json("./datasets/News_Category_Dataset_IS_course.json", lines=True)

    new_data = df[['headline', 'short_description', 'category']].copy()

    # Replace None with a default string and apply function
    new_data['clean_head'] = new_data['headline'].fillna('').apply(preprocess_text)
    new_data['clean_desc'] = new_data['short_description'].fillna('').apply(preprocess_text)

    new_data['clean_text'] = new_data['clean_head'] + ' ' + new_data['clean_head'] + ' ' + new_data['clean_desc']
    #print(new_data[['clean_head','headline', 'clean_desc','short_description']][:10])
    return new_data

def get_average_w2v(tokens, w2v_model):
        vector_sum = 0
        count = 0
        for word in tokens:
            if word in w2v_model.wv:
                vector_sum += w2v_model.wv[word]
                count += 1
        if count != 0:
            return vector_sum / count
        else:
            return [0] * 100  # Return zero vector if no word found


def get_data_w2v():
    new_data = get_data()

    X_train, X_test, y_train, y_test = train_test_split(new_data[['clean_head', 'clean_desc', 'headline', 'short_description']], new_data['category'], test_size=0.2, random_state=42)

    X_train['clean_text'] = X_train['clean_head'] + ' ' + X_train['clean_head'] + ' ' + X_train['clean_desc']
    X_test['clean_text'] = X_test['clean_head'] + ' ' + X_test['clean_head'] + ' ' + X_test['clean_desc']
    
    tokenized_train_text = [text.split() for text in X_train['clean_text']]
    tokenized_test_text = [text.split() for text in X_test['clean_text']]

    w2v_model_train = Word2Vec(tokenized_train_text, vector_size=100, window=5, min_count=1, workers=4, epochs=10)
    w2v_model_test = Word2Vec(tokenized_test_text, vector_size=100, window=5, min_count=1, workers=4, epochs=10)

    # Add Word2Vec representations to DataFrame
    X_train_w2v = [get_average_w2v(text, w2v_model_train) for text in tokenized_train_text]
    X_test_w2v = [get_average_w2v(text, w2v_model_test) for text in tokenized_test_text]

    # Logistic Regression model
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train_w2v, y_train)
    logistic_predictions = logistic_model.predict(X_test_w2v)
    logistic_accuracy = accuracy_score(y_test, logistic_predictions)
    print("Logistic Regression Accuracy:", logistic_accuracy)

if __name__ == "__main__":
    #get_data_w2v()
    get_data().to_csv('cleaned.csv', index=False)