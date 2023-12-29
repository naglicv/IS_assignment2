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

    return new_data


def get_data_w2v():
    new_data = get_data()[:1000]

    X_train, X_test, y_train, y_test = train_test_split(new_data[['clean_head', 'clean_desc', 'headline', 'short_description']], new_data['category'], test_size=0.2, random_state=42)

    X_train['clean_text'] = X_train['clean_head'] + ' ' + X_train['clean_head'] + ' ' + X_train['clean_desc']
    X_test['clean_text'] = X_test['clean_head'] + ' ' + X_test['clean_head'] + ' ' + X_test['clean_desc']
    
    tokenized_train_text = [text.split() for text in X_train['clean_text']]
    tokenized_test_text = [text.split() for text in X_test['clean_text']]

    w2v_model_train = Word2Vec(tokenized_train_text, vector_size=100, window=5, min_count=1, workers=4, epochs=10)
    w2v_model_test = Word2Vec(tokenized_test_text, vector_size=100, window=5, min_count=1, workers=4, epochs=10)

    all_words = w2v_model_train.wv.index_to_key
    word_vectors = {word: w2v_model_train.wv[word] for word in all_words}

    print(word_vectors)

    # Print the word vectors
    '''for word in word_vectors.items():
        print(f"Word: {word}")
        print(f"Vector: {vector}")
        print("\n")  # Add a newline for better readability

    similar_words = w2v_model_train.wv.most_similar('cat', topn=10)
    for word, similarity in similar_words:
        print(f"Similar word: {word}, Similarity: {similarity}")'''

if __name__ == "__main__":
    get_data_w2v()