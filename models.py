from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

vectorizer = TfidfVectorizer()

# Random forest classificator
def random_forest(X_train_vec, X_test_vec, y_train, y_test):
    print("starting random forest")
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train_vec, y_train)
    rf_predictions = rf_model.predict(X_test_vec)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    print("Random Forest Accuracy:", rf_accuracy)
    return rf_accuracy

# XGBoosting
def XGB(X_train, X_test, y_train, y_test):
    # XGBoost
    model_xgb = XGBClassifier()
    model_xgb.fit(X_train, y_train)
    pred_xgb = model_xgb.predict(X_test)
    result_xgb = accuracy_score(y_test, pred_xgb)
    print("XGBoost Accuracy:", result_xgb)
    return result_xgb


