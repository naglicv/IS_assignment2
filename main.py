import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset
import json

df = pd.read_json("./datasets/News_Category_Dataset_IS_course.json", lines=True)
 
print(df[:10]['date'].to_string()) 

