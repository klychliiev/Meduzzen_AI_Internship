import pandas as pd 
import os 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from text_cleaning import fetch_body, preprocess_text

SHORT_EMAIL_DATASET_PATH = "short_dataset.csv"

def process_dataset():
    df = pd.read_csv(SHORT_EMAIL_DATASET_PATH)
    df = df.drop('Unnamed: 0', axis=1)
    df = df.sample(frac=1)
    df.reset_index(drop=True, inplace=True)
    df['Email_body'] = df['Content'].apply(fetch_body)
    df['Clean_texts'] = df['Email_body'].apply(preprocess_text)
    return df 

def prepare_classifier(df):
    X_train, X_test, y_train, y_test = train_test_split(df['Email_body'], df['Category'], random_state = 0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    linear_svc = LinearSVC()
    clf = linear_svc.fit(X_train_tfidf, y_train)
    return clf, count_vect

df = process_dataset()
clf, count_vect = prepare_classifier(df)
