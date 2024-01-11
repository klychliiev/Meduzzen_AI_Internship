import pandas as pd 
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from text_cleaning import fetch_body, preprocess_text

dataset_path = "short_dataset.csv"

def process_dataset():
    """
    Reads dataset stored in csv format using pandas, converts it to pandas DataFrame,
    Drops unnecessary columns, shuffles df and applies functions for email body extraction
    (fetch_body) and text preprocessing (preprocess_text).
    
    Returns: 
        df (pd.DataFrame): DataFrame containing preprocessed emails.
    
    """
    df = pd.read_csv(dataset_path)
    df = df.drop('Unnamed: 0', axis=1)
    df = df.sample(frac=1)
    df.reset_index(drop=True, inplace=True)
    df['Email_body'] = df['Content'].apply(fetch_body)
    df['Clean_texts'] = df['Email_body'].apply(preprocess_text)
    return df 

def prepare_classifier(df: pd.DataFrame):
    """
    Prepares data for classification and defines LinearSVC classifier. 
    
    Args:
        df (pd.DataFrame): input data stored as a pandas.DataFrame.

    Returns:
        clf object (sklearn.svm._classes.LinearSVC): LinearSVC classifier for email category prediction
        count_vect object (sklearn.feature_extraction.text.CountVectorizer): 
            numerical representation of text for ML

    """
    X_train, X_test, y_train, y_test = train_test_split(df['Clean_texts'], df['Category'], random_state = 0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    linear_svc = LinearSVC()
    clf = linear_svc.fit(X_train_tfidf, y_train)
    return clf, count_vect

df = process_dataset()
clf, count_vect = prepare_classifier(df)
