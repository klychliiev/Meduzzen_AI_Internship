import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings("ignore")


dir1 = "emails_short_ds"

my_dict = {
    'Category':[],
    'Email':[],
    'Content':[]
}

for folder in os.listdir(dir1):

    # print(folder)

    clean_folder = folder.split('_')[1]

    fold = os.path.join(dir1, folder)

    for file in os.listdir(fold):
        # check if it's a folder
        file_item = os.path.join(fold, file)
        if os.path.isdir(file_item):
            for f in os.listdir(file_item):
                if f.endswith('.TXT'):
                    # print(str(f))

                    cont = os.path.join(file_item, f)
                    with open(cont, 'r') as f:
                        cont = f.read()
                        # print(cont)

                    my_dict['Category'].append(clean_folder)
                    my_dict['Email'].append(str(f))
                    my_dict['Content'].append(cont)

        # check if it's a single .txt email
        if file.endswith('.TXT'):
            cont = os.path.join(fold, file)
            with open(cont, 'r') as f:
                cont = f.read()
        
                my_dict['Category'].append(clean_folder)
                my_dict['Email'].append(str(f))
                my_dict['Content'].append(cont)
                
            # print(str(file))
            # my_dict[clean_folder].append(file)

df = pd.DataFrame(my_dict)

shuffled_df = df.sample(frac=1)

def fetch_body(text):

    my_text = text.split('Nachricht      : ')[-1]

    return my_text.split('--')[0]

shuffled_df['Email_body'] = shuffled_df['Content'].apply(fetch_body)

categories_list = list(shuffled_df['Category'].unique())

label_encoder = LabelEncoder()

shuffled_df['Category_ID'] = label_encoder.fit_transform(shuffled_df['Category'])


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))
features = tfidf.fit_transform(shuffled_df.Email_body).toarray()
labels = shuffled_df.Category

X_train, X_test, y_train, y_test = train_test_split(shuffled_df['Email_body'], shuffled_df['Category'], random_state = 0)
count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

print(cv_df.groupby('model_name').accuracy.mean())
