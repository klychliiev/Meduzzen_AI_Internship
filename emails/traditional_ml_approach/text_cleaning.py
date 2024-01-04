import re 
import spacy 

de_nlp = spacy.load("de_core_news_sm")

# function to fetch the email body 
def fetch_body(text):
    my_text = text.split('Nachricht      : ')[-1]
    text =  my_text.split('--')[0]
    return text.replace('\n', ' ')


# function to preprocess emails (text normalization)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'^a-zäöüß0-9\s', '', text)
    text = re.sub(r'\s', ' ', text).strip()
    doc = de_nlp(text)
    no_stopwords_sent = [str(word.lemma_).lower() for word in doc if not word.is_stop]
    text = ' '.join(no_stopwords_sent)
    text = re.sub('--', '', text.replace('\n', ''))
    text = re.sub(' +', ' ', text)
    return text  


